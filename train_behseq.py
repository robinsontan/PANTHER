

"""
Main entry point for model training. Please refer to README.md for usage instructions.

python train_behseq.py --gin_config_file=...... --master_port=12345 --debug
"""
import logging
import os
import random
from collections import OrderedDict
from datetime import date, datetime
from typing import Optional

import polars
import torch.nn.functional as F
from glom import glom
from tqdm import tqdm

# Experiment tracking tools
from trainer.utils import (
    init_aim_run,
    init_mlflow_run,
    log_csv,
    log_metrics,
    log_params,
    log_parquet,
    log_state_dict,
    update_records,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Hide excessive tensorflow debug messages
os.environ["MLFLOW_TRACKING_URI"] = "./mlruns/"
# os.environ["MLFLOW_TRACKING_URI"]="http://localhost:5000"
# mlflow.set_tracking_uri('http://localhost:5000')

import sys
import time

import fbgemm_gpu  # noqa: F401, E402
import gin
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from absl import app, flags
from torch.nn.parallel import DistributedDataParallel as DDP

from data.eval import (
    _avg,
)
from eval_tool import *
from modeling.sequential.autoregressive_losses import (
    SampledSoftmaxLoss,
)
from modeling.sequential.embedding_modules import (
    ConcatenateEmbeddingModule,
    EmbeddingModule,
)
from modeling.sequential.encoder_utils import get_sequential_encoder
from modeling.sequential.input_features_preprocessors import (
    LearnablePositionalEmbeddingInputFeaturesPreprocessor,
)
from modeling.sequential.output_postprocessors import (
    L2NormEmbeddingPostprocessor,
    LayerNormEmbeddingPostprocessor,
)
from modeling.sequential.regularization_losses import ContrastiveRegularizationLoss
from modeling.similarity_utils import get_similarity_function

logging.basicConfig(
    # stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s: %(relativeCreated)d: %(levelname)s: %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S',
)


flags.DEFINE_string("gin_config_file", None, "Path to the config file.")
flags.DEFINE_integer("master_port", 12347, "Master port.")
flags.DEFINE_boolean("debug", False, "Master port.")
flags.DEFINE_boolean("inference_mode", False, "Master port.")
flags.DEFINE_string("model_save_path",None, "model_save_path.")

FLAGS = flags.FLAGS

abbreviate_number = lambda n: next(
    f'{n/10**(3*i):.1f}{["","K","M","B","T"][i]}'
    for i in range(5)
    if n < 10 ** (3 * (i + 1))
)


def setup(rank: int, world_size: int, master_port: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


@gin.configurable
def get_lr_scheduler_wsd(
    opt,
    factor=1.0,
    warmup_steps=1e4,  # 10k
    stable_steps=8.75e4,  # 87.5K
    decay_steps=1.25e4,  # 12.5K
):
    from torch.optim.lr_scheduler import ConstantLR, LinearLR
    warmup_steps, stable_steps, decay_steps = int(warmup_steps * factor), int(stable_steps * factor), int(decay_steps * factor)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        opt,
        schedulers=[
            LinearLR(
                opt,
                start_factor=1 / warmup_steps,
                end_factor=1,
                total_iters=warmup_steps,
            ),
            ConstantLR(opt, factor=1, total_iters=stable_steps),
            LinearLR(opt, start_factor=1, end_factor=0.1, total_iters=decay_steps),
            ConstantLR(opt, factor=0.1, total_iters=stable_steps),
        ],
        milestones=[
            warmup_steps,
            stable_steps + warmup_steps,
            stable_steps + warmup_steps + decay_steps,
        ],
    )
    return scheduler


@gin.configurable
def train_fn(
    rank: int,
    world_size: int,
    master_port: int,
    exp_name: str = None,
    dataset_name: str = "biz-payer-beh-seq",
    use_user_attrs: bool = False,
    tokenizer_path: str = "prune95",
    max_sequence_length: int = 256,
    positional_sampling_ratio: float = 1.0,
    local_batch_size: int = 128,
    eval_batch_size: int = 128,
    eval_user_max_batch_size: Optional[int] = None,
    main_module: str = "SASRec",
    main_module_bf16: bool = False,
    checkpoint_path: Optional[str] = None,
    dropout_rate: float = 0.2,
    user_embedding_norm: str = "l2_norm",
    loss_module: str = "SampledSoftmaxLoss",
    contrastive_regularization_strength: int = 0,
    num_negatives: int = 1,
    test_n_targets: int = 1,  # test N
    loss_activation_checkpoint: bool = False,
    min_context_len: int = 0,
    temperature: float = 0.05,
    num_epochs: int = 101,
    learning_rate: float = 1e-3,
    num_warmup_steps: int = 0,
    weight_decay: float = 1e-3,
    top_k_method: str = "MIPSBruteForceTopK",
    log_interval: int = 1000,  # log per 1000 steps
    eval_interval: int = 100,  # evaluation per 100 steps?
    full_eval_every_n: int = 1,
    save_ckpt_every_n: int = 1000,
    partial_eval_num_iters: int = 32,
    embedding_module_type: str = "local",
    item_embedding_dim: int = 240,
    feature_embedding_dim: int = 0,  # Specify feature_embedding_dim for ConcatenateEmbeddingModule
    interaction_module_type: str = "",
    gr_output_length: int = 10,
    enable_tf32: bool = False,
    random_seed: int = 42,
    debug: bool = False,
    model_save_path: str = None,
    attr_loss: bool = True,
    attr_loss_factor: float = 0.5,
    attr_metrics: bool = True,
    attr_cat_embed: bool = False,
    use_continuous_attrs: bool = False,
    balance_attr_loss_by_vocabsize: bool = False,
    use_local_traindata: bool = True,
    use_local_testdata: bool = True,
    enable_mix_precision: bool = True,
    train_step: int = 500000,
    inference_mode : bool = False,
    block_type = ["hstu"],
    use_eval_user=False,
    local_data_path: str = None,
) -> None:
    if debug:
        logging.info(
            "Note: DEBUG mode is on! Model will not be saved and events will not be recorded."
        )
    use_user_attrs = 'contrast_useremb_pos' in block_type
    if rank == 0:
        logging.info(f"{locals()}")
        # logging.info(f"use attr_loss: {attr_loss}")
        # logging.info(f"use_continuous_attrs: {use_continuous_attrs}")
        # logging.info(f"balance_attr_loss_by_vocabsize: {balance_attr_loss_by_vocabsize}")
        # logging.info(f"use_user_attrs: {use_user_attrs}")
    attr_metrics = attr_loss
    if attr_metrics and (not attr_loss):
        raise ValueError("attr_metrics is True but attr_loss is False.")
    os.environ["LOCAL_RANK"] = str(rank)
    saved_args = locals()

    device = rank
    if rank == 0 and not debug and not inference_mode:
        experiment_tracker = {
            # "mlflow":{"run_id": init_mlflow_run(debug=debug, tracking_uri="./mlruns/")},
            "aim": {"run": init_aim_run(debug=debug, exp_name=exp_name, repo=".aim")}
        }
        log_params(experiment_tracker, saved_args, "train_config")
    else:
        experiment_tracker = None

    # to enable more deterministic results.
    random.seed(random_seed)
    torch.backends.cuda.matmul.allow_tf32 = enable_tf32
    torch.backends.cudnn.allow_tf32 = enable_tf32

    logging.info(f"Training model on rank {rank}.")
    if not debug:
        setup(rank, world_size, master_port)

    embedding_num_dict = torch.load(os.path.join(local_data_path, 'embedding_num_dict.pt'))
    max_item_id = embedding_num_dict['vocab_size']['beh_seq']
    all_item_ids = [i+1 for i in range(max_item_id)]
    if embedding_module_type == "ConcatenateEmbeddingModule":
        embedding_module: EmbeddingModule = ConcatenateEmbeddingModule(
            num_items=max_item_id,
            item_embedding_dim=item_embedding_dim,
            use_user_attrs=use_user_attrs,
            attr_loss=attr_loss,
            block_type=block_type,
            feat_meta_dict=embedding_num_dict,
        )
    else:
        raise ValueError(f"Unknown embedding_module_type {embedding_module_type}")

    interaction_module, interaction_module_debug_str = get_similarity_function(
        module_type=interaction_module_type,
        query_embedding_dim=embedding_module.input_dim,
        item_embedding_dim=embedding_module.input_dim,
    )

    assert (
        user_embedding_norm == "l2_norm" or user_embedding_norm == "layer_norm"
    ), f"Not implemented for {user_embedding_norm}"
    output_postproc_module = (
        L2NormEmbeddingPostprocessor(
            embedding_dim=embedding_module.input_dim,
            eps=1e-6,
        )
        if user_embedding_norm == "l2_norm"
        else LayerNormEmbeddingPostprocessor(
            embedding_dim=embedding_module.input_dim,
            eps=1e-6,
        )
    )
    input_preproc_module = LearnablePositionalEmbeddingInputFeaturesPreprocessor(
        max_sequence_len=max_sequence_length + gr_output_length + 1,
        embedding_dim=embedding_module.input_dim,
        dropout_rate=dropout_rate,
    )  # ! actions/ratings are not involved in the input features preprocessor, only item embeddings. !

    model = get_sequential_encoder(
        module_type=main_module,
        max_sequence_length=max_sequence_length,
        max_output_length=gr_output_length + 1,
        embedding_module=embedding_module,
        interaction_module=interaction_module,
        input_preproc_module=input_preproc_module,
        output_postproc_module=output_postproc_module,
        verbose=False,
        experiment_tracker=experiment_tracker,
        block_type=block_type,
    )

    # num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"model.named_parameters(): {[np[0] for np in model.named_parameters()]}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    net_params = sum(
        [
            p.numel()
            for n, p in model.named_parameters()
            if "_embedding_module" not in n and p.requires_grad
        ]
    )
    if rank == 0:
        logging.info(f"Enable_mix_precision: {enable_mix_precision}")
        logging.info(f"Number of parameters: {total_params:,}")
        logging.info(f"Number of parameters w/o embedding module: {net_params:,}")
        logging.info(f"Batchsize: {local_batch_size}")
        log_params(
            experiment_tracker,
            {
                "params_total": total_params,
                "params_net": net_params,
                "n_vocabulary": embedding_module._item_emb.num_embeddings,
            },
            "model_statistics",
        )

    # loss
    loss_debug_str = loss_module
    if loss_module == "SampledSoftmaxLoss":
        loss_debug_str = "ssl"
        if temperature != 1.0:
            loss_debug_str += f"-t{temperature}"
        ar_loss = SampledSoftmaxLoss(
            num_to_sample=num_negatives,
            softmax_temperature=temperature,
            model=model,
            activation_checkpoint=loss_activation_checkpoint,
            min_context_len=min_context_len,
        )
        loss_debug_str += (
            f"-n{num_negatives}{'-ac' if loss_activation_checkpoint else ''}"
        )
    else:
        raise ValueError(f"Unrecognized loss module {loss_module}.")

    from modeling.sequential import create_negatives_sampler

    negatives_sampler = create_negatives_sampler(
        max_item_id=max_item_id,
        item_emb=model._embedding_module._item_emb,
        all_item_ids=all_item_ids,
    )
    negatives_sampler_dict = None
    # if attr_loss:
    #     negatives_sampler_dict = {k:create_negatives_sampler() for k in action_def}
    #     negatives_sampler_dict = {k:v.to(device) for k, v in negatives_sampler_dict.items()}

    # Creates model and moves it to GPU with id rank
    if main_module_bf16:
        model = model.to(torch.bfloat16)
    model = model.to(device)
    ar_loss = ar_loss.to(device)
    negatives_sampler = negatives_sampler.to(device)
    if not debug:
        model = DDP(model, device_ids=[rank], broadcast_buffers=False)
        model_pointer = model.module
    else:
        model_pointer = model

    batch_id = 0
    if inference_mode:
        checkpoint_path = os.environ.get("CHECKPOINT_PATH", None)
        if checkpoint_path is None:
            raise ValueError("checkpoint_path is not specified.") 
    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location=f'cuda:{rank}')
        model_state = state_dict["model_state_dict"]
        if debug:
            model_state = {k[7:]: v for k, v in state_dict["model_state_dict"].items() if k.startswith("module.")}
        load_model_except_embedding(model, model_state, filter_key='._embedding_module')
        if rank == 0:
            logging.info(f"Loaded checkpoint from {checkpoint_path}")
    # TODO: wrap in create_optimizer.
    # params = [  # embedding module 和 transformer 设置不同学习率
    #     {
    #         "params": [v for k, v in model.named_parameters() if "embedding" in k],
    #         "lr": 0.5 * learning_rate,
    #     },
    #     {
    #         "params": [v for k, v in model.named_parameters() if "embedding" not in k],
    #         "lr": learning_rate,
    #     },
    # ]
    if not inference_mode:
        # opt = torch.optim.AdamW(params, betas=(0.9, 0.98), weight_decay=weight_decay)
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.98),
            weight_decay=weight_decay,
        )
        if enable_mix_precision:
            scaler = torch.amp.GradScaler()
        # scheduler = get_lr_scheduler_wsd(opt, factor=128 / local_batch_size)  # warmup-stable-decay
        # for _ in range(batch_id):
        #     scheduler.step()

    date_str = datetime.now().strftime(
        "%Y-%m-%d-%H%M%S"
    )
    model_subfolder = f"{dataset_name}-l{max_sequence_length}"
    if not os.path.exists(f"./exps/{model_subfolder}"):
        os.makedirs(f"./exps/{model_subfolder}", exist_ok=True)

    last_training_time = time.time()

    total_batch_id = 0
    log_loss = []
    # local_train_data_path = os.path.join(local_data_path, 'beh_seq_train.parquet')
    # local_eval_data_path = os.path.join(local_data_path, 'beh_seq_test.parquet')
    local_train_data_path = os.path.join(local_data_path, 'beh_seq_combined.parquet')
    local_eval_data_path = os.path.join(local_data_path, 'beh_seq_combined.parquet')
    train_data_sampler, train_data_loader = get_local_data_loader(rank, world_size, datapath=local_train_data_path, max_length=max_sequence_length + gr_output_length + 1, batch_size=local_batch_size
                        # , eval_flag=False
                        , eval_flag=True
                        )
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(num_epochs):
        if total_batch_id >= train_step:
            break
        if train_data_sampler is not None:
            train_data_sampler.set_epoch(epoch)
        model.train()
        for batch_id, data in enumerate(train_data_loader): # local training data
            if (total_batch_id % eval_interval) == 0:
                with torch.autocast(
                    "cuda",
                    enabled=enable_mix_precision,
                    # dtype=torch.float16,
                    dtype=torch.bfloat16,
                ), torch.no_grad():
                    eval_st = time.time()
                    eval_dict_all, all_targets, all_targets_feat, log_metric_dict = eval_func(rank, world_size, gr_output_length, eval_user_max_batch_size, main_module_bf16, top_k_method, attr_metrics, device, all_item_ids, model, negatives_sampler, negatives_sampler_dict, use_continuous_attrs, eval_batch_size, tokenizer_path, max_sequence_length, test_n_targets, debug, local_eval_data_path, inference_mode, embedding_num_dict=embedding_num_dict)
                    if rank == 0:
                        logging_text = get_eval_info(eval_dict_all, all_targets, all_targets_feat, total_batch_id)
                        if experiment_tracker is not None:
                            glom(experiment_tracker, "aim.run").log_info(logging_text)
                            log_metrics(experiment_tracker, "eval", total_batch_id, log_metric_dict)
                        eval_time_text = f"\neval_time: {(time.time() - eval_st):.2f}s"
                        logging.info(logging_text + eval_time_text)
                if inference_mode:
                    break
                    # continue
                # == END eval
                # is_bad_data = torch.tensor((row is None), dtype=torch.int).to(rank)
                # if not debug:
                #     dist.all_reduce(is_bad_data, op=dist.ReduceOp.MAX)
                # if is_bad_data.item() > 0:
                #     if rank == 0:
                #         logging.info(f"\n=======\nrank-{rank}: bad data, skipping....\n=======\n")
                #     continue
            with torch.autocast(
                "cuda",
                enabled=enable_mix_precision,
                dtype=torch.float16,
                # dtype=torch.bfloat16,
            ):
                torch.cuda.empty_cache()
                model.train()
                _, seq_features = get_input(embedding_num_dict, data)
                B, N = seq_features.past_ids.shape
                opt.zero_grad()
                input_embeddings = model_pointer.get_item_embeddings(
                    seq_features.past_ids,
                    features=seq_features.past_payloads["features"],
                    user_attrs=(
                        seq_features.past_payloads["user_attrs"]
                        if use_user_attrs
                        else None
                    ),
                )
                supervision_embeddings = model_pointer.get_item_embeddings(
                    seq_features.past_ids,
                    features=None,  # seq_features.past_payloads['features'] # if embedding_module_type == "local_cross_feature_concat" else None
                )
                # from thop import profile
                # macs, params = profile(model, inputs=(seq_features.past_lengths, seq_features.past_ids,input_embeddings,seq_features.past_payloads))
                # from thop import clever_format
                # macs, params = clever_format([macs, params], "%.3f")
                # print(macs,params)
                # exit()
                seq_embeddings = model(
                # seq_embeddings, cached_states, addiction_loss = model(
                    past_lengths=seq_features.past_lengths,
                    past_ids=seq_features.past_ids,
                    past_embeddings=input_embeddings,
                    past_payloads=seq_features.past_payloads,
                    return_cache_states=False,
                )
                if main_module != "SASRec":
                    seq_embeddings, cached_states, addiction_loss = seq_embeddings
                supervision_ids = seq_features.past_ids

                if negatives_sampler.sampling_strategy == "in-batch":
                    in_batch_ids = supervision_ids.view(-1)
                    negatives_sampler.process_batch(
                        ids=in_batch_ids,
                        presences=(in_batch_ids != 0),
                        embeddings=supervision_embeddings.view(-1, supervision_embeddings.size(-1)),
                    )
                else:
                    negatives_sampler._item_emb = (
                        model_pointer._embedding_module._item_emb
                    )

                # 设置计算 loss 时的上下文长度，沿着 N 维度用 1 标出参与计算的元素。
                ar_mask = supervision_ids[:, 1:] != 0  # non-padding token
                try:
                    loss = ar_loss(
                        lengths=seq_features.past_lengths,  # [B],
                        output_embeddings=seq_embeddings[:, :-1, :supervision_embeddings.size(-1)],  # [B, N-1, D]
                        supervision_ids=supervision_ids[:, 1:],  # [B, N-1]
                        supervision_embeddings=supervision_embeddings[
                            :, 1:, :
                        ],  # [B, N - 1, D]
                        supervision_weights=ar_mask.float(),
                        negatives_sampler=negatives_sampler,
                    )
                    aloss = loss
                except Exception as e:
                    print(e)
                    print("skip this batch")
                    torch.cuda.empty_cache()
                    continue
            # loss.backward()
            # import ipdb;ipdb.set_trace()
            if enable_mix_precision:
                scaler.scale(loss).backward()
            else:
                # __import__('ipdb').set_trace()
                try:
                    loss.backward()
                    # 在反向传播后，优化器更新前添加
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)  # 限制梯度范数
                except Exception as e:
                    print(e)
                    print("skip this batch")
                    torch.cuda.empty_cache()
                    continue
            if any([p.grad.isinf().any() or p.grad.isnan().any() for p in model.parameters()]):
                print("Current batch size: ", data['beh_seq'].shape)
                # import ipdb;ipdb.set_trace()
                print("skip this batch")
                torch.cuda.empty_cache()
                continue


            # Optional linear warmup.  # Starting with lower learning rate
            
            # if batch_id < num_warmup_steps:
            #     lr_scalar = min(1.0, float(batch_id + 1) / num_warmup_steps)
            #     for pg in opt.param_groups:
            #         pg["lr"] = lr_scalar * learning_rate
            #     lr = lr_scalar * learning_rate
            # else:
            #     lr = scheduler.get_last_lr()[0]  # learning_rate
            lr = learning_rate

            if world_size > 1:
                aloss = _avg(aloss, world_size)
                loss = _avg(loss, world_size)
            if rank == 0:
                log_loss.append(loss)

            if total_batch_id % 10 == 0 and rank == 0:
                log_metrics(
                    experiment_tracker,
                    "train",
                    total_batch_id,
                    {"ar_loss": aloss},
                )

            if (total_batch_id % log_interval) == 0 and rank == 0:
                log_info = f"rank-{rank}: batch-stat (train): e0-s{total_batch_id} in {time.time() - last_training_time:.2f}s, "
                log_info += f"lr: {lr:.2e}, arloss: {aloss:.3f}, "
                log_info += f"loss: {sum(log_loss)/len(log_loss):.6f}, "
                logging.info(datetime.now().strftime("%Y-%m-%d %H:%M:%S") +' '+log_info)

                log_metrics(
                    experiment_tracker,
                    "train",
                    total_batch_id,
                    {
                        "loss": sum(log_loss) / len(log_loss),
                        "lr": lr,
                        "time_per_step": (
                            (time.time() - last_training_time) / log_interval
                            if total_batch_id > 1
                            else 0
                        ),
                    },
                )
                last_training_time = time.time()
                log_loss = []

            if enable_mix_precision:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()

            if (
                rank == 0 and (total_batch_id % save_ckpt_every_n) == 0 and not debug and not inference_mode
            ):  # Save every `save_ckpt_every_n` batches.
                state_dict = {
                    "epoch": epoch,
                    "batch_id": batch_id,
                    "model_state_dict": model.state_dict(),
                }
                logging.info(f"Saving checkpoint at {model_save_path}")
                torch.save(state_dict, f"{model_save_path}/ep{epoch}_b{total_batch_id}")

            total_batch_id += 1
            if total_batch_id >= train_step:
                break
            # scheduler.step()
            # if debug:
            #     break
        # if debug:
        #     break

        # if (
        #     rank == 0 and not debug and not inference_mode
        # ):  # and epoch > 0 and (epoch % save_ckpt_every_n) == 0:  # Save every epoch
        #     print("Save model: ", f"{model_save_path}/ep{epoch}")
        #     torch.save(
        #         {
        #             "epoch": epoch,
        #             "model_state_dict": model.state_dict(),
        #         },
        #         f"{model_save_path}/ep{epoch}",
            # )
        last_training_time = time.time()

    if rank == 0 and not debug and not inference_mode:
        # torch.save(
        #     {
        #         "epoch": epoch,
        #         "model_state_dict": model.state_dict(),
        #     },
        #     f"{model_save_path}/ep{epoch}",
        # )
        logging.info("Training end...")
    if not debug:
        cleanup()


def mp_train_fn(
    rank: int,
    world_size: int,
    master_port: int,
    gin_config_file: Optional[str],
    debug: bool,
    inference_mode: bool,
    model_save_path: Optional[str],
) -> None:
    if gin_config_file is not None:
        # Hack as absl doesn't support flag parsing inside multiprocessing.
        if rank == 0:
            logging.info(f"Rank {rank}: loading gin config from {gin_config_file}")
        gin_path = os.path.abspath(gin_config_file)
        old_directory = os.getcwd()
        new_directory = os.path.join(os.getcwd(),'configs',gin_config_file.split('/')[-2])
        os.chdir(new_directory)
        gin.parse_config_file(gin_path)
        os.chdir(old_directory)
        
        exp_name = gin_config_file.split(".")[0]
    if model_save_path is not None:
        os.makedirs(model_save_path, exist_ok=True)
    train_fn(rank, world_size, master_port, model_save_path=model_save_path, debug=debug, exp_name=exp_name, inference_mode=inference_mode)


def main(argv):
    world_size = torch.cuda.device_count()

    # DEBUG Mode ===============================================================
    if FLAGS.debug and world_size == 1:
        print("DEBUG mode is on!")
        
        gin_path = os.path.abspath(FLAGS.gin_config_file)
        old_directory = os.getcwd()
        new_directory = os.path.join(os.getcwd(),'configs',FLAGS.gin_config_file.split('/')[-2])
        os.chdir(new_directory)
        gin.parse_config_file(gin_path)
        os.chdir(old_directory)
        train_fn(0, 1, FLAGS.master_port, debug=True, inference_mode=FLAGS.inference_mode)
        return
    # ==========================================================================

    mp.set_start_method("forkserver")
    mp.spawn(
        mp_train_fn,
        args=(world_size, FLAGS.master_port, FLAGS.gin_config_file, FLAGS.debug, FLAGS.inference_mode,FLAGS.model_save_path),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    app.run(main)
