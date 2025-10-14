

"""
Main entry point for model training. Please refer to README.md for usage instructions.
"""

import glob
import logging
import os
import random
from datetime import date, datetime
from typing import Optional, Union

import mlflow
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Hide excessive tensorflow debug messages
import shutil
import sys
import time

import fbgemm_gpu  # noqa: F401, E402
import gin
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from absl import app, flags
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.eval import (
    _avg,
    _sum,
    add_to_summary_writer,
    eval_metrics_v2_from_tensors,
    get_eval_state,
    log_to_mlflow,
)
from data.reco_dataset import get_reco_dataset
from indexing.utils import get_top_k_module
from modeling.sequential.autoregressive_losses import (
    BCELoss,
    InBatchNegativesSampler,
    LocalNegativesSampler,
    SampledSoftmaxLoss,
)
from modeling.sequential.embedding_modules import (
    EmbeddingModule,
    FeatureEmbeddingModule,
    LocalCrossFeatureEmbeddingModule,
    LocalEmbeddingModule,
)
from modeling.sequential.encoder_utils import get_sequential_encoder
from modeling.sequential.features import (
    get_feature_processer,  #  movielens_seq_features_from_row, biz_payer_beh_seq_features_from_row
)
from modeling.sequential.input_features_preprocessors import (
    LearnablePositionalEmbeddingInputFeaturesPreprocessor,
)
from modeling.sequential.output_postprocessors import (
    L2NormEmbeddingPostprocessor,
    LayerNormEmbeddingPostprocessor,
)
from modeling.similarity_utils import get_similarity_function
from trainer.data_collator import behseq_iceberg_collator
from trainer.data_loader import create_data_loader

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s: %(relativeCreated)d: %(levelname)s: %(message)s')
mlflow.utils.logging_utils.suppress_logs('mlflow', '.*')
mlflow.utils.logging_utils.suppress_logs('conf.DistributedConfigHelper', 'fetch local .*')
MLFLOW_EXPERIMENT_ID = "287824017476210272"

flags.DEFINE_string("gin_config_file", None, "Path to the config file.")
flags.DEFINE_integer("master_port", 12355, "Master port.")

FLAGS = flags.FLAGS

abbreviate_number = lambda n: next(f'{n/10**(3*i):.1f}{["","K","M","B","T"][i]}' for i in range(5) if n < 10**(3*(i+1)))

def setup(rank: int, world_size: int, master_port: int) -> None:
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(master_port)
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()

def update_records(records, new_record):
    assert type(records) == type(new_record)
    if isinstance(records, list):
        records += new_record
    elif isinstance(records, dict):
        for key in new_record:
            records[key] = update_records(records[key], new_record[key]) if key in records else new_record[key]
    return records

@gin.configurable
def inference(
    rank: int,
    world_size: int,
    master_port: int,
    dataset_name: str = "biz-payer-beh-seq-v0615-330-60",
    tokenizer_path: str = "prune3000",
    max_sequence_length: int = 256,
    positional_sampling_ratio: float = 1.0,
    eval_batch_size: int = 10,
    eval_user_max_batch_size: Optional[int] = None,
    top_k_method: str = "MIPSBruteForceTopK",
    gr_output_length: int = 10,
    enable_tf32: bool = False,
    random_seed: int = 42,
    mlflow_run_id: str = "",
    certain_step: Union[int, str] = "*",
    write_result_to_mlflow: bool = False,
) -> None:
    device = rank
    mlflow_run_path = f"mlruns/{MLFLOW_EXPERIMENT_ID}/{mlflow_run_id}"
    if write_result_to_mlflow and rank == 0:
        run = mlflow.start_run(log_system_metrics=False, run_id=mlflow_run_id)
        logging.info(f"Logging to mlflow run {run.data.tags.get('mlflow.runName')}")
        if not os.path.exists(os.path.join(mlflow_run_path, "metrics/archive")):
            shutil.copytree(metrics_path, os.path.join(mlflow_run_path, "metrics/archive"))  # backup original metrics
    
    ckpt_list = glob.glob(os.path.join(mlflow_run_path, f"artifacts/checkpoint/{certain_step}/state_dict.pth"))
    ckpt_list = sorted(ckpt_list, key=lambda x: int(os.path.dirname(x).split('/')[-1]))
    batch_ids = [int(os.path.dirname(x).split('/')[-1]) for x in ckpt_list]
    metrics_path = os.path.join(mlflow_run_path, "metrics/eval")
    
    metrics_names = os.listdir(metrics_path)
    metrics = {
        mn: pd.read_csv(os.path.join(metrics_path, mn), header=None, sep=' ', names=['timestamp', 'value', 'step'])
        for mn in metrics_names
    }

    for mn, md in metrics.items():
        metrics[mn] = md[md['step'].isin(batch_ids)]

    logging.info(f"{len(ckpt_list)} checkpoints found.")
    main_module_bf16 = False
    # to enable more deterministic results.
    random.seed(random_seed)
    torch.backends.cuda.matmul.allow_tf32 = enable_tf32
    torch.backends.cudnn.allow_tf32 = enable_tf32

    setup(rank, world_size, master_port)

    dataset = get_reco_dataset(
        dataset_name=dataset_name,
        max_sequence_length=max_sequence_length,
        chronological=True,
        positional_sampling_ratio=positional_sampling_ratio,
    )

    eval_data_sampler, eval_data_loader = create_data_loader(
        dataset.eval_dataset,
        batch_size=eval_batch_size,
        world_size=world_size,
        rank=rank,
        shuffle=False,  # True # needed for partial eval
        drop_last=world_size > 1,
    )

    from trainer.tokenizer import get_tokenizer
    tokenizer = get_tokenizer(path=tokenizer_path)
    n_targets = 10
    collate_fn = behseq_iceberg_collator(
        tokenizer=tokenizer,
        ignore_last_n=1, 
        padding_length=max_sequence_length+1, 
        chronological=True, 
        shift_id_by=0,
        n_targets=n_targets
    )
    eval_data_loader.collate_fn = collate_fn
    dataset.max_item_id = len(collate_fn._tokenizer)
    dataset.all_item_ids = [x + 1 for x in range(len(collate_fn._tokenizer))]

    
    seq_features_from_row = get_feature_processer()['biz-payer-beh-seq']

    from modeling.sequential import create_model, create_negatives_sampler
    model = create_model(max_item_id=dataset.max_item_id)
    negatives_sampler = create_negatives_sampler(
        max_item_id=dataset.max_item_id, 
        item_emb=model._embedding_module._item_emb,
        all_item_ids=dataset.all_item_ids
    )

    model, negatives_sampler = model.to(device), negatives_sampler.to(device)
    model = DDP(model, device_ids=[rank], broadcast_buffers=False)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    net_params = sum([p.numel() for n,p in model.named_parameters() if '_embedding_module' not in n and p.requires_grad])
    if rank == 0:
        logging.info(f"Number of parameters: {total_params:,}\nNumber of parameters w/o embedding module: {net_params:,}")
        logging.debug("Start evaluation...")
    
    torch.autograd.set_detect_anomaly(True)
    # Creates model and moves it to GPU with id rank
    for ckpt_path in ckpt_list:  # 评测每个checkpoint
        state_dict = torch.load(ckpt_path)
        batch_id = state_dict['batch_id']
        if batch_id == 0:
            continue
        state_dict = state_dict['model_state_dict']
        # state_dict = {k[len('module.'):] : v for k,v in state_dict.items()}
        model.load_state_dict(state_dict)

        eval_dict_all = None
        eval_details = {'eval_ranks':[], 'eval_features':dict()}
        num_seqs = 0
        model.eval()
        torch.cuda.empty_cache()
        target_next_n = 10
        for eval_row in eval_data_loader:
            # __import__('ipdb').set_trace()
            seq_features, target_ids, target_ratings = seq_features_from_row(
                eval_row, device=device, max_output_length=gr_output_length + 1,
            )
            num_seqs += len(target_ids)
            # all_item_ids = eval_row['target_ids'].tolist()
            all_item_ids = dataset.all_item_ids

            eval_state = get_eval_state(
                model=model.module,
                all_item_ids=all_item_ids,
                negatives_sampler=negatives_sampler,
                top_k_module_fn=lambda item_embeddings, item_ids: get_top_k_module(
                    top_k_method=top_k_method,
                    model=model.module,
                    item_embeddings=item_embeddings,
                    item_ids=item_ids,
                ),
                device=device,
                float_dtype=torch.bfloat16 if main_module_bf16 else None,
            )
            # __import__('ipdb').set_trace()
            eval_dict, eval_detail = eval_metrics_v2_from_tensors(
                eval_state, model.module, seq_features, target_ids=target_ids, target_ratings=target_ratings,
                user_max_batch_size=eval_user_max_batch_size,
                dtype=torch.bfloat16 if main_module_bf16 else None,
                filter_invalid_ids=False,
                return_details=True,
                tokenizer = tokenizer,
            )
            # print(f"Average-{eval_dict['hr@1'].size(0)} HR@1: {eval_dict['hr@1'].sum()/eval_dict['hr@1'].size(0)}")
            
            # uin, past_length, eval_rank
            eval_details['eval_ranks'].extend([(int(uin), int(sq), int(er)) for uin, sq, er in 
                                               zip(seq_features.past_payloads['uins'], seq_features.past_lengths, eval_detail['eval_ranks'])])
            eval_details['eval_features'] = update_records(eval_details['eval_features'], eval_detail['eval_features'])

            if eval_dict_all is None:
                eval_dict_all = {}
                for k, v in eval_dict.items():
                    eval_dict_all[k] = []

            for k, v in eval_dict.items():
                eval_dict_all[k] = eval_dict_all[k] + [v]
            del eval_dict, eval_row
            # break
        print(f"Done: Evaluated on {num_seqs} sequences.")
        if rank == 0:
            eval_ranks = pd.DataFrame(eval_details['eval_ranks'])
            os.makedirs(os.path.join(mlflow_run_path, "artifacts/eval/"), exist_ok=True)
            eval_ranks.to_csv(os.path.join(mlflow_run_path, f"artifacts/eval/ranks-t{n_targets}-{batch_id}.csv"), header=['uin', 'past_length', 'eval_rank'], index=False)
            eval_features = pd.DataFrame(eval_details['eval_features'])
            eval_features.to_csv(os.path.join(mlflow_run_path, f"artifacts/eval/features-t{n_targets}-{batch_id}.csv"), index=False)
        for k, v in eval_dict_all.items():
            eval_dict_all[k] = torch.cat(v, dim=-1)
            eval_dict_all[k] = _avg(eval_dict_all[k], world_size)
        print(f"Step {batch_id}: {'-'*20}")
        print({k:v.tolist() for k, v in eval_dict_all.items()})
        if rank == 0 and write_result_to_mlflow:
            for mn, md in metrics.items():
                md.loc[md['step']==batch_id, 'value'] = eval_dict_all[mn.replace('-', '@')].tolist()
                md.to_csv(os.path.join(metrics_path, mn), header=None, sep=' ', index=False)
            # log_to_mlflow(batch_id, eval_dict_all, prefix="eval", world_size=world_size, mlflow_run_id=mlflow_run_id)
    cleanup()


def mp_inference_fn(
    rank: int,
    world_size: int,
    master_port: int,
    gin_config_file: Optional[str],
) -> None:
    if gin_config_file is not None:
        # Hack as absl doesn't support flag parsing inside multiprocessing.
        if rank == 0:
            logging.info(f"Rank {rank}: loading gin config from {gin_config_file}")
        gin.parse_config_file(gin_config_file)
    
    inference(rank, world_size, master_port)


def main(argv):
    world_size = torch.cuda.device_count()

    mp_inference_fn(0, 1, FLAGS.master_port, FLAGS.gin_config_file)
    exit()

    mp.set_start_method('forkserver')
    mp.spawn(mp_inference_fn,
             args=(world_size, FLAGS.master_port, FLAGS.gin_config_file),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    app.run(main)
