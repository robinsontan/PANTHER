import logging
import os

import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

from data.eval import (
    _avg,
    _concat,
    _merge_length,
    eval_metrics_v2_from_tensors,
    get_eval_state,
)
from indexing.utils import get_top_k_module


def load_model_except_embedding(model, state_dict, filter_key='embedding', rank=0):
    filtered_state_dict = {k: v for k, v in state_dict.items() if filter_key not in k}
    
    model.load_state_dict(filtered_state_dict, strict=False)
    
    if rank == 0:
        logging.info("Unsuccessfully loaded weights:")
        for k in filtered_state_dict.keys():
            if k not in model.state_dict():
                logging.info(f"{k}")
                
class batch_data(object):
    def __init__(self, past_ids, past_payloads, past_lengths):
        self.past_ids = past_ids
        self.past_payloads = past_payloads
        self.past_lengths = past_lengths
        
class ParquetDataset(Dataset):
    def __init__(self, parquet_path, max_length=256, eval_flag=False):
        self.dataframe = pd.read_parquet(parquet_path)
        self.max_length = max_length
        self.eval_flag = eval_flag
        self.target_num = 10
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        row = self.dataframe.iloc[idx]
        row_dict = row.to_dict()
        ans_dict = dict()
        for key in row_dict.keys():
            if key == 'seq_length':
                continue
            value = row_dict[key]
            if isinstance(value, str):
                try:
                    tmpresult = [int(x) for x in value.split(',')]
                except ValueError:
                    continue
                if self.eval_flag:
                    if len(tmpresult) >= self.max_length + self.target_num:
                        tmpresult = tmpresult[-(self.max_length + self.target_num):]
                        if key == 'beh_seq':
                            ans_dict['target_ids'] = tmpresult[-self.target_num:]
                        tmpresult = tmpresult[:-self.target_num]
                    else:
                        if key == 'beh_seq':
                            ans_dict['target_ids'] = tmpresult[-self.target_num:]
                        tmpresult = tmpresult[:-self.target_num] 
                        tmpresult += [0] * (self.max_length - len(tmpresult))
                else:
                    raise Exception(f"self.eval_flag is not True.")
                ans_dict[key] = tmpresult
            else:
                ans_dict[key] = value
        return ans_dict

def collate_fn(batch, device):
    """
    自定义的collate_fn函数，用于将Dataset返回的字典列表合并为一个批次。
    
    参数:
    batch (list of dict): 一个批次的数据，每个元素是一个字典
    
    返回:
    dict: 合并后的批次数据，字典的键是列名，值是该列的所有值组成的Tensor
    """
    batch_dict = {}
    
    # 获取所有的键
    keys = batch[0].keys()
    for key in keys:
        values = [d[key] for d in batch]
        if isinstance(values[0], list):
            batch_dict[key] = torch.tensor(values).to(device)
        else:
            batch_dict[key] = torch.tensor(values).view(-1, 1).to(device)
    batch_dict['seq_length'] = (batch_dict['beh_seq'] != 0).sum(dim=1)
    return batch_dict

def get_local_data_loader(rank, world_size, datapath, max_length, batch_size, eval_flag=False):
    dataset = ParquetDataset(datapath, max_length=max_length, eval_flag=eval_flag)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=0,
        drop_last=False,
    )
    return sampler, DataLoader(
        dataset,
        batch_size=batch_size,
        # num_workers=16,
        sampler=sampler,
        shuffle=False,
        persistent_workers=False,
        collate_fn=lambda x: collate_fn(x, f"cuda:{rank}")
        )
# 示例用法
# dataset = ParquetDataset('path/to/your/file.parquet')
# dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

# for batch in dataloader:
#     print(batch)  # 打印每个批次的数据 
def eval_func(rank, world_size, gr_output_length, eval_user_max_batch_size, main_module_bf16, top_k_method, attr_metrics, device, all_item_ids, model, negatives_sampler, negatives_sampler_dict, use_continuous_attrs, 
              eval_batch_size, tokenizer_path, max_sequence_length, test_n_targets, debug, local_eval_data_path, inference_mode, embedding_num_dict):
    if rank == 0:
        logging.info("EVAL_DATA_PATH provided. Using the provided test dataset.......")
    _, eval_data_loader = get_local_data_loader(rank, world_size, local_eval_data_path, batch_size=eval_batch_size, max_length=max_sequence_length + gr_output_length + 1, eval_flag=True)
    if rank == 0:
        logging.info("Start evaluation...")
    model.eval()
    torch.cuda.empty_cache()

    eval_state = get_eval_state(
        model=model.module if not debug else model,
        all_item_ids=all_item_ids,
        negatives_sampler=negatives_sampler,
        top_k_module_fn=lambda item_embeddings, item_ids: get_top_k_module(
            top_k_method=top_k_method,
            model=model.module if not debug else model,
            item_embeddings=item_embeddings,
            item_ids=item_ids,
        ),
        device=device,
        float_dtype=torch.bfloat16 if main_module_bf16 else None,
    )
    eval_dict_all = None
    if inference_mode:
        per_length_correct = [[0, 0] for _ in range(gr_output_length + max_sequence_length + 1)] #[total_num, correct_num]
    for eval_batch_id, data in enumerate(eval_data_loader):
        target_ids, seq_features = get_input(embedding_num_dict, data)
        eval_result = eval_metrics_v2_from_tensors(
            eval_state,
            model.module if not debug else model,
            seq_features,
            target_ids=target_ids,
            dtype=torch.bfloat16 if main_module_bf16 else None,
            filter_invalid_ids=False,
        )
        if inference_mode:
            for k, v in zip(seq_features.past_lengths, eval_result["Acc_Next1"]):
                if v:
                    per_length_correct[k.item()][1] += 1
                per_length_correct[k.item()][0] += 1
        eval_dict = eval_result
        if eval_dict_all is None:
            eval_dict_all = {}
            for k, v in eval_dict.items():
                eval_dict_all[k] = []

        for k, v in eval_dict.items():
            eval_dict_all[k] = eval_dict_all[k] + [v]

        del eval_dict
        if debug:
            break
    
    # all_targets = torch.cat(all_targets, dim=-1)
    # all_targets = _concat(all_targets, world_size)
    # all_targets_feat_dict = dict()
    # for k in all_targets_feat[0].keys():
    #     all_targets_feat_dict[k] = torch.cat([i[k].view(-1) for i in all_targets_feat], dim=-1)
    #     all_targets_feat_dict[k] = _concat(all_targets_feat_dict[k], world_size)
    # all_targets_feat = all_targets_feat_dict
    log_metric_dict = dict()
    for k, v in eval_dict_all.items():
        eval_dict_all[k] = torch.cat(v, dim=-1)
        if "Acc" in k or "HR" in k:
            eval_dict_all[k] = _avg(eval_dict_all[k], world_size)
            log_metric_dict[k] = eval_dict_all[k]
        elif "id" in k:
            eval_dict_all[k] = _concat(eval_dict_all[k], world_size)
    # return eval_dict_all, all_targets, all_targets_feat,log_metric_dict
    if inference_mode:
        per_length_correct = torch.tensor(per_length_correct, device=eval_result["Acc_Next1"].device)
        per_length_correct = _merge_length(per_length_correct, world_size)
        eval_dict_all['per_length_correct'] = per_length_correct
    return eval_dict_all, None, None, log_metric_dict

def get_input(embedding_num_dict, data):
    target_ids = None
    features = dict()
    user_attrs = dict()
    if 'target_ids' in data:
        target_ids = data['target_ids']
    for k in embedding_num_dict['seq_columns']:
        features[k] = data[k]
    for k in embedding_num_dict['element_column']:
        user_attrs[k] = data[k]
    seq_features = batch_data(past_ids=data['beh_seq'], past_payloads={
            'features': features, 'user_attrs': user_attrs, 'timestamps': data['unix_time'] if 'unix_time' in data else None
            }, past_lengths=data['seq_length'])
    return target_ids,seq_features

def get_eval_info(eval_dict_all, all_targets, all_targets_feat, batch_id):
    correct_info, incorrect_info = None, None
    logging_text = f"Eval: step-{batch_id}: \n"
    acc_metrics = ""
    str_metrics = ""
    for k, v in eval_dict_all.items():
        if isinstance(v, str):
            str_metrics+= f"{k:}\n{v},\n"
        else:
            acc_metrics+= f"{k:} {v:.6f},\n"
    return (
        f"\n{logging_text}"
        + "============================== Acc Metrics ==============================\n"
        + acc_metrics
        + "=========================================================================\n"
        + "=========================== Statistics Metrics ===========================\n"
        + str_metrics
        + (correct_info if correct_info else "")
        + (incorrect_info if incorrect_info else "")
        + "=========================================================================\n"
    )


def find_frequent(tensor, de_tokenizer, target):
    unique_elements, counts = torch.unique(tensor, return_counts=True)
    
    idx = (unique_elements == target)
    if idx.sum() > 0:
    
        frequent = counts[idx]
        
        proportion = (frequent / counts.sum()).item()
    else:
        proportion = 0
    return f"The frequent of token:  ({target}:{de_tokenize([target], de_tokenizer=de_tokenizer)[0]}) with proportion: {proportion:.4f} in {counts.sum().item()} tokens"

def find_topk(target, mask, de_tokenizer, k=3, pred=None):
    if (mask).sum()<k:
        return f"There are no enough value for top-{k}.\n"
    target = target[mask]
    if pred is not None:
        pred = pred[mask]
    unique_elements, counts = torch.unique(target, return_counts=True)
    counts_sum = counts.sum().item()
    counts, idx = torch.topk(counts, k=k)
    unique_elements = unique_elements[idx]
    unique_elements_id = unique_elements.cpu().tolist()
    unique_elements = de_tokenize(unique_elements_id, de_tokenizer=de_tokenizer)
    ans = f"Top K={k} Correctly Predicted :\n" if pred is None else f"Top K={k} Incorrectly Predicted Tokens:\n"
    for i in range(k):
        ans += f"Top {i+1} token pair: ({unique_elements_id[i]}, {unique_elements[i]})   Proportion: {counts[i].item() / counts_sum:.4f} in {counts_sum} tokens"
        if pred is not None:
            target_id = unique_elements_id[i]
            pred_tmp = pred[target == target_id]
            unique_elements_pred_tmp, counts_pred_tmp = torch.unique(pred_tmp, return_counts=True)
            ans += f"   Most Frequent Incorrect Prediction: ({unique_elements_pred_tmp[0].item()}, {de_tokenize([unique_elements_pred_tmp[0].item()], de_tokenizer=de_tokenizer)[0]})"
        ans+="\n"
    return ans
    
def find_most_frequent(tensor, de_tokenizer):
    # 使用torch.unique来获取张量中的唯一值及其出现的次数
    unique_elements, counts = torch.unique(tensor, return_counts=True)
    
    # 找到出现频次最高的元素的索引
    max_idx = torch.argmax(counts)
    
    # 获取频次最高的元素
    most_frequent_element = unique_elements[max_idx]
    # 计算该元素的占比
    proportion = counts[max_idx].float() / tensor.size(0)
    return (most_frequent_element.item(),de_tokenize([most_frequent_element.item()], de_tokenizer=de_tokenizer)[0]),proportion.item(),f"({most_frequent_element.item()}:{de_tokenize([most_frequent_element.item()], de_tokenizer=de_tokenizer)[0]}) with proportion: {proportion.item():.4f} in {counts.sum().item()} tokens"

def setup(rank: int, world_size: int, master_port: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
