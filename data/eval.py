

import logging
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple

import mlflow
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from indexing.candidate_index import CandidateIndex, TopKModule
from modeling.ndp_module import NDPModule
from modeling.sequential.features import SequentialFeatures

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

@dataclass
class EvalState:
    all_item_ids: Set[int]
    candidate_index: CandidateIndex
    top_k_module: TopKModule


@dataclass
class EvalResult:
    metrics: Dict[str, float]
    ranks: List[Tuple[int, int, int]] = None  # uin, past_length, rank
    features: OrderedDict = None

def get_eval_state(
    model: NDPModule,
    all_item_ids: List[int],  # [X]
    negatives_sampler,
    top_k_module_fn: Callable[[torch.Tensor, torch.Tensor], TopKModule],
    device: torch.device,
    float_dtype: Optional[torch.dtype] = None,
    feat_key=None
) -> EvalState:
    # Exhaustively eval all items (incl. seen ids).
    eval_negatives_ids = torch.as_tensor(all_item_ids).to(device).unsqueeze(0)  # [1, X]
    if feat_key:
        eval_negative_embeddings = negatives_sampler.normalize_embeddings(
            model._embedding_module.get_feature_embeddings(eval_negatives_ids, feat_key=feat_key)  # pass features
        )
    else:
        eval_negative_embeddings = negatives_sampler.normalize_embeddings(
            model.get_item_embeddings(eval_negatives_ids, features=None)  # pass features
        )
    if float_dtype is not None:
        eval_negative_embeddings = eval_negative_embeddings.to(float_dtype)
    candidates = CandidateIndex(
        ids=eval_negatives_ids,
        embeddings=eval_negative_embeddings,
    )
    return EvalState(
        all_item_ids=set(all_item_ids),
        candidate_index=candidates,
        top_k_module=top_k_module_fn(eval_negative_embeddings, eval_negatives_ids)
    )


# @torch.no_grad
def calculate_accuracy_of_prediction(pre, target, n):
    """
    计算预测的下一个行为出现在接下来的N次行为中的准确率。
    
    参数:
    pre (torch.Tensor): 模型预测的下一个行为，形状为 [batch_size, 1]
    target (torch.Tensor): 实际的接下来的k次行为，形状为 [batch_size, k]
    n (int): 关注的接下来的行为次数，k >= n
    """
    # 确保n不大于target的第二个维度
    assert n <= target.size(1)
    # 检查预测的行为是否在每个样本的目标行为的前n个中
    match = (pre == target[:, :n]).any(dim=1)
    return match

# @torch.no_grad
def eval_metrics_v2_from_tensors(
    eval_state: EvalState,
    model: NDPModule,
    seq_features: SequentialFeatures,
    target_ids: torch.Tensor,  # [B, K]
    filter_invalid_ids: bool = True,
    dtype: Optional[torch.dtype] = None,
) -> Dict[str, List[float]]:
    """
    Args:
        eval_negatives_ids: Optional[Tensor]. If not present, defaults to eval over
            the entire corpus (`num_items`) excluding all the items that users have
            seen in the past (historical_ids, target_ids). This is consistent with
            papers like SASRec and TDM but may not be fair in practice as retrieval
            modules don't have access to read state during the initial fetch stage.
        filter_invalid_ids: bool. If true, filters seen ids by default.
    Returns:
        keyed metric -> list of values for each example.
    """
    B, n_targets = target_ids.shape
    device = target_ids.device
    for target_id in target_ids:
        target_id = int(target_id[0])
        if target_id not in eval_state.all_item_ids:
            print(f"missing target_id {target_id}")

    # computes ro- part exactly once.
    shared_input_embeddings = model.encode(
        past_lengths=seq_features.past_lengths,
        past_ids=seq_features.past_ids,
        past_embeddings=model.get_item_embeddings(
            seq_features.past_ids, 
            features=seq_features.past_payloads['features'], 
            user_attrs=seq_features.past_payloads.get('user_attrs', None)),
        past_payloads=seq_features.past_payloads,
    )

    if dtype is not None:
        shared_input_embeddings = shared_input_embeddings.to(dtype)
    k = 50
    eval_top_k_ids_n, _, _ = eval_state.candidate_index.get_top_k_outputs(
            query_embeddings=shared_input_embeddings[..., :model._embedding_module._item_embedding_dim],
            top_k_module=eval_state.top_k_module,
            k=k, 
            invalid_ids=seq_features.past_ids if filter_invalid_ids else None,
            return_embeddings=False,
        )
    eval_top_k_ids = eval_top_k_ids_n[:,:1]
    cal_1 = calculate_accuracy_of_prediction(eval_top_k_ids, target_ids, 1)
    cal_5 = calculate_accuracy_of_prediction(eval_top_k_ids, target_ids, 5)
    cal_n_targets = calculate_accuracy_of_prediction(eval_top_k_ids, target_ids, n_targets)
    hr_match = (target_ids[:, :1] == eval_top_k_ids_n[:, :k])
    hr_1 = hr_match[:, :1].any(dim=1)
    hr_5 = hr_match[:, :5].any(dim=1)
    hr_10 = hr_match[:, :10].any(dim=1)
    hr_50 = hr_match[:, :50].any(dim=1)
    output = {
        "HR@1": hr_1,  # 预测的top1行为是否在接下来命中
        "HR@5": hr_5,  # 预测的top5行为是否在接下来命中
        "HR@10": hr_10,  # 预测的top10行为是否在接下来命中
        "HR@50": hr_50,  # 预测的top50行为是否在接下来命中
        "Acc_Next1": cal_1,  # 预测的下次行为是否命中接下来一次支付行为
        "Acc_Next5": cal_5,  # 预测的下次行为在将来 5 次支付行为中出现
        f"Acc_Next{n_targets}": cal_n_targets,  # 预测的下次行为在将来 n_targets 次支付行为中出现
        # f"All_id_beh": eval_top_k_ids.view(-1),
        # f"Correct_id_beh": cal_1.view(-1)
    }
    return output
    
def preprocess_and_evaluate(predictions, targets, mask, k, device, ref, attrs):

    def string_to_tensor(str_list):
        ans = []
        unkmask = []
        for s in str_list: 
            if len(s.split('_')) != len(mask):
                ans.append([0] * len(mask))
                unkmask.append(False)
            else:
                ans.append(list(map(int, s.split('_'))))
                unkmask.append(True)
        unkmask, ans = torch.tensor(unkmask, dtype=torch.bool, device=device), torch.tensor(ans, dtype=torch.long, device=device).squeeze(0)
        if len(ans.shape) == 1:
            ans = ans.unsqueeze(0)
        return unkmask, ans
    # 应用mask并选择列
    def apply_mask(data_tensor, mask):
        return data_tensor[:, mask]
    
    pre_unkmask, predictions_tensor = string_to_tensor(predictions)
    # 计算准确率
    out = []
    for pre_mask_line, r, pred, target_list in zip(pre_unkmask, ref, predictions_tensor, targets):
        if pre_mask_line:
            out.append(torch.tensor([r.item()] * pred.size(-1), dtype=torch.bool, device=device))
            continue
        target_unkmask, target_tensor = string_to_tensor(target_list[:k])
        target_tensor = target_tensor[target_unkmask]
        if target_tensor.numel() > 0:
            matches = (pred == target_tensor).any(dim=0)
            out.append(matches)
        else:
            out.append(torch.tensor([r.item()] * pred.size(-1), dtype=torch.bool, device=device))
    out = torch.stack(out, dim=0).T
    feat_wise_correct = {k:vec for k, vec in zip(attrs, out)}
    
    # 将预测值转换并应用mask
    pre_unkmask, predictions_tensor = string_to_tensor(predictions)
    predictions_tensor = apply_mask(predictions_tensor, mask)
    # 计算准确率
    out = []
    for r, pred, target_list in zip(ref, predictions_tensor, targets):
        target_unkmask, target_tensor = string_to_tensor(target_list[:k])
        target_tensor = apply_mask(target_tensor, mask)[target_unkmask]
        if target_tensor.numel() > 0:
            matches = (pred == target_tensor).all(dim=1)
            out.append(matches.any().item())
        else:
            out.append(r.item())
    out = torch.tensor(out, dtype=torch.bool, device=device)
    out[~pre_unkmask] = ref[~pre_unkmask]
    return out, feat_wise_correct


def eval_recall_metrics_from_tensors(
    eval_state: EvalState,
    model: NDPModule,
    seq_features: SequentialFeatures,
    user_max_batch_size: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
) -> Dict[str, torch.Tensor]:
    target_ids = seq_features.past_ids[:, -1].unsqueeze(1)
    filtered_past_ids = seq_features.past_ids.detach().clone()
    filtered_past_ids[:, -1] = torch.zeros_like(target_ids.squeeze(1))
    return eval_metrics_v2_from_tensors(
        eval_state=eval_state,
        model=model,
        seq_features=SequentialFeatures(
            past_lengths=seq_features.past_lengths - 1,
            past_ids=filtered_past_ids,
            past_embeddings=seq_features.past_embeddings,
            past_payloads=seq_features.past_payloads,
        ),
        target_ids=target_ids,
        user_max_batch_size=user_max_batch_size,
        dtype=dtype,
    )


def _merge_length(x: torch.Tensor, world_size: int) -> float:
    if world_size > 1:
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    zero_sum_mask = (x[:, 0] == 0)
    x[:, 0] += 1
    return torch.where(zero_sum_mask, -1.0, (x[:, 1] / (x[:, 0])))

def _avg(x: torch.Tensor, world_size: int) -> float:
    _sum_and_numel = torch.tensor([x.sum(), x.numel()], dtype=torch.float32, device=x.device)
    if world_size > 1:
        dist.all_reduce(_sum_and_numel, op=dist.ReduceOp.SUM)
    return _sum_and_numel[0] / _sum_and_numel[1]

def _avg_mse(x: torch.Tensor, world_size: int) -> float:
    if world_size > 1:
        dist.all_reduce(x, op=dist.ReduceOp.AVG)
    return x

def _concat(x: torch.Tensor, world_size: int) -> torch.Tensor:
    x = x.view(-1)
    if world_size > 1:
        gather_list = [torch.zeros_like(x, device=x.device, dtype=x.dtype) for _ in range(world_size)]
        dist.all_gather(gather_list, x)
    else:
        return x
    return torch.cat(gather_list)


def _sum(x: torch.Tensor, world_size: int) -> int:
    _sum_value = torch.tensor([x.sum()], device=x.device)
    if world_size > 1:
        dist.all_reduce(_sum_value, op=dist.ReduceOp.SUM)
    return _sum_value


def add_to_summary_writer(
    writer: SummaryWriter,
    batch_id: int,
    metrics: Dict[str, torch.Tensor],
    prefix: str,
    world_size: int,
) -> None:
    for key, values in metrics.items():
        # avg_value = _avg(values, world_size)  
        # do the communcations once after each training step, instead of several times including (tensorboard, logging, mlflow, ...)
        if writer is not None:
            writer.add_scalar(f"{prefix}/{key}", values, batch_id)


def log_to_mlflow(
    batch_id: int,
    metrics: Dict[str, torch.Tensor],
    prefix: str,
    world_size: int,
    mlflow_run_id: Optional[str] = None
) -> None:
    for key, values in metrics.items():
        # avg_value = _avg(values, world_size)
        if os.environ.get("LOCAL_RANK", None) == "0":
            # with mlflow.start_run(run_id=mlflow_run_id):
            mlflow.log_metric(f"{prefix}/{key.replace('@','-')}", values, step=batch_id, run_id=mlflow_run_id)
