"""
The original generative-recommenders used a custom map-style dataset (data.dataset.DatasetV2) where the data is processed (truncation, padding) in the __getitem__ method of dataset.

The iceberg dataset (ProjectTableIterableDataset) that we have to use is closely coupled with the pyiceberg package. To process the input sequence, we can employ data collator for dataloader instead.

data collator function:
- return samples in the form:
ret = {     "user_id": user_id,
            "historical_ids": torch.tensor(historical_ids, dtype=torch.int64),
            "historical_ratings": torch.tensor(historical_ratings, dtype=torch.int64),
            "historical_timestamps": torch.tensor(historical_timestamps, dtype=torch.int64),
            "history_lengths": history_length,
            "target_ids": target_ids,
            "target_ratings": target_ratings,
            "target_timestamps": target_timestamps,
        }
- for row in iter(dataloader): row: Dict[str, torch.tensor(BxN)]
- _truncate_or_pad_seq

Besides, we need merchant feature embedding module, which maps the features of merchant to embedding vector.
"""

from typing import Dict, List, Optional, Tuple, Union
import re
from datetime import datetime
from joblib import Parallel, delayed

from collections import OrderedDict
import logging
import torch
from torch.utils.data import default_collate
from trainer.tokenizer import attr_columns, feature_columns, tokenize, bucketize_amount, ind_ind_feature_tokenizer, user_attr_tokenizer, continuous_attrs, ignore_loss_attrs
import numpy as np
import os
import numpy as np
import time
import torch.nn.functional as F

def optimize_risk_level_replace(seq):
    seq = seq.replace('A', '1')
    seq = seq.replace('B', '2')
    seq = seq.replace('C', '3')
    seq = seq.replace('D', '4')
    seq = seq.replace('S', '5')
    return seq

def create_adjustable_integer_bucket_boundaries(vocab_size, time_gap_max):
    num_points = vocab_size
    log_boundaries = np.linspace(np.log(0.5), np.log(time_gap_max), num_points)
    exp_boundaries = np.exp(log_boundaries)
    integer_boundaries = np.round(exp_boundaries).astype(int)
    unique_boundaries = np.unique(integer_boundaries)
    return torch.tensor(unique_boundaries)
USE_AMOUNT = True

def eval_as_list(x, ignore_last_n) -> List[int]:
    if x == "":
        return []
    x = x.split(',')[:-ignore_last_n] if ignore_last_n > 0 else x.split(',')
    y_list = [int(num) if num != 'unk' else -1 for num in x]
    return y_list

def eval_int_list_noreverse(x, target_len: int, ignore_last_n: int, shift_id_by: int, sampling_kept_mask: Optional[List[bool]]) -> Tuple[List[int], int]:
    y = eval_as_list(x, ignore_last_n=ignore_last_n)
    if sampling_kept_mask is not None:
        y = [x for x, kept in zip(y, sampling_kept_mask) if kept]
    y_len = len(y)
    if shift_id_by > 0:
        y = [x + shift_id_by for x in y]
    return y, y_len

def eval_str_list_noreverse(x, target_len: int, ignore_last_n: int, shift_id_by: int, sampling_kept_mask: Optional[List[bool]]) -> Tuple[List[str], int]:
    y, y_len = eval_int_list_noreverse(x, target_len, ignore_last_n, shift_id_by, sampling_kept_mask)
    return [str(yy) for yy in y], y_len

def _truncate_or_pad_seq(y: List[int], target_len: int, chronological: bool) -> List[int]:
    y_len = len(y)
    if y_len < target_len:
        y = y + [0] * (target_len - y_len)  # padding: 0 should be the padding token
    else:
        if not chronological:
            y = y[:target_len]  # 从新到旧，timestamp从大到小，截断出target_len长度的最近的行为序列
        else:
            y = y[-target_len:]  # 从旧到新，timestamp从小到大，截断出target_len长度的最近的行为序列
    assert len(y) == target_len
    return y

def _truncate_or_pad_seq_tensor(y, target_len: int):
    y_len = y.size(-1)
    if y_len < target_len:
        y = F.pad(y, (0, target_len - y_len), value=0)
    else:
        y = y[..., -target_len:]  # 从旧到新，timestamp从小到大，截断出target_len长度的最近的行为序列
    assert y.size(-1) == target_len
    return y

def argsort(seq, reverse=True):
    # reverse: A Boolean. False will sort ascending 小到大, True will sort descending 大到小.
    return sorted(range(len(seq)), key=seq.__getitem__, reverse=reverse)  
def str2time(dstr):
    return datetime(*[int(dstr[:4]), int(dstr[4:6]), int(dstr[6:8]), int(dstr[8:10])])
# def reorder(seq, inds):  # No need to reorder
#     if isinstance(seq, dict):
#         return {k:reorder(v, inds) for k, v in seq.items()}
#     res = [seq[i] if i < len(seq) else 1 for i in inds]
#     return torch.stack(res) if torch.is_tensor(seq) else res
def seq_mask(seq, mask):
    if isinstance(seq, dict):
        return {k:seq_mask(v, mask) for k,v in seq.items()}
    elif torch.is_tensor(seq) and seq.dim() == mask.dim():
        return torch.where(mask, 0, seq)
    else:
        return seq


def user_attrs_collator(data):
    user_attrs = {k:data[k] for k in attr_columns}
    tokenized_user_attrs = {k: user_attr_tokenizer.get(k+'_'+str(v), 0) for k, v in user_attrs.items()}
    return tokenized_user_attrs

def check_timestamps_sorted(timestamps):
    differences = timestamps[1:] - timestamps[:-1]
    if (differences >= 0).all():
        return None
    else:
        if int(os.environ.get('LOCAL_RANK')) == 0:
            logging.info(f"\n===============\ntimestamps not sorted!!!\n===============\n")
        return argsort(timestamps, reverse=False)

def get_timestamps_diff(timestamps):
    differences = timestamps[1:] - timestamps[:-1]
    assert (differences >= 0).all(), f"\n===============\ntimestamps not sorted!!!\n===============\n"
    return torch.cat([torch.tensor([0], device=differences.device, dtype=differences.dtype), differences]).to(dtype=torch.float)

class behseq_iceberg_collator:
    def __init__(
        self,
        tokenizer,
        ignore_last_n = 1,
        padding_length = 257,
        chronological = True,
        shift_id_by = 0,
        n_targets = 1,
    ):
        self._ignore_last_n = ignore_last_n
        self._padding_length = padding_length
        self._chronological = chronological
        self._shift_id_by = shift_id_by
        self._n_targets = n_targets
        if tokenizer == None:
            __import__('traceback').print_stack()
            raise ValueError("Tokenizer not specified happened.........")
        # import os
        # print(f"RANK: {os.environ.get('LOCAL_RANK')}, tokenizer: {len(tokenizer) if tokenizer else 0}")
        self._tokenizer = tokenizer
        self.timediff_bucket  = None
        if os.environ.get('TIMEDIFF_BUCKETS', None) is not None:
            self.timediff_bucket = create_adjustable_integer_bucket_boundaries(int(os.environ.get('TIMEDIFF_BUCKETS', None)) - 1, 12 * 31 * 24 * 60)

    def __one_call(
        self, 
        data: Dict[str, Union[str, int]],
    ):
        user_id = data['uin_secret'] if 'uin_secret' in data.keys() else data['uin']
        seq_length = data['seq_length']
        sampling_kept_mask = None
        user_attrs = user_attrs_collator(data)
        timestamps = eval_int_list_noreverse(  # timestamp
            data['uhb_timestamp_seq'], self._padding_length, self._ignore_last_n, 0, sampling_kept_mask=sampling_kept_mask
        )[0]  # (N)
        timestamps = [int(str2time(str(ymdh)).timestamp()) for ymdh in timestamps]
        amount_history = eval_int_list_noreverse(  # payment amount
                data['uhb_amount_seq'], self._padding_length, self._ignore_last_n, 0, sampling_kept_mask=sampling_kept_mask
                )[0]

        if 'uhb_amount_bucket_seq' in data.keys():
            amounts_bucket = eval_str_list_noreverse(  # payment amount
                data['uhb_amount_bucket_seq'], self._padding_length, self._ignore_last_n, 0, sampling_kept_mask=sampling_kept_mask
            )[0]
        else:
            amounts_bucket = bucketize_amount(amount_history)

        data['uhb_mch_sec_risklvl_seq'] = optimize_risk_level_replace(data['uhb_mch_sec_risklvl_seq'])
        # Cartesian producted and tokenized features (feature_columns).
        features = OrderedDict((k, eval_str_list_noreverse(data[k], self._padding_length, self._ignore_last_n, 0, sampling_kept_mask=sampling_kept_mask)[0]) 
                                for k in ignore_loss_attrs + feature_columns)
        features['uhb_amount_bucket_seq'] = amounts_bucket
        beh_tokens = OrderedDict((k, features[k]) for k in feature_columns)
        beh_tokens['uhb_amount_bucket_seq'] = amounts_bucket
        tokenized_feature_ids, _ = tokenize(beh_tokens, return_token_str=True, tokenizer=self._tokenizer)  # (NxM) -> N, cartesian product + tokenization
        features = {k: [ind_ind_feature_tokenizer.get(k + '_' + x, 0) for x in v] for k, v in features.items()}
        timestamps = torch.tensor(timestamps,dtype=torch.int64)
        reorder_idx = check_timestamps_sorted(timestamps)
        
        tokenized_feature_ids = torch.tensor(tokenized_feature_ids, dtype=torch.int64)
        features = {k: torch.tensor(v,dtype=torch.int64) for k, v in features.items()}
        amount_history = torch.tensor(amount_history,dtype=torch.int64)
        
        if reorder_idx:
            timestamps = timestamps[reorder_idx]
            tokenized_feature_ids = tokenized_feature_ids[reorder_idx]
            features = {k: v[reorder_idx] for k, v in features.items()}
            amount_history = amount_history[reorder_idx]
        if self.timediff_bucket is not None:
            timestamps_diff = get_timestamps_diff(timestamps)
            features["timestamps_diff"] = torch.bucketize(timestamps_diff, self.timediff_bucket.to(device=timestamps_diff.device)) + 1
        
        historical_timestamps = timestamps[:-self._n_targets]  # 历史时间
        historical_features_ids = tokenized_feature_ids[:-self._n_targets]  # 历史行为，此处从新到旧，timestamp 从大到小
        historical_features = {k:v[:-self._n_targets] for k, v in features.items()}
        historical_amounts = amount_history[:-self._n_targets]

        target_timestamps = timestamps[-self._n_targets:]  # timestamps[0]  # 最新N个时间
        target_features_ids = tokenized_feature_ids[-self._n_targets:]  # tokenized_feature_ids[0]  # 最新N个行为
        target_features = {k:v[-self._n_targets:] for k,v in features.items()}
        target_amounts = amount_history[-self._n_targets:]  # amount_history[0]

        max_seq_len = self._padding_length - 1

        history_length = min(len(historical_timestamps), max_seq_len)

        historical_features_ids = _truncate_or_pad_seq_tensor(historical_features_ids, max_seq_len)

        historical_amounts = _truncate_or_pad_seq_tensor(historical_amounts, max_seq_len)

        historical_features = {k: _truncate_or_pad_seq_tensor(v, max_seq_len) for k, v in historical_features.items()}

        historical_timestamps = _truncate_or_pad_seq_tensor(historical_timestamps, max_seq_len)

        ret = {
            "user_id": user_id,
            "user_attrs": user_attrs,
            "historical_ids": historical_features_ids,
            "historical_features": historical_features,
            "historical_amounts": historical_amounts,
            "historical_timestamps": historical_timestamps,
            "history_lengths": history_length,
            "target_features": {k:v for k,v in target_features.items()},
            "target_ids": target_features_ids,
            "target_ratings": target_features_ids,
            "target_amounts": target_amounts,
            "target_timestamps": target_timestamps,
            # 'feature_token_str': feature_token_str[:50],
        }
        # print(ret)
        return ret
    
    def __call__(
        self, 
        batch: List[Dict[str, Union[str, int]]]
    ):
        # one_call = lambda data: self.__one_call(data)

        # try:
        collate_result = [self.__one_call(b) for b in batch]
        return default_collate(collate_result)
        # except Exception as e:
        #     if int(os.environ.get('LOCAL_RANK')) == 0:
        #         logging.error(f"{e}")
        #     return None
