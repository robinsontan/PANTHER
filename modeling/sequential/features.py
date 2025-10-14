

import abc
from typing import Dict, NamedTuple, Optional, Tuple

import torch


class SequentialFeatures(NamedTuple):
    # (B,) x int64. Requires past_lengths[i] > 0 \forall i.
    past_lengths: torch.Tensor
    # (B, N,) x int64. 0 denotes valid ids.
    past_ids: torch.Tensor
    # (B, N, D) x float.
    past_embeddings: Optional[torch.Tensor]
    # Implementation-specific payloads.
    # e.g., past timestamps, past event_types (e.g., clicks, likes), etc.
    past_payloads: Dict[str, torch.Tensor]


def get_feature_processer():
    return {
        "ml-1m": movielens_seq_features_from_row, 
        "ml-20m": movielens_seq_features_from_row, 
        "ml-1b": movielens_seq_features_from_row, 
        "amzn-books": movielens_seq_features_from_row, 
        "biz-payer-beh-seq": biz_payer_beh_seq_features_from_row
    }


def movielens_seq_features_from_row(
    row,
    device: torch.device,
    max_output_length: int,
) -> Tuple[SequentialFeatures, torch.Tensor, torch.Tensor]:
    historical_lengths = row["history_lengths"].to(device)  # [B]
    historical_ids = row["historical_ids"].to(device)  # [B, N]
    historical_ratings = row["historical_ratings"].to(device)
    historical_timestamps = row["historical_timestamps"].to(device)
    target_ids = row["target_ids"].to(device).unsqueeze(1)  # [B, 1]
    target_ratings = row["target_ratings"].to(device).unsqueeze(1)
    target_timestamps = row["target_timestamps"].to(device).unsqueeze(1)
    if max_output_length > 0:
        B = historical_lengths.size(0)
        historical_ids = torch.cat([
            historical_ids,
            torch.zeros((B, max_output_length), dtype=historical_ids.dtype, device=device),
        ], dim=1)
        historical_ratings = torch.cat([
            historical_ratings,
            torch.zeros((B, max_output_length), dtype=historical_ratings.dtype, device=device),
        ], dim=1)
        historical_timestamps = torch.cat([
            historical_timestamps,
            torch.zeros((B, max_output_length), dtype=historical_timestamps.dtype, device=device),
        ], dim=1)
        historical_timestamps.scatter_(
            dim=1,
            index=historical_lengths.view(-1, 1),
            src=target_timestamps.view(-1, 1),
        )
        # print(f"historical_ids.size()={historical_ids.size()}, historical_timestamps.size()={historical_timestamps.size()}")
    features = SequentialFeatures(
        past_lengths=historical_lengths,
        past_ids=historical_ids,
        past_embeddings=None,
        past_payloads={
            "timestamps": historical_timestamps,
            "ratings": historical_ratings,
        },
    )
    return features, target_ids, target_ratings


def biz_payer_beh_seq_features_from_row(
    row,
    device: torch.device,
    max_output_length: int,
) -> Tuple[SequentialFeatures, torch.Tensor, torch.Tensor]:
    # print(row)
    uins = row["user_id"]
    # __import__('ipdb').set_trace()
    user_attrs = {k: v.to(device).unsqueeze(-1) for k,v in row["user_attrs"].items()} if 'user_attrs' in row else None
    historical_lengths = row["history_lengths"].to(device)  # [B]
    historical_ids = row["historical_ids"].to(device)  # [B, N]  Historical feature ids
    historical_features = {k:v.to(device) for k,v in row["historical_features"].items()}
    historical_timestamps = row["historical_timestamps"].to(device)
    historical_amounts = row["historical_amounts"].to(device)
    target_ids = row["target_ids"].to(device) # .unsqueeze(-1)  # [B, 1]
    target_features = {k:v.to(device).unsqueeze(-1) for k,v in row["target_features"].items()}
    target_amounts = row["target_amounts"].to(device)
    target_timestamps = row["target_timestamps"].to(device).unsqueeze(-1)

    if max_output_length > 0:
        B = historical_lengths.size(0)
        historical_ids = torch.cat([
            historical_ids,
            torch.zeros((B, max_output_length), dtype=historical_ids.dtype, device=device),
        ], dim=1)
        historical_timestamps = torch.cat([
            historical_timestamps,
            torch.zeros((B, max_output_length), dtype=historical_timestamps.dtype, device=device),
        ], dim=1)
        historical_features = {k: torch.cat([v, torch.zeros((B, max_output_length), dtype=v.dtype, device=device)], dim=1) for k,v in historical_features.items()}
        # Make sure the ind_feature_token_ids have the same length with historical_ids

        historical_timestamps.scatter_(
            dim=1,
            index=historical_lengths.view(-1, 1),
            src=target_timestamps.view(-1, 1),
        )
        # print(f"historical_ids.size()={historical_ids.size()}, historical_timestamps.size()={historical_timestamps.size()}")
    features = SequentialFeatures(
        past_lengths=historical_lengths,
        past_ids=historical_ids,
        past_embeddings=None,
        past_payloads={
            "timestamps": historical_timestamps,
            "uins": uins,
            "user_attrs": user_attrs,
            "ratings": torch.zeros_like(historical_timestamps),
            "features": historical_features,
            "target_features": target_features,
            "target_timestamps": target_timestamps,
        },
    )
    return features, target_ids, 0
