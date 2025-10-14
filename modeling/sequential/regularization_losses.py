from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import abc
from copy import deepcopy
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class SequenceAugmentation(abc.ABC):
    @abc.abstractmethod
    def __call__(self, seq):
        pass


class DropBehavior(SequenceAugmentation):
    def __init__(
        self, 
        strength: int = 1,
    ):
        self._strength = strength

    def __call__(self, input_embeddings, seq_features):
        # seq: input embeddings BxNxD
        # augment it
        _,N,_ = input_embeddings.size()
        seq_lengths = seq_features.past_lengths
        Nl = seq_lengths.max()
        pad_embedding = input_embeddings[0,-1,:].unsqueeze(0)
        n_drops = torch.poisson((self._strength *  seq_lengths / Nl)**0.5).to(int) + 1
        n_keeps = N - n_drops  # In the drop behavior augmentation, drop at least one behavior
        try:
            keep_inds = [torch.randperm(sl)[:n].sort()[0].to(input_embeddings.device) for sl, n in zip(seq_lengths, n_keeps)]
            aug_embedding = torch.stack([torch.cat([torch.index_select(s, 0, ki), pad_embedding.repeat(N-len(ki), 1)])
                                     for s, ki in zip(input_embeddings, keep_inds)], dim=0)
            aug_ids = torch.stack([torch.cat([torch.index_select(s, 0, ki), torch.zeros(N-len(ki)).to(s.device)])
                                     for s, ki in zip(seq_features.past_ids, keep_inds)], dim=0)
            aug_payloads = {
                "timestamps": torch.stack([torch.cat([torch.index_select(ts, 0, ki), torch.zeros(N-len(ki)).to(ts.device)])
                                     for ts, ki in zip(seq_features.past_payloads['timestamps'], keep_inds)], dim=0)
            }
            aug_lengths = seq_lengths - n_drops  # [B]
        except Exception as e:
            print(f"!!! Exception at modeling.sequential.regularization_losses.DropBehavior: {e}")
            aug_embedding, aug_ids, aug_lengths, aug_payloads = input_embeddings, seq_features.past_ids, seq_lengths, seq_features.past_payloads
        return aug_embedding, aug_ids, aug_lengths, aug_payloads


class ContrastiveRegularizationLoss(torch.nn.Module):
    def __init__(
        self, 
        model,
        contrast_layer: int = -1,  # -3
        aug_strength: int = 1
        ):
        super().__init__()
        self._model = model
        self._contrast_layer = contrast_layer
        self._aug = DropBehavior(strength = aug_strength)

    def forward(
        self, 
        input_embeddings: torch.Tensor,  #BxNxD
        cached_states: torch.Tensor,  # BxNxD
        seq_features,
    ) -> torch.Tensor:
        orig_index = torch.ops.fbgemm.asynchronous_complete_cumsum(seq_features.past_lengths)[1:] - 1
        orig_seq_embeddings = torch.index_select(cached_states[self._contrast_layer][-1], dim=0, index=orig_index)  # [B, D, ]

        aug_input_embedding, aug_past_ids, aug_lengths, aug_payloads = self._aug(
            input_embeddings,   # B,N,D
            seq_features
        )

        _, ang_cached_states = self._model(
                past_lengths=aug_lengths,
                past_ids=aug_past_ids,
                past_embeddings=aug_input_embedding,
                past_payloads=aug_payloads,
                return_cache_states = True
        )
        aug_index = torch.ops.fbgemm.asynchronous_complete_cumsum(aug_lengths)[1:] - 1
        aug_seq_embeddings = torch.index_select(ang_cached_states[self._contrast_layer][-1], dim=0, index=aug_index)

        #  Contrast seq_embedding and aug_seq_embedding
        emb_sim_loss = 1-F.cosine_similarity(orig_seq_embeddings, aug_seq_embeddings)

        return emb_sim_loss.mean()
