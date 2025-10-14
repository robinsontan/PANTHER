

"""
Implements HSTU (Hierarchical Sequential Transduction Unit) in 
Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations
(https://arxiv.org/abs/2402.17152).
"""

import abc
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from indexing.candidate_index import CandidateIndex
from modeling.initialization import truncated_normal
from modeling.ndp_module import NDPModule
from modeling.sequential.embedding_modules import EmbeddingModule
from modeling.sequential.features import SequentialFeatures
from modeling.sequential.input_features_preprocessors import (
    InputFeaturesPreprocessorModule,
)
from modeling.sequential.mytool import *
from modeling.sequential.output_postprocessors import OutputPostprocessorModule
from modeling.sequential.utils import (
    batch_scatter_embeddings,
    get_current_embeddings,
    jagged_or_dense_index_select_dim0,
    jagged_or_dense_repeat_interleave_dim0,
)
from modeling.similarity_module import GeneralizedInteractionModule

TIMESTAMPS_KEY = "timestamps"

class RelativeAttentionBiasModule(torch.nn.Module):

    @abc.abstractmethod
    def forward(
        self,
        all_timestamps: torch.Tensor,
    ) -> torch.Tensor:
       """
       Args:
           all_timestamps: [B, N] x int64
       Returns:
           torch.float tensor broadcastable to [B, N, N]
       """
       pass


class RelativePositionalBias(RelativeAttentionBiasModule):

    def __init__(self, max_seq_len: int) -> None:
        super().__init__()

        self._max_seq_len: int = max_seq_len
        self._w = torch.nn.Parameter(
            torch.empty(2 * max_seq_len - 1).normal_(mean=0, std=0.02),
        )

    def forward(
        self,
        all_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        del all_timestamps
        n: int = self._max_seq_len
        t = F.pad(self._w[:2 * n - 1], [0, n]).repeat(n)
        t = t[..., :-n].reshape(1, n, 3 * n - 2)
        r = (2 * n - 1) // 2
        return t[..., r:-r]


class RelativeBucketedTimeAndPositionBasedBias(RelativeAttentionBiasModule):
    """
    Bucketizes timespans based on ts(next-item) - ts(current-item).
    """
    def __init__(
        self,
        max_seq_len: int,
        num_buckets: int,
        bucketization_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        super().__init__()

        self._max_seq_len: int = max_seq_len
        self._ts_w = torch.nn.Parameter(
            torch.empty(num_buckets + 1).normal_(mean=0, std=0.02),
        )
        self._pos_w = torch.nn.Parameter(
            torch.empty(2 * max_seq_len - 1).normal_(mean=0, std=0.02),
        )
        self._num_buckets: int = num_buckets
        self._bucketization_fn: Callable[[torch.Tensor], torch.Tensor] = bucketization_fn

    def forward(
        self,
        all_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            all_timestamps: (B, N).
        Returns:
            (B, N, N).
        """
        B = all_timestamps.size(0)
        N = self._max_seq_len
        t = F.pad(self._pos_w[:2 * N - 1], [0, N]).repeat(N)
        t = t[..., :-N].reshape(1, N, 3 * N - 2)
        r = (2 * N - 1) // 2

        # [B, N + 1] to simplify tensor manipulations.
        ext_timestamps = torch.cat([all_timestamps, all_timestamps[:, N-1:N]], dim=1)
        # causal masking. Otherwise [:, :-1] - [:, 1:] works
        bucketed_timestamps = torch.clamp(
            self._bucketization_fn(ext_timestamps[:, 1:].unsqueeze(2) - ext_timestamps[:, :-1].unsqueeze(1)),
            min=0,
            max=self._num_buckets,
        ).detach()
        rel_pos_bias = t[:, :, r:-r]
        rel_ts_bias = torch.index_select(self._ts_w, dim=0, index=bucketed_timestamps.view(-1)).view(B, N, N)
        return rel_pos_bias + rel_ts_bias


HSTUCacheState = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class SequentialTransductionUnitJagged(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        linear_hidden_dim: int,
        attention_dim: int,
        dropout_ratio: float,
        attn_dropout_ratio: float,
        num_heads: int,
        linear_activation: str,
        relative_attention_bias_module: Optional[RelativeAttentionBiasModule] = None,
        normalization: str = "rel_bias",
        linear_config: str = "uvqk",
        concat_ua: bool = False,
        epsilon: float = 1e-6,
        max_length: Optional[int] = None,
        block_type: str = "hstu",  # New parameter to control block type
        residual: bool = True,
        max_sequence_len = 256,
        n_patterns = None,
        conv_kernelsize = None,
        attn_scale: int = None,  # Default value for scale
        attn_mod: int = None,  # Default value for sparse
        attn_mod_id: int = None,  # Default value for sparse
    ) -> None:
        super().__init__()
        self._embedding_dim: int = embedding_dim
        self._linear_dim: int = linear_hidden_dim
        self._attention_dim: int = attention_dim
        self._dropout_ratio: float = dropout_ratio
        self._attn_dropout_ratio: float = attn_dropout_ratio
        self._num_heads: int = num_heads
        self._rel_attn_bias: Optional[RelativeAttentionBiasModule] = relative_attention_bias_module
        self._normalization: str = normalization
        self._linear_config: str = linear_config
        self._block_type: str = block_type  # Store block type
        self._residual: bool = residual
        if self._linear_config == "uvqk":
            if block_type == "attention_gate":
                self._uvqk = torch.nn.Parameter(
                    torch.empty((embedding_dim, linear_hidden_dim * 2 * num_heads + attention_dim * num_heads * 2 + attention_dim * 2)).normal_(mean=0, std=0.02),
                )
            else:
                self._uvqk = torch.nn.Parameter(
                    torch.empty((embedding_dim, linear_hidden_dim * 2 * num_heads + attention_dim * num_heads * 2)).normal_(mean=0, std=0.02),
                )
        else:
            raise ValueError(f"Unknown linear_config {self._linear_config}")
        self._linear_activation: str = linear_activation
        self._concat_ua: bool = concat_ua
        self._o = torch.nn.Linear(in_features=linear_hidden_dim * num_heads * (3 if concat_ua else 1), out_features=embedding_dim)
        torch.nn.init.xavier_uniform_(self._o.weight)
        self._eps: float = epsilon
        
        self.attn_scale = attn_scale
        self.attn_mod = attn_mod
        self.attn_mod_id = attn_mod_id

        if block_type in ["scale", "sparse"]:
            self.gate = GatedUnit(embedding_dim)
            
        if block_type == "pattern":
            self.pattern_matrix = nn.Parameter(torch.randn(embedding_dim, n_patterns))
            self.w_m = nn.Embedding(max_sequence_len + 12, n_patterns)
            
        if block_type == "conv_softmax":
            self.conv_kernelsize = conv_kernelsize
            self.conv = CausalConv1D(embedding_dim, n_patterns, self.conv_kernelsize)
            
            # self.pattern_matrix = nn.Parameter(torch.randn(n_patterns, embedding_dim))
            # torch.nn.init.xavier_uniform_(self.pattern_matrix)
            self.project_v = nn.Linear(embedding_dim * conv_kernelsize, embedding_dim)
            
            self.batch_norm = nn.BatchNorm1d(n_patterns)
            self.module_gate = GatedUnit(embedding_dim)
            
        if block_type == "xattn_pattern":
            self.conv_kernelsize = conv_kernelsize
            self.conv = CausalConv1D(embedding_dim, embedding_dim, self.conv_kernelsize)
            self.pattern_matrix = nn.Parameter(torch.randn(n_patterns, embedding_dim))
            torch.nn.init.xavier_uniform_(self.pattern_matrix)
            self.pattern_cross_attn_layer = nn.MultiheadAttention(embedding_dim, num_heads, dropout_ratio, batch_first=True)
            self.module_gate = GatedUnit(embedding_dim)
            
        if block_type == "dynamic_conv":
            kernel_size_list = [i for i in range(32) if i % 2 == 1]
            self.conv = DynamicConvolution(embedding_dim, embedding_dim, kernel_size_list)
            self.module_gate = GatedUnit(embedding_dim)
            
    def _norm_input(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, normalized_shape=[self._embedding_dim], eps=self._eps)

    def _norm_attn_output(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, normalized_shape=[self._linear_dim * self._num_heads], eps=self._eps)

    def forward(
        self,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        all_timestamps: Optional[torch.Tensor],
        invalid_attn_mask: torch.Tensor,
        delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache: Optional[HSTUCacheState] = None,
        return_cache_states: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: (\sum_i N_i, D) x float.
            x_offsets: (B + 1) x int32.
            all_timestamps: optional (B, N) x int64.
            invalid_attn_mask: (B, N, N) x float, each element in {0, 1}.
            delta_x_offsets: optional 2-tuple ((B,) x int32, (B,) x int32).
                For the 1st element in the tuple, each element is in [0, x_offsets[-1]). For the
                2nd element in the tuple, each element is in [0, N).
            cache: Optional 4-tuple of (v, padded_q, padded_k, output) from prior runs,
                where all except padded_q, padded_k are jagged.
        Returns:
            x' = f(x), (\sum_i N_i, D) x float.
        """
        n: int = invalid_attn_mask.size(-1)
        if delta_x_offsets is not None:
            # In this case, for all the following code, x, u, v, q, k become restricted to
            # [delta_x_offsets[0], :].
            assert cache is not None
            x = x[delta_x_offsets[0], :]
            cached_v, cached_q, cached_k, cached_outputs = cache
        normed_x = self._norm_input(x)

        attn_scale, attn_mod, attn_mod_id = self.attn_scale, self.attn_mod, self.attn_mod_id
        if "conv_softmax" in self._block_type:
            padded_x = torch.ops.fbgemm.jagged_to_padded_dense(values=normed_x, offsets=[x_offsets], max_lengths=[n], padding_value=0.0) #B, N, D
            pattern_x = self.conv(padded_x) # B, N, P
            pattern_x = self.batch_norm(pattern_x.transpose(1, 2)).transpose(1, 2) # B, N, P
            pattern_x = F.softmax(pattern_x, dim=-1) # B, N, P
            
            pattern_matrix = self.project_v(self.conv.weight.flatten(start_dim=1)) # P, D
            # pattern_matrix = self.pattern_matrix
            pattern_x = torch.einsum('bnp,pd->bnd', pattern_x, pattern_matrix)# B, N, D
            pattern_x = torch.ops.fbgemm.dense_to_jagged(pattern_x, [x_offsets])[0]
            normed_x = self.module_gate(normed_x, pattern_x)
            
        if "xattn_pattern" in self._block_type:
            padded_x = torch.ops.fbgemm.jagged_to_padded_dense(values=normed_x, offsets=[x_offsets], max_lengths=[n], padding_value=0.0)
            padded_x = self.conv(padded_x)
            pattern_matrix = self.pattern_matrix.unsqueeze(0).repeat(padded_x.size(0), 1, 1)
            pattern_out, _ = self.pattern_cross_attn_layer(padded_x, pattern_matrix, pattern_matrix)
            pattern_out = torch.ops.fbgemm.dense_to_jagged(pattern_out, [x_offsets])[0]
            normed_x = self.module_gate(normed_x, pattern_out)
            
        if "dynamic_conv" in self._block_type:
            padded_x = torch.ops.fbgemm.jagged_to_padded_dense(values=normed_x, offsets=[x_offsets], max_lengths=[n], padding_value=0.0)
            padded_x = self.conv(padded_x)
            new_normed_x = torch.ops.fbgemm.dense_to_jagged(padded_x, [x_offsets])[0]
            normed_x = self.module_gate(new_normed_x, normed_x)
            
        if self._linear_config == "uvqk":
            batched_mm_output = torch.mm(normed_x, self._uvqk)
            if self._linear_activation == "silu":
                batched_mm_output = F.silu(batched_mm_output)
            elif self._linear_activation == "none":
                batched_mm_output = batched_mm_output

            if "attention_gate" in self._block_type:
                u, v, q, k, gq, gk = torch.split(
                    batched_mm_output,
                    [self._linear_dim * self._num_heads, self._linear_dim * self._num_heads, self._attention_dim * self._num_heads, self._attention_dim * self._num_heads, self._attention_dim, self._attention_dim],
                    dim=1,
                )
            else:
                u, v, q, k = torch.split(
                    batched_mm_output,
                    [self._linear_dim * self._num_heads, self._linear_dim * self._num_heads, self._attention_dim * self._num_heads, self._attention_dim * self._num_heads],
                    dim=1,
                )
        else:
            raise ValueError(f"Unknown self._linear_config {self._linear_config}")

        if delta_x_offsets is not None:
            v = cached_v.index_copy_(dim=0, index=delta_x_offsets[0], source=v)

        B: int = x_offsets.size(0) - 1
        if self._normalization == "rel_bias" or self._normalization == "hstu_rel_bias":
            if delta_x_offsets is not None:
                padded_q, padded_k = cached_q, cached_k
                flattened_offsets = delta_x_offsets[1] + torch.arange(start=0, end=B * n, step=n, device=delta_x_offsets[1].device, dtype=delta_x_offsets[1].dtype)
                padded_q = padded_q.view(B * n, -1).index_copy_(
                    dim=0, index=flattened_offsets, source=q,
                ).view(B, n, -1)
                padded_k = padded_k.view(B * n, -1).index_copy_(
                    dim=0, index=flattened_offsets, source=k,
                ).view(B, n, -1)
            else:
                padded_q = torch.ops.fbgemm.jagged_to_padded_dense(
                    values=q, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
                )
                padded_k = torch.ops.fbgemm.jagged_to_padded_dense(
                    values=k, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
                )
            if "expm" in self._block_type:
                qk_attn = torch.einsum(
                "bnhd,bmhd->bhnm",
                l2_normalize_last_dim(padded_q.view(B, n, self._num_heads, self._attention_dim)),
                l2_normalize_last_dim(padded_k.view(B, n, self._num_heads, self._attention_dim)),
                )
                
                qk_attn += 1.0
                qk_attn = qk_attn / (qk_attn.sum(dim=-1, keepdim=True) + 1e-12)
            else:
                qk_attn = torch.einsum(
                    "bnhd,bmhd->bhnm",
                    padded_q.view(B, n, self._num_heads, self._attention_dim),
                    padded_k.view(B, n, self._num_heads, self._attention_dim),
                )
            if all_timestamps is not None:
                qk_attn = qk_attn + self._rel_attn_bias(all_timestamps).unsqueeze(1)
            qk_attn = F.silu(qk_attn) / n
            
            if "attention_gate" in self._block_type:
                padded_gate_q = torch.ops.fbgemm.jagged_to_padded_dense(values=gq, offsets=[x_offsets], max_lengths=[n], padding_value=0.0)
                padded_gate_k = torch.ops.fbgemm.jagged_to_padded_dense(values=gk, offsets=[x_offsets], max_lengths=[n], padding_value=0.0)

                attn_gate = torch.einsum(
                    "bnhd,bmhd->bhnm",
                    padded_gate_q.view(B, n, 1, self._attention_dim),
                    padded_gate_k.view(B, n, 1, self._attention_dim),
                )
                attn_gate = F.sigmoid(attn_gate) # [B, 1, N, N]
                qk_attn = qk_attn * attn_gate # gate operation
                
            elif "pattern" in self._block_type:
                padded_x = torch.ops.fbgemm.jagged_to_padded_dense(
                    values=normed_x, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
                    )
                pattern_scores = torch.matmul(padded_x, self.pattern_matrix)
                pattern_scores = F.softmax(pattern_scores, dim=-1) # [B, N, P]
                wm = self.w_m(torch.arange(padded_x.size(1), device=x.device)).T # [P, N]

                qk_attn = torch.matmul(pattern_scores, wm).unsqueeze(1) * qk_attn

            elif "scale" in self._block_type and attn_scale > 0 and attn_scale < qk_attn.size(3) - 1:
                device = qk_attn.device
                idx = torch.arange(qk_attn.size(3), device=device)
                mask = (idx[None, None, :] >= (idx[:, None] - attn_scale)) & (idx[None, None, :] <= (idx[:, None] + attn_scale))
                qk_attn *= mask.unsqueeze(0)

            elif "sparse" in self._block_type and attn_mod_id >= 0:
                device = qk_attn.device
                idx = torch.arange(qk_attn.size(3), device=device)
                mask = ((idx[None, :] - idx[:, None]) % attn_mod == attn_mod_id) | torch.eye(qk_attn.size(3), dtype=torch.bool, device=device)
                qk_attn *= mask.unsqueeze(0).unsqueeze(0)

            qk_attn = qk_attn * invalid_attn_mask.unsqueeze(0).unsqueeze(0)
            attn_output = torch.ops.fbgemm.dense_to_jagged(
                torch.einsum(
                    "bhnm,bmhd->bnhd",
                    qk_attn,
                    torch.ops.fbgemm.jagged_to_padded_dense(v, [x_offsets], [n]).reshape(B, n, self._num_heads, self._linear_dim)
                ).reshape(B, n, self._num_heads * self._linear_dim),
                [x_offsets],
            )[0]
        else:
            raise ValueError(f"Unknown normalization method {self._normalization}")

        attn_output = attn_output if delta_x_offsets is None else attn_output[delta_x_offsets[0], :]
        if self._concat_ua:
            a = self._norm_attn_output(attn_output)
            o_input = torch.cat([u, a, u * a], dim=-1)
        else:
            o_input = u * self._norm_attn_output(attn_output)

        new_outputs = self._o(
            F.dropout(
                o_input,
                p=self._dropout_ratio,
                training=self.training,
            )
        )

        if "scale" in self._block_type or "sparse" in self._block_type:
            new_outputs = self.gate(x, new_outputs)

        if self._residual:
            new_outputs += x

        if delta_x_offsets is not None:
            new_outputs = cached_outputs.index_copy_(dim=0, index=delta_x_offsets[0], source=new_outputs)

        if return_cache_states and delta_x_offsets is None:
            v = v.contiguous()

        return new_outputs, (v, padded_q, padded_k, new_outputs)
