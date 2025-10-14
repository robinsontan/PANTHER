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

TIMESTAMPS_KEY = "unix_time"

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
        t = F.pad(self._w[: 2 * n - 1], [0, n]).repeat(n)
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
        self._bucketization_fn: Callable[[torch.Tensor], torch.Tensor] = (
            bucketization_fn
        )

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

        # [B, N + 1] to simplify tensor manipulations.
        ext_timestamps = torch.cat(
            [all_timestamps, all_timestamps[:, N - 1 : N]], dim=1
        )
        # causal masking. Otherwise [:, :-1] - [:, 1:] works
        bucketed_timestamps = torch.clamp(
            self._bucketization_fn(
                ext_timestamps[:, 1:].unsqueeze(2) - ext_timestamps[:, :-1].unsqueeze(1)
            ),
            min=0,
            max=self._num_buckets,
        ).detach()
        rel_pos_bias = torch.flip(self._pos_w.unfold(0,N,1), (0,))
        rel_ts_bias = torch.index_select(
            self._ts_w, dim=0, index=bucketed_timestamps.view(-1)
        ).view(B, N, N)
        return rel_pos_bias + rel_ts_bias


HSTUCacheState = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class SequentialTransductionUnitJagged(torch.nn.Module):
    def __init__(
        self,
        item_embedding_dim: int,
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
        block_type = [],  # New parameter to control block type
        residual: bool = True,
        max_sequence_len = 256,
        n_patterns = None,
        conv_kernelsize = None,
        attn_scale: int = None,  # Default value for scale
        attn_mod: int = None,  # Default value for sparse
        attn_mod_id: int = None,  # Default value for sparse
        dim_down: bool = False,
        dim_down_dim: int = None,
        backbone_pointer = None,
    ) -> None:
        super().__init__()
        self._item_embedding_dim: int = item_embedding_dim
        self._embedding_dim: int = embedding_dim
        self._linear_dim: int = linear_hidden_dim
        self._attention_dim: int = attention_dim
        self._dropout_ratio: float = dropout_ratio
        self._attn_dropout_ratio: float = attn_dropout_ratio
        self._num_heads: int = num_heads
        self._rel_attn_bias: Optional[RelativeAttentionBiasModule] = relative_attention_bias_module
        self._normalization: str = normalization
        self._linear_config: str = linear_config
        self.block_type = block_type  # Store block type
        self._residual: bool = residual
        self._dim_down = dim_down
        self._linear_activation: str = linear_activation
        self._concat_ua: bool = concat_ua
        self._eps: float = epsilon
        
        self.attn_scale = attn_scale
        self.attn_mod = attn_mod
        self.attn_mod_id = attn_mod_id
        if "conv_attn" in block_type:
            xattn_embedding_dim = (embedding_dim // num_heads) * num_heads
            self.in_xattn_proj = nn.Linear(embedding_dim, xattn_embedding_dim)
            self.layer = Conv1dMultiHeadAttention(xattn_embedding_dim, embedding_dim, linear_hidden_dim, attention_dim, num_heads, 3, dropout_ratio, self.training)
            return
        if "subseq_attngate_v2" in block_type:
            self.layer_subseq_attngate = Conv1d_attngate_v2(embedding_dim, num_heads, 3)
        if "subseq_attngate" in block_type or "subseq_attngate_v3" in block_type or "subseq_attngate_v4" in block_type:
            self.layer_subseq_attngate = Conv1d_attngate(embedding_dim, attention_dim, num_heads, 3)
        if "subseq_attngate_v5" in block_type:
            self.layer_subseq_attngate = Conv1d_attngate(embedding_dim, attention_dim, num_heads, 6)
        if "subseq_attngate_v6" in block_type:
            self.layer_subseq_attngate = Conv1d_attngate_v6(embedding_dim, attention_dim, num_heads, 3)
        if "attention_gate" in self.block_type or "subseq_attngate_v6" in block_type or "subseq_attngate_v5" in block_type or "subseq_attngate" in self.block_type or "subseq_attngate_v2" in self.block_type or "subseq_attngate_shift_v2" in self.block_type or "pattern_v2" in self.block_type:
            self.attn_gate_factor = nn.Parameter(torch.empty(1).normal_(mean=0, std=0.02))
        if "subseq_emb" in block_type:
            self.subseq_emb_agg = GatedUnit(embedding_dim, item_embedding_dim*3)
        if "subseq_emb_v2" in block_type:
            self.subseq_emb_agg = xattn_block(input_dim=embedding_dim, sub_seqemb_dim=item_embedding_dim, num_heads=num_heads, hstu_attn=True)
        if "subseq_emb_v3" in block_type:
            self.subseq_emb_conv = CausalConv1D(item_embedding_dim, item_embedding_dim, 3, residual=True, has_bias=True, use_act=True)
            self.subseq_emb_agg = GatedUnit(embedding_dim, item_embedding_dim)
        if "subseq_emb_v4" in block_type:
            self.subseq_emb_conv = CausalConv1D(embedding_dim, embedding_dim, 3, residual=True, has_bias=True, use_act=True)
            # self.subseq_emb_agg = GatedUnit(embedding_dim)
        if "subseq_emb_v5" in block_type:
            self.subseq_emb_agg = GatedUnit(embedding_dim)
        if "subseq_emb_v6" in block_type:
            self.subseq_emb_conv_gate = CausalConv1D(embedding_dim, embedding_dim, 3, residual=False, has_bias=True, use_act=False)
        if "subseq_emb_v7" in block_type or "subseq_emb_v8" in block_type or "subseq_emb_v9" in block_type :
            self.subseq_emb_conv_gate = nn.Linear(embedding_dim, embedding_dim)
        if "subseq_emb_sim" in block_type and "v1" in block_type:
            self.gate = GatedUnit(embedding_dim, item_embedding_dim * 11)
        if "subseq_emb_sim" in block_type and "v3" in block_type:
            self.gate = GatedUnit(embedding_dim, item_embedding_dim)
        if "subseq_emb_sim" in block_type and "v3_1" in block_type:
            self.gate = GatedUnit(embedding_dim, item_embedding_dim*5)
        if "subseq_attnbasev2" in self.block_type:
            num_bit_count = 9
            self.mlp_merge = nn.Linear(num_bit_count + 1, 1)
            self.gate = GatedUnit(1)
        self.module_gate = None
        if "pattern_enc" in block_type:
            
            self.enc_p = nn.Linear(item_embedding_dim*3, embedding_dim)
            if self._embedding_dim % self._num_heads != 0:
                xattn_embedding_dim = (embedding_dim // num_heads) * num_heads
                self.in_xattn_proj = nn.Linear(embedding_dim, xattn_embedding_dim)
                self.out_xattn_proj = nn.Linear(xattn_embedding_dim, embedding_dim)
                self.xattn_embedding_dim = xattn_embedding_dim
                self.pattern_cross_attn_layer = nn.MultiheadAttention(xattn_embedding_dim, num_heads, dropout_ratio, batch_first=True)
            else:
                self.pattern_cross_attn_layer = nn.MultiheadAttention(self._embedding_dim, num_heads, dropout_ratio, batch_first=True)
            self.module_gate = GatedUnit(embedding_dim)

        if "scale" in block_type or "sparse" in block_type:
            self.gate = GatedUnit(embedding_dim)

        if "pattern" in block_type or "pattern_v2" in self.block_type:
            self.pattern_matrix = nn.Parameter(torch.randn(embedding_dim, n_patterns)) # TODO：Whether the pattern matrix is ​​shared between different layers
            self.w_m = nn.Embedding(max_sequence_len + 12, n_patterns)
        
        if "vae_pattern" in block_type:
            latent_dim = 8
            self.vae_encoder = VAE_pattern(embedding_dim, conv_kernelsize[0], latent_dim=latent_dim)
            self.proj_pattern = nn.Linear(latent_dim, embedding_dim)
            self.module_gate = GatedUnit(embedding_dim)

        if "conv_softmax" in block_type:
            # self.conv_kernelsize = conv_kernelsize
            # model_list = [ConvSoftmax(embedding_dim, n_patterns, k, residual=True) for k in conv_kernelsize]
            # self.conv = nn.Sequential(*model_list)
            self.conv = ConvSoftmax(embedding_dim, n_patterns, 3, residual=False)
            self.module_gate = GatedUnit(embedding_dim)

        if "xattn_pattern" in block_type:
            self.conv_kernelsize = conv_kernelsize
            # model_list = [CausalConv1D(embedding_dim, embedding_dim, k, residual=True, has_bias=True, use_act=True) for k in conv_kernelsize]
            # self.conv = nn.Sequential(*model_list)
            self.conv = CausalConv1D(embedding_dim, embedding_dim, 3, residual=True, has_bias=True, use_act=True)
            if self._embedding_dim % self._num_heads != 0:
                xattn_embedding_dim = (embedding_dim // num_heads) * num_heads
                self.in_xattn_proj = nn.Linear(embedding_dim, xattn_embedding_dim)
                self.out_xattn_proj = nn.Linear(xattn_embedding_dim, embedding_dim)
                self.xattn_embedding_dim = xattn_embedding_dim
                self.pattern_matrix = nn.Parameter(torch.randn(n_patterns, xattn_embedding_dim))
                self.pattern_cross_attn_layer = nn.MultiheadAttention(xattn_embedding_dim, num_heads, dropout_ratio, batch_first=True)
            else:
                self.pattern_matrix = nn.Parameter(torch.randn(n_patterns, embedding_dim))
                self.pattern_cross_attn_layer = nn.MultiheadAttention(self._embedding_dim, num_heads, dropout_ratio, batch_first=True)
            torch.nn.init.xavier_uniform_(self.pattern_matrix)
                
            self.module_gate = GatedUnit(embedding_dim)
            
        if "xattn_pattern_v2" in block_type:
            self.conv_kernelsize = conv_kernelsize
            self.conv = CausalConv1D(embedding_dim, embedding_dim, 3, residual=False, has_bias=True, use_act=True)
            self.module_gate = GatedUnit(embedding_dim)

        if "xattn_pattern_v3" in block_type:
            self.conv = CausalConv1D(embedding_dim, embedding_dim, 3, residual=True, has_bias=True, use_act=True)
            self.pattern_matrix = nn.Parameter(torch.randn(n_patterns, embedding_dim))
            self.pattern_cross_attn_layer = xattn_block(input_dim=embedding_dim, sub_seqemb_dim=embedding_dim, num_heads=num_heads, attention_dim=attention_dim, hstu_attn=True)
            torch.nn.init.xavier_uniform_(self.pattern_matrix)
            self.module_gate = GatedUnit(embedding_dim)
            
        if "xattn_pattern_v4" in block_type:
            self.conv = CausalConv1D(embedding_dim, embedding_dim, 3, residual=True, has_bias=True, use_act=True)
            self.pattern_matrix = nn.Parameter(torch.randn(n_patterns, embedding_dim))
            self.pattern_cross_attn_layer = xattn_block(input_dim=embedding_dim, sub_seqemb_dim=embedding_dim, num_heads=num_heads, attention_dim=attention_dim, hstu_attn=True)
            torch.nn.init.xavier_uniform_(self.pattern_matrix)
            
        if "xattn_pattern_v5" in block_type:
            self.conv = CausalConv1D(embedding_dim, embedding_dim, 3, residual=True, has_bias=True, use_act=True)
            self.pattern_matrix = nn.Parameter(torch.randn(n_patterns, embedding_dim))
            self.pattern_cross_attn_layer = xattn_block(input_dim=embedding_dim, sub_seqemb_dim=embedding_dim, num_heads=num_heads, attention_dim=attention_dim, hstu_attn=False)
            torch.nn.init.xavier_uniform_(self.pattern_matrix)
                
        if "dynamic_conv" in block_type:
            kernel_size_list = [i for i in range(32) if i % 2 == 1]
            self.conv = DynamicConvolution(embedding_dim, embedding_dim, kernel_size_list)
            self.module_gate = GatedUnit(embedding_dim)
            
        # if "icb_filter" in block_type:
        #     self.icb_layer = Icb_block(input_dim=embedding_dim, kernel_size=1, inter_kernel_size=3, need_global_inter=True)
        # if "icb_filter_v2" in block_type:
        #     self.icb_layer = Icb_block(input_dim=embedding_dim, kernel_size=3, need_global_inter=False)
        if "xattn_pattern_baseline" in block_type:
            self.xattn_pattern_baseline_ffn_layer = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.LeakyReLU())
            self.xattn_pattern_baseline_layer = selfattn_block(backbone_pointer=backbone_pointer, block_type=block_type, input_dim=embedding_dim, num_heads=num_heads, attention_dim=attention_dim, dropout_ratio=dropout_ratio, merge_method=None, residual=False, training=self.training)
            # self.module_gate = GatedUnit(embedding_dim)
        if "subseq_attngate_baseline" in block_type:
            self.subseq_attngate_baseline_ffn_layer = nn.Sequential(nn.Linear(embedding_dim, int(0.36 * embedding_dim)), nn.LeakyReLU(), nn.Linear(int(0.36 * embedding_dim), embedding_dim))
            self.subseq_attngate_baseline_layer = selfattn_block(backbone_pointer=backbone_pointer, block_type=block_type, input_dim=embedding_dim, num_heads=num_heads, attention_dim=attention_dim, dropout_ratio=dropout_ratio, merge_method=None, residual=False, training=self.training)
            # self.subseq_emb_agg = GatedUnit(embedding_dim)
        if "subseq_attngate_baseline_mbd" in block_type:
            self.subseq_attngate_baseline_ffn_layer = nn.Sequential(nn.Linear(embedding_dim, int(0.3 * embedding_dim)), nn.LeakyReLU(), nn.Linear(int(0.3 * embedding_dim), embedding_dim))
            self.subseq_attngate_baseline_layer = selfattn_block(backbone_pointer=backbone_pointer, block_type=block_type, input_dim=embedding_dim, num_heads=num_heads, attention_dim=attention_dim, dropout_ratio=dropout_ratio, merge_method=None, residual=False, training=self.training)
            # self.subseq_emb_agg = GatedUnit(embedding_dim)
        if 'moe4hstu' in block_type: # cct
            self.hstu_moe_ffn_layer = nn.Sequential(nn.Linear(embedding_dim, int(0.36 * embedding_dim)), nn.LeakyReLU(), nn.Linear( int(0.36 * embedding_dim), embedding_dim))
            self.hstu_moe_layer = selfattn_block(backbone_pointer=backbone_pointer, block_type=block_type, input_dim=embedding_dim, num_heads=num_heads, attention_dim=attention_dim, dropout_ratio=dropout_ratio, merge_method=None, residual=False, training=self.training)
        if 'moe4hstu_mbd' in block_type:
            self.hstu_moe_ffn_layer = nn.Sequential(nn.Linear(embedding_dim, int(0.3 * embedding_dim)), nn.LeakyReLU(), nn.Linear( int(0.3 * embedding_dim), embedding_dim))
            self.hstu_moe_layer = selfattn_block(backbone_pointer=backbone_pointer, block_type=block_type, input_dim=embedding_dim, num_heads=num_heads, attention_dim=attention_dim, dropout_ratio=dropout_ratio, merge_method=None, residual=False, training=self.training)
        
        self.mha_layer = selfattn_block(backbone_pointer=backbone_pointer, block_type=block_type, input_dim=embedding_dim, num_heads=num_heads, attention_dim=attention_dim, linear_attn=("linear_attn" in block_type), dropout_ratio=dropout_ratio, merge_method=('mean' if self.module_gate is None else 'module'), residual=False, training=self.training)
        
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
        sub_seq_emb=None,
        repeat_mask=None,
        repeat_count=None,
        seq_length=None,
        user_emb=None,
        id_embs=None
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
        B: int = x_offsets.size(0) - 1
        n: int = invalid_attn_mask.size(-1)
        addiction_loss = dict()
        attn_gate = None
        seq_out = None
        padded_x = None
        attn_scale = self.attn_scale
        scale_flag = "scale" in self.block_type and attn_scale >= 0 and attn_scale < n - 1 and False
        
        if delta_x_offsets is not None:
            # In this case, for all the following code, x, u, v, q, k become restricted to
            # [delta_x_offsets[0], :].
            assert cache is not None
            x = x[delta_x_offsets[0], :]
            cached_v, cached_q, cached_k, cached_outputs = cache
        normed_x = self._norm_input(x)
        # if "icb_filter" in self.block_type:
        #     padded_x = torch.ops.fbgemm.jagged_to_padded_dense(values=normed_x, offsets=[x_offsets], max_lengths=[n], padding_value=0.0) #B, N, D
        #     padded_x = self.icb_layer(padded_x)
        #     normed_x = torch.ops.fbgemm.dense_to_jagged(padded_x, [x_offsets])[0]
            
        if "subseq_attngate" in self.block_type or "subseq_attngate_v5" in self.block_type or "subseq_attngate_v6" in self.block_type or "subseq_attngate_v2" in self.block_type:
            padded_x = torch.ops.fbgemm.jagged_to_padded_dense(values=normed_x, offsets=[x_offsets], max_lengths=[n], padding_value=0.0) if padded_x is None else padded_x #B, N, D
            attn_gate = self.layer_subseq_attngate(padded_x)
            attn_gate = F.sigmoid(attn_gate / F.sigmoid(self.attn_gate_factor)) # [B, H, N, N]
        if "subseq_attngate_v3" in self.block_type:
            padded_x = torch.ops.fbgemm.jagged_to_padded_dense(values=normed_x, offsets=[x_offsets], max_lengths=[n], padding_value=0.0) if padded_x is None else padded_x #B, N, D
            attn_gate = self.layer_subseq_attngate(padded_x)
            attn_gate = F.sigmoid(attn_gate) # [B, H, N, N]
        if "subseq_attngate_v4" in self.block_type:
            padded_x = torch.ops.fbgemm.jagged_to_padded_dense(values=normed_x, offsets=[x_offsets], max_lengths=[n], padding_value=0.0) if padded_x is None else padded_x #B, N, D
            attn_gate = self.layer_subseq_attngate(padded_x) # [B, H, N, N]
        if "sequential_pr" not in self.block_type:
            addiction_loss, seq_out = seq_operator(self, x_offsets, sub_seq_emb, n, addiction_loss, normed_x, padded_x)
        if "xattn_pattern_baseline" in self.block_type:
            seq_out = layernorm(self.xattn_pattern_baseline_layer(x=self.xattn_pattern_baseline_ffn_layer(x), attnmask=invalid_attn_mask, x_offsets=x_offsets, B=B, n=n))
        # if "transformer" in self.block_type:
        if all_timestamps is not None and self._rel_attn_bias is not None:
            attn_bias = self._rel_attn_bias(all_timestamps).unsqueeze(1)
        else:
            attn_bias = None
        out = self.mha_layer(x=x, attnmask=invalid_attn_mask, x_offsets=x_offsets, B=B, n=n, attn_gate=attn_gate, seq_out=seq_out, merge_seq_out_module=self.module_gate, attn_bias=attn_bias, user_emb=user_emb,id_embs=id_embs)
        if "subseq_emb" in self.block_type:
            out = self.subseq_emb_agg(out, sub_seq_emb)
        elif "subseq_emb_v2" in self.block_type:
            out = self.subseq_emb_agg(out, sub_seq_emb)
        elif "subseq_emb_v3" in self.block_type:
            sub_seq_emb = torch.ops.fbgemm.dense_to_jagged(self.subseq_emb_conv(sub_seq_emb), [x_offsets])[0]
            out = self.subseq_emb_agg(out, sub_seq_emb)
        elif "subseq_emb_v4" in self.block_type:
            padded_x = torch.ops.fbgemm.jagged_to_padded_dense(values=normed_x, offsets=[x_offsets], max_lengths=[n], padding_value=0.0) if padded_x is None else padded_x #B, N, D
            sub_seq_emb = torch.ops.fbgemm.dense_to_jagged(self.subseq_emb_conv(padded_x), [x_offsets])[0]
            out = out * F.sigmoid(sub_seq_emb)
            # out = self.subseq_emb_agg(out, sub_seq_emb)
        elif "subseq_emb_v5" in self.block_type:
            out = self.subseq_emb_agg(out, x)
        elif "subseq_emb_v6" in self.block_type:
            padded_x = torch.ops.fbgemm.jagged_to_padded_dense(values=normed_x, offsets=[x_offsets], max_lengths=[n], padding_value=0.0) if padded_x is None else padded_x #B, N, D
            out = F.sigmoid(torch.ops.fbgemm.dense_to_jagged(self.subseq_emb_conv_gate(padded_x), [x_offsets])[0]) * (out + x)
        elif "subseq_emb_v7" in self.block_type:
            out = F.sigmoid(self.subseq_emb_conv_gate(normed_x)) * (out + x)
        elif "subseq_emb_v8" in self.block_type:
            out = F.sigmoid(self.subseq_emb_conv_gate(normed_x)) * out + x
        elif "subseq_emb_v9" in self.block_type:
            out = self.subseq_emb_conv_gate(normed_x) * out + x
        elif "subseq_attngate_baseline" in self.block_type or 'subseq_attngate_baseline_mbd' in self.block_type:
            out += self.subseq_attngate_baseline_layer(x=self.subseq_attngate_baseline_ffn_layer(x), attnmask=invalid_attn_mask, x_offsets=x_offsets, B=B, n=n) + x
        elif "moe4hstu" in self.block_type or 'moe4hstu_mbd' in self.block_type:
            out += self.hstu_moe_layer(x=self.hstu_moe_ffn_layer(x), attnmask=invalid_attn_mask, x_offsets=x_offsets, B=B, n=n) + x
        else:
            out += x
        if "sequential_pr" in self.block_type:
            out_attn = layernorm(out)
            addiction_loss, out = seq_operator(self, x_offsets, sub_seq_emb, n, addiction_loss, out_attn, padded_x=None)
            out = out + out_attn
        
            
        return out, None, addiction_loss

class HSTUJagged(torch.nn.Module):

    def __init__(
        self,
        modules,
        autocast_dtype: torch.dtype,
        block_type,
        input_dim,
    ) -> None:
        super().__init__()

        self._attention_layers: torch.nn.ModuleList = torch.nn.ModuleList(modules=modules)
        self._autocast_dtype: torch.dtype = autocast_dtype
        self.block_type = block_type
        if "vae_encdec" in self.block_type:
            self.mu_head, self.var_head = nn.Linear(input_dim, input_dim), nn.Linear(input_dim, input_dim)
        # if "subseq_emb_sim" in self.block_type:
        #     self.sim_factor_list = nn.Parameter(torch.randn((len(self._attention_layers), )))

    def jagged_forward(
        self,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        all_timestamps: Optional[torch.Tensor],
        invalid_attn_mask: torch.Tensor,
        delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache: Optional[List[HSTUCacheState]] = None,
        return_cache_states: bool = False,
        sub_seq_emb=None,
        repeat_mask=None,
        repeat_count=None,
        seq_length=None,
        user_emb=None,
        id_embs=None
    ) -> Tuple[torch.Tensor, List[HSTUCacheState]]:
        """
        Args:
            x: (\sum_i N_i, D) x float
            x_offsets: (B + 1) x int32
            all_timestamps: (B, 1 + N) x int64
            invalid_attn_mask: (B, N, N) x float, each element in {0, 1}
            return_cache_states: bool. True if we should return cache states.

        Returns:
            x' = f(x), (\sum_i N_i, D) x float
        """
        cache_states: List[HSTUCacheState] = []

        addiction_loss = dict()
        
        for i, layer in enumerate(self._attention_layers):
            x, cache_states_i, addiction_loss_layer = layer(
                x=x,
                x_offsets=x_offsets,
                all_timestamps=all_timestamps,
                invalid_attn_mask=invalid_attn_mask,
                delta_x_offsets=delta_x_offsets,
                cache=cache[i] if cache is not None else None,
                return_cache_states=return_cache_states,
                sub_seq_emb=sub_seq_emb,
                repeat_mask=repeat_mask,
                repeat_count=repeat_count,
                seq_length=seq_length,
                user_emb=user_emb,
                id_embs=id_embs
                # sub_seq_emb=F.softmax(sub_seq_emb[i], dim=-1) * self.sim_factor_list[i] if "subseq_emb_sim" in self.block_type else sub_seq_emb,
            )
            if return_cache_states:
                cache_states.append(cache_states_i)
            for k, v in addiction_loss_layer.items():
                if k in addiction_loss:
                    addiction_loss[k].append(v)
                else:
                    addiction_loss[k] = [v]
        for k, v in addiction_loss.items():
            if isinstance(v, list):
                addiction_loss[k] = torch.stack(v).mean()

        return x, cache_states, addiction_loss

    def forward(
        self,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        all_timestamps: Optional[torch.Tensor],
        invalid_attn_mask: torch.Tensor,
        delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache: Optional[List[HSTUCacheState]] = None,
        return_cache_states: bool = False,
        sub_seq_emb=None,
        sub_seq_emb_jagged=True,
        repeat_mask=None,
        repeat_count=None,
        seq_length=None,
        user_emb=None,
        id_embs=None
    ) -> Tuple[torch.Tensor, List[HSTUCacheState]]:
        """
        Args:
            x: (B, N, D) x float.
            x_offsets: (B + 1) x int32.
            all_timestamps: (B, 1 + N) x int64
            invalid_attn_mask: (B, N, N) x float, each element in {0, 1}.
        Returns:
            x' = f(x), (B, N, D) x float
        """
        if len(x.size()) == 3:
            x = torch.ops.fbgemm.dense_to_jagged(x, [x_offsets])[0]
        if (sub_seq_emb is not None) and sub_seq_emb_jagged:
            sub_seq_emb = torch.ops.fbgemm.dense_to_jagged(sub_seq_emb, [x_offsets])[0]

        jagged_x, cache_states, addiction_loss = self.jagged_forward(
            x=x,
            x_offsets=x_offsets,
            all_timestamps=all_timestamps,
            invalid_attn_mask=invalid_attn_mask,
            delta_x_offsets=delta_x_offsets,
            cache=cache,
            return_cache_states=return_cache_states,
            sub_seq_emb=sub_seq_emb,
            repeat_mask=repeat_mask,
            repeat_count=repeat_count,
            seq_length=seq_length,
            user_emb=user_emb,
            id_embs=id_embs
        )
        y = torch.ops.fbgemm.jagged_to_padded_dense(
            values=jagged_x,
            offsets=[x_offsets],
            max_lengths=[invalid_attn_mask.size(1)],
            padding_value=0.0,
        )
        return y, cache_states, addiction_loss


class HSTU(GeneralizedInteractionModule):
    """
    Implements HSTU (Hierarchical Sequential Transduction Unit) in 
    Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations,
    https://arxiv.org/abs/2402.17152.

    Note that this implementation is intended for reproducing experiments in
    the traditional sequential recommender setting (Section 4.1.1), and does
    not yet use optimized kernels discussed in the paper.
    """

    def __init__(
        self,
        max_sequence_len: int,
        max_output_len: int,
        embedding_dim: int,
        num_blocks: int,
        num_heads: int,
        linear_dim: int,
        attention_dim: int,
        normalization: str,
        linear_config: str,
        linear_activation: str,
        linear_dropout_rate: float,
        attn_dropout_rate: float,
        embedding_module: EmbeddingModule,
        similarity_module: NDPModule,
        input_features_preproc_module: InputFeaturesPreprocessorModule,
        output_postproc_module: OutputPostprocessorModule,
        enable_relative_attention_bias: bool = True,
        concat_ua: bool = False,
        verbose: bool = True,
        block_type = ["hstu"], # ["scale", "sparse", ""]
        scale_up_factor: int = 2,
        residual: bool = True,
        n_patterns: int = None,
        reverse: bool = False,
        num_conv_blocks: int = 1,
    ) -> None:
        super().__init__(ndp_module=similarity_module)

        self._embedding_dim: int = embedding_module.input_dim
        self._item_embedding_dim: int = embedding_module.item_embedding_dim
        self._max_sequence_length: int = max_sequence_len
        self._embedding_module: EmbeddingModule = embedding_module
        self._input_features_preproc: InputFeaturesPreprocessorModule = input_features_preproc_module
        self._output_postproc: OutputPostprocessorModule = output_postproc_module
        self._num_blocks: int = num_blocks
        self._num_heads: int = num_heads
        self._dqk: int = attention_dim
        self._dv: int = linear_dim
        self._linear_activation: str = linear_activation
        self._linear_dropout_rate: float = linear_dropout_rate
        self._attn_dropout_rate: float = attn_dropout_rate
        self._enable_relative_attention_bias: bool = enable_relative_attention_bias
        self.block_type = block_type
        self.use_user_attrs = embedding_module.use_user_attrs
        
        attn_mod = num_blocks - 1
        config_dict = {
            "backbone_pointer": self,
            "item_embedding_dim": self._item_embedding_dim,
            "linear_hidden_dim": linear_dim,
            "attention_dim": attention_dim,
            "normalization": normalization,
            "linear_config": linear_config,
            "linear_activation": linear_activation,
            "num_heads": num_heads,
            "relative_attention_bias_module": None,
            # "relative_attention_bias_module": (RelativeBucketedTimeAndPositionBasedBias(
            #     max_seq_len=max_sequence_len + max_output_len,  # accounts for next item.
            #     num_buckets=128,
            #     bucketization_fn=lambda x: (torch.log(torch.abs(x).clamp(min=1)) / 0.301).long(),
            # ) if enable_relative_attention_bias else None),
            "dropout_ratio": linear_dropout_rate,
            "attn_dropout_ratio": attn_dropout_rate,
            "concat_ua": concat_ua,
            "block_type": block_type,
            "residual": residual,
            "attn_mod": attn_mod,
        }
        import numpy as np
        conv_kernelsize_list = np.linspace(2, 16, num_blocks, dtype=int).reshape(num_blocks, 1).tolist()
        conv_kernelsize_list = [[i] for i in [2, 3, 4 ,5, 6, 8, 10, 12, 16, 21]]
        self.conv_kernelsize_list = conv_kernelsize_list
        # conv_kernelsize_list = np.linspace(2, max_sequence_len, num_blocks * num_conv_blocks, dtype=int).reshape(num_blocks,num_conv_blocks).tolist()
        attn_mod_id_list = [i for i in range(attn_mod)] + [-1]
        n_patterns_list = [n_patterns] * num_blocks
        if scale_up_factor ** (num_blocks - 1) < (max_sequence_len + max_output_len) // 2:
            scale_list = np.linspace(1, (max_sequence_len + max_output_len) // 2, num_blocks, dtype=int).reshape(-1).tolist()
        else:
            scale_list = [int(scale_up_factor ** i) for i in range(num_blocks)]
        if reverse:
            conv_kernelsize_list = [i[::-1] for i in conv_kernelsize_list]
            conv_kernelsize_list = conv_kernelsize_list[::-1]
            attn_mod_id_list = attn_mod_id_list[::-1]
            scale_list = scale_list[::-1]
            n_patterns_list = n_patterns_list[::-1]
        print("conv_kernelsize_list: ", conv_kernelsize_list)
        if 'scale' in self.block_type:
            print("scale_list: ", scale_list)
        print("n_patterns: ", n_patterns)
        if "sparse" in self.block_type:
            print("attn_mod_id_list: ", attn_mod_id_list)
        print("enable_relative_attention_bias: ", enable_relative_attention_bias)
        self._hstu = HSTUJagged(
            modules=[
                SequentialTransductionUnitJagged(**config_dict, 
                conv_kernelsize = conv_kernelsize_list[i], 
                attn_scale = scale_list[i],
                attn_mod_id = attn_mod_id_list[i],
                n_patterns = n_patterns_list[i],
                embedding_dim=self._embedding_dim,
                ) for i in range(num_blocks)],
            autocast_dtype=None,
            block_type=block_type,
            input_dim=self._embedding_dim,
        )
        print("Block Type: ", block_type)
        # causal forward, w/ +1 for padding.
        self._verbose: bool = verbose
                
        self.reset_params()

    def reset_params(self):
        for name, params in self.named_parameters():
            # if ("_hstu" in name) or ("_embedding_module" in name):
            #     if self._verbose:
            #         print(f"Skipping init for {name}")
            #     continue
            try:
                torch.nn.init.xavier_normal_(params.data)
                if self._verbose:
                    print(f"Initialize {name} as xavier normal: {params.data.size()} params")
            except:
                if self._verbose:
                    print(f"Failed to initialize {name}: {params.data.size()} params")

    def get_item_embeddings(self, ids: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._embedding_module.get_item_embeddings(ids, **kwargs)

    def debug_str(self) -> str:
        debug_str = (
            f"HSTU-b{self._num_blocks}-h{self._num_heads}-dqk{self._dqk}-dv{self._dv}"
            + f"-l{self._linear_activation}d{self._linear_dropout_rate}"
            + f"-ad{self._attn_dropout_rate}"
        )
        if not self._enable_relative_attention_bias:
            debug_str += "-norab"
        return debug_str
    
    
    def generate_user_embeddings(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
        delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache: Optional[List[HSTUCacheState]] = None,
        return_cache_states: bool = False,
    ) -> Tuple[torch.Tensor, List[HSTUCacheState]]:
        """
        [B, N] -> [B, N, D].
        """
        device = past_lengths.device
        float_dtype = past_embeddings.dtype
        B, N, _ = past_embeddings.size()
        x_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(past_lengths)
        

        past_lengths, user_embeddings, valid_mask = self._input_features_preproc(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=None,
        )
        # id_embs=self._embedding_module.get_user_id_embeddings(user_ids=past_payloads["uins"])
        id_embs=None

        float_dtype = user_embeddings.dtype
        user_embeddings, cached_states, addiction_loss = self._hstu(
            x=user_embeddings,
            x_offsets=x_offsets,
            all_timestamps=(
                past_payloads[TIMESTAMPS_KEY]
                if past_payloads and (TIMESTAMPS_KEY in past_payloads) else None
            ),
            invalid_attn_mask=1.0 - torch.triu(
                torch.ones((N, N), dtype=torch.bool),
                diagonal=1,
            ).to(float_dtype).to(device),
            delta_x_offsets=delta_x_offsets,
            cache=cache,
            return_cache_states=return_cache_states,
            sub_seq_emb=None,
            repeat_mask=None,
            repeat_count=None,
            sub_seq_emb_jagged=('subseq_emb' in self.block_type or "subseq_emb_attn" in self.block_type or "subseq_emb_sim" in self.block_type) and isinstance(sub_seq_emb, torch.Tensor) and (len(sub_seq_emb.size()) == 3), 
            seq_length=past_lengths, 
            user_emb=self._embedding_module.get_user_attrs(past_payloads["user_attrs"]) if self.use_user_attrs else None,
            id_embs=id_embs,
        )
        return self._output_postproc(user_embeddings), cached_states, addiction_loss

    

    def forward(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
        batch_id: Optional[int] = None,
        return_cache_states: bool = False,
    ) -> torch.Tensor:
        """
        Runs the main encoder.

        Args:
            past_lengths: (B,) x int64
            past_ids: (B, N,) x int64 where the latest engaged ids come first. In
                particular, past_ids[i, past_lengths[i] - 1] should correspond to
                the latest engaged values.
            past_embeddings: (B, N, D) x float or (\sum_b N_b, D) x float.
            past_payloads: implementation-specific keyed tensors of shape (B, N, ...).

        Returns:
            encoded_embeddings of [B, N, D].
        """
        encoded_embeddings, cached_states, addiction_loss = self.generate_user_embeddings(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
            return_cache_states=return_cache_states,
        )
        if self.training:
            return (encoded_embeddings, cached_states, addiction_loss)
        else:
            return (encoded_embeddings, cached_states)

    def _encode(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
        delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]],
        cache: Optional[List[HSTUCacheState]],
        return_cache_states: bool,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[HSTUCacheState]]]:
        """
        Args:
            past_lengths: (B,) x int64.
            past_ids: (B, N,) x int64.
            past_embeddings: (B, N, D,) x float.
            past_payloads: implementation-specific keyed tensors of shape (B, N, ...).
            return_cache_states: bool.

        Returns:
            (B, D) x float, representing embeddings for the current state.
        """
        encoded_seq_embeddings, cache_states, _ = self.generate_user_embeddings(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
            delta_x_offsets=delta_x_offsets,
            cache=cache,
            return_cache_states=return_cache_states,
        )   # [B, N, D]
        current_embeddings = get_current_embeddings(lengths=past_lengths, encoded_embeddings=encoded_seq_embeddings)
        if return_cache_states:
            return current_embeddings, cache_states
        else:
            return current_embeddings

    def encode(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
        delta_x_offsets: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache: Optional[List[HSTUCacheState]] = None,
        return_cache_states: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[HSTUCacheState]]]:
        """
        Runs encoder to obtain the current hidden states.

        Args:
            past_lengths: (B,) x int.
            past_ids: (B, N,) x int.
            past_embeddings: (B, N, D) x float.
            past_payloads: implementation-specific keyed tensors of shape (B, N, ...).

        Returns:
            (B, D,) x float, representing encoded states at the most recent time step.
        """
        return self._encode(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
            delta_x_offsets=delta_x_offsets,
            cache=cache,
            return_cache_states=return_cache_states,
        )
