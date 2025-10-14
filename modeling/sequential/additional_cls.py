class SequentialTransductionUnitJagged_expm(SequentialTransductionUnitJagged):
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

        if self._linear_config == "uvqk":
            batched_mm_output = torch.mm(normed_x, self._uvqk)
            if self._linear_activation == "silu":
                batched_mm_output = F.silu(batched_mm_output)
            elif self._linear_activation == "none":
                batched_mm_output = batched_mm_output
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
                
            qk_attn = torch.einsum(
                "bnhd,bmhd->bhnm",
                l2_normalize_last_dim(padded_q.view(B, n, self._num_heads, self._attention_dim)),
                l2_normalize_last_dim(padded_k.view(B, n, self._num_heads, self._attention_dim)),
            )
            
            qk_attn += 1.0
            qk_attn = qk_attn / (qk_attn.sum(dim=-1, keepdim=True) + 1e-12)
            if all_timestamps is not None:
                qk_attn = qk_attn + self._rel_attn_bias(all_timestamps).unsqueeze(1)
            qk_attn = F.silu(qk_attn) / n
            qk_attn = qk_attn * invalid_attn_mask.unsqueeze(0).unsqueeze(0)
            attn_output = torch.ops.fbgemm.dense_to_jagged(
                torch.einsum(
                    "bhnm,bmhd->bnhd",
                    qk_attn,
                    torch.ops.fbgemm.jagged_to_padded_dense(v, [x_offsets], [n]).reshape(B, n, self._num_heads, self._linear_dim)
                ).reshape(B, n, self._num_heads * self._linear_dim),
                [x_offsets],
            )[0]
        elif self._normalization == "softmax_rel_bias":
            if delta_x_offsets is not None:
                B = x_offsets.size() - 1
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

            qk_attn = torch.einsum("bnd,bmd->bnm", padded_q, padded_k)
            if self._rel_attn_bias is not None:
                qk_attn = qk_attn + self._rel_attn_bias(all_timestamps)
            qk_attn = F.softmax(qk_attn / math.sqrt(self._attention_dim), dim=-1)
            qk_attn = qk_attn * invalid_attn_mask
            attn_output = torch.ops.fbgemm.dense_to_jagged(
                torch.bmm(qk_attn, torch.ops.fbgemm.jagged_to_padded_dense(v, [x_offsets], [n])),
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
        ) + x

        if delta_x_offsets is not None:
            new_outputs = cached_outputs.index_copy_(dim=0, index=delta_x_offsets[0], source=new_outputs)

        if return_cache_states and delta_x_offsets is None:
            v = v.contiguous()

        return new_outputs, (v, padded_q, padded_k, new_outputs)
    
class SequentialTransductionUnitJagged_with_recurrentgate(SequentialTransductionUnitJagged):
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
    ) -> None:
        super().__init__(
            embedding_dim=embedding_dim,
            linear_hidden_dim=linear_hidden_dim,
            attention_dim=attention_dim,
            dropout_ratio=dropout_ratio,
            attn_dropout_ratio=attn_dropout_ratio,
            num_heads=num_heads,
            linear_activation=linear_activation,
            relative_attention_bias_module=relative_attention_bias_module,
            normalization=normalization,
            linear_config=linear_config,
            concat_ua=concat_ua,
            epsilon=epsilon,
            max_length=max_length,
        )
        self.recurrent_gate = Recurrentgate(embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, embedding_dim)
        self.linear_o = nn.Linear(embedding_dim, embedding_dim)
        
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

        if self._linear_config == "uvqk":
            batched_mm_output = torch.mm(normed_x, self._uvqk)
            if self._linear_activation == "silu":
                batched_mm_output = F.silu(batched_mm_output)
            elif self._linear_activation == "none":
                batched_mm_output = batched_mm_output
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

            qk_attn = torch.einsum(
                "bnhd,bmhd->bhnm",
                padded_q.view(B, n, self._num_heads, self._attention_dim),
                padded_k.view(B, n, self._num_heads, self._attention_dim),
            )
            if all_timestamps is not None:
                qk_attn = qk_attn + self._rel_attn_bias(all_timestamps).unsqueeze(1)
            qk_attn = F.silu(qk_attn) / n
            qk_attn = qk_attn * invalid_attn_mask.unsqueeze(0).unsqueeze(0)
            attn_output = torch.ops.fbgemm.dense_to_jagged(
                torch.einsum(
                    "bhnm,bmhd->bnhd",
                    qk_attn,
                    torch.ops.fbgemm.jagged_to_padded_dense(v, [x_offsets], [n]).reshape(B, n, self._num_heads, self._linear_dim)
                ).reshape(B, n, self._num_heads * self._linear_dim),
                [x_offsets],
            )[0]
        elif self._normalization == "softmax_rel_bias":
            if delta_x_offsets is not None:
                B = x_offsets.size() - 1
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

            qk_attn = torch.einsum("bnd,bmd->bnm", padded_q, padded_k)
            if self._rel_attn_bias is not None:
                qk_attn = qk_attn + self._rel_attn_bias(all_timestamps)
            qk_attn = F.softmax(qk_attn / math.sqrt(self._attention_dim), dim=-1)
            qk_attn = qk_attn * invalid_attn_mask
            attn_output = torch.ops.fbgemm.dense_to_jagged(
                torch.bmm(qk_attn, torch.ops.fbgemm.jagged_to_padded_dense(v, [x_offsets], [n])),
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


        new_outputs = self.recurrent_gate(normed_x, self._o(
            F.dropout(
                o_input,
                p=self._dropout_ratio,
                training=self.training,
            )
        )) * F.gelu(self.linear1(normed_x))
        new_outputs = self.linear_o(new_outputs)
        
        new_outputs += x

        if delta_x_offsets is not None:
            new_outputs = cached_outputs.index_copy_(dim=0, index=delta_x_offsets[0], source=new_outputs)

        if return_cache_states and delta_x_offsets is None:
            v = v.contiguous()

        return new_outputs, (v, padded_q, padded_k, new_outputs)

class Recurrentgate(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.recurrent_gate = nn.Linear(hidden_dim, hidden_dim)
        self.input_gate = nn.Linear(hidden_dim, hidden_dim)
        self.A = nn.Parameter(torch.randn(hidden_dim))
        
    def forward(self, x, ref_x):
        r = torch.sigmoid(self.recurrent_gate(x))
        i = torch.sigmoid(self.input_gate(x))
        log_a = 8.0 * r * torch.log(torch.sigmoid(self.A))
        a = torch.exp(log_a)
        return a * ref_x + torch.sqrt(1 - a**2) * i * x
        
class RG_LRU(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.recurrent_gate = nn.Linear(hidden_dim, hidden_dim)
        self.input_gate = nn.Linear(hidden_dim, hidden_dim)
        
        self.A = nn.Parameter(torch.randn(hidden_dim))
        self.c = 8

    def forward(self, x):
        # 初始化隐藏状态
        h_prev = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        r = torch.sigmoid(self.recurrent_gate(x))
        i = torch.sigmoid(self.input_gate(x))
        log_a = self.c * r * torch.log(torch.sigmoid(self.A))
        a = torch.exp(log_a)
        
        outputs = []
        for t in range(x.size(1)):
            it = i[:, t, :]
            at = a[:, t, :]
            
            ht = at * h_prev + torch.sqrt(1 - at**2) * it * x[:, t, :]
            outputs.append(ht)
            h_prev = ht
        
        # 将输出堆叠成一个张量
        outputs = torch.stack(outputs, dim=1)
        return outputs
class RecurrentBlock(nn.Module):
    def __init__(self, hidden_dim, d_rnn):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, d_rnn)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim, d_rnn)
        self.rg_lru = RG_LRU(d_rnn)
        self.temp_conv1d = nn.Conv1d(d_rnn, d_rnn, kernel_size=3, padding=1, stride=1)
        self.o_linear = nn.Linear(d_rnn, hidden_dim)
        

    def forward(self, x, x_offsets, invalid_attn_mask):
        if len(x.size()) == 2:
            x = torch.ops.fbgemm.jagged_to_padded_dense(
                values=x,
                offsets=[x_offsets],
                max_lengths=[invalid_attn_mask.size(1)],
                padding_value=0.0,
            )
        b1 = self.linear1(x)
        b1 = self.gelu(b1)
        
        b2 = self.linear2(x)
    
        b2 = x.transpose(1, 2)  # Conv1d expects (batch, channels, length)
        b2 = self.temp_conv1d(b2)
        b2 = b2.transpose(1, 2)  # Convert back to (batch, length, channels)
        b2 = self.rg_lru(b2)
        
        x_out = self.o_linear(b1 * b2)
        return x_out

