import torch.nn as nn
import torch
import torch.nn.functional as F
import math

class SPRM(nn.Module):
    def __init__(self, embedding_dim, n_patterns, num_heads, attention_dim):
        super(SPRM, self).__init__()
        self.conv = CausalConv1D(embedding_dim, embedding_dim, 3, residual=True, has_bias=True, use_act=True)
        self.pattern_matrix = torch.nn.Parameter(torch.empty((n_patterns, embedding_dim)).normal_(mean=0, std=0.02))
        self.pattern_cross_attn_layer = xattn_block(input_dim=embedding_dim, sub_seqemb_dim=embedding_dim, num_heads=num_heads, attention_dim=attention_dim, hstu_attn=False)

    def forward(self, normed_x, x_offsets, n):
        padded_x = torch.ops.fbgemm.jagged_to_padded_dense(values=normed_x, offsets=[x_offsets], max_lengths=[n], padding_value=0.0)
        padded_x = self.conv(padded_x)
        padded_x = torch.ops.fbgemm.dense_to_jagged(padded_x, [x_offsets])[0]
        pattern_matrix = self.pattern_matrix
        pattern_matrix = layernorm(pattern_matrix)
        seq_out = self.pattern_cross_attn_layer(padded_x, pattern_matrix)
        seq_out = layernorm(seq_out)
        return seq_out
    
class selfattn_block(nn.Module):
    def __init__(self, backbone_pointer, block_type, input_dim, num_heads, attention_dim, training, dropout_ratio=0.1, merge_method='mean', residual=False, linear_attn=False):
        super(selfattn_block, self).__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.attention_dim = attention_dim
        self.scale = attention_dim ** -0.5
        self._dropout_ratio = dropout_ratio
        self.training = training
        self._residual = residual
        self.merge_method = merge_method
        self.linear_attn = linear_attn
        self.block_type = block_type
        
        if "transformer" in self.block_type or "tf_gate" in self.block_type:
            self.proj_layer = nn.Parameter(torch.randn(self.input_dim, attention_dim * num_heads * 3))
            self.hstu_attn = False
        elif "mlp_transformer" in self.block_type:
            self.proj_layer = nn.Parameter(torch.randn(self.input_dim, attention_dim * num_heads * 3))
            self.mlp = nn.Sequential(nn.Linear(self.input_dim, self.input_dim), nn.ReLU(), nn.Linear(self.input_dim, self.input_dim))
            self.hstu_attn = False
        else:
            self.proj_layer = nn.Parameter(torch.randn(self.input_dim, attention_dim * num_heads * 4))
            self.hstu_attn = True
        
        for b in self.block_type:
            if "mlp_transformer_" in b:
                mul_ffn_dim=float(b.replace('mlp_transformer_',''))
                self.mlp = nn.Sequential(nn.Linear(self.input_dim, int(self.input_dim*mul_ffn_dim)), nn.ReLU(), nn.Linear(int(self.input_dim*mul_ffn_dim), self.input_dim))
            
        torch.nn.init.xavier_uniform_(self.proj_layer)
        if 'xattn_useremb' in self.block_type:
            self.xattn_useremb_layer = xattn_block(input_dim=self.input_dim, sub_seqemb_dim=backbone_pointer._embedding_module.user_attr_dim, num_heads=num_heads, attention_dim=attention_dim, hstu_attn=False)
        elif 'mlp_useremb' in self.block_type:
            self.xattn_useremb_layer = nn.Linear(backbone_pointer._embedding_module.user_attr_dim * backbone_pointer._embedding_module.user_attr_num, self.input_dim)
        if 'mlp_idemb' in self.block_type:
            # 使用mlp将id embedding映射到attention_dim
            self.idemb_layer=nn.Linear(8,self.input_dim)
        self.out_proj = nn.Linear(attention_dim * num_heads, input_dim)
        torch.nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x, attnmask, x_offsets, B, n, attn_gate=None, seq_out=None, merge_seq_out_module=None, attn_bias=None, user_emb=None,id_embs=None):
        # x : N, D
        x = layernorm(x)
        if user_emb is not None:
            user_emb = layernorm(user_emb)
        if 'tf_gate' in self.block_type:
            v, q, k = torch.split(F.silu(torch.mm(x, self.proj_layer)).view(x.size(0), self.num_heads, -1), [self.attention_dim] * 3, dim=-1)
        elif "transformer" in self.block_type or "mlp_transformer" in self.block_type:
            v, q, k = torch.split(torch.mm(x, self.proj_layer).view(x.size(0), self.num_heads, -1), [self.attention_dim] * 3, dim=-1)
        else:
            u, v, q, k = torch.split(F.silu(torch.mm(x, self.proj_layer).view(x.size(0), self.num_heads, -1)), [self.attention_dim] * 4, dim=-1)
        padded_v = torch.ops.fbgemm.jagged_to_padded_dense(
            values=v.flatten(-2), offsets=[x_offsets], max_lengths=[n], padding_value=0.0
        ).view(B, n, self.num_heads, self.attention_dim)
        padded_q = torch.ops.fbgemm.jagged_to_padded_dense(
                        values=q.flatten(-2), offsets=[x_offsets], max_lengths=[n], padding_value=0.0
                    ).view(B, n, self.num_heads, self.attention_dim)
        padded_k = torch.ops.fbgemm.jagged_to_padded_dense(
            values=k.flatten(-2), offsets=[x_offsets], max_lengths=[n], padding_value=0.0
        ).view(B, n, self.num_heads, self.attention_dim)
        attn_scores = torch.einsum("bnhd,bmhd->bhnm", padded_q, padded_k)
        if 'tf_gate' in self.block_type or self.hstu_attn:
            if attn_bias is not None:
                attn_scores += attn_bias
            attn_scores = F.silu(attn_scores) / n * attnmask.unsqueeze(0).unsqueeze(0)
        else:
            attn_scores = attn_scores * self.scale
            attn_scores = attn_scores.masked_fill(attnmask.expand(attn_scores.size(0), attn_scores.size(1), -1, -1) == 0, float('-inf'))
            attn_scores = F.softmax(attn_scores, dim=-1)
        if attn_gate is not None:
            attn_scores = attn_scores * attn_gate
        attnout = torch.einsum("bhnm,bmhd->bnhd", attn_scores, padded_v).flatten(-2)
            
        attnout = torch.ops.fbgemm.dense_to_jagged(attnout,[x_offsets])[0] # _, h*d
        if self.hstu_attn:
            attnout = layernorm(attnout)
            attnout = attnout * u.flatten(-2)
        if 'tf_gate' in self.block_type:
            attnout = layernorm(attnout)
        attnout = self.out_proj(F.dropout(
                attnout,
                p=self._dropout_ratio,
                training=self.training,
            ))
        if "mlp_transformer" in self.block_type:
            attnout = self.mlp(attnout)
        if 'xattn_useremb' in self.block_type or 'mlp_useremb' in self.block_type:
            attnout = torch.ops.fbgemm.jagged_to_padded_dense(
                    values=attnout, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
                )
            if 'xattn_useremb' in self.block_type:
                useremb_out = self.xattn_useremb_layer.padded_forward(attnout, user_emb)
            elif 'mlp_useremb' in self.block_type:
                useremb_out = self.xattn_useremb_layer(user_emb).unsqueeze(1)
            attnout += useremb_out
            attnout = torch.ops.fbgemm.dense_to_jagged(attnout,[x_offsets])[0]
        if 'mlp_idemb' in self.block_type:
            attnout = torch.ops.fbgemm.jagged_to_padded_dense(
                    values=attnout, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
                )
            # 过mlp
            uinemb_out=self.idemb_layer(id_embs).unsqueeze(1)
            
            attnout += uinemb_out
            attnout = torch.ops.fbgemm.dense_to_jagged(attnout,[x_offsets])[0]
        if seq_out is not None:
            if self.merge_method == 'mean':
                attnout = (attnout + seq_out) / 2
            elif self.merge_method == 'module':
                attnout = merge_seq_out_module(attnout, seq_out)
        return attnout
    
class xattn_block(nn.Module):
    def __init__(self, input_dim ,sub_seqemb_dim, num_heads, attention_dim, hstu_attn=True):
        super(xattn_block, self).__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.sub_seqemb_dim = sub_seqemb_dim
        self.attention_dim = attention_dim
        self.scale = attention_dim ** -0.5
        self.hstu_attn = hstu_attn
        if hstu_attn:
            self.query_w = nn.Linear(self.input_dim, attention_dim * num_heads * 2, bias=False)
        else:
            self.query_w = nn.Linear(self.input_dim, attention_dim * num_heads, bias=False)
        self.kv_w = nn.Linear(self.sub_seqemb_dim, attention_dim * num_heads * 2, bias=False)
            
        self.out_proj = nn.Linear(attention_dim * num_heads, input_dim)
        self.reset_params()
        
    def reset_params(self) -> None:
        nn.init.normal_(self.query_w.weight, 0, 0.02)
        nn.init.normal_(self.kv_w.weight, 0, 0.02)
        torch.nn.init.xavier_uniform_(self.out_proj.weight)
        
    def forward(self, x, subseq_emb):
        # x : N, D; subseq_emb : N, W, D or W, D
        subseq_len = subseq_emb.size(1)
        x = layernorm(x)
        subseq_emb = layernorm(subseq_emb)
        if self.hstu_attn:
            q, u = torch.split(self.query_w(x).view(x.size(0), self.num_heads, -1), [self.attention_dim] * 2, dim=-1)
        else:
            q = self.query_w(x).view(-1, self.num_heads, self.attention_dim)
        # Compute attention scores
        if len(subseq_emb.size()) == 2:
            k, v = torch.split(self.kv_w(subseq_emb).view(subseq_emb.size(0), self.num_heads, -1), [self.attention_dim] * 2, dim=-1)
            attn_scores = self.scale * torch.einsum("nhd,whd->nhw", q, k)
        else:
            k, v = torch.split(self.kv_w(subseq_emb).view(subseq_emb.size(0), subseq_emb.size(1), self.num_heads, -1), [self.attention_dim] * 2, dim=-1)
            attn_scores = self.scale * torch.einsum("nhd,nwhd->nhw", q, k)
        if self.hstu_attn:
            attn_scores = F.silu(attn_scores) / subseq_len
        else:
            attn_scores = F.softmax(attn_scores, dim=-1)
        if len(subseq_emb.size()) == 2:
            attnout = torch.einsum("nhw,whd->nhd", attn_scores, v).flatten(1)
        else:
            attnout = torch.einsum("nhw,nwhd->nhd", attn_scores, v).flatten(1)
            
        if self.hstu_attn:
            attnout = layernorm(attnout)
            attnout = attnout * u.flatten(1)
        attnout = self.out_proj(attnout)
        return attnout
    
    def padded_forward(self, x, subseq_emb):
        # x : B, N, D; subseq_emb : B, L, D
        subseq_len = subseq_emb.size(1)
        x = layernorm(x)
        subseq_emb = layernorm(subseq_emb)
        if self.hstu_attn:
            q, u = torch.split(self.query_w(x).view(x.size(0), x.size(1), self.num_heads, -1), [self.attention_dim] * 2, dim=-1)
        else:
            q = self.query_w(x).view(x.size(0), x.size(1), self.num_heads, self.attention_dim)
        # Compute attention scores
        k, v = torch.split(self.kv_w(subseq_emb).view(subseq_emb.size(0), subseq_emb.size(1), self.num_heads, -1), [self.attention_dim] * 2, dim=-1)
        attn_scores = self.scale * torch.einsum("bnhd,bmhd->bhnm", q, k)
        if self.hstu_attn:
            attn_scores = F.silu(attn_scores) / subseq_len
        else:
            attn_scores = F.softmax(attn_scores, dim=-1)
        attnout = torch.einsum("bhnm,bmhd->bnhd", attn_scores, v).flatten(-2)
            
        if self.hstu_attn:
            attnout = layernorm(attnout)
            attnout = attnout * u.flatten(-2)
        attnout = self.out_proj(attnout)
        return attnout
    
def get_subseq_attngate(x, scale_factor, w_size=3, shift=True):
    # x : B, N, D
    subseq_x = x.unfold(dimension=1, size=w_size, step=1).flatten(2) # B, N-w_size+1, D * w_size
    subseq_x = l2_normalize_last_dim(subseq_x)
    attn_gate = torch.einsum('bmd,bnd->bmn', subseq_x, subseq_x)
    attn_gate = F.sigmoid(attn_gate / F.sigmoid(scale_factor)) # B, N-w_size+1, N-w_size+1
    
    if shift:
        attn_gate = attn_gate[:, :, :-1] # shift right
        attn_gate = F.pad(attn_gate, (1, 0), value=1.0) # B, N-w_size+1, N-w_size+1
    
    attn_gate = F.pad(attn_gate, (w_size-1, 0, w_size-1, 0), value=1.0) # B, N, N
    
    return attn_gate
    
def get_scale_attnmask(n, length, w, device='cpu'):
    length = torch.cumsum(length[:-1], 0)
    ilow = F.pad(length, [1, 0], value=0).to(device).view(-1, 1, 1)
    ihight = ilow + w
    ii = torch.arange(n).to(device).view(1, -1, 1)
    jj = torch.arange(w + 1).to(device).view(1, 1, -1)
    mask = (ilow <= ii) * (ii < ihight)
    mask = mask & (jj > (ii - ilow))
    return ~mask.any(0)

def get_qkattn(q, k, w_size, seq_length):
    # k : n, h, d
    k = torch.stack([torch.roll(k, i, 0) for i in range(w_size + 1)])
    qk_attn = (q.unsqueeze(0) * k).sum(-1) #w_size+1, n, h
    qk_attn = qk_attn.permute(1, 0, 2) # n, w_size+1, h
    qk_attn = qk_attn * get_scale_attnmask(qk_attn.size(0), seq_length, w_size, qk_attn.device).unsqueeze(-1)
    return qk_attn

def get_attn_out(qk_attn, v, w_size):
    # v : n, h, d; qk_attn: n, w_size+1, h
    v = torch.stack([torch.roll(v, i, 0) for i in range(w_size + 1)], dim=1) #n, w_size+1, h, d
    return (qk_attn.unsqueeze(-1) * v).sum(1) #n, h, d

def seq_operator(layer_module, x_offsets, sub_seq_emb, n, addiction_loss, normed_x, padded_x=None):
    seq_out = None
    padded_x = torch.ops.fbgemm.jagged_to_padded_dense(values=normed_x, offsets=[x_offsets], max_lengths=[n], padding_value=0.0) if padded_x is None else padded_x
    # if "vae_pattern" in layer_module.block_type:
    #     padded_x, mu, logvar, addiction_loss['kld_loss'] = layer_module.vae_encoder(padded_x)
    #     pattern_x = layer_module.proj_pattern(padded_x)
    #     pattern_x = torch.ops.fbgemm.dense_to_jagged(pattern_x, [x_offsets])[0]
    #     normed_x = layernorm(layer_module.module_gate(normed_x, pattern_x))

    if "conv_softmax" in layer_module.block_type:
        padded_x = torch.ops.fbgemm.jagged_to_padded_dense(values=normed_x, offsets=[x_offsets], max_lengths=[n], padding_value=0.0) #B, N, D
        pattern_x = layer_module.conv(padded_x) # B, N, D
        pattern_x = torch.ops.fbgemm.dense_to_jagged(pattern_x, [x_offsets])[0]
        seq_out = pattern_x
        
    if "xattn_pattern" in layer_module.block_type:
        padded_x = layer_module.conv(padded_x)
        padded_x = torch.ops.fbgemm.dense_to_jagged(padded_x, [x_offsets])[0]
        pattern_matrix = layer_module.pattern_matrix
        if layer_module._embedding_dim % layer_module._num_heads != 0:
            padded_x = layer_module.in_xattn_proj(padded_x)
        pattern_matrix = layernorm(pattern_matrix)
        pattern_out = layer_module.pattern_cross_attn_layer(padded_x, pattern_matrix, pattern_matrix, need_weights=False)[0]
        if layer_module._embedding_dim % layer_module._num_heads != 0:
            pattern_out = layer_module.out_xattn_proj(pattern_out)
        seq_out = pattern_out
        
    if "xattn_pattern_v2" in layer_module.block_type:
        pattern_x = layer_module.conv(padded_x, padding=False) # B, N - k + 1, d
        sim = torch.einsum('bnd, bmd -> bnm', pattern_x, pattern_x)
        num_valid = sim.size(-2)
        
        hstu_attn_mask = torch.tril(torch.ones(num_valid, num_valid, dtype=torch.bool, device=sim.device))
        hstu_attn_mask = hstu_attn_mask & (~torch.eye(num_valid, dtype=torch.bool, device=sim.device))
        sim = sim.masked_fill(hstu_attn_mask.expand(sim.size(0), -1, -1) == 0, 0.0)
        
        ref_seq = padded_x[:, -num_valid:, :]
        sim, ref_seq = sim[:, :, :-1], ref_seq[:, 1:, :]
        pattern_out = torch.einsum('bnm, bmd -> bnd', sim, ref_seq)
        pattern_out = F.pad(pattern_out, (0, 0, padded_x.size(1)-num_valid, 0), value=0.0)
        
        pattern_out = torch.ops.fbgemm.dense_to_jagged(pattern_out, [x_offsets])[0]
        seq_out = pattern_out
        
    if "xattn_pattern_v3" in layer_module.block_type or "xattn_pattern_v4" in layer_module.block_type or  "xattn_pattern_v5" in layer_module.block_type:
        padded_x = layer_module.conv(padded_x)
        padded_x = torch.ops.fbgemm.dense_to_jagged(padded_x, [x_offsets])[0]
        pattern_matrix = layer_module.pattern_matrix
        pattern_matrix = layernorm(pattern_matrix)
        seq_out = layer_module.pattern_cross_attn_layer(padded_x, pattern_matrix)
        
    # if "pattern_enc" in layer_module.block_type:
    #     sub_seq_emb = layernorm(sub_seq_emb)
    #     sub_seq_emb = layer_module.enc_p(sub_seq_emb)
    #     padded_x = torch.ops.fbgemm.jagged_to_padded_dense(values=normed_x, offsets=[x_offsets], max_lengths=[n], padding_value=0.0)
    #     if layer_module._embedding_dim % layer_module._num_heads != 0:
    #         padded_x = layer_module.in_xattn_proj(padded_x)
    #         sub_seq_emb = layer_module.in_xattn_proj(sub_seq_emb)
    #     L = padded_x.size(1)
    #     attn_mask = get_causal_mask(L, padded_x.device)
    #     pattern_out = layer_module.pattern_cross_attn_layer(padded_x, sub_seq_emb, sub_seq_emb, attn_mask=attn_mask, need_weights=False)[0]
    #     if layer_module._embedding_dim % layer_module._num_heads != 0:
    #         pattern_out = layer_module.out_xattn_proj(pattern_out)
    #     pattern_out = torch.ops.fbgemm.dense_to_jagged(pattern_out, [x_offsets])[0]
    #     seq_out = pattern_out
        # normed_x = layer_module.module_gate(normed_x, pattern_out)
        
    # if "dynamic_conv" in layer_module.block_type:
    #     padded_x = layer_module.conv(padded_x)
    #     new_normed_x = torch.ops.fbgemm.dense_to_jagged(padded_x, [x_offsets])[0]
    #     normed_x = layer_module.module_gate(new_normed_x, normed_x)
    seq_out = layernorm(seq_out) if (seq_out is not None) else None
    return addiction_loss, seq_out
    # return sub_seq_emb, normed_x, addiction_loss, seq_out
    
    
def attn_operator(layer_module, x_offsets, repeat_mask, repeat_count, n, addiction_loss, normed_x, attn_gate, B, qk_attn):
    attn_scale, attn_mod, attn_mod_id = layer_module.attn_scale, layer_module.attn_mod, layer_module.attn_mod_id
    if "attention_gate" in layer_module.block_type or "subseq_attngate" in layer_module.block_type or "subseq_attngate_shift_v2" in layer_module.block_type:
        qk_attn = qk_attn * attn_gate # gate operation
            
    if "pattern_v2" in layer_module.block_type:
        padded_x = torch.ops.fbgemm.jagged_to_padded_dense(
                values=normed_x, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
                )
        pattern_scores = torch.matmul(padded_x, layer_module.pattern_matrix) # [B, N, D] * [D, P] = [B, N, P]
        pattern_scores = F.softmax(pattern_scores, dim=-1) # [B, N, P]
        wm = layer_module.w_m(torch.arange(padded_x.size(1), device=padded_x.device)).T # [P, N]
        gate = torch.einsum("bnp, pm -> bnm",pattern_scores, wm).unsqueeze(1) # [B, N, P] * [P, N] = [B, N, N]
        gate = F.sigmoid(gate / F.sigmoid(layer_module.attn_gate_factor)) # [B, H, N, N]
        qk_attn = gate * qk_attn
    
    if "pattern" in layer_module.block_type:
        padded_x = torch.ops.fbgemm.jagged_to_padded_dense(
                values=normed_x, offsets=[x_offsets], max_lengths=[n], padding_value=0.0
                )
        pattern_scores = torch.matmul(padded_x, layer_module.pattern_matrix) # [B, N, D] * [D, P] = [B, N, P]
        pattern_scores = F.softmax(pattern_scores, dim=-1) # [B, N, P]
        wm = layer_module.w_m(torch.arange(padded_x.size(1), device=padded_x.device)).T # [P, N]
        qk_attn = torch.matmul(pattern_scores, wm).unsqueeze(1) * qk_attn # [B, N, P] * [P, N] = [B, N, N]

    if "scale" in layer_module.block_type and attn_scale >= 0 and attn_scale < qk_attn.size(-1) - 1:
        device = qk_attn.device
        idx = torch.arange(qk_attn.size(-1), device=device)
        mask = (idx[None, None, :] >= (idx[:, None] - attn_scale)) & (idx[None, None, :] <= (idx[:, None] + attn_scale))
        qk_attn = qk_attn * mask.unsqueeze(0)

    if "sparse" in layer_module.block_type and attn_mod_id >= 0:
        device = qk_attn.device
        idx = torch.arange(qk_attn.size(-1), device=device)
        mask = ((idx[None, :] - idx[:, None]) % attn_mod == attn_mod_id) | torch.eye(qk_attn.size(-1), dtype=torch.bool, device=device)
        qk_attn = qk_attn * mask.unsqueeze(0).unsqueeze(0)
            
    if "topkattn" in layer_module.block_type and attn_scale > 0 and attn_scale < qk_attn.size(-1):
        _, indice = torch.topk(qk_attn, attn_scale, -1, sorted=False)
        mask = torch.zeros_like(qk_attn, dtype=torch.float)
        mask.scatter_(-1, indice, 1.0)
        qk_attn = qk_attn * mask
            
    if "subseq_attnbasev1" in layer_module.block_type and repeat_mask.any():
        addiction_loss["repeat_attn_loss"] = -torch.log(F.softmax(qk_attn, dim=-1)[repeat_mask.unsqueeze(1).expand(-1, qk_attn.size(1), -1, -1)]+1e-12).mean()
    if "subseq_attnbasev2" in layer_module.block_type and repeat_mask.any():
        repeat_mask_withhead = repeat_mask.unsqueeze(1).expand(-1, qk_attn.size(1), -1, -1)# B*H*N*N
        qkattn_part = qk_attn.view(-1, 1)[repeat_mask_withhead.reshape(-1)]# _,1
        qk_attn_new = layer_module.mlp_merge(torch.cat([repeat_count, qkattn_part], dim=-1))# _,1
            
        qkattn_part = layer_module.gate(qkattn_part, qk_attn_new)# _,1
        qk_attn_new = qk_attn.clone()
        qk_attn_new[repeat_mask_withhead] = qkattn_part.flatten()
        qk_attn = torch.where(repeat_mask_withhead, qk_attn_new, qk_attn)
    return qk_attn, addiction_loss

def get_causal_mask(L,device):
    attn_mask = torch.tril(torch.ones((L, L), dtype=torch.bool), diagonal=0)
    attn_mask = ~attn_mask
    attn_mask = attn_mask.to(device)
    return attn_mask
class VAE_pattern(nn.Module):
    def __init__(self, input_dim=160, kernel_size=3, hidden_dim=64, latent_dim=8):
        super(VAE_pattern, self).__init__()
        self.conv1 = CausalConv1D(input_dim, hidden_dim, kernel_size)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)
        # self.conv3 = CausalConv1D(latent_dim, hidden_dim, kernel_size)
        # self.conv4 = CausalConv1D(hidden_dim, input_dim, kernel_size)

    def encode(self, x):
        h1 = torch.relu(self.conv1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # def decode(self, z):
    #     h3 = torch.relu(self.conv3(z))
    #     return torch.sigmoid(self.conv4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        pattern = self.reparameterize(mu, logvar) # B,N,P
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return pattern, mu, logvar, kld_loss
        # return self.decode(z), mu, logvar
    
    
class Icb_block(nn.Module):
    def __init__(self, input_dim, kernel_size=1, inter_kernel_size=3, need_global_inter=True):
        super(Icb_block, self).__init__()
        self.input_dim = input_dim
        self.need_global_inter = need_global_inter
        self.self_conv = CausalConv1D(self.input_dim, input_dim, kernel_size)
        self.inter_conv = CausalConv1D(self.input_dim, input_dim, inter_kernel_size) if need_global_inter else None
        self.act = nn.GELU()
        self.out_conv = CausalConv1D(self.input_dim, input_dim, 1)
    
    def forward(self, x, inter_x=None):
        batch_size, seq_length, embed_dim = x.shape
        x = self.self_conv(x)
        if inter_x is None:
            inter_x = self.inter_conv(x)
        output = self.act(inter_x) * x + inter_x * self.act(x)
        output = self.out_conv(output)
        return output
    
class Conv1dMultiHeadAttention(nn.Module):
    def __init__(self, input_dim ,embed_dim, linear_hidden_dim, attention_dim,  num_heads, kernel_size=1, dropout=0.1, training=True):
        super(Conv1dMultiHeadAttention, self).__init__()
        assert input_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = linear_hidden_dim
        self.head_attention_dim = attention_dim
        self.kernel_size = kernel_size
        self.scale = self.head_dim ** -0.5
        self.training = training

        self.query_conv = CausalConv1D(self.input_dim, attention_dim * num_heads, kernel_size)
        self.key_conv = CausalConv1D(self.input_dim, attention_dim * num_heads, kernel_size)
        self.value_conv = CausalConv1D(self.input_dim, linear_hidden_dim * num_heads, kernel_size)

        self.out_proj = nn.Linear(self.head_dim * self.num_heads, embed_dim)
    
    def forward(self, x, attn_mask):
        batch_size, seq_length, embed_dim = x.shape

        # Apply convolutional layers to get Q, K, V
        q = self.query_conv(x).permute(0, 2, 1)
        k = self.key_conv(x).permute(0, 2, 1)
        v = self.value_conv(x).permute(0, 2, 1)

        # Change shape to (batch_size, num_heads, seq_length, head_dim)
        q = q.view(batch_size, self.num_heads, self.head_attention_dim, seq_length).permute(0, 1, 3, 2) # B,H,N,dq
        k = k.view(batch_size, self.num_heads, self.head_attention_dim, seq_length).permute(0, 1, 3, 2) # B,H,N,dk
        v = v.view(batch_size, self.num_heads, self.head_dim, seq_length).permute(0, 1, 3, 2) # B,H,N,dv

        # Compute attention scores
        attn_scores = self.scale * torch.einsum("bhnd,bhmd->bhnm", q, k)
        attn_scores = attn_scores.masked_fill(attn_mask.expand(attn_scores.size(0), attn_scores.size(1), -1, -1) == 0, float('-inf'))
        attn_probs = F.softmax(attn_scores, dim=-1)

        # Compute attention output
        context = torch.einsum("bhnm,bhmd->bhnd", attn_probs, v)
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, self.head_dim * self.num_heads)

        # Final linear projection
        output = self.out_proj(context)
        return output
    
class Conv1d_attngate_v2(nn.Module):
    def __init__(self, input_dim, num_heads, kernel_size=3):
        super(Conv1d_attngate_v2, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.kernel_size = kernel_size

        self.qk_conv = CausalConv1D(input_dim, 2 * num_heads, kernel_size, has_bias=True)

    def forward(self, x):
        batch_size, seq_length, embed_dim = x.shape

        # Apply convolutional layers to get Q, K
        q, k = torch.split(self.qk_conv(x), [self.num_heads] * 2, dim=-1)
        q = q.view(batch_size, seq_length, self.num_heads, 1)
        k = k.view(batch_size, seq_length, 1, self.num_heads)
        attn_scores =  q + k
        return attn_scores
    
class Conv1d_attngate_v6(nn.Module):
    def __init__(self, input_dim, attention_dim,  num_heads, kernel_size=1, hideen_kernel_size=64):
        super(Conv1d_attngate_v6, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        self.kernel_size = kernel_size
        self.scale = attention_dim ** -0.5

        self.qk_conv_hidden = CausalConv1D(input_dim, attention_dim, hideen_kernel_size)
        self.qk_conv = CausalConv1D(attention_dim, attention_dim * num_heads * 2, kernel_size)

    def forward(self, x):
        batch_size, seq_length, embed_dim = x.shape
        x = F.leaky_relu(layernorm(self.qk_conv_hidden(x)))
        # Apply convolutional layers to get Q, K
        q, k = torch.split(self.qk_conv(x), [self.num_heads * self.attention_dim] * 2, dim=-1)
        # Change shape to (batch_size, num_heads, seq_length, head_dim)
        q = q.view(batch_size, seq_length, self.num_heads, self.attention_dim)
        k = k.view(batch_size, seq_length, self.num_heads, self.attention_dim)

        # Compute attention scores
        attn_scores = self.scale * torch.einsum("bnhd,bmhd->bhnm", q, k)
        return attn_scores
class Conv1d_attngate(nn.Module):
    def __init__(self, input_dim, attention_dim,  num_heads, kernel_size=1):
        super(Conv1d_attngate, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        self.kernel_size = kernel_size
        self.scale = attention_dim ** -0.5

        self.qk_conv = CausalConv1D(input_dim, attention_dim * num_heads * 2, kernel_size)

    def forward(self, x):
        batch_size, seq_length, embed_dim = x.shape

        # Apply convolutional layers to get Q, K
        q, k = torch.split(self.qk_conv(x), [self.num_heads * self.attention_dim] * 2, dim=-1)
        # Change shape to (batch_size, num_heads, seq_length, head_dim)
        q = q.view(batch_size, seq_length, self.num_heads, self.attention_dim)
        k = k.view(batch_size, seq_length, self.num_heads, self.attention_dim)

        # Compute attention scores
        attn_scores = self.scale * torch.einsum("bnhd,bmhd->bhnm", q, k)
        return attn_scores

def token_dropout(tensor, p=0.5):
    mask = (torch.rand(tensor.shape[:-1], dtype=tensor.dtype, device=tensor.device) > p).to(dtype=tensor.dtype)
    mask = mask.unsqueeze(-1)
    return tensor * mask

class ConvSoftmax(nn.Module):
    def __init__(self, embedding_dim, n_patterns, conv_kernelsize, residual=True):
        super(ConvSoftmax, self).__init__()
        self.conv_kernelsize = conv_kernelsize
        self.embedding_dim = embedding_dim
        self.conv = CausalConv1D(embedding_dim, n_patterns, conv_kernelsize, residual=False, has_bias=False, use_act=False)
        self.project_v = nn.Linear(embedding_dim * conv_kernelsize, embedding_dim)
        self.residual = residual
        # self.pattern_matrix = nn.Parameter(torch.randn(n_patterns, embedding_dim))
        # torch.nn.init.xavier_uniform_(self.pattern_matrix)
        # self.batch_norm = nn.BatchNorm1d(n_patterns)
        
    def forward(self, padded_x):
        # Apply causal convolution
        pattern_x = self.conv(padded_x)  # shape: (B, N, P)
        
        # Scaling
        scale = torch.sqrt(torch.tensor(self.conv_kernelsize * self.embedding_dim, dtype=pattern_x.dtype, device=pattern_x.device))
        pattern_x = pattern_x / scale

        # Apply softmax
        pattern_x = F.softmax(pattern_x, dim=-1)  # shape: (B, N, P)
        
        # Project the convolution weights
        pattern_matrix = self.project_v(self.conv.weight.flatten(start_dim=1))  # shape: (P, D)
        
        # Compute the final output using einsum
        pattern_x = torch.einsum('bnp,pd->bnd', pattern_x, pattern_matrix)  # shape: (B, N, D)
        if self.residual:
            pattern_x = pattern_x + padded_x
        return pattern_x
    
class CausalConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual=False, has_bias=False, use_act=False, init_method="normal"):
        super(CausalConv1D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.residual = residual
        
        # 定义权重矩阵
        self.weight = torch.nn.Parameter(torch.empty((out_channels, in_channels, kernel_size)).normal_(mean=0, std=0.02))
        self.bias = None
        self.has_bias = has_bias
        if has_bias:
            self.bias = torch.nn.Parameter(torch.empty((out_channels,)).normal_(mean=0, std=0.02))
        self.use_act = use_act
            
    def forward(self, x, padding=True):
        # 输入形状: (batch, seq, in_channels)
        batch, seq, in_channels = x.shape
        input_x = x.clone()
        
        x = x.permute(0, 2, 1)
        if padding:
            pad = (self.kernel_size - 1, 0)
            x = F.pad(x, pad, "constant", 0)

        x = F.conv1d(x, self.weight, self.bias)

        # 返回到原始形状：[batch, seqlen, hiddendim/outputchannel]
        x = x.permute(0, 2, 1)
        
        if self.residual:
            x = x + input_x
            
        if self.use_act:
            x = F.leaky_relu(x)
        return x
    
class DynamicConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_list):
        super(DynamicConvolution, self).__init__()
        self.conv_list = torch.nn.ModuleList([CausalConv1D(in_channels, out_channels, i, residual=True, has_bias=True, use_act=True) for i in kernel_size_list])

    def forward(self, x):
        # outputs = [layer(x) for layer in self.conv_list]
        futures = [torch.jit.fork(layer, x) for layer in self.conv_list]
        outputs = [torch.jit.wait(future) for future in futures]
        outputs = torch.stack(outputs, dim=0)
        outputs = outputs.mean(dim=0)
        return outputs
    
def l2_normalize_last_dim(x):
    norm = x.norm(p=2, dim=-1, keepdim=True)
    normalized_tensor = x / (norm + 1e-12)
    return normalized_tensor
def get_split_size(seq_len, w):
    part = seq_len // w
    mod = seq_len % w
    if mod > 0:
        return [w] * part + [mod]
    else:
        return [w] * part
def split_linear(tensor, w, layer):
    batchsize_list = get_split_size(tensor.size(0), w)
    tensor = tensor.split(batchsize_list)
    tensor = [layer(i) for i in tensor]
    return torch.cat(tensor, dim=0)
def sequence_equality(tensor, w, seqlen=267, padding=False):
    if isinstance(w, list):
        return [sequence_equality(tensor, i) for i in w]
    B, N = tensor.shape
    # 使用unfold获得形状为[B, N-w+1, w]的张量
    sub_sequences = tensor.unfold(dimension=1, size=w, step=1)
    
    # 将sub_sequences扩展到形状为[B, N-w+1, N-w+1, w],两个N-w+1的维度表示可以比较的两个子序列
    ss_expanded1 = sub_sequences.unsqueeze(2).expand(-1, -1, N-w+1, -1)
    ss_expanded2 = sub_sequences.unsqueeze(1).expand(-1, N-w+1, -1, -1)
    
    # 比较扩展的两个序列,如果所有的位都一致,则返回True,否则返回False
    # 结果shape为[B, N-w+1, N-w+1]
    equal_sequences = torch.all(ss_expanded1 == ss_expanded2, dim=3)
    if padding:
        equal_sequences = F.pad(equal_sequences, (seqlen - equal_sequences.size(-1), 0, seqlen - equal_sequences.size(-1), 0), value=False)
    return equal_sequences

class GatedUnit(nn.Module):
    def __init__(self, embed_dim, embed_dim2=None):
        super(GatedUnit, self).__init__()
        self.proj = None
        if embed_dim2 and embed_dim2 != embed_dim:
            self.proj = nn.Linear(embed_dim + embed_dim2, embed_dim)
            self.gate = nn.Linear(embed_dim + embed_dim2, embed_dim + embed_dim2)
        else:
            self.gate = nn.Linear(embed_dim * 2, embed_dim)
    
    def forward(self, x, prev_output):
        # x: Input from the current layer
        # prev_output: Output from the previous layer
        combined = torch.cat([x, prev_output], dim=-1)
        gate_values = torch.sigmoid(self.gate(combined))
        if self.proj:
            output = gate_values * combined
            return self.proj(output)
        return gate_values * x + (1 - gate_values) * prev_output
    
def layernorm(x):
    return F.layer_norm(x, normalized_shape=[x.size(-1)])

class SimpleEncoderCausal(nn.Module):
    def __init__(self, embed_dim, num_heads, layernum=2):
        super(SimpleEncoderCausal, self).__init__()
        self.embed_dim = embed_dim
        self.attnlayer = nn.ModuleList([nn.MultiheadAttention(self.embed_dim, num_heads, batch_first=True) for i in range(layernum)])
        self.ffn = nn.ModuleList([nn.Linear(self.embed_dim, self.embed_dim) for i in range(layernum)])
    
    def forward(self, x):
        attn_mask = get_causal_mask(x.size(-2), x.device)
        lastx = x
        for attn, line in zip(self.attnlayer, self.ffn):
            x = attn(lastx, lastx, lastx, attn_mask=attn_mask, need_weights=False)[0]
            x = layernorm(lastx + x)
            x = line(x)
            x = layernorm(lastx + x)
            lastx = x
        return x
    
def limit_true_values_per_row(tensor, maxvalue):
    true_counts = tensor.sum(dim=-1, keepdim=True)
    if true_counts.max() < maxvalue:
        return tensor
    true_counts = true_counts.repeat(1, tensor.size(-1))
    random_tensor = torch.rand(tensor.shape, device=tensor.device)
    
    random_scores = torch.where(tensor, random_tensor, torch.ones_like(tensor))
    
    _, ind = torch.topk(random_scores, maxvalue, largest=False, dim=-1)
    new_value = torch.zeros_like(tensor).bool().flatten(0, -2)
    row_ind = torch.arange(new_value.size(0), device=new_value.device).view(-1, 1).repeat(1, maxvalue).view(-1)
    new_value[row_ind, ind.flatten()] = True
    new_value.view_as(tensor)
    new_tensor = torch.where(true_counts > maxvalue, new_value, tensor)
    
    return new_tensor

def find_duplicates(x):
    result = torch.zeros_like(x, dtype=bool)
    for i in range(1, x.size(1)): 
        result[:, i] = (x[:, :i] == x[:, i].unsqueeze(1)).any(dim=1)
    return result
def RegularizationLoss(x, norm_type='L1'):
    if norm_type == 'L1':
        loss = torch.sum(torch.abs(x))
    elif norm_type == 'L2':
        loss = torch.sum(x ** 2)
    return loss

def get_subseq_embedding(module, past_ids: torch.Tensor, sub_seqlen=3, padding=True) -> torch.Tensor:
    if isinstance(sub_seqlen, list):
        return [get_subseq_embedding(past_ids, seqlen, False) for seqlen in sub_seqlen]
    B,N = past_ids.size()
    if padding:
        pad = (sub_seqlen - 1, 0)
        past_ids = F.pad(past_ids, pad, "constant", 0)# B, sub_seqlen-1 + N
    past_ids = past_ids.unfold(1, sub_seqlen, 1) # B, _, sub_seqlen
    sub_seq_emb = module.get_item_embeddings(past_ids)# B, _, sub_seqlen, D
    return sub_seq_emb
    
def subseq_emb_v3(module, past_ids):
    seqlen = past_ids.size(-1) # N
    user_emb = module.get_item_embeddings(past_ids)
    pos_emb = torch.arange(seqlen, device=user_emb.device).view(1, -1).expand(past_ids.size(0), -1)
    x = module.userseq_pos(pos_emb) + user_emb
    user_emb = layernorm(user_emb)
    
    
    kernelsize_list = [1]
    sub_seq_mask = sequence_equality(past_ids, kernelsize_list)
    sub_seq_mask = torch.stack([F.pad(i, (seqlen - i.size(-1), 0, seqlen - i.size(-1), 0), value=False) for i in sub_seq_mask]) # kernel_num, B, N, N
    
    hstu_attn_mask = torch.tril(torch.ones(seqlen, seqlen, dtype=torch.bool, device=sub_seq_mask.device))
    hstu_attn_mask = hstu_attn_mask & (~torch.eye(seqlen, dtype=torch.bool, device=sub_seq_mask.device))
    sub_seq_mask = sub_seq_mask & (hstu_attn_mask[None, None, :, :])# kernel_num, B, N, N
    sub_seq_mask = sub_seq_mask.permute(1, 2, 0, 3)
    sub_seq_mask[past_ids == 0] = False
    sub_seq_mask = sub_seq_mask.permute(2, 0, 1, 3)
    
    sub_seq_mask = sub_seq_mask.any(dim=0, keepdim=True)
    sub_seq_mask, user_emb_shift = sub_seq_mask[..., :-1], user_emb[:, 1:, :]# kernel_num, B, N, N-1;   B, N-1, D
    
    max_emb_num = 5
    sub_seq_mask = limit_true_values_per_row(sub_seq_mask.flatten(0, -2), max_emb_num).view_as(sub_seq_mask)# kernel_num, B, N, N-1
    origin_size = sub_seq_mask.size()[:-1]
    past_length = sub_seq_mask.float().sum(dim=-1).view(-1).int()
    
    repeat_subseq_emb = user_emb_shift.unsqueeze(0).unsqueeze(2).expand(sub_seq_mask.size() + (user_emb_shift.size(-1),))
    repeat_subseq_emb = repeat_subseq_emb[sub_seq_mask] # _,D
    
    x_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(past_length)
    repeat_subseq_emb = torch.ops.fbgemm.jagged_to_padded_dense(values=repeat_subseq_emb, offsets=[x_offsets], max_lengths=[max_emb_num]
                        , padding_value=0.0).view(origin_size + (max_emb_num, repeat_subseq_emb.size(-1))) # kernel_num, B, N, L, D
    repeat_subseq_emb = repeat_subseq_emb.permute(1,2,0,3,4).flatten(2,3)#  B, N, (kernel_num*L), D
    
    
    x = layernorm(x)
    x = module.encoderlayer(x)
    
    sim_score = torch.einsum("bnmd,bnd->bnm", module.subseq_key(repeat_subseq_emb), x)
    sim_score /= torch.sqrt(torch.tensor(x.size(-1), dtype=torch.float, device=x.device))
    sim_score = F.softmax(sim_score, dim=-1)
    x = torch.einsum("bnm,bnmd->bnd", sim_score, module.subseq_value(repeat_subseq_emb))
    return x

def subseq_emb_v3_1(module, past_ids):
    seqlen = past_ids.size(-1) # N
    user_emb = module.get_item_embeddings(past_ids)
    user_emb = layernorm(user_emb)
    
    
    kernelsize_list = [2]
    sub_seq_mask = sequence_equality(past_ids, kernelsize_list)
    
    
    
    sub_seq_mask = torch.stack([F.pad(i, (seqlen - i.size(-1), 0, seqlen - i.size(-1), 0), value=False) for i in sub_seq_mask]) # kernel_num, B, N, N
    
    hstu_attn_mask = torch.tril(torch.ones(seqlen, seqlen, dtype=torch.bool, device=sub_seq_mask.device))
    hstu_attn_mask = hstu_attn_mask & (~torch.eye(seqlen, dtype=torch.bool, device=sub_seq_mask.device))
    sub_seq_mask = sub_seq_mask & (hstu_attn_mask[None, None, :, :])# kernel_num, B, N, N
    sub_seq_mask, user_emb_shift = sub_seq_mask[..., :-1], user_emb[:, 1:, :]# kernel_num, B, N, N-1;   B, N-1, D
    
    max_emb_num = 5
    sub_seq_mask = limit_true_values_per_row(sub_seq_mask.flatten(0, -2), max_emb_num).view_as(sub_seq_mask)# kernel_num, B, N, N-1
    origin_size = sub_seq_mask.size()[:-1]
    past_length = sub_seq_mask.float().sum(dim=-1).view(-1).int()
    
    
    past_length_float = sub_seq_mask.float().sum(dim=-1)[..., 10:]
    past_length_float = past_length_float.permute(1,2 ,0)[(past_ids > 0)[..., 10:]]
    
    repeat_subseq_emb = user_emb_shift.unsqueeze(0).unsqueeze(2).expand(sub_seq_mask.size() + (user_emb_shift.size(-1),))
    repeat_subseq_emb = repeat_subseq_emb[sub_seq_mask] # _,D
    
    x_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(past_length)
    repeat_subseq_emb = torch.ops.fbgemm.jagged_to_padded_dense(values=repeat_subseq_emb, offsets=[x_offsets], max_lengths=[max_emb_num]
                        , padding_value=0.0).view(origin_size + (max_emb_num, repeat_subseq_emb.size(-1))) # kernel_num, B, N, L, D, L==max_emb_num
    repeat_subseq_emb = repeat_subseq_emb.permute(1,2,0,3,4).flatten(2, -1)#  B, N, (kernel_num*L*D)
    
    return repeat_subseq_emb
# def insert_zeros(sequence, k):
#     if k == 0:
#         return sequence
#     batch_size, seqlen, dim = sequence.shape
#     # Create a tensor to hold the result
#     result_shape = (batch_size, seqlen + (seqlen - 1) * k, dim)
#     result = torch.zeros(result_shape, dtype=sequence.dtype, device=sequence.device)

#     result[:, ::(k + 1), :] = sequence

#     return result

# def remove_zeros(padded_sequence, k):
#     if k == 0:
#         return padded_sequence
#     batch_size, new_seqlen, dim = padded_sequence.shape
#     original_seqlen = (new_seqlen + k) // (k + 1)

#     # Create a tensor to hold the result
#     result = torch.zeros((batch_size, original_seqlen, dim), dtype=padded_sequence.dtype, device=padded_sequence.device)

#     # Fill the result tensor
#     result = padded_sequence[:, ::(k + 1), :]

#     return result
