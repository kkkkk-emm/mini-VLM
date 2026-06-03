import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from .config import VLMConfig
except ImportError:
    from config import VLMConfig

class RMSNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.eps = cfg.lm_rms_eps
        self.weight = nn.Parameter(torch.ones(cfg.lm_hidden_dim))

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(input_dtype)

class RotaryEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.lm_hidden_dim % cfg.lm_n_heads == 0
        self.dim = cfg.lm_hidden_dim // cfg.lm_n_heads
        self.max_seq_len = cfg.lm_max_position_embeddings
        self.base = cfg.lm_re_base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        self.attention_scaling = cfg.lm_attn_scaling

    def forward(self, position_ids: torch.Tensor):
        """
        position_ids:[batch_size, seq_len]
        """
        bsz, seq_len = position_ids.shape
        inv_freq = self.inv_freq # [D // 2]
        
        flat_position_ids = position_ids.reshape(-1).float() # [B * seq_len]
        freqs = flat_position_ids.unsqueeze(-1) * inv_freq.unsqueeze(0) # [B * seq_len, D // 2]
        freqs = freqs.reshape(bsz, seq_len, -1)
        
        emb = torch.cat([freqs, freqs], dim=-1)

        cos = torch.cos(emb) * self.attention_scaling
        sin = torch.sin(emb) * self.attention_scaling
        return cos, sin

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_embd(q, k, cos, sin, unsequeeze_dim: int=1):
    # 确保cos和sin的维度与q和k匹配
    cos = cos.unsqueeze(unsequeeze_dim) # [B, 1, seq_len, D]
    sin = sin.unsqueeze(unsequeeze_dim) # [B, 1, seq_len, D]

    q_embd = q * cos + rotate_half(q) * sin
    k_embd = k * cos + rotate_half(k) * sin
    return q_embd, k_embd

class LanguageModelGroupQueryAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.lm_n_heads
        self.embd_dim = cfg.lm_hidden_dim
        self.n_kv_heads = cfg.lm_n_kv_heads
        self.dropout = cfg.lm_dropout
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        assert self.embd_dim % self.n_heads == 0, "embd_dim must be divisible by n_heads"
        
        self.n_groups = self.n_heads // self.n_kv_heads
        self.head_dim = self.embd_dim // self.n_heads
        
        self.q_proj = nn.Linear(self.embd_dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.embd_dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.embd_dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.embd_dim, bias=False)

        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

        self.sdpa = hasattr(F, 'scaled_dot_product_attention')
        if not self.sdpa:
            print("Using custom attention implementation")

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attention_mask=None, block_kv_cache=None):
        is_prefill = block_kv_cache is None
        bsz, seq_len, dim = x.size()

        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2) # [bsz, n_heads, seq_len, head_dim]
        k = self.k_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2) # [bsz, n_kv_heads, seq_len, head_dim]
        v = self.v_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2) # [bsz, n_kv_heads, seq_len, head_dim]

        q, k = apply_rotary_pos_embd(q, k, cos, sin) # 应用旋转位置编码
        
        # 查看是否可以使用kv cache
        if not is_prefill and block_kv_cache['key'] is not None:
            k = torch.cat([block_kv_cache['key'], k], dim=-2)
            v = torch.cat([block_kv_cache['value'], v], dim=-2)
            block_kv_cache['key'] = k
            block_kv_cache['value'] = v
        else:
            block_kv_cache = {'key': k, 'value': v}
        
        # 将k,v扩展至与q相同数量
        k = k.repeat_interleave(self.n_groups, dim=1)
        v = v.repeat_interleave(self.n_groups, dim=1)

        seq_len_total = k.size(-2)

        additive_attn_mask = None # 用来屏蔽无意义的token，比如padding
        if attention_mask is not None:
            mask = attention_mask[..., :seq_len_total] # 只取前seq_len_total个元素
            mask = mask.unsqueeze(1).unsqueeze(2).float()  # [bsz, 1, 1, seq_len_total]
            additive_attn_mask = (1.0 - mask) * torch.finfo(q.dtype).min # 将掩码转换为负无穷

        is_causal = (seq_len == seq_len_total and seq_len > 1) # 第一次没有kv_cache，前后都有token可查看；有kv_cache，每次只会送一个token，所以没有未来的token可查看
        if self.sdpa and x.device.type != "mps":
            qkv = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=additive_attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal
            )
        else:
            attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if is_causal:
                casual_mask = torch.tril(torch.ones(seq_len_total, seq_len_total, dtype=torch.bool, device=x.device))
                attn = attn.masked_fill(casual_mask == 0, float('-inf'))

            if additive_attn_mask is not None:
                attn = attn + additive_attn_mask # 屏蔽无意义token
            
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            qkv = attn @ v # [bsz, n_heads, seq_len, head_dim]
        
        qkv = qkv.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_heads * self.head_dim)
        output = self.o_proj(qkv)
        output = self.resid_dropout(output)
        return output, block_kv_cache

class LanguageModelMLP(nn.Module):
    def __init__(self, cfg, inter_dim: int = None):
        super().__init__()
        self.embd_dim = cfg.lm_hidden_dim
        self.inter_dim = inter_dim or cfg.lm_inter_dim

        self.activate_fn = F.silu
        self.gate_proj = nn.Linear(self.embd_dim, self.inter_dim, bias=False)
        self.up_proj = nn.Linear(self.embd_dim, self.inter_dim, bias=False)
        self.down_proj = nn.Linear(self.inter_dim, self.embd_dim, bias=False)

    def forward(self, x):
        gate = self.activate_fn(self.gate_proj(x))
        up = self.up_proj(x)
        fuse = gate * up
        return self.down_proj(fuse)

class LanguageModelMoE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.num_experts = cfg.lm_num_experts
        self.gate = nn.Linear(cfg.lm_hidden_dim, self.num_experts, bias=False)
        self.experts = nn.ModuleList([LanguageModelMLP(cfg, cfg.lm_moe_inter_dim) for _ in range(self.num_experts)])

    def forward(self, x, attention_mask=None):
        bsz, seq_len, hidden_dim = x.shape
        x_flat = x.reshape(-1, hidden_dim)
        gate_logits = self.gate(x_flat)
        scores = F.softmax(gate_logits, dim=-1)
        top_k_weight, top_k_indices = torch.topk(scores, self.config.lm_num_experts_per_tok, dim=-1)
        if self.config.lm_norm_topk_prob:
            top_k_weight = top_k_weight / (top_k_weight.sum(dim=-1, keepdim=True) + 1e-20)

        y = torch.zeros_like(x_flat)
        unused_expert_sum = None
        for i, expert in enumerate(self.experts):
            token_idx, top_k_slot = torch.where(top_k_indices == i)
            if token_idx.numel() > 0:
                weight = top_k_weight[token_idx, top_k_slot].unsqueeze(-1)
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                expert_sum = sum(p.sum() for p in expert.parameters())
                unused_expert_sum = expert_sum if unused_expert_sum is None else unused_expert_sum + expert_sum

        if unused_expert_sum is not None:
            y = y + 0.0 * unused_expert_sum

        if self.training:
            if attention_mask is not None:
                valid_token_mask = attention_mask[..., -seq_len:].reshape(-1).bool()
                valid_scores = scores[valid_token_mask]
                valid_top_k_indices = top_k_indices[valid_token_mask]
            else:
                valid_scores = scores
                valid_top_k_indices = top_k_indices

            if valid_scores.numel() > 0:
                load = F.one_hot(valid_top_k_indices, num_classes=self.num_experts).float().mean(dim=(0, 1))
                aux_loss = (load * valid_scores.mean(dim=0)).sum() * self.num_experts
            else:
                aux_loss = scores.new_zeros(())
        else:
            aux_loss = scores.new_zeros(())

        return y.view(bsz, seq_len, hidden_dim), aux_loss


class LanguageModelBlock(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.attn = LanguageModelGroupQueryAttention(cfg)
        self.use_moe = cfg.lm_use_moe
        self.mlp = LanguageModelMoE(cfg) if cfg.lm_use_moe else LanguageModelMLP(cfg)
        self.norm1 = RMSNorm(cfg)
        self.norm2 = RMSNorm(cfg)

    def forward(self, x, cos, sin, attention_mask=None, block_kv_cache=None):
        residual = x # [B, S, D]
        attn_out, block_kv_cache = self.attn(self.norm1(x), cos, sin, attention_mask, block_kv_cache)
        x = residual + attn_out
        residual = x
        if self.use_moe:
            mlp_out, router_aux_loss = self.mlp(self.norm2(x), attention_mask=attention_mask)
        else:
            mlp_out = self.mlp(self.norm2(x))
            router_aux_loss = x.new_zeros(())
        x = residual + mlp_out
        return x, block_kv_cache, router_aux_loss

class LanguageModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.lm_use_tokens = cfg.lm_use_tokens # 判断是否传入的是token而非embedding
        self.lm_tie_weights = cfg.lm_tie_weights # 判断是否需要Embedding和head共用权重
        self.lm_router_aux_loss_coef = cfg.lm_router_aux_loss_coef
        self.blocks = nn.ModuleList([LanguageModelBlock(cfg) for _ in range(cfg.lm_n_blocks)])
        self.norm = RMSNorm(cfg)
        self.token_embedding = nn.Embedding(cfg.lm_vocab_size, cfg.lm_hidden_dim)
        self.rotary_embd = RotaryEmbedding(cfg)
        self.head = nn.Linear(cfg.lm_hidden_dim, cfg.lm_vocab_size, bias=False)
        if self.lm_tie_weights:
            self.head.weight = self.token_embedding.weight
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, RMSNorm):
            m.weight.data.fill_(1.0)
    
    def forward(self, x, attention_mask=None, block_kv_cache=None, start_pos=0):
        if self.lm_use_tokens:
            x = self.token_embedding(x) # [B, S, D_lm]
        
        bsz, seq_len, _ = x.shape
        freqs = torch.arange(start=start_pos, end=start_pos + seq_len, device=x.device).unsqueeze(0).expand(bsz, seq_len) # [B, S]
        cos, sin = self.rotary_embd(freqs) # [B, S, D_rotary]

        if block_kv_cache is None:
            block_kv_cache = [None] * len(self.blocks)

        router_aux_losses = []
        for i, block in enumerate(self.blocks):
            x, block_kv_cache[i], router_aux_loss = block(x, cos, sin, attention_mask, block_kv_cache[i])
            router_aux_losses.append(router_aux_loss)
        x = self.norm(x)
        if self.lm_use_tokens:
            x = self.head(x) # [B, S, V]
        router_aux_loss = torch.stack(router_aux_losses).mean() * self.lm_router_aux_loss_coef
        return x, block_kv_cache, router_aux_loss

if __name__ == "__main__":
    # 测试decoder
    cfg = VLMConfig()
    decoder = LanguageModel(cfg)
    x = torch.randn(2, 10, cfg.lm_hidden_dim)
    output = decoder(x)
    print(output[0].shape)
