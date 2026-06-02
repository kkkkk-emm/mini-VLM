from turtle import forward

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import VLMConfig

class ViTPatchEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.img_size = cfg.vit_img_size
        self.patch_size = cfg.vit_patch_size
        self.nums_patches = (cfg.vit_img_size // cfg.vit_patch_size) ** 2
        self.embed_dim = cfg.vit_hidden_dim
        self.cls_flag = cfg.vit_cls_flag

        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0
        )

        if self.cls_flag:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.position_embedding = nn.Parameter(torch.zeros(1, self.nums_patches + 1, self.embed_dim))
        else:
            self.position_embedding = nn.Parameter(torch.zeros(1, self.nums_patches, self.embed_dim))

    def forward(self, x):
        x = self.conv(x) # [batch_size, embed_dim, vit_img_size // vit_patch_size, vit_img_size // vit_patch_size]
        x = x.flatten(2).transpose(1, 2) # [batch_size, nums_patches, embed_dim]

        if self.cls_flag:
            cls_tokens = self.cls_token.expand(x.size(0), -1, -1) # [batch_size, 1, embed_dim]
            x = torch.cat((cls_tokens, x), dim=1) # [batch_size, nums_patches + 1, embed_dim]

        x += self.position_embedding
        return x

class ViTMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.activate_fn = nn.GELU(approximate='tanh')
        self.fc1 = nn.Linear(cfg.vit_hidden_dim, cfg.vit_inter_dim)
        self.fc2 = nn.Linear(cfg.vit_inter_dim, cfg.vit_hidden_dim)
        self.dropout = nn.Dropout(cfg.vit_dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activate_fn(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class ViTMultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.vit_n_heads
        self.embd_dim = cfg.vit_hidden_dim
        assert self.embd_dim % self.n_heads == 0
        self.head_dim = self.embd_dim // self.n_heads
        self.qkv = nn.Linear(self.embd_dim, 3 * self.embd_dim, bias=True)
        self.out = nn.Linear(self.embd_dim, self.embd_dim, bias=True)
        # dropout层
        self.attn_dropout = nn.Dropout(cfg.vit_dropout)
        self.resid_dropout = nn.Dropout(cfg.vit_dropout)

        self.sdpa = hasattr(F, 'scaled_dot_product_attention')
        if not self.sdpa:
            print("Using custom attention implementation.")

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x) # [B, T, 3 * C]
        q, k, v = qkv.split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) # [B, n_heads, T, head_dim]
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) # [B, n_heads, T, head_dim]
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) # [B, n_heads, T, head_dim]
        if self.sdpa:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=False
            )
        else:
            qk = q @ k.transpose(-2, -1) # [B, n_heads, T, T]
            qk = qk / (self.head_dim ** 0.5)
            attn = F.softmax(qk, dim=-1)
            attn = self.attn_dropout(attn)
            x = attn @ v # [B, n_heads, T, head_dim]

        x = x.transpose(1, 2).contiguous().view(B, T, C) # [B, T, C]
        x = self.out(x)
        x = self.resid_dropout(x)
        return x

class ViTBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.ln1 = nn.LayerNorm(cfg.vit_hidden_dim, cfg.vit_ln_eps)
        self.attn = ViTMultiHeadAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.vit_hidden_dim, cfg.vit_ln_eps)
        self.mlp = ViTMLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class ViT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.cls_flag = cfg.vit_cls_flag
        self.patch_embedding = ViTPatchEmbedding(cfg)
        self.dropout = nn.Dropout(cfg.vit_dropout)
        self.blocks = nn.ModuleList([ViTBlock(cfg) for _ in range(cfg.vit_n_blocks)])
        self.norm = nn.LayerNorm(cfg.vit_hidden_dim, cfg.vit_ln_eps)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0) # 初始化β
            torch.nn.init.ones_(m.weight) # 初始化γ
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.patch_embedding(x) # [batch_size, nums_patches, embed_dim]
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        if self.cls_flag:
            x = self.norm(x[:, 0]) # [batch_size, embed_dim]
        else:
            x = self.norm(x) # [batch_size, nums_patches, embed_dim]
        return x

if __name__ == "__main__":
    # 测试ViT是否正常
    cfg = VLMConfig()
    vit = ViT(cfg)
    x = torch.randn(2, 3, 512, 512)
    output = vit(x)
    print(output.shape)