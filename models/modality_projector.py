import torch
import torch.nn as nn

from config import VLMConfig

class ModalityProjector(nn.Module):
    def __init__(self, cfg: VLMConfig):
        super().__init__()
        self.input_dim = cfg.vit_hidden_dim * (cfg.mp_pixel_shuffle_factor ** 2)
        self.output_dim = cfg.lm_hidden_dim
        self.projector = nn.Linear(self.input_dim, self.output_dim)
        self.apply(self._init_weights) # 初始化线性层权重
        self.factor = cfg.mp_pixel_shuffle_factor
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def pixel_shuffle(self, x):
        """
        x: (bsz, token_num, hidden_dim)
        压缩图片token的数量从而降低复杂度
        """
        bsz, token_num, hidden_dim = x.shape
        h = w = int(token_num ** 0.5)
        assert h * w == token_num
        x = x.view(bsz, h, w, hidden_dim)
        h_out = h // self.factor
        w_out = w // self.factor
        x = x.reshape(bsz, h_out, self.factor, w_out, self.factor, hidden_dim)
        x = x.permute(0, 1, 3, 2, 4, 5) # (bsz, h', w', factor, factor, hidden_dim)
        x = x.reshape(bsz, h_out * w_out, hidden_dim * self.factor**2)
        return x

    def forward(self, x):
        x = self.pixel_shuffle(x) # (bsz, h' * w', hidden_dim * factor^2)
        x = self.projector(x) # (bsz, h' * w', output_dim)
        return x

if __name__ == "__main__":
    # (1, 1024, 768)
    cfg = config.VLMConfig()
    projector = ModalityProjector(cfg)
    x = torch.randn(1, 1024, 768)
    output = projector(x)
    print(output.shape)
