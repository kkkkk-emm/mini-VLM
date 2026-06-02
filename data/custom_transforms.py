import torch
import torch.nn as nn
import math
from torchvision.transforms.functional import resize, InterpolationMode
from einops import rearrange
from PIL import Image
from typing import Union, Tuple

class DynamicResize(nn.Module):
    def __init__(self, patch_size: int, max_size: int, resize_to_max=False, 
                 interpolation=InterpolationMode.BICUBIC):
        super().__init__()
        self.patch_size = patch_size
        self.max_size = max_size
        self.resize_to_max = resize_to_max
        self.interpolation = interpolation

    def _get_new_hw(self, h: int, w: int):
        """
        计算新的高度和宽度
        """
        long, short = (h, w) if h >= w else (w, h)
        long = self.max_size if self.resize_to_max else min(self.max_size, math.ceil(long / self.patch_size) * self.patch_size)
        scale = long / max(w, h)
        short = math.ceil(short * scale / self.patch_size) * self.patch_size
        return (long, short) if h >= w else (short, long)

    def forward(self, img: Union[Image.Image, torch.Tensor]):
        if isinstance(img, Image.Image):
            w, h = img.size
            new_h, new_w = self._get_new_hw(h, w)
            return resize(img, [new_h, new_w], self.interpolation)
        if not isinstance(img, torch.Tensor):
            raise TypeError(f"Input must be a PIL Image or a torch.Tensor. Got {type(img)}")
        batched = img.ndim == 4
        if img.ndim not in [3, 4]:
            raise ValueError(f"Input tensor must be 3D or 4D. Got {img.ndim}D")
        imgs = img if batched else img.unsqueeze(0)
        _, _, h, w = imgs.shape
        new_h, new_w = self._get_new_hw(h, w)
        out = resize(imgs, [new_h, new_w], self.interpolation)
        return out if batched else out.squeeze(0)

class SplitImage(nn.Module):
    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor):
        """
        将图像分割成补丁(B, C, H, W) -> (B * num_patches, patch_size, patch_size)
        """
        if x.ndim == 3:
            x = x.unsqueeze(0)
        b, c, h, w = x.shape
        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError(f"Image size must be divisible by patch size. Got image size {h}x{w}, patch size {self.patch_size}")
        hh = h // self.patch_size
        ww = w // self.patch_size
        x = rearrange(x, "b c (hh ph) (ww pw) -> (b hh ww) c ph pw", ph=self.patch_size, pw=self.patch_size)
        return x, (hh, ww)
    
class GlobalAndSplitImages(nn.Module):
    def __init__(self, patch_size: int):
        super().__init__()
        self.split_image = SplitImage(patch_size)
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor):
        """
        添加全局特征
        """
        patchs, grid = self.split_image(x)
        if grid == (1, 1):
            return patchs, grid
        global_patch = resize(x, [self.patch_size, self.patch_size])
        if global_patch.ndim == 3:
            global_patch = global_patch.unsqueeze(0)
        return torch.cat([global_patch, patchs], dim=0), grid
        
