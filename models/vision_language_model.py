import os
import json
import torch 
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from models.utils import top_k_top_p_filtering
try:
    from .backbone_loader import load_hf_backbones
    from .config import VLMConfig
    from .language_model import LanguageModel
    from .vision_transformer import ViT
    from .modality_projector import ModalityProjector
except ImportError:
    from backbone_loader import load_hf_backbones
    from config import VLMConfig
    from language_model import LanguageModel
    from vision_transformer import ViT
    from modality_projector import ModalityProjector
from data.processors import get_tokenizer
from safetensors.torch import load_model

class VisionLanguageModel(nn.Module):
    """视觉-语言模型主类，整合视觉编码器、模态投影器与解码器语言模型。

    该类封装了将图像特征注入到语言模型的流程，并提供 `forward` 与
    `generate` 两种接口：前者用于带标签的训练，后者用于自回归推理/生成。
    """
    def __init__(self, cfg: VLMConfig, tokenizer_path: Optional[str] = None):
        super().__init__()
        self.cfg = cfg
        tokenizer_source = tokenizer_path or cfg.lm_tokenizer
        self.tokenizer = get_tokenizer(tokenizer_source, cfg.vlm_extra_tokens, cfg.lm_chat_template)
        cfg.lm_vocab_size = len(self.tokenizer)
        self.vision_encoder = ViT(cfg)
        self.decoder = LanguageModel(cfg)
        self.projector = ModalityProjector(cfg)

    def _process_images(self, images, device):
        """规范化并将输入的 image 数据转换为单个张量，移动到 `device`。

        支持的输入形式包括单张/多张 PIL.Image、bytes、或已经拼接的张量列表。
        若 images 为空或处理后为空，则返回 None。
        """
        if isinstance(images, list):
            if images and isinstance(images[0], list):
                images = [img for sublist in images for img in sublist]
            
            if not images:
                return None
            else:
                return torch.cat(images, dim=0).to(device)
        return images

    def _replace_img_tokens_with_embd(self, input_ids, token_embd, img_embds):
        """将序列中占位的图像 token embedding 替换为视觉特征投影后的向量。

        参数:
            input_ids: LongTensor，输入 id 序列。
            token_embd: FloatTensor，初始的 token embedding 张量。
            img_embds: FloatTensor，经视觉编码与 projector 得到的图像 embedding。

        返回:
            替换后新的 token embedding 张量。
        """
        update_token_embd = token_embd.clone()
        mask = (input_ids == self.tokenizer.image_token_id)
        update_token_embd[mask] = img_embds.view(-1, img_embds.size(-1)).to(update_token_embd.dtype)
        return update_token_embd

    def forward(self, input_ids, images, attention_mask=None, target_ids=None):
        """模型前向计算：将图像特征注入 token embedding 并计算 logits（可选返回 loss）。

        参数:
            input_ids: LongTensor，输入 token id，形状 [B, T]
            images: 图像或图像列表，支持多种格式（见 `_process_images`）。
            attention_mask: 可选的 attention mask。
            target_ids: 可选的标签 id，用于计算交叉熵 loss（训练时使用）。

        返回:
            logits: FloatTensor，形状 [B, T, V]
            loss: Optional[Tensor]，当 `target_ids` 提供时返回交叉熵 loss 与 router aux loss 之和。
        """
        images_tensors = self._process_images(images, input_ids.device)
        token_embd = self.decoder.token_embedding(input_ids)
        if images_tensors is not None:
            images_embd = self.vision_encoder(images_tensors) # [N_Chunks, T_feat, D_vit]
            images_embd = self.projector(images_embd) # [N_Chunks, T_img, D_lm]
            token_embd = self._replace_img_tokens_with_embd(input_ids, token_embd, images_embd) # [batch_size, seq_len, D_lm]
        
        hidden_status, _, router_aux_loss = self.decoder(token_embd, attention_mask=attention_mask) # [batch_size, seq_len, D_lm]
        logits = self.decoder.head(hidden_status) # [batch_size, seq_len, vocab_size]
        loss = None
        if target_ids is not None:
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1), ignore_index=-100)
            loss = ce_loss + router_aux_loss
        return logits, loss

    @torch.inference_mode()
    def generate(self, input_ids, images, attention_mask=None, max_new_tokens=5, top_k=50, top_p=0.9, temperature=0.5, greedy=False):
        """自回归生成接口。

        参数:
            input_ids: LongTensor，输入 prompt 的 token id，形状 [B, T_prefill]
            images: 图像张量或列表，会被 `_process_images` 规范化。
            attention_mask: 可选的 attention mask。
            max_new_tokens: int，要生成的最大 token 数。
            top_k, top_p, temperature: 采样参数。
            greedy: bool，是否使用贪心解码（True 则忽略采样参数）。

        返回:
            生成的 token id 张量，形状 [B, T_gen]（若没有生成则返回空张量）。
        """

        # 处理图像
        images_tensor = self._process_images(images, input_ids.device)
        token_embd = self.decoder.token_embedding(input_ids) # [batch_size, seq_len, D_lm]
        if images_tensor is not None:
            images_embd = self.vision_encoder(images_tensor) # [B, nums_patches, D_vit]
            images_embd = self.projector(images_embd) # [B, mp_image_token_length, D_lm]
            token_embd = self._replace_img_tokens_with_embd(input_ids, token_embd, images_embd) # [batch_size, seq_len, D_lm]
        
        # 初始化自回归计算
        current_total_sel_len = token_embd.size(1)
        batch_size = token_embd.size(0)
        
        # prefill 计算
        prefill_output, block_kv_cache, _ = self.decoder(
            token_embd,
            attention_mask=attention_mask,
            block_kv_cache=None,
            start_pos=0
        ) # prefill_output: [B, T_prefill, V_lm] if lm_use_tokens else [B, T_prefill, D_lm]
        last_token_from_prefill = prefill_output[:, -1, :]

        if not self.decoder.lm_use_tokens:
            currrent_logits = self.decoder.head(last_token_from_prefill)
        else:
            currrent_logits = last_token_from_prefill
        
        generated_token_list = [] # 新生成的token ids, [[B, 1]]
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = eos_token_id
        finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        for _ in range(max_new_tokens):
            if greedy:
                generated_token = currrent_logits.argmax(dim=-1, keepdim=True)
            else:
                flitered_logits = top_k_top_p_filtering(currrent_logits, top_k=top_k, top_p=top_p)
                generated_token = torch.multinomial(F.softmax(flitered_logits / temperature, dim=-1), num_samples=1)
            
            if eos_token_id is not None:
                if finished.any() and pad_token_id is not None:
                    generated_token = torch.where(
                        finished.unsqueeze(1),
                        torch.full_like(generated_token, pad_token_id),
                        generated_token,
                    )
                finished = finished | (generated_token.squeeze(1) == eos_token_id)

            generated_token_list.append(generated_token)
            if eos_token_id is not None and bool(finished.all()):
                break

            generated_token_embd = self.decoder.token_embedding(generated_token)
            
            current_total_sel_len += 1
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype, device=attention_mask.device)], dim=1)
            
            next_output, block_kv_cache, _ = self.decoder(
                generated_token_embd,
                attention_mask=attention_mask,
                block_kv_cache=block_kv_cache,
                start_pos=current_total_sel_len - 1
            )
            next_output = next_output[:, -1, :]
            if self.decoder.lm_use_tokens:
                currrent_logits = next_output # [B, V_lm]
            else:
                currrent_logits = self.decoder.head(next_output) # [B, V_lm]
        
        if not generated_token_list:
            return torch.empty((batch_size, 0), dtype=torch.long, device=input_ids.device)
        generated_token_list = torch.cat(generated_token_list, dim=1) # [B, T_gen]

        # 处理EOS
        if self.tokenizer.eos_token_id is not None and generated_token_list.numel() > 0:
            seq_len = generated_token_list.size(1)
            device = generated_token_list.device

            eos_mask = (generated_token_list == self.tokenizer.eos_token_id) # [B, T_gen]
            indexs = torch.arange(seq_len, device=device) # [T_gen]
            masked_indexs = torch.where(eos_mask, indexs.unsqueeze(0).expand_as(generated_token_list), seq_len) # [B, T_gen]
            min_for_row = masked_indexs.min(dim=1, keepdim=True).values # [B]

            indexs_new = torch.arange(seq_len, device=device) # [T_gen]
            mask = (indexs_new.unsqueeze(0) > min_for_row) # [B, T_gen]
            generated_token_list = generated_token_list.masked_fill(mask, self.tokenizer.pad_token_id)

        return generated_token_list

    @classmethod
    def from_pretrained(
        cls, repo: str, *, revison: Optional[str] = None
    ):
        """从本地目录或 HuggingFace Hub 加载预训练模型与 tokenizer。

        参数:
            repo: 本地路径或 HF 仓库 id。
            revison: 可选的 HF 版本/commit/分支。

        返回:
            已加载的 `VisionLanguageModel` 实例（权重已加载）。
        """
        # 加载本地模型
        if os.path.exists(repo):
            config_path = os.path.join(repo, "config.json")
            weight_path = os.path.join(repo, "model.safetensors")

            if not os.path.exists(config_path) or not os.path.exists(weight_path):
                raise ValueError("Invalid model path. Please check the config.json and model.safetensors files.")
        # 加载huggingface模型
        else:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(
                repo_id=repo, filename="config.json", revision=revison
            )
            weight_path = hf_hub_download(
                repo_id=repo, filename="model.safetensors", revision=revison
            )
        
        with open(config_path, "r") as f:
            config = VLMConfig(**json.load(f))

        tokenizer_path = repo if os.path.exists(repo) else None
        model = cls(config, tokenizer_path=tokenizer_path)
        load_model(model, weight_path)

        return model

    @classmethod
    def from_hf_backbones(cls, cfg: VLMConfig):
        """基于 HuggingFace backbone 配置构建模型实例，并可选择加载骨干权重。

        参数:
            cfg: `VLMConfig` 实例。

        返回:
            构建好的 `VisionLanguageModel` 实例。
        """
        model = cls(cfg)
        if cfg.vlm_load_backbone_weights:
            load_hf_backbones(model)
        return model
