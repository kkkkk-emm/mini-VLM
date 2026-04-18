from tkinter import NO
import os
import json
import torch 
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from models.utils import top_k_top_p_filtering
from config import VLMConfig
from language_model import LanguageModel
from vision_transformer import ViT
from modality_projector import ModalityProjector
from data.processors import get_tokenizer
from safetensors.torch import load_model

class VisionLanguageModel(nn.Module):
    def __init__(self, cfg: VLMConfig):
        super().__init__()
        self.vision_encoder = ViT(cfg)
        self.decoder = LanguageModel(cfg)
        self.projector = ModalityProjector(cfg)
        self.tokenizer = get_tokenizer(cfg.lm_tokenizer, cfg.vlm_extra_tokens, cfg.lm_chat_template)

    def _process_images(self, images, device):
        if isinstance(images, list):
            if images and isinstance(images[0], list):
                images = [img for sublist in images for img in sublist]
            
            if not images:
                return None
            else:
                return torch.cat(images, dim=0).to(device)
        return images

    def _replace_img_tokens_with_embd(self, input_ids, token_embd, img_embds):
        """
        将token_embd中的图像token替换为对应的图像embedding
        """
        update_token_embd = token_embd.clone()
        mask = (input_ids == self.tokenizer.image_token_id)
        update_token_embd[mask] = img_embds.view(-1, img_embds.size(-1)).to(update_token_embd.dtype)
        return update_token_embd

    def forward(self, input_ids, images, attention_mask=None, target_ids=None):
        images_tensors = self._process_images(images, input_ids.device)
        token_embd = self.decoder.token_embedding(input_ids)
        if images_tensors:
            images_embd = self.vision_encoder(images_tensors) # [N_Chunks, T_feat, D_vit]
            images_embd = self.projector(images_embd) # [N_Chunks, T_img, D_lm]
            token_embd = self._replace_img_tokens_with_embd(input_ids, token_embd, images_embd) # [batch_size, seq_len, D_lm]
        
        hidden_status, _ = self.decoder(token_embd, attention_mask=attention_mask) # [batch_size, seq_len, D_lm]
        logits = self.decoder.head(hidden_status) # [batch_size, seq_len, vocab_size]
        loss = None
        if target_ids:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1), ignore_index=-100)
        return logits, loss

    @torch.inference_mode()
    def generate(self, input_ids, images, attention_mask=None, max_new_tokens=5, top_k=50, top_p=0.9, temperature=0.5, greedy=False):
        """
        input_ids: [batch_size, seq_len]
        images: [N_Chunks, 3, height, width]
        """

        # 处理图像
        images_tensor = self._process_images(images, input_ids.device)
        token_embd = self.decoder.token_embedding(input_ids)
        if images_tensor is not None:
            images_embd = self.vision_encoder(images_tensor) # [N_Chunks, T_feat, D_vit]
            images_embd = self.projector(images_embd) # [N_Chunks, T_img, D_lm]
            token_embd = self._replace_img_tokens_with_embd(input_ids, token_embd, images_embd) # [batch_size, seq_len, D_lm]
        
        # 初始化自回归计算
        current_total_sel_len = token_embd.size(1)
        batch_size = token_embd.size(0)
        
        # prefill 计算
        prefill_output, kv_cache_lise = self.decoder(
            token_embd,
            attention_mask=attention_mask,
            kv_cache=None,
            start_pos=0
        ) # prefill_output: [B, T_prefill, V_lm] if lm_use_tokens else [B, T_prefill, D_lm]
        last_token_from_prefill = prefill_output[:, -1, :]

        if not self.decoder.lm_use_tokens:
            currrent_logits = self.decoder.head(last_token_from_prefill)
        else:
            currrent_logits = last_token_from_prefill
        
        generated_token_list = [] # 新生成的token ids, [[B, 1]]
        for _ in range(max_new_tokens):
            if greedy:
                generated_token = currrent_logits.argmax(dim=-1, keepdim=True)
            else:
                flitered_logits = top_k_top_p_filtering(currrent_logits, top_k=top_k, top_p=top_p)
                generated_token = torch.multinomial(F.softmax(flitered_logits / temperature, dim=-1), num_samples=1)
            
            generated_token_list.append(generated_token)
            generated_token_embd = self.decoder.token_embedding(generated_token)
            
            current_total_sel_len += 1
            if attention_mask:
                attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype, device=attention_mask.device)], dim=1)
            
            next_output, kv_cache_lise = self.decoder(
                generated_token_embd,
                attention_mask=attention_mask,
                kv_cache=kv_cache_lise,
                start_pos=current_total_sel_len - 1
            )
            if self.decoder.cfg.lm_use_tokens:
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
            mask = (indexs_new > min_for_row.unsqueeze(0).expand_as(generated_token_list)) # [B, T_gen]
            generated_token_list = generated_token_list.masked_fill(mask, self.tokenizer.pad_token_id)

        return generated_token_list

    @classmethod
    def from_pretrained(
        cls, repo: str, *, revison: Optional[str] = None
    ):
        """
        加载模型
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

        model = cls(config)
        load_model(model, weight_path)

        return model