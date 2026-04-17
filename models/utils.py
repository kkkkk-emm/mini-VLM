import torch
import torch.nn.functional as F



def top_k_top_p_filtering(logits, top_k=50, top_p=0.9):
    """
    对 logits 进行 top-k 和 top-p 过滤
    """
    top_k = min(top_k, logits.size(-1))
    # Top-k filtering
    if top_k > 0:
        indexs_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
        logits = logits.masked_fill(indexs_to_remove, float('-inf'))

    # Top-p filtering
    if top_p < 1.0:
        sorted_logits, sorted_index = torch.sort(logits, descending=True)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

        sorted_index_to_remove = cumulative_probs > top_p
        sorted_index_to_remove[..., 0] = False # 至少选一个

        indexs_to_remove = sorted_index_to_remove.scatter(-1, sorted_index, sorted_index_to_remove)
        logits = logits.masked_fill(indexs_to_remove, float('-inf'))

    return logits