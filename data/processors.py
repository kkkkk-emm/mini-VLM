from transformers import AutoTokenizer
from torchvision.transforms import transforms

from data.custom_transforms import DynamicResize, GlobalAndSplitImages

TOKENIZER_CACHE = {}

def get_tokenizer(tokenizer_name, extra_tokens, chat_template):
    """
    获取tokenizer,并且缓存以提高性能
    """
    if tokenizer_name not in TOKENIZER_CACHE:
        tokenizer_args = {"use_fast": True}
        if extra_tokens:
            tokenizer_args["additional_special_tokens"] = extra_tokens
        if chat_template:
            tokenizer_args["chat_template"] = chat_template
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_args)
        TOKENIZER_CACHE[tokenizer_name] = tokenizer
    return TOKENIZER_CACHE[tokenizer_name]

def get_image_processor(max_image_size, splitted_image_size, resize_to_max_side_len=False):
    """
    获取图像处理器
    """
    return transforms.Compose([
        DynamicResize(splitted_image_size, max_image_size, resize_to_max_side_len), # [[N, 3, P, P]]
        transforms.ToTensor(),
        GlobalAndSplitImages(splitted_image_size)
    ])

def get_image_string(tokenizer, splitted_image_counts, mp_image_token_length):
    """
    根据分割后的图像数量和每个图像的token长度生成图像字符串
    """
    image_strings = ""
    for idx, (nh, nw) in enumerate(splitted_image_counts):
        if len(splitted_image_counts) > 1:
            image_strings += f"<image {idx}>"
        if hasattr(tokenizer, "global_image_token"):
            image_strings += tokenizer.global_image_token
            image_strings += tokenizer.image_token * mp_image_token_length
            if nh == 1 and nw == 1:
                # 只有一个patch
                continue
        for i in range(nh):
            for j in range(nw):
                image_strings += f"<image c{i + 1}r{j + 1}>"
                image_strings += tokenizer.image_token * mp_image_token_length
    return image_strings
