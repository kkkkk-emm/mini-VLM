import argparse
from email import message
from email.mime import image
from opcode import hasarg
from PIL import Image
import torch

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

from data.processors import get_image_processor, get_image_string, get_tokenizer
from models.vision_language_model import VisionLanguageModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None, 
                        help="Path to the model checkpoint")
    parser.add_argument("--hf_model", type=str, default="lusxvr/nanoVLM-230M-8k", 
                        help="Path to the Hugging Face model")
    parser.add_argument("--image", type=str, default="assets/image.png", 
                        help="Path to the input image")
    parser.add_argument("--prompt", type=str, default="What is in the image?", 
                        help="The prompt for image generation")
    parser.add_argument("--generations", type=int, default=5, 
                        help="Number of output texts to generate")
    parser.add_argument("--max_new_tokens", type=int, default=300, 
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--measure_vram", action="store_true", 
                        help="Measure VRAM usage")
    return parser.parse_args()


def main():
    args = parse_args()
    # 运行环境
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    # 加载模型
    source = args.checkpoint if args.checkpoint else args.hf_model
    print(f"Loading model from: {source}")
    if args.measure_vram and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model = VisionLanguageModel.from_pretrained(source).to(device)
    model.eval()
    # 计算显存占用
    if args.measure_vram and torch.cuda.is_available():
        torch.cuda.synchronize() # 确保模型异步加载完成
        vram_usage = torch.cuda.max_memory_allocated()
        print(f"VRAM usage: {vram_usage / (1024 ** 2):.2f} MB")
    # 获取tokenizer和图形处理器
    tokenizer = get_tokenizer(model.cfg.lm_tokenizer, model.cfg.vlm_extra_tokens, model.cfg.lm_chat_template)
    resize_to_max_side_len = False
    if hasattr(model.cfg, "resize_to_max_side_len"):
        resize_to_max_side_len = model.cfg.resize_to_max_side_len
    image_processor = get_image_processor(model.cfg.max_image_size, model.cfg.splitted_image_size, resize_to_max_side_len)
    # 对图像进行处理
    image = Image.open(args.image).convert("RGB")
    processed_image, splitted_image_ratio = image_processor(image) # (N_num, 3, P, P), (nh, nw)
    if not hasattr(tokenizer, "global_image_token"):
        processed_image = processed_image[1:]
    # 处理prompt
    image_string = get_image_string(tokenizer, [splitted_image_ratio], model.cfg.mp_image_token_length)
    messages = [{"role": "user", "content": image_string + args.prompt}]
    encode_prompt = tokenizer.apply_chat_template([messages], tokenized=True, add_generation_prompt=True, return_dict=True)
    # 将数据送入内存
    tokens = torch.tensor(encode_prompt["input_ids"], dtype=torch.long, device=device) # [1, L]
    img_t = processed_image.to(device) # [N_num, 3, P, P]
    print("\ninput: ", {args.prompt}, "\n output: ")
    for i in range(args.generations):
        gen = model.generate(tokens, img_t, max_new_tokens=args.max_new_tokens) # [1, L]
        out = tokenizer.decode(gen, skip_special_tokens=True)[0]
        if args.measure_vram and torch.cuda.is_available():
            torch.cuda.synchronize()
            max_vram_usage = torch.cuda.max_memory_allocated(device)
            current_vram_usage = torch.cuda.memory_allocated(device)
            print(f"generation {i+1}")
            print(f"Max VRAM usage: {max_vram_usage / (1024 ** 2):.2f} MB")
            print(f"Current VRAM usage: {current_vram_usage / (1024 ** 2):.2f} MB")
        else:
            print(f"generation {i+1}: {out}")


if __name__ == "__main__":
    main()