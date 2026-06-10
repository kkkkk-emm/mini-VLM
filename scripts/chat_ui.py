#!/usr/bin/env python3
"""Gradio chat UI for mini-VLM image question answering."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.processors import get_image_processor, get_image_string
from models.vision_language_model import VisionLanguageModel


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the mini-VLM Gradio chat UI.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Complete trained VLM checkpoint directory containing config.json and model.safetensors.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host address for the Gradio server.")
    parser.add_argument("--port", type=int, default=7860, help="Port for the Gradio server.")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Default max_new_tokens value.")
    parser.add_argument("--top-k", type=int, default=50, help="Default top_k value.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Default top_p value.")
    parser.add_argument("--temperature", type=float, default=0.5, help="Default temperature value.")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding by default.")
    args = parser.parse_args(argv)

    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.top_k < 0:
        parser.error("--top-k must be non-negative")
    if not 0.0 < args.top_p <= 1.0:
        parser.error("--top-p must be in the range (0, 1]")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    if args.port <= 0:
        parser.error("--port must be positive")
    return args


def build_prompt_inputs(
    *,
    model: Any,
    image: Image.Image,
    question: str,
) -> tuple[dict[str, Any], torch.Tensor]:
    """Build tokenizer inputs and processed image tensor with generate.py-compatible logic."""

    image = image.convert("RGB")
    resize_to_max_side_len = getattr(model.cfg, "resize_to_max_side_len", False)
    image_processor = get_image_processor(
        model.cfg.max_img_size,
        model.cfg.vit_img_size,
        resize_to_max_side_len,
    )
    processed_image, split_ratio = image_processor(image)
    tokenizer = model.tokenizer
    if not hasattr(tokenizer, "global_image_token") and split_ratio != (1, 1):
        processed_image = processed_image[1:]

    image_string = get_image_string(tokenizer, [split_ratio], model.cfg.mp_image_token_length)
    prompt = image_string + question.strip()
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )
    return encoded, processed_image


def prepare_vlm_inputs(
    *,
    model: Any,
    image: Image.Image,
    question: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    encoded, processed_image = build_prompt_inputs(model=model, image=image, question=question)
    input_ids = encoded["input_ids"]
    if input_ids and isinstance(input_ids[0], int):
        input_ids = [input_ids]

    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None and attention_mask and isinstance(attention_mask[0], int):
        attention_mask = [attention_mask]

    tokens = torch.tensor(input_ids, dtype=torch.long, device=device)
    attention = (
        torch.tensor(attention_mask, dtype=torch.long, device=device)
        if attention_mask is not None
        else torch.ones_like(tokens, dtype=torch.long, device=device)
    )
    return tokens, attention, processed_image.to(device)


def generate_answer(
    *,
    model: Any,
    device: torch.device,
    image: Image.Image,
    question: str,
    max_new_tokens: int,
    top_k: int,
    top_p: float,
    temperature: float,
    greedy: bool,
) -> str:
    input_ids, attention_mask, images = prepare_vlm_inputs(
        model=model,
        image=image,
        question=question,
        device=device,
    )
    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=input_ids,
            images=images,
            attention_mask=attention_mask,
            max_new_tokens=int(max_new_tokens),
            top_k=int(top_k),
            top_p=float(top_p),
            temperature=float(temperature),
            greedy=bool(greedy),
        )
    return model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def handle_chat_message(
    question: str | None,
    image: Image.Image | None,
    history: list[dict[str, str]] | None,
    model: Any,
    device: torch.device,
    max_new_tokens: int,
    top_k: int,
    top_p: float,
    temperature: float,
    greedy: bool,
) -> tuple[list[dict[str, str]], str]:
    history = list(history or [])
    question = (question or "").strip()
    if not question:
        history.append({"role": "assistant", "content": "Please enter a question."})
        return history, ""

    history.append({"role": "user", "content": question})
    if image is None:
        history.append({"role": "assistant", "content": "Please upload an image first."})
        return history, ""

    try:
        answer = generate_answer(
            model=model,
            device=device,
            image=image,
            question=question,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            greedy=greedy,
        )
    except Exception as error:
        answer = f"Generation failed: {error}"

    history.append({"role": "assistant", "content": answer})
    return history, ""


def build_demo(model: Any, device: torch.device, args: argparse.Namespace):
    import gradio as gr

    with gr.Blocks(title="mini-VLM") as demo:
        gr.HTML(
            "<div style='text-align:center;margin:18px 0 10px;'>"
            "<h1 style='font-size:32px;font-weight:700;margin:0;'>mini-VLM</h1>"
            "</div>"
        )
        with gr.Row():
            with gr.Column(scale=1, min_width=260):
                gr.Markdown("### Generation")
                max_new_tokens = gr.Slider(
                    1,
                    512,
                    value=args.max_new_tokens,
                    step=1,
                    label="max_new_tokens",
                )
                top_k = gr.Slider(0, 200, value=args.top_k, step=1, label="top_k")
                top_p = gr.Slider(0.05, 1.0, value=args.top_p, step=0.01, label="top_p")
                temperature = gr.Slider(
                    0.05,
                    2.0,
                    value=args.temperature,
                    step=0.05,
                    label="temperature",
                )
                greedy = gr.Checkbox(value=args.greedy, label="greedy")

            with gr.Column(scale=5):
                chatbot = gr.Chatbot(
                    label="Chatbot",
                    elem_classes=["mini-vlm-chat"],
                    height=620,
                )
                with gr.Row(elem_classes=["mini-vlm-input-row"]):
                    image = gr.Image(
                        label="Image",
                        type="pil",
                        sources=["upload", "clipboard"],
                        height=110,
                        scale=1,
                    )
                    question = gr.Textbox(
                        label="",
                        placeholder="Ask a question, then click Send.",
                        scale=6,
                        lines=2,
                    )
                    send = gr.Button("Send", variant="primary", scale=1)
                    clear = gr.Button("Clear", scale=1)

        inputs = [
            question,
            image,
            chatbot,
            max_new_tokens,
            top_k,
            top_p,
            temperature,
            greedy,
        ]

        def submit(question, image, history, max_new_tokens, top_k, top_p, temperature, greedy):
            return handle_chat_message(
                question,
                image,
                history,
                model,
                device,
                max_new_tokens,
                top_k,
                top_p,
                temperature,
                greedy,
            )

        send.click(submit, inputs=inputs, outputs=[chatbot, question])
        question.submit(submit, inputs=inputs, outputs=[chatbot, question])
        clear.click(lambda: ([], ""), outputs=[chatbot, question])

    return demo


def load_model(checkpoint: str, device: torch.device) -> VisionLanguageModel:
    model = VisionLanguageModel.from_pretrained(checkpoint).to(device)
    model.eval()
    return model


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = select_device()
    print(f"Using device: {device}")
    print(f"Loading model from: {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    demo = build_demo(model, device, args)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
