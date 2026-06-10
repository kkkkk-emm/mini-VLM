import pytest
import torch
from PIL import Image

from scripts.chat_ui import handle_chat_message, parse_args, prepare_vlm_inputs


class FakeTokenizer:
    image_token_id = 99
    image_token = "<|image|>"
    global_image_token = "<|global_image|>"

    def __init__(self):
        self.messages = None
        for row in range(1, 9):
            for col in range(1, 9):
                setattr(self, f"r{row}c{col}", f"<row_{row}_col_{col}>")

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.template_kwargs = kwargs
        return {"input_ids": [[1, self.image_token_id, 2]], "attention_mask": [[1, 1, 1]]}

    def batch_decode(self, generated_ids, skip_special_tokens=True):
        return ["a short answer"]


class FakeModel:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.cfg = type(
            "Cfg",
            (),
            {
                "max_img_size": 128,
                "vit_img_size": 64,
                "resize_to_max_side_len": False,
                "mp_image_token_length": 1,
            },
        )()
        self.generate_kwargs = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return torch.tensor([[7, 8]])


def test_parse_args_requires_checkpoint():
    with pytest.raises(SystemExit):
        parse_args([])

    args = parse_args(["--checkpoint", "ckpt"])

    assert args.checkpoint == "ckpt"
    assert args.host == "127.0.0.1"
    assert args.port == 7860
    assert args.share is False


def test_prepare_vlm_inputs_matches_generate_prompt_shape():
    model = FakeModel()
    image = Image.new("RGB", (80, 80), color=(30, 90, 120))

    input_ids, attention_mask, images = prepare_vlm_inputs(
        model=model,
        image=image,
        question="What is shown?",
        device=torch.device("cpu"),
    )

    assert input_ids.shape == (1, 3)
    assert attention_mask.shape == (1, 3)
    assert images.ndim == 4
    assert images.shape[-2:] == (64, 64)
    assert model.tokenizer.messages[0]["role"] == "user"
    assert model.tokenizer.messages[0]["content"].endswith("What is shown?")
    assert model.tokenizer.template_kwargs["add_generation_prompt"] is True


def test_handle_chat_message_requires_image():
    history, textbox = handle_chat_message(
        question="What is shown?",
        image=None,
        history=[],
        model=FakeModel(),
        device=torch.device("cpu"),
        max_new_tokens=12,
        top_k=20,
        top_p=0.8,
        temperature=0.7,
        greedy=False,
    )

    assert textbox == ""
    assert history[-1] == {"role": "assistant", "content": "Please upload an image first."}


def test_handle_chat_message_forwards_sidebar_generation_parameters():
    model = FakeModel()
    image = Image.new("RGB", (80, 80), color=(30, 90, 120))

    history, textbox = handle_chat_message(
        question="Answer briefly.",
        image=image,
        history=[],
        model=model,
        device=torch.device("cpu"),
        max_new_tokens=12,
        top_k=20,
        top_p=0.8,
        temperature=0.7,
        greedy=True,
    )

    assert textbox == ""
    assert history == [
        {"role": "user", "content": "Answer briefly."},
        {"role": "assistant", "content": "a short answer"},
    ]
    assert model.generate_kwargs["max_new_tokens"] == 12
    assert model.generate_kwargs["top_k"] == 20
    assert model.generate_kwargs["top_p"] == 0.8
    assert model.generate_kwargs["temperature"] == 0.7
    assert model.generate_kwargs["greedy"] is True
