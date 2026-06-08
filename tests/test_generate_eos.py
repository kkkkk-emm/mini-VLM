import torch

from models.vision_language_model import VisionLanguageModel


class DummyTokenizer:
    eos_token_id = 2
    pad_token_id = 0


class DummyDecoder:
    lm_use_tokens = True

    def __init__(self, logits_by_call):
        self.logits_by_call = logits_by_call
        self.calls = 0

    def token_embedding(self, token_ids):
        return torch.zeros(
            token_ids.size(0),
            token_ids.size(1),
            1,
            dtype=torch.float32,
            device=token_ids.device,
        )

    def __call__(
        self,
        token_embd,
        attention_mask=None,
        block_kv_cache=None,
        start_pos=0,
    ):
        logits = self.logits_by_call[self.calls].to(token_embd.device)
        self.calls += 1
        return logits, block_kv_cache or [], torch.zeros((), device=token_embd.device)


def _logits_for_next_tokens(next_tokens, *, seq_len=1, vocab_size=5):
    logits = torch.full((len(next_tokens), seq_len, vocab_size), -100.0)
    for row, token_id in enumerate(next_tokens):
        logits[row, -1, token_id] = 100.0
    return logits


def _dummy_model(decoder):
    model = VisionLanguageModel.__new__(VisionLanguageModel)
    model.tokenizer = DummyTokenizer()
    model.decoder = decoder
    model._process_images = lambda images, device: None
    return model


def test_generate_breaks_immediately_when_all_rows_emit_eos():
    decoder = DummyDecoder(
        logits_by_call=[
            _logits_for_next_tokens([DummyTokenizer.eos_token_id], seq_len=3),
            _logits_for_next_tokens([4]),
        ]
    )
    model = _dummy_model(decoder)

    generated = model.generate(
        input_ids=torch.tensor([[1, 1, 1]]),
        images=None,
        max_new_tokens=5,
        greedy=True,
    )

    assert generated.tolist() == [[DummyTokenizer.eos_token_id]]
    assert decoder.calls == 1


def test_generate_pads_finished_rows_until_remaining_rows_finish():
    decoder = DummyDecoder(
        logits_by_call=[
            _logits_for_next_tokens([DummyTokenizer.eos_token_id, 3], seq_len=2),
            _logits_for_next_tokens([4, DummyTokenizer.eos_token_id]),
            _logits_for_next_tokens([4, 4]),
        ]
    )
    model = _dummy_model(decoder)

    generated = model.generate(
        input_ids=torch.tensor([[1, 1], [1, 1]]),
        images=None,
        max_new_tokens=5,
        greedy=True,
    )

    assert generated.tolist() == [
        [DummyTokenizer.eos_token_id, DummyTokenizer.pad_token_id],
        [3, DummyTokenizer.eos_token_id],
    ]
    assert decoder.calls == 2
