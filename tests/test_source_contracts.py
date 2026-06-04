import unittest
from pathlib import Path
import sys

from evaluate import (
    build_prompt,
    parse_choice,
    parse_pope_answer,
    parse_yes_no,
    resolve_dataset_files,
    summarize_records,
)


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
if str(MODELS) not in sys.path:
    sys.path.insert(0, str(MODELS))

from config import VLMConfig


def read_text(relative_path):
    return (ROOT / relative_path).read_text(encoding="utf-8")


class MoESourceContractTests(unittest.TestCase):
    def test_config_exposes_dense_default_and_moe_parameters(self):
        source = read_text("models/config.py")

        self.assertIn("lm_use_moe: bool = False", source)
        self.assertIn("lm_num_experts: int = 8", source)
        self.assertIn("lm_num_experts_per_tok: int = 2", source)
        self.assertIn("lm_moe_inter_dim: int = 1280", source)
        self.assertIn("lm_norm_topk_prob: bool = True", source)
        self.assertIn("lm_router_aux_loss_coef: float = 0.01", source)
        self.assertIn("def __post_init__(self):", source)

    def test_config_rejects_more_selected_experts_than_available_experts(self):
        with self.assertRaises(ValueError):
            VLMConfig(lm_num_experts=2, lm_num_experts_per_tok=3)

    def test_default_chat_template_defers_to_smollm_tokenizer(self):
        self.assertIsNone(VLMConfig().lm_chat_template)

    def test_decoder_selects_moe_and_returns_auxiliary_loss(self):
        source = read_text("models/language_model.py")

        self.assertIn(
            "self.mlp = LanguageModelMoE(cfg) if cfg.lm_use_moe else LanguageModelMLP(cfg)",
            source,
        )
        self.assertIn("top_k_weight, top_k_indices = torch.topk(scores", source)
        self.assertIn("def forward(self, x, attention_mask=None):", source)
        self.assertIn("return x, block_kv_cache, router_aux_loss", source)

    def test_vlm_adds_auxiliary_loss_and_uses_cache_contract(self):
        source = read_text("models/vision_language_model.py")

        self.assertIn("hidden_status, _, router_aux_loss = self.decoder(", source)
        self.assertIn("loss = ce_loss + router_aux_loss", source)
        self.assertIn("block_kv_cache=None", source)
        self.assertIn("block_kv_cache=block_kv_cache", source)
        self.assertIn("next_output = next_output[:, -1, :]", source)
        self.assertNotIn("self.decoder.cfg", source)

    def test_tensor_conditions_are_explicit(self):
        lm_source = read_text("models/language_model.py")
        vlm_source = read_text("models/vision_language_model.py")

        self.assertNotIn("if attention_mask:", lm_source)
        self.assertNotIn("if additive_attn_mask:", lm_source)
        self.assertNotIn("if images_tensors:", vlm_source)
        self.assertNotIn("if target_ids:", vlm_source)
        self.assertNotIn("if attention_mask:", vlm_source)

    def test_image_string_and_cli_generation_blockers_are_fixed(self):
        processors_source = read_text("data/processors.py")
        generate_source = read_text("generate.py")

        self.assertIn("for idx, (nh, nw) in enumerate(splitted_image_counts):", processors_source)
        self.assertIn("tokenizer.batch_decode(", generate_source)
        self.assertIn("splitted_image_ratio != (1, 1)", generate_source)


class EvaluationAdapterTests(unittest.TestCase):
    def test_script_is_renamed_to_generic_evaluate_entrypoint(self):
        self.assertTrue((ROOT / "evaluate.py").is_file())
        self.assertFalse((ROOT / "evaluate_mmstar.py").exists())

    def test_default_dataset_paths_resolve_downloaded_benchmarks(self):
        self.assertEqual(len(resolve_dataset_files("mmstar")), 1)
        self.assertEqual(len(resolve_dataset_files("mme")), 4)
        self.assertEqual(len(resolve_dataset_files("pope")), 3)
        self.assertEqual(len(resolve_dataset_files("mme", "data/MME")), 4)
        self.assertEqual(len(resolve_dataset_files("pope", "data/POPE")), 3)

    def test_benchmark_answer_parsers(self):
        self.assertEqual(parse_choice("The answer is C."), "C")
        self.assertEqual(parse_choice("(A)"), "A")
        self.assertEqual(parse_choice("I think C because..."), "")
        self.assertEqual(parse_yes_no("Yes."), "yes")
        self.assertEqual(parse_yes_no("No, there is not."), "no")
        self.assertEqual(parse_yes_no("The answer is yes"), "")
        self.assertEqual(parse_pope_answer("There is not a dog. It may be a cat."), "no")
        self.assertEqual(parse_pope_answer("Maybe"), "yes")

    def test_prompts_match_benchmark_answer_type(self):
        self.assertIn("A, B, C, or D", build_prompt("mmstar", "Question", "strict"))
        self.assertIn("exactly Yes or No", build_prompt("mme", "Question", "strict"))
        self.assertIn("exactly Yes or No", build_prompt("pope", "Question", "strict"))

    def test_mmstar_summary_reports_accuracy_and_diagnostic(self):
        summary = summarize_records(
            "mmstar",
            [
                {
                    "correct": False,
                    "parsed_answer": "",
                    "category": "c1",
                    "l2_category": "l1",
                    "diagnostic_choice": "B",
                    "diagnostic_correct": True,
                }
            ],
            mode="multimodal_with_forced_choice",
            evaluation_config={},
        )

        self.assertEqual(summary["official_accuracy"], 0.0)
        self.assertEqual(summary["forced_choice_diagnostic"]["accuracy"], 1.0)

    def test_mme_summary_uses_accuracy_plus_paired_image_accuracy(self):
        summary = summarize_records(
            "mme",
            [
                {"question_id": "img1", "category": "existence", "correct": True},
                {"question_id": "img1", "category": "existence", "correct": True},
                {"question_id": "img2", "category": "existence", "correct": True},
                {"question_id": "img2", "category": "existence", "correct": False},
            ],
            mode="multimodal",
            evaluation_config={},
        )

        existence = summary["category"][0]
        self.assertEqual(existence["accuracy"], 75.0)
        self.assertEqual(existence["accuracy_plus"], 50.0)
        self.assertEqual(existence["score"], 125.0)
        self.assertEqual(summary["total_score"], 125.0)

    def test_pope_summary_reports_binary_classification_metrics(self):
        summary = summarize_records(
            "pope",
            [
                {"answer": "yes", "parsed_answer": "yes", "category": "random"},
                {"answer": "yes", "parsed_answer": "no", "category": "random"},
                {"answer": "no", "parsed_answer": "yes", "category": "random"},
                {"answer": "no", "parsed_answer": "no", "category": "random"},
            ],
            mode="multimodal",
            evaluation_config={},
        )

        self.assertEqual(summary["overall"]["accuracy"], 0.5)
        self.assertEqual(summary["overall"]["precision"], 0.5)
        self.assertEqual(summary["overall"]["recall"], 0.5)
        self.assertEqual(summary["overall"]["f1"], 0.5)
        self.assertEqual(summary["overall"]["yes_ratio"], 0.5)


if __name__ == "__main__":
    unittest.main()
