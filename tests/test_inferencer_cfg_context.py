import unittest
from unittest import mock

from PIL import Image
import torch

from any2any.inferencer import InterleaveInferencer


def _context(*items):
    return {"items": list(items)}


class _ImagePairHarness:
    def update_context_image(self, image, context, **kwargs):
        context["samples"] = context.get("samples", []) + [torch.rand(())]
        return context


class _FeedHarness:
    def __init__(self):
        self.pair_calls = []

    def update_context_codebook_image_pair(
        self, image, modality, gen_context, cfg_text_context
    ):
        self.pair_calls.append(modality)
        gen_context["items"].append(modality)
        cfg_text_context["items"].append(modality)
        return gen_context, cfg_text_context

    def update_context_image_pair(
        self, image, gen_context, cfg_text_context, **kwargs
    ):
        modality = kwargs.get("modality_type")
        self.pair_calls.append(modality)
        gen_context["items"].append(modality)
        cfg_text_context["items"].append(modality)
        return gen_context, cfg_text_context

    def update_context_text(self, text, context, modality_type=None):
        context["items"].append((modality_type, text))
        return context


class _CodebookPairHarness:
    def __init__(self):
        self.encode_calls = 0

    def _encode_codebook_condition_from_image(self, image, modality):
        self.encode_calls += 1
        return torch.tensor([3, 1, 4]), torch.device("cpu")

    def _append_codebook_condition(self, token_ids, modality, context, device):
        context["tokens"] = token_ids.clone()
        context["modality"] = modality
        return context


class InferencerCfgContextTest(unittest.TestCase):
    def test_image_pair_reuses_sample_and_advances_rng_once(self):
        torch.manual_seed(2026)
        expected_sample = torch.rand(())
        expected_next = torch.rand(())

        torch.manual_seed(2026)
        harness = _ImagePairHarness()
        gen_context, cfg_text_context = InterleaveInferencer.update_context_image_pair(
            harness, object(), {}, {}, vae=True, vit=False
        )

        self.assertTrue(
            torch.equal(gen_context["samples"][0], cfg_text_context["samples"][0])
        )
        self.assertTrue(torch.equal(gen_context["samples"][0], expected_sample))
        self.assertTrue(torch.equal(torch.rand(()), expected_next))

    def test_codebook_pair_encodes_once_and_reuses_tokens(self):
        harness = _CodebookPairHarness()
        gen_context, cfg_text_context = (
            InterleaveInferencer.update_context_codebook_image_pair(
                harness, object(), "dinolocal", {}, {}
            )
        )

        self.assertEqual(harness.encode_calls, 1)
        self.assertTrue(torch.equal(gen_context["tokens"], cfg_text_context["tokens"]))
        self.assertEqual(gen_context["modality"], "dinolocal")
        self.assertEqual(cfg_text_context["modality"], "dinolocal")

    def test_gt_feature_conditions_update_both_cfg_branches(self):
        image = Image.new("RGB", (8, 8))
        flag_names = {
            "dino": "use_gt_dino_condition",
            "dinolocal": "use_gt_dinolocal_condition",
            "clip": "use_gt_clip_condition",
            "imagebind": "use_gt_imagebind_condition",
            "imagebindlocal": "use_gt_imagebindlocal_condition",
        }

        for modality, flag_name in flag_names.items():
            with self.subTest(modality=modality):
                harness = _FeedHarness()
                flags = {name: False for name in flag_names.values()}
                flags[flag_name] = True
                gen_context, cfg_text_context, _, _ = InterleaveInferencer._feed_conditions(
                    harness,
                    [image],
                    _context("prompt"),
                    _context(),
                    _context(),
                    modality_type_dict={"condition": [modality]},
                    understanding_output=False,
                    use_instruction=False,
                    use_condition_instruction=False,
                    use_det_image=False,
                    do_modality_norm=False,
                    image_shapes=(8, 8),
                    cond_feature_mode="both",
                    **flags,
                )

                self.assertEqual(harness.pair_calls, [modality])
                self.assertEqual(gen_context["items"], ["prompt", modality])
                self.assertEqual(cfg_text_context["items"], [modality])

    @mock.patch("any2any.inferencer.get_det_image")
    def test_detection_image_updates_both_cfg_branches(self, get_det_image):
        get_det_image.return_value = Image.new("RGB", (8, 8))
        harness = _FeedHarness()

        gen_context, cfg_text_context, _, _ = InterleaveInferencer._feed_conditions(
            harness,
            ["boxes<sep>serialized"],
            _context(),
            _context(),
            _context(),
            modality_type_dict={"condition": ["det"]},
            understanding_output=False,
            use_instruction=False,
            use_condition_instruction=False,
            use_det_image=True,
            do_modality_norm=False,
            image_shapes=(8, 8),
        )

        self.assertEqual(harness.pair_calls, ["rgb"])
        self.assertEqual(cfg_text_context["items"], ["rgb"])
        self.assertIn("rgb", gen_context["items"])


if __name__ == "__main__":
    unittest.main()
