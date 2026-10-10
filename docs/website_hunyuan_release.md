# MODUS-Hunyuan website handoff

## Goal

Add MODUS-Hunyuan to `modus-multimodal.epfl.ch` without replacing or visually
demoting MODUS-BAGEL. The page should answer one question quickly: what changes
when the same MODUS recipe is applied to a HunyuanImage-3.0 backbone?

## Minimal page change

1. Keep the current hero, paper story and BAGEL results intact.
2. Add a compact **Models** row below the hero with two equal cards:
   **MODUS-BAGEL** (original, default) and **MODUS-Hunyuan** (new, optional).
   Each card needs only a model link, code link and license label.
3. Add one **Backbone comparison** section with a single two-column figure:
   **Independent prediction / Chained generation**. Do not add a second gallery
   or a long table.
4. Add one sentence below the figure: both checkpoints use the same 16-modality
   interface; Hunyuan requires a separate runtime and substantially more GPU
   memory. Link installation details to the repository README.

Suggested section copy:

> In our evaluated tasks, MODUS-Hunyuan is stronger at independent prediction,
> while MODUS-BAGEL remains more consistent in chained generation.

## What the comparison figure must show

Use matched inputs and label every column directly; do not imply that one model
wins every task.

| Panel | Claim | Representative evidence |
|---|---|---|
| Independent prediction | Hunyuan is stronger on learned visual features and direct structured prediction. | DINO-local cosine `0.896 vs 0.827`; COCO detection AP `0.374 vs 0.279`. |
| Chained generation | BAGEL preserves cross-step consistency better on the evaluated chain. | `RGB → COCO detections → depth`, five matched samples: pixel-r `0.716 vs 0.855` (Hunyuan vs BAGEL). |

Small labels under the two panels are enough: **independent prediction** and
**chain consistency**. Avoid paragraphs inside the figure. Do not add a
“Similar” category or claim text-to-image parity without a dedicated matched
evaluation.

## Evidence and asset source of truth

The release candidate is **MODUS-Hunyuan step 16,750 EMA**, not step 20,000.
The self-contained internal comparison page already embeds the reviewed images:

`MODUS/artifacts/modus_hunyuan_vs_modus_bagel_comparison.html`

Matched chained-generation aggregates and the paired contact sheet are under:

`MODUS/eval_outputs/release_compare/chain_multisample/summary/`

These are production inputs for the website agent. Do not publish filesystem
paths, seeds, debug captions or the full internal report on the website.

## Links and release order

- Code: `https://github.com/EPFL-VILAB/Modus`
- Original model: `https://huggingface.co/EPFL-VILAB/MODUS`
- New model: `https://huggingface.co/EPFL-VILAB/MODUS-Hunyuan`
- Hunyuan license note: repository `HUNYUAN_LICENSE_NOTICE.md`

Stage the website and verify the comparison first. Upload/publish the model
after review, then enable the MODUS-Hunyuan model link and publish the website
and repository update together. Keep unreleased links hidden rather than broken.

## Acceptance checklist

- Existing BAGEL copy, links, demo and figures are unchanged.
- The two claims above match the figure and use the 16,750 EMA results.
- Mobile layout stacks independent prediction before chained generation.
- Images have concise alt text and no essential conclusion is color-only.
- There is one comparison figure, one short paragraph and no duplicated gallery.
- Model and license links resolve before production deploy.
