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
3. Add one **Backbone comparison** section with a single three-column figure:
   **Better / Similar / Worse**. Do not add a second gallery or a long table.
4. Add one sentence below the figure: both checkpoints use the same 16-modality
   interface; Hunyuan requires a separate runtime and substantially more GPU
   memory. Link installation details to the repository README.

Suggested section copy:

> MODUS-Hunyuan improves feature-map reconstruction and structured prediction,
> while preserving the core text-to-image capability. Chained generation is
> comparable overall, but MODUS-BAGEL remains more consistent on some
> detection-mediated chains.

## What the comparison figure must show

Use matched inputs and label every column directly; do not imply that one model
wins every task.

| Panel | Claim | Representative evidence |
|---|---|---|
| Better | Hunyuan is stronger on learned visual features and direct structured prediction. | DINO-local cosine `0.896 vs 0.827`; COCO detection AP `0.374 vs 0.279`. |
| Similar | Base text-to-image quality is retained. | One matched prompt pair; describe this as qualitative parity, not a benchmark win. ImageBind-global is also effectively tied: `0.899 vs 0.896`. |
| Worse | BAGEL is more reliable on the detection-mediated chain. | `RGB → COCO detections → depth`, five matched samples: pixel-r `0.716 vs 0.855` (Hunyuan vs BAGEL). |

Small labels under the three panels are enough: **feature fidelity**, **T2I
quality**, and **chain consistency**. Avoid paragraphs inside the figure.

## Evidence and asset source of truth

The release candidate is **MODUS-Hunyuan step 16,750 EMA**, not step 20,000.
The self-contained internal comparison page already embeds the reviewed images:

`MODUS/artifacts/modus_hunyuan_vs_modus_bagel_comparison.html`

Matched chained-generation aggregates and the paired contact sheet are under:

`MODUS/eval_outputs/release_compare/chain_multisample/summary/`

T2I candidates are under:

`MODUS/eval_outputs/release_compare/modus_hunyuan_16750_t2i/` and
`MODUS/eval_outputs/release_compare/modus_bagel_public_t2i/`

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
- The three claims above match the figure and use the 16,750 EMA results.
- Mobile layout stacks the three panels in Better / Similar / Worse order.
- Images have concise alt text and no essential conclusion is color-only.
- There is one comparison figure, one short paragraph and no duplicated gallery.
- Model and license links resolve before production deploy.
