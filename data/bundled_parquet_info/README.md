# bundled_parquet_info — repo-tracked parquet metadata (auto-fallback)

Repo-tracked copies of the parquet-info JSONs used by BAGEL training.
**The code reads this directory automatically**: `data/dataset_base.py`
falls back to `data/bundled_parquet_info/<basename>` whenever the
configured live path (`./datasets/blip3o/parquet_info/…`, a symlink chain
that only exists on the original cluster) is missing. When the live path
exists it still wins — behavior on the original cluster is unchanged.

## Files

| File | Files listed | Rows | Used by |
|---|---|---|---|
| `…_grounding_canny_dino_global_clip448_imagebind.json` | 2891 (1000 `sa_*` + 1891 `webdataset_*`) | 29.2M | 13mod: all target groups except grounding (any2rgb / depth / normal / caption / canny / dino / dinolocal / clip / imagebind / imagebindlocal) |
| `…_grounding2_canny_dino_global_clip448_imagebind.json` | 1000 (`sa_*` only) | 11.1M | 13mod: `unified_any2grounding` only |
| `…_clip448_imagebind_samseg_samedge_cocodet.json` | — | — | 16mod runs (adds cocodet / samseg / samedge) |
| `llava_onevision_vqa.json` | — | — | VLM-SFT (llava-onevision) |

**Why grounding has two JSONs:** grounding annotations (GLaMM phrase+bbox,
matched by UID) only exist for the SA-1B portion of the parquet set. The
`webdataset_*` shards have no `grounding` column at all (verified via
parquet footer schema). So when grounding is the generation TARGET, the
loader must restrict itself to the `sa_*` files — that restricted file
list is the `grounding2` JSON. Both JSONs point into the SAME `data_dir`.

## Setting up on a new machine

1. Get the parquet data itself (≈ several TB, NOT in this repo):
   `parquet_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global_clip448_imagebind/`
   (2891 parquet files).
2. Place (or symlink) it at
   `datasets/blip3o/parquet_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global_clip448_imagebind/`
   relative to the repo root.
3. parquet-info JSONs: **no action needed** — the loader automatically
   falls back to this directory when the live
   `datasets/blip3o/parquet_info/` path is absent
   (`data/dataset_base.py`, "Fall back to the copy bundled in the repo").
4. Note the JSON keys are relative paths
   (`./datasets/blip3o/parquet_.../sa_000000.parquet`), so always launch
   training from the repo root.

If the parquet set ever changes, regenerate with
`data/any2any_preprocess/generate_parquet_json.py` and refresh these copies.
