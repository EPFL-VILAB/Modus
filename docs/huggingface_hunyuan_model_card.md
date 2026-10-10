---
license: other
license_name: tencent-hunyuan-community-license
license_link: https://huggingface.co/EPFL-VILAB/MODUS-Hunyuan/blob/main/LICENSE_HUNYUAN
base_model:
  - tencent/HunyuanImage-3.0-Instruct
tags:
  - modus
  - hunyuan
  - multimodal
  - any-to-any
  - image-to-image
---

# MODUS-Hunyuan

MODUS-Hunyuan applies the MODUS any-to-any adaptation recipe to the
HunyuanImage-3.0 backbone. It supports independent prediction and chained
generation across the same 16-modality interface as MODUS-BAGEL: text,
caption, RGB, depth, normals, detection, segmentation, edges, and global or
local visual features.

This release contains the EMA weights from 16,750 completed optimizer steps.
It is exported as a self-contained BF16 inference checkpoint with its
tokenizer, VAE, modality registry, MODUS adapter, and 40 weight shards. It does
not contain optimizer, scheduler, data-loader, or training state.

## Use

Inference code and examples are provided in the
[MODUS repository](https://github.com/EPFL-VILAB/Modus). The checkpoint is
approximately 158 GiB and the released inference path was validated on four
GPUs.

```bash
hf download EPFL-VILAB/MODUS-Hunyuan --local-dir models/modus-hunyuan

python infer.py --backend hunyuan --condition rgb --target depth \
  checkpoint_path=models/modus-hunyuan \
  input_image=test_images/01_basil_cathedral.jpg
```

## Verification

The export was checked bitwise against the source EMA checkpoint: 5,198 model
tensors, including 5 MODUS adapter tensors and 280 VAE tensors, passed. The
three inference examples documented in the MODUS README were also executed
successfully against this export.

## License and restrictions

The model is a derivative of HunyuanImage-3.0 and is distributed under the
[Tencent Hunyuan Community License](LICENSE_HUNYUAN). That license includes
territorial and use restrictions; read it before downloading, using, or
redistributing the model. In particular, the licensed Territory excludes the
European Union, United Kingdom, and South Korea.

This model is provided by the EPFL Visual Intelligence and Learning Lab.
Tencent is not affiliated with, associated with, sponsoring, or endorsing
MODUS or this release. See [`NOTICE`](NOTICE) for the required attribution.

## Citation

```bibtex
@article{ye2026modus,
  title   = {MODUS: Decoder-only Any-to-Any Modeling of Diverse Modalities},
  author  = {Ye, Mingqiao and An, Zhaochong and Gao, Zhitong and Liu, Xian
             and Fleuret, Fran\c{c}ois and Li, Chuan and Zadeh, Amir
             and Belongie, Serge and Dehghan, Afshin and Allardice, Jesse
             and Mizrahi, David and Kar, O\u{g}uzhan Fatih and Bachmann, Roman
             and Zamir, Amir},
  journal = {arXiv preprint arXiv:2607.25948},
  year    = {2026},
}
```
