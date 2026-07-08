# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from .vlm_dataset import SftJSONLIterableDataset, SftParquetIterableDataset
from .interleave_datasets.any2any_dataset import UnifiedAny2AnyIterableDataset
import torch
import os


Any2AnyDatasetClass = UnifiedAny2AnyIterableDataset

DATASET_REGISTRY = {
    'vlm_sft': SftJSONLIterableDataset,
    'vlm_sft_parquet': SftParquetIterableDataset,
    'unified_any2rgb': Any2AnyDatasetClass,
    'unified_any2depth': Any2AnyDatasetClass,
    'unified_any2normal': Any2AnyDatasetClass,
    'unified_any2caption': Any2AnyDatasetClass,
    'unified_any2det': Any2AnyDatasetClass,
    'unified_any2seg': Any2AnyDatasetClass,
    'unified_any2grounding': Any2AnyDatasetClass,
    'unified_any2canny': Any2AnyDatasetClass,
    'unified_any2dino': Any2AnyDatasetClass,
    'unified_any2dinolocal': Any2AnyDatasetClass,
    'unified_any2clip': Any2AnyDatasetClass,
    'unified_any2imagebind': Any2AnyDatasetClass,
    'unified_any2imagebindlocal': Any2AnyDatasetClass,
    'unified_any2cocodet': Any2AnyDatasetClass,
    'unified_any2samseg': Any2AnyDatasetClass,
    'unified_any2samedge': Any2AnyDatasetClass,
}


DATASET_INFO = {
    'vlm_sft': {
        'text_only_benchmark': {
            'data_dir': './datasets/text_only_benchmark/images',
            'jsonl_path': './datasets/text_only_benchmark/label/labels.jsonl',
        },
        'ai2d(cauldron,llava_format)': {
			'data_dir': './datasets/llava_onevision_vqa/ai2d(cauldron,llava_format)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/ai2d(cauldron,llava_format)/label/labels.jsonl',
		},
        'ai2d(gpt4v)': {
			'data_dir': './datasets/llava_onevision_vqa/ai2d(gpt4v)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/ai2d(gpt4v)/label/labels.jsonl',
		},
        'ai2d(internvl)': {
			'data_dir': './datasets/llava_onevision_vqa/ai2d(internvl)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/ai2d(internvl)/label/labels.jsonl',
		},
        'allava_instruct_laion4v': {
			'data_dir': './datasets/llava_onevision_vqa/allava_instruct_laion4v/images',
			'jsonl_path': './datasets/llava_onevision_vqa/allava_instruct_laion4v/label/labels.jsonl',
		},
        'allava_instruct_vflan4v': {
			'data_dir': './datasets/llava_onevision_vqa/allava_instruct_vflan4v/images',
			'jsonl_path': './datasets/llava_onevision_vqa/allava_instruct_vflan4v/label/labels.jsonl',
		},
        'aokvqa(cauldron,llava_format)': {
			'data_dir': './datasets/llava_onevision_vqa/aokvqa(cauldron,llava_format)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/aokvqa(cauldron,llava_format)/label/labels.jsonl',
		},
        'chart2text(cauldron)': {
			'data_dir': './datasets/llava_onevision_vqa/chart2text(cauldron)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/chart2text(cauldron)/label/labels.jsonl',
		},
        'chartqa(cauldron,llava_format)': {
			'data_dir': './datasets/llava_onevision_vqa/chartqa(cauldron,llava_format)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/chartqa(cauldron,llava_format)/label/labels.jsonl',
		},
        'chrome_writting': {
			'data_dir': './datasets/llava_onevision_vqa/chrome_writting/images',
			'jsonl_path': './datasets/llava_onevision_vqa/chrome_writting/label/labels.jsonl',
		},
        'CLEVR-Math(MathV360K)': {
			'data_dir': './datasets/llava_onevision_vqa/CLEVR-Math(MathV360K)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/CLEVR-Math(MathV360K)/label/labels.jsonl',
		},
        'clevr(cauldron,llava_format)': {
			'data_dir': './datasets/llava_onevision_vqa/clevr(cauldron,llava_format)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/clevr(cauldron,llava_format)/label/labels.jsonl',
		},
        'diagram_image_to_text(cauldron)': {
			'data_dir': './datasets/llava_onevision_vqa/diagram_image_to_text(cauldron)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/diagram_image_to_text(cauldron)/label/labels.jsonl',
		},
        'dvqa(cauldron,llava_format)': {
			'data_dir': './datasets/llava_onevision_vqa/dvqa(cauldron,llava_format)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/dvqa(cauldron,llava_format)/label/labels.jsonl',
		},
        'Evol-Instruct-GPT4-Turbo': {
			'data_dir': './datasets/llava_onevision_vqa/Evol-Instruct-GPT4-Turbo/images',
			'jsonl_path': './datasets/llava_onevision_vqa/Evol-Instruct-GPT4-Turbo/label/labels.jsonl',
		},
        'figureqa(cauldron,llava_format)': {
			'data_dir': './datasets/llava_onevision_vqa/figureqa(cauldron,llava_format)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/figureqa(cauldron,llava_format)/label/labels.jsonl',
		},
        'FigureQA(MathV360K)': {
			'data_dir': './datasets/llava_onevision_vqa/FigureQA(MathV360K)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/FigureQA(MathV360K)/label/labels.jsonl',
		},
        'geo170k(align)': {
			'data_dir': './datasets/llava_onevision_vqa/geo170k(align)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/geo170k(align)/label/labels.jsonl',
		},
        'geo170k(qa)': {
			'data_dir': './datasets/llava_onevision_vqa/geo170k(qa)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/geo170k(qa)/label/labels.jsonl',
		},
        'geo3k': {
			'data_dir': './datasets/llava_onevision_vqa/geo3k/images',
			'jsonl_path': './datasets/llava_onevision_vqa/geo3k/label/labels.jsonl',
		},
        'Geometry3K(MathV360K)': {
			'data_dir': './datasets/llava_onevision_vqa/Geometry3K(MathV360K)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/Geometry3K(MathV360K)/label/labels.jsonl',
		},
        'geomverse(cauldron)': {
			'data_dir': './datasets/llava_onevision_vqa/geomverse(cauldron)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/geomverse(cauldron)/label/labels.jsonl',
		},
        'GeoQA+(MathV360K)': {
			'data_dir': './datasets/llava_onevision_vqa/GeoQA+(MathV360K)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/GeoQA+(MathV360K)/label/labels.jsonl',
		},
        'GEOS(MathV360K)': {
			'data_dir': './datasets/llava_onevision_vqa/GEOS(MathV360K)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/GEOS(MathV360K)/label/labels.jsonl',
		},
        'hateful_memes(cauldron,llava_format)': {
			'data_dir': './datasets/llava_onevision_vqa/hateful_memes(cauldron,llava_format)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/hateful_memes(cauldron,llava_format)/label/labels.jsonl',
		},
        'hitab(cauldron,llava_format)': {
			'data_dir': './datasets/llava_onevision_vqa/hitab(cauldron,llava_format)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/hitab(cauldron,llava_format)/label/labels.jsonl',
		},
        'hme100k': {
			'data_dir': './datasets/llava_onevision_vqa/hme100k/images',
			'jsonl_path': './datasets/llava_onevision_vqa/hme100k/label/labels.jsonl',
		},
        'iam(cauldron)': {
			'data_dir': './datasets/llava_onevision_vqa/iam(cauldron)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/iam(cauldron)/label/labels.jsonl',
		},
        'iconqa(cauldron,llava_format)': {
			'data_dir': './datasets/llava_onevision_vqa/iconqa(cauldron,llava_format)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/iconqa(cauldron,llava_format)/label/labels.jsonl',
		},
        'IconQA(MathV360K)': {
			'data_dir': './datasets/llava_onevision_vqa/IconQA(MathV360K)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/IconQA(MathV360K)/label/labels.jsonl',
		},
        'iiit5k': {
			'data_dir': './datasets/llava_onevision_vqa/iiit5k/images',
			'jsonl_path': './datasets/llava_onevision_vqa/iiit5k/label/labels.jsonl',
		},
        'image_textualization(filtered)': {
			'data_dir': './datasets/llava_onevision_vqa/image_textualization(filtered)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/image_textualization(filtered)/label/labels.jsonl',
		},
        'infographic_vqa': {
			'data_dir': './datasets/llava_onevision_vqa/infographic_vqa/images',
			'jsonl_path': './datasets/llava_onevision_vqa/infographic_vqa/label/labels.jsonl',
		},
        'infographic_vqa_llava_format': {
			'data_dir': './datasets/llava_onevision_vqa/infographic_vqa_llava_format/images',
			'jsonl_path': './datasets/llava_onevision_vqa/infographic_vqa_llava_format/label/labels.jsonl',
		},
        'infographic(gpt4v)': {
			'data_dir': './datasets/llava_onevision_vqa/infographic(gpt4v)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/infographic(gpt4v)/label/labels.jsonl',
		},
        'intergps(cauldron,llava_format)': {
			'data_dir': './datasets/llava_onevision_vqa/intergps(cauldron,llava_format)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/intergps(cauldron,llava_format)/label/labels.jsonl',
		},
        'k12_printing': {
			'data_dir': './datasets/llava_onevision_vqa/k12_printing/images',
			'jsonl_path': './datasets/llava_onevision_vqa/k12_printing/label/labels.jsonl',
		},
        'llava_wild_4v_12k_filtered': {
			'data_dir': './datasets/llava_onevision_vqa/llava_wild_4v_12k_filtered/images',
			'jsonl_path': './datasets/llava_onevision_vqa/llava_wild_4v_12k_filtered/label/labels.jsonl',
		},
        'llava_wild_4v_39k_filtered': {
			'data_dir': './datasets/llava_onevision_vqa/llava_wild_4v_39k_filtered/images',
			'jsonl_path': './datasets/llava_onevision_vqa/llava_wild_4v_39k_filtered/label/labels.jsonl',
		},
        'llavar_gpt4_20k': {
			'data_dir': './datasets/llava_onevision_vqa/llavar_gpt4_20k/images',
			'jsonl_path': './datasets/llava_onevision_vqa/llavar_gpt4_20k/label/labels.jsonl',
		},
        'lrv_chart': {
			'data_dir': './datasets/llava_onevision_vqa/lrv_chart/images',
			'jsonl_path': './datasets/llava_onevision_vqa/lrv_chart/label/labels.jsonl',
		},
        'lrv_normal(filtered)': {
			'data_dir': './datasets/llava_onevision_vqa/lrv_normal(filtered)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/lrv_normal(filtered)/label/labels.jsonl',
		},
        'magpie_pro(l3_80b_mt)': {
			'data_dir': './datasets/llava_onevision_vqa/magpie_pro(l3_80b_mt)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/magpie_pro(l3_80b_mt)/label/labels.jsonl',
		},
        'magpie_pro(l3_80b_st)': {
			'data_dir': './datasets/llava_onevision_vqa/magpie_pro(l3_80b_st)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/magpie_pro(l3_80b_st)/label/labels.jsonl',
		},
        'magpie_pro(qwen2_72b_st)': {
			'data_dir': './datasets/llava_onevision_vqa/magpie_pro(qwen2_72b_st)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/magpie_pro(qwen2_72b_st)/label/labels.jsonl',
		},
        'mapqa(cauldron,llava_format)': {
			'data_dir': './datasets/llava_onevision_vqa/mapqa(cauldron,llava_format)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/mapqa(cauldron,llava_format)/label/labels.jsonl',
		},
        'MapQA(MathV360K)': {
			'data_dir': './datasets/llava_onevision_vqa/MapQA(MathV360K)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/MapQA(MathV360K)/label/labels.jsonl',
		},
        'mathqa': {
			'data_dir': './datasets/llava_onevision_vqa/mathqa/images',
			'jsonl_path': './datasets/llava_onevision_vqa/mathqa/label/labels.jsonl',
		},
        'MathV360K_TQA': {
			'data_dir': './datasets/llava_onevision_vqa/MathV360K_TQA/images',
			'jsonl_path': './datasets/llava_onevision_vqa/MathV360K_TQA/label/labels.jsonl',
		},
        'MathV360K_VQA-AS': {
			'data_dir': './datasets/llava_onevision_vqa/MathV360K_VQA-AS/images',
			'jsonl_path': './datasets/llava_onevision_vqa/MathV360K_VQA-AS/label/labels.jsonl',
		},
        'MathV360K_VQA-RAD': {
			'data_dir': './datasets/llava_onevision_vqa/MathV360K_VQA-RAD/images',
			'jsonl_path': './datasets/llava_onevision_vqa/MathV360K_VQA-RAD/label/labels.jsonl',
		},
        'mavis_math_metagen': {
			'data_dir': './datasets/llava_onevision_vqa/mavis_math_metagen/images',
			'jsonl_path': './datasets/llava_onevision_vqa/mavis_math_metagen/label/labels.jsonl',
		},
        'mavis_math_rule_geo': {
			'data_dir': './datasets/llava_onevision_vqa/mavis_math_rule_geo/images',
			'jsonl_path': './datasets/llava_onevision_vqa/mavis_math_rule_geo/label/labels.jsonl',
		},
        'multihiertt(cauldron)': {
			'data_dir': './datasets/llava_onevision_vqa/multihiertt(cauldron)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/multihiertt(cauldron)/label/labels.jsonl',
		},
        'orand_car_a': {
			'data_dir': './datasets/llava_onevision_vqa/orand_car_a/images',
			'jsonl_path': './datasets/llava_onevision_vqa/orand_car_a/label/labels.jsonl',
		},
        'PMC-VQA(MathV360K)': {
			'data_dir': './datasets/llava_onevision_vqa/PMC-VQA(MathV360K)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/PMC-VQA(MathV360K)/label/labels.jsonl',
		},
        'raven(cauldron)': {
			'data_dir': './datasets/llava_onevision_vqa/raven(cauldron)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/raven(cauldron)/label/labels.jsonl',
		},
        'rendered_text(cauldron)': {
			'data_dir': './datasets/llava_onevision_vqa/rendered_text(cauldron)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/rendered_text(cauldron)/label/labels.jsonl',
		},
        'robut_sqa(cauldron)': {
			'data_dir': './datasets/llava_onevision_vqa/robut_sqa(cauldron)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/robut_sqa(cauldron)/label/labels.jsonl',
		},
        'robut_wikisql(cauldron)': {
			'data_dir': './datasets/llava_onevision_vqa/robut_wikisql(cauldron)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/robut_wikisql(cauldron)/label/labels.jsonl',
		},
        'robut_wtq(cauldron,llava_format)': {
			'data_dir': './datasets/llava_onevision_vqa/robut_wtq(cauldron,llava_format)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/robut_wtq(cauldron,llava_format)/label/labels.jsonl',
		},
        'scienceqa(cauldron,llava_format)': {
			'data_dir': './datasets/llava_onevision_vqa/scienceqa(cauldron,llava_format)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/scienceqa(cauldron,llava_format)/label/labels.jsonl',
		},
        'scienceqa(nona_context)': {
			'data_dir': './datasets/llava_onevision_vqa/scienceqa(nona_context)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/scienceqa(nona_context)/label/labels.jsonl',
		},
        'screen2words(cauldron)': {
			'data_dir': './datasets/llava_onevision_vqa/screen2words(cauldron)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/screen2words(cauldron)/label/labels.jsonl',
		},
        'sharegpt4o': {
			'data_dir': './datasets/llava_onevision_vqa/sharegpt4o/images',
			'jsonl_path': './datasets/llava_onevision_vqa/sharegpt4o/label/labels.jsonl',
		},
        'sharegpt4v(coco)': {
			'data_dir': './datasets/llava_onevision_vqa/sharegpt4v(coco)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/sharegpt4v(coco)/label/labels.jsonl',
		},
        'sharegpt4v(knowledge)': {
			'data_dir': './datasets/llava_onevision_vqa/sharegpt4v(knowledge)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/sharegpt4v(knowledge)/label/labels.jsonl',
		},
        'sharegpt4v(llava)': {
			'data_dir': './datasets/llava_onevision_vqa/sharegpt4v(llava)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/sharegpt4v(llava)/label/labels.jsonl',
		},
        'sharegpt4v(sam)': {
			'data_dir': './datasets/llava_onevision_vqa/sharegpt4v(sam)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/sharegpt4v(sam)/label/labels.jsonl',
		},
        'sroie': {
			'data_dir': './datasets/llava_onevision_vqa/sroie/images',
			'jsonl_path': './datasets/llava_onevision_vqa/sroie/label/labels.jsonl',
		},
        'st_vqa(cauldron,llava_format)': {
			'data_dir': './datasets/llava_onevision_vqa/st_vqa(cauldron,llava_format)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/st_vqa(cauldron,llava_format)/label/labels.jsonl',
		},
        'Super-CLEVR(MathV360K)': {
			'data_dir': './datasets/llava_onevision_vqa/Super-CLEVR(MathV360K)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/Super-CLEVR(MathV360K)/label/labels.jsonl',
		},
        'tabmwp(cauldron)': {
			'data_dir': './datasets/llava_onevision_vqa/tabmwp(cauldron)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/tabmwp(cauldron)/label/labels.jsonl',
		},
        'TabMWP(MathV360K)': {
			'data_dir': './datasets/llava_onevision_vqa/TabMWP(MathV360K)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/TabMWP(MathV360K)/label/labels.jsonl',
		},
        'tallyqa(cauldron,llava_format)': {
			'data_dir': './datasets/llava_onevision_vqa/tallyqa(cauldron,llava_format)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/tallyqa(cauldron,llava_format)/label/labels.jsonl',
		},
        'textcaps': {
			'data_dir': './datasets/llava_onevision_vqa/textcaps/images',
			'jsonl_path': './datasets/llava_onevision_vqa/textcaps/label/labels.jsonl',
		},
        'textocr(gpt4v)': {
			'data_dir': './datasets/llava_onevision_vqa/textocr(gpt4v)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/textocr(gpt4v)/label/labels.jsonl',
		},
        'tqa(cauldron,llava_format)': {
			'data_dir': './datasets/llava_onevision_vqa/tqa(cauldron,llava_format)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/tqa(cauldron,llava_format)/label/labels.jsonl',
		},
        'UniGeo(MathV360K)': {
			'data_dir': './datasets/llava_onevision_vqa/UniGeo(MathV360K)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/UniGeo(MathV360K)/label/labels.jsonl',
		},
        'ureader_cap': {
			'data_dir': './datasets/llava_onevision_vqa/ureader_cap/images',
			'jsonl_path': './datasets/llava_onevision_vqa/ureader_cap/label/labels.jsonl',
		},
        'ureader_ie': {
			'data_dir': './datasets/llava_onevision_vqa/ureader_ie/images',
			'jsonl_path': './datasets/llava_onevision_vqa/ureader_ie/label/labels.jsonl',
		},
        'vision_flan(filtered)': {
			'data_dir': './datasets/llava_onevision_vqa/vision_flan(filtered)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/vision_flan(filtered)/label/labels.jsonl',
		},
        'vistext(cauldron)': {
			'data_dir': './datasets/llava_onevision_vqa/vistext(cauldron)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/vistext(cauldron)/label/labels.jsonl',
		},
        'visual7w(cauldron,llava_format)': {
			'data_dir': './datasets/llava_onevision_vqa/visual7w(cauldron,llava_format)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/visual7w(cauldron,llava_format)/label/labels.jsonl',
		},
        'visualmrc(cauldron)': {
			'data_dir': './datasets/llava_onevision_vqa/visualmrc(cauldron)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/visualmrc(cauldron)/label/labels.jsonl',
		},
        'VisualWebInstruct(filtered)': {
			'data_dir': './datasets/llava_onevision_vqa/VisualWebInstruct(filtered)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/VisualWebInstruct(filtered)/label/labels.jsonl',
		},
        'VizWiz(MathV360K)': {
			'data_dir': './datasets/llava_onevision_vqa/VizWiz(MathV360K)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/VizWiz(MathV360K)/label/labels.jsonl',
		},
        'vqarad(cauldron,llava_format)': {
			'data_dir': './datasets/llava_onevision_vqa/vqarad(cauldron,llava_format)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/vqarad(cauldron,llava_format)/label/labels.jsonl',
		},
        'vsr(cauldron,llava_format)': {
			'data_dir': './datasets/llava_onevision_vqa/vsr(cauldron,llava_format)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/vsr(cauldron,llava_format)/label/labels.jsonl',
		},
        'websight(cauldron)': {
			'data_dir': './datasets/llava_onevision_vqa/websight(cauldron)/images',
			'jsonl_path': './datasets/llava_onevision_vqa/websight(cauldron)/label/labels.jsonl',
		},
    },
    # vlm_sft bundled into parquet shards (streaming-friendly, S3-ready).
    # Written by data/any2any_preprocess/generate_parquet_vlm_sft.py; sample
    # selection (shuffle_lines + num_used_data) is baked in at conversion.
    'vlm_sft_parquet': {
        'llava_onevision_vqa': {
            'data_dir': './datasets/llava_onevision_vqa_parquet/train',
            'parquet_info_path': './datasets/llava_onevision_vqa_parquet/parquet_info/llava_onevision_vqa.json',
        },
    },
    'unified_any2rgb': {
        'blip3o_example_toydataset': {
            'data_dir': './datasets/blip3o/parquet_toy_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global/train',
            'num_files': 1,
            'num_total_samples': 1000,
            'parquet_info_path': './datasets/blip3o/parquet_toy_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global/train/blip3o_example_toydataset.json',
        },
        'blip3o_example_toydataset_13modality_clip448_imagebind': {
            'data_dir': './datasets/blip3o/parquet_toy_13modality_clip448_imagebind/train',
            'num_files': 1,
            'num_total_samples': 1000,
            'parquet_info_path': './datasets/blip3o/parquet_toy_13modality_clip448_imagebind/train/blip3o_example_toydataset.json',
        },
        'blip3o':{
            'data_dir': './datasets/blip3o/parquet',
            'num_files': 2891,
            'num_total_samples': 25_000_000,
            'parquet_info_path': './datasets/blip3o/parquet_info/blip3o.json',
        },
        'blip3o_rgb_caption_depth_normal_det_seg':{
            'data_dir': './datasets/blip3o/parquet_rgb_caption_depth_normal_det_seg',
            'num_files': 2891,
            'num_total_samples': 25_000_000,
            'parquet_info_path': './datasets/blip3o/parquet_info/blip3o_rgb_caption_depth_normal_det_seg.json',
        },
        'blip3o_rgb_caption_depth_normal_det_seg_grounding':{
            'data_dir': './datasets/blip3o/parquet_rgb_caption_depth_normal_det_seg_grounding',
            'num_files': 2891,
            'num_total_samples': 25_000_000,
            'parquet_info_path': './datasets/blip3o/parquet_info/blip3o_rgb_caption_depth_normal_det_seg_grounding.json',
        },
        'blip3o_rgb_caption_depth_normal_det_seg_grounding_canny_dino':{
            'data_dir': './datasets/blip3o/parquet_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global',
            'num_files': 2891,
            'num_total_samples': 25_000_000,
            'parquet_info_path': './datasets/blip3o/parquet_info/blip3o_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global.json',
        },
        'blip3o_rgb_caption_depth_normal_det_seg_grounding_canny_dino_clip448_imagebind':{
            'data_dir': './datasets/blip3o/parquet_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global_clip448_imagebind',
            'num_files': 2891,
            'num_total_samples': 25_000_000,
            'parquet_info_path': './datasets/blip3o/parquet_info/blip3o_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global_clip448_imagebind.json',
        },
        'blip3o_rgb_caption_depth_normal_det_seg_grounding_canny_dino_clip448_imagebind_aligned':{
            'data_dir': './datasets/blip3o/parquet_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global_clip448_imagebind_aligned',
            'num_files': 1000,
            'num_total_samples': 3_899_886,
            'parquet_info_path': './datasets/blip3o/parquet_info/parquet_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global_clip448_imagebind_aligned.json',
        }
    },
    'unified_any2depth': {
        'blip3o_example_toydataset': {
            'data_dir': './datasets/blip3o/parquet_toy_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global/train',
            'num_files': 1,
            'num_total_samples': 1000,
            'parquet_info_path': './datasets/blip3o/parquet_toy_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global/train/blip3o_example_toydataset.json',
        },
        'blip3o':{
            'data_dir': './datasets/blip3o/parquet',
            'num_files': 2891,
            'num_total_samples': 25_000_000,
            'parquet_info_path': './datasets/blip3o/parquet_info/blip3o.json',
        },
        'blip3o_rgb_caption_depth_normal_det_seg':{
            'data_dir': './datasets/blip3o/parquet_rgb_caption_depth_normal_det_seg',
            'num_files': 2891,
            'num_total_samples': 25_000_000,
            'parquet_info_path': './datasets/blip3o/parquet_info/blip3o_rgb_caption_depth_normal_det_seg.json',
        },
        'blip3o_rgb_caption_depth_normal_det_seg_grounding':{
            'data_dir': './datasets/blip3o/parquet_rgb_caption_depth_normal_det_seg_grounding',
            'num_files': 2891,
            'num_total_samples': 25_000_000,
            'parquet_info_path': './datasets/blip3o/parquet_info/blip3o_rgb_caption_depth_normal_det_seg_grounding.json',
        },
        'blip3o_rgb_caption_depth_normal_det_seg_grounding_canny_dino':{
            'data_dir': './datasets/blip3o/parquet_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global',
            'num_files': 2891,
            'num_total_samples': 25_000_000,
            'parquet_info_path': './datasets/blip3o/parquet_info/blip3o_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global.json',
        },
        'blip3o_rgb_caption_depth_normal_det_seg_grounding_canny_dino_clip448_imagebind':{
            'data_dir': './datasets/blip3o/parquet_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global_clip448_imagebind',
            'num_files': 2891,
            'num_total_samples': 25_000_000,
            'parquet_info_path': './datasets/blip3o/parquet_info/blip3o_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global_clip448_imagebind.json',
        },
        'blip3o_rgb_caption_depth_normal_det_seg_grounding_canny_dino_clip448_imagebind_aligned':{
            'data_dir': './datasets/blip3o/parquet_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global_clip448_imagebind_aligned',
            'num_files': 1000,
            'num_total_samples': 3_899_886,
            'parquet_info_path': './datasets/blip3o/parquet_info/parquet_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global_clip448_imagebind_aligned.json',
        }
    },
    'unified_any2normal': {
        'blip3o_example_toydataset': {
            'data_dir': './datasets/blip3o/parquet_toy_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global/train',
            'num_files': 1,
            'num_total_samples': 1000,
            'parquet_info_path': './datasets/blip3o/parquet_toy_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global/train/blip3o_example_toydataset.json',
        },
        'blip3o':{
            'data_dir': './datasets/blip3o/parquet',
            'num_files': 2891,
            'num_total_samples': 25_000_000,
            'parquet_info_path': './datasets/blip3o/parquet_info/blip3o.json',
        },
        'blip3o_rgb_caption_depth_normal_det_seg':{
            'data_dir': './datasets/blip3o/parquet_rgb_caption_depth_normal_det_seg',
            'num_files': 2891,
            'num_total_samples': 25_000_000,
            'parquet_info_path': './datasets/blip3o/parquet_info/blip3o_rgb_caption_depth_normal_det_seg.json',
        },
        'blip3o_rgb_caption_depth_normal_det_seg_grounding':{
            'data_dir': './datasets/blip3o/parquet_rgb_caption_depth_normal_det_seg_grounding',
            'num_files': 2891,
            'num_total_samples': 25_000_000,
            'parquet_info_path': './datasets/blip3o/parquet_info/blip3o_rgb_caption_depth_normal_det_seg_grounding.json',
        },
        'blip3o_rgb_caption_depth_normal_det_seg_grounding_canny_dino':{
            'data_dir': './datasets/blip3o/parquet_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global',
            'num_files': 2891,
            'num_total_samples': 25_000_000,
            'parquet_info_path': './datasets/blip3o/parquet_info/blip3o_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global.json',
        },
        'blip3o_rgb_caption_depth_normal_det_seg_grounding_canny_dino_clip448_imagebind':{
            'data_dir': './datasets/blip3o/parquet_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global_clip448_imagebind',
            'num_files': 2891,
            'num_total_samples': 25_000_000,
            'parquet_info_path': './datasets/blip3o/parquet_info/blip3o_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global_clip448_imagebind.json',
        },
        'blip3o_rgb_caption_depth_normal_det_seg_grounding_canny_dino_clip448_imagebind_aligned':{
            'data_dir': './datasets/blip3o/parquet_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global_clip448_imagebind_aligned',
            'num_files': 1000,
            'num_total_samples': 3_899_886,
            'parquet_info_path': './datasets/blip3o/parquet_info/parquet_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global_clip448_imagebind_aligned.json',
        }
    },
    'unified_any2caption': {
        'blip3o_example_toydataset': {
            'data_dir': './datasets/blip3o/parquet_toy_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global/train',
            'num_files': 1,
            'num_total_samples': 1000,
            'parquet_info_path': './datasets/blip3o/parquet_toy_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global/train/blip3o_example_toydataset.json',
        },
        'blip3o_example_toydataset_13modality_clip448_imagebind': {
            'data_dir': './datasets/blip3o/parquet_toy_13modality_clip448_imagebind/train',
            'num_files': 1,
            'num_total_samples': 1000,
            'parquet_info_path': './datasets/blip3o/parquet_toy_13modality_clip448_imagebind/train/blip3o_example_toydataset.json',
        },
        'blip3o':{
            'data_dir': './datasets/blip3o/parquet',
            'num_files': 2891,
            'num_total_samples': 25_000_000,
            'parquet_info_path': './datasets/blip3o/parquet_info/blip3o.json',
        },
        'blip3o_rgb_caption_depth_normal_det_seg':{
            'data_dir': './datasets/blip3o/parquet_rgb_caption_depth_normal_det_seg',
            'num_files': 2891,
            'num_total_samples': 25_000_000,
            'parquet_info_path': './datasets/blip3o/parquet_info/blip3o_rgb_caption_depth_normal_det_seg.json',
        },
        'blip3o_rgb_caption_depth_normal_det_seg_grounding':{
            'data_dir': './datasets/blip3o/parquet_rgb_caption_depth_normal_det_seg_grounding',
            'num_files': 2891,
            'num_total_samples': 25_000_000,
            'parquet_info_path': './datasets/blip3o/parquet_info/blip3o_rgb_caption_depth_normal_det_seg_grounding.json',
        },
        'blip3o_rgb_caption_depth_normal_det_seg_grounding_canny_dino':{
            'data_dir': './datasets/blip3o/parquet_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global',
            'num_files': 2891,
            'num_total_samples': 25_000_000,
            'parquet_info_path': './datasets/blip3o/parquet_info/blip3o_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global.json',
        },
        'blip3o_rgb_caption_depth_normal_det_seg_grounding_canny_dino_clip448_imagebind':{
            'data_dir': './datasets/blip3o/parquet_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global_clip448_imagebind',
            'num_files': 2891,
            'num_total_samples': 25_000_000,
            'parquet_info_path': './datasets/blip3o/parquet_info/blip3o_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global_clip448_imagebind.json',
        },
        'blip3o_rgb_caption_depth_normal_det_seg_grounding_canny_dino_clip448_imagebind_aligned':{
            'data_dir': './datasets/blip3o/parquet_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global_clip448_imagebind_aligned',
            'num_files': 1000,
            'num_total_samples': 3_899_886,
            'parquet_info_path': './datasets/blip3o/parquet_info/parquet_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global_clip448_imagebind_aligned.json',
        }
    },
    'unified_any2det': {
        'blip3o_rgb_caption_depth_normal_det_seg':{
            'data_dir': './datasets/blip3o/parquet_rgb_caption_depth_normal_det_seg',
            'num_files': 2891,
            'num_total_samples': 25_000_000,
            'parquet_info_path': './datasets/blip3o/parquet_info/blip3o_rgb_caption_depth_normal_det_seg.json',
        }
    },
    'unified_any2seg': {
        'blip3o_example_toydataset': {
            'data_dir': './datasets/blip3o/parquet_toy_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global/train',
            'num_files': 1,
            'num_total_samples': 1000,
            'parquet_info_path': './datasets/blip3o/parquet_toy_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global/train/blip3o_example_toydataset.json',
        },
        'blip3o_rgb_caption_depth_normal_det_seg':{
            'data_dir': './datasets/blip3o/parquet_rgb_caption_depth_normal_det_seg',
            'num_files': 2891,
            'num_total_samples': 25_000_000,
            'parquet_info_path': './datasets/blip3o/parquet_info/blip3o_rgb_caption_depth_normal_det_seg.json',
        },
        'blip3o_rgb_caption_depth_normal_det_seg_grounding':{
            'data_dir': './datasets/blip3o/parquet_rgb_caption_depth_normal_det_seg_grounding',
            'num_files': 2891,
            'num_total_samples': 25_000_000,
            'parquet_info_path': './datasets/blip3o/parquet_info/blip3o_rgb_caption_depth_normal_det_seg_grounding.json',
        },
        'blip3o_rgb_caption_depth_normal_det_seg_grounding_canny_dino':{
            'data_dir': './datasets/blip3o/parquet_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global',
            'num_files': 2891,
            'num_total_samples': 25_000_000,
            'parquet_info_path': './datasets/blip3o/parquet_info/blip3o_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global.json',
        },
        'blip3o_rgb_caption_depth_normal_det_seg_grounding_canny_dino_clip448_imagebind':{
            'data_dir': './datasets/blip3o/parquet_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global_clip448_imagebind',
            'num_files': 2891,
            'num_total_samples': 25_000_000,
            'parquet_info_path': './datasets/blip3o/parquet_info/blip3o_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global_clip448_imagebind.json',
        },
        'blip3o_rgb_caption_depth_normal_det_seg_grounding_canny_dino_clip448_imagebind_aligned':{
            'data_dir': './datasets/blip3o/parquet_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global_clip448_imagebind_aligned',
            'num_files': 1000,
            'num_total_samples': 3_899_886,
            'parquet_info_path': './datasets/blip3o/parquet_info/parquet_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global_clip448_imagebind_aligned.json',
        }
    },
    'unified_any2grounding': {
        'blip3o_example_toydataset': {
            'data_dir': './datasets/blip3o/parquet_toy_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global/train',
            'num_files': 1,
            'num_total_samples': 1000,
            'parquet_info_path': './datasets/blip3o/parquet_toy_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global/train/blip3o_example_toydataset.json',
        },
        'blip3o_example_toydataset_13modality_clip448_imagebind': {
            'data_dir': './datasets/blip3o/parquet_toy_13modality_clip448_imagebind/train',
            'num_files': 1,
            'num_total_samples': 1000,
            'parquet_info_path': './datasets/blip3o/parquet_toy_13modality_clip448_imagebind/train/blip3o_example_toydataset.json',
        },
        'blip3o_rgb_caption_depth_normal_det_seg_grounding':{
            'data_dir': './datasets/blip3o/parquet_rgb_caption_depth_normal_det_seg_grounding',
            'num_files': 1000,
            'num_total_samples': 11_000_000,
            'parquet_info_path': './datasets/blip3o/parquet_info/blip3o_rgb_caption_depth_normal_det_seg_grounding2.json',
        },
        'blip3o_rgb_caption_depth_normal_det_seg_grounding_canny_dino':{
            'data_dir': './datasets/blip3o/parquet_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global',
            'num_files': 1000,
            'num_total_samples': 11_000_000,
            'parquet_info_path': './datasets/blip3o/parquet_info/blip3o_rgb_caption_depth_normal_det_seg_grounding2_canny_dino_global.json',
        },
        'blip3o_rgb_caption_depth_normal_det_seg_grounding_canny_dino_clip448_imagebind':{
            'data_dir': './datasets/blip3o/parquet_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global_clip448_imagebind',
            'num_files': 1000,
            'num_total_samples': 11_000_000,
            'parquet_info_path': './datasets/blip3o/parquet_info/blip3o_rgb_caption_depth_normal_det_seg_grounding2_canny_dino_global_clip448_imagebind.json',
        },
        'blip3o_rgb_caption_depth_normal_det_seg_grounding_canny_dino_clip448_imagebind_aligned':{
            'data_dir': './datasets/blip3o/parquet_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global_clip448_imagebind_aligned',
            'num_files': 1000,
            'num_total_samples': 3_899_886,
            'parquet_info_path': './datasets/blip3o/parquet_info/parquet_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global_clip448_imagebind_aligned.json',
        }
    },
    'unified_any2canny': {
        'blip3o_example_toydataset': {
            'data_dir': './datasets/blip3o/parquet_toy_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global/train',
            'num_files': 1,
            'num_total_samples': 1000,
            'parquet_info_path': './datasets/blip3o/parquet_toy_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global/train/blip3o_example_toydataset.json',
        },
        'blip3o_rgb_caption_depth_normal_det_seg_grounding_canny_dino':{
            'data_dir': './datasets/blip3o/parquet_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global',
            'num_files': 2891,
            'num_total_samples': 25_000_000,
            'parquet_info_path': './datasets/blip3o/parquet_info/blip3o_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global.json',
        },
        'blip3o_rgb_caption_depth_normal_det_seg_grounding_canny_dino_clip448_imagebind':{
            'data_dir': './datasets/blip3o/parquet_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global_clip448_imagebind',
            'num_files': 2891,
            'num_total_samples': 25_000_000,
            'parquet_info_path': './datasets/blip3o/parquet_info/blip3o_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global_clip448_imagebind.json',
        },
        'blip3o_rgb_caption_depth_normal_det_seg_grounding_canny_dino_clip448_imagebind_aligned':{
            'data_dir': './datasets/blip3o/parquet_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global_clip448_imagebind_aligned',
            'num_files': 1000,
            'num_total_samples': 3_899_886,
            'parquet_info_path': './datasets/blip3o/parquet_info/parquet_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global_clip448_imagebind_aligned.json',
        }
    },
    'unified_any2dino': {
        'blip3o_example_toydataset': {
            'data_dir': './datasets/blip3o/parquet_toy_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global/train',
            'num_files': 1,
            'num_total_samples': 1000,
            'parquet_info_path': './datasets/blip3o/parquet_toy_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global/train/blip3o_example_toydataset.json',
        },
        'blip3o_example_toydataset_13modality_clip448_imagebind': {
            'data_dir': './datasets/blip3o/parquet_toy_13modality_clip448_imagebind/train',
            'num_files': 1,
            'num_total_samples': 1000,
            'parquet_info_path': './datasets/blip3o/parquet_toy_13modality_clip448_imagebind/train/blip3o_example_toydataset.json',
        },
        'blip3o_rgb_caption_depth_normal_det_seg_grounding_canny_dino':{
            'data_dir': './datasets/blip3o/parquet_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global',
            'num_files': 2891,
            'num_total_samples': 25_000_000,
            'parquet_info_path': './datasets/blip3o/parquet_info/blip3o_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global.json',
        },
        'blip3o_rgb_caption_depth_normal_det_seg_grounding_canny_dino_clip448_imagebind':{
            'data_dir': './datasets/blip3o/parquet_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global_clip448_imagebind',
            'num_files': 2891,
            'num_total_samples': 25_000_000,
            'parquet_info_path': './datasets/blip3o/parquet_info/blip3o_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global_clip448_imagebind.json',
        },
        'blip3o_rgb_caption_depth_normal_det_seg_grounding_canny_dino_clip448_imagebind_aligned':{
            'data_dir': './datasets/blip3o/parquet_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global_clip448_imagebind_aligned',
            'num_files': 1000,
            'num_total_samples': 3_899_886,
            'parquet_info_path': './datasets/blip3o/parquet_info/parquet_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global_clip448_imagebind_aligned.json',
        }
    },
    'unified_any2dinolocal': {
        'blip3o_example_toydataset_13modality_clip448_imagebind': {
            'data_dir': './datasets/blip3o/parquet_toy_13modality_clip448_imagebind/train',
            'num_files': 1,
            'num_total_samples': 1000,
            'parquet_info_path': './datasets/blip3o/parquet_toy_13modality_clip448_imagebind/train/blip3o_example_toydataset.json',
        },
        'blip3o_rgb_caption_depth_normal_det_seg_grounding_canny_dino':{
            'data_dir': './datasets/blip3o/parquet_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global',
            'num_files': 2891,
            'num_total_samples': 25_000_000,
            'parquet_info_path': './datasets/blip3o/parquet_info/blip3o_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global.json',
        },
        'blip3o_rgb_caption_depth_normal_det_seg_grounding_canny_dino_clip448_imagebind':{
            'data_dir': './datasets/blip3o/parquet_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global_clip448_imagebind',
            'num_files': 2891,
            'num_total_samples': 25_000_000,
            'parquet_info_path': './datasets/blip3o/parquet_info/blip3o_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global_clip448_imagebind.json',
        },
        'blip3o_rgb_caption_depth_normal_det_seg_grounding_canny_dino_clip448_imagebind_aligned':{
            'data_dir': './datasets/blip3o/parquet_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global_clip448_imagebind_aligned',
            'num_files': 1000,
            'num_total_samples': 3_899_886,
            'parquet_info_path': './datasets/blip3o/parquet_info/parquet_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global_clip448_imagebind_aligned.json',
        }
    },
    'unified_any2clip': {
        'blip3o_example_toydataset_13modality_clip448_imagebind': {
            'data_dir': './datasets/blip3o/parquet_toy_13modality_clip448_imagebind/train',
            'num_files': 1,
            'num_total_samples': 1000,
            'parquet_info_path': './datasets/blip3o/parquet_toy_13modality_clip448_imagebind/train/blip3o_example_toydataset.json',
        },
        'blip3o_rgb_caption_depth_normal_det_seg_grounding_canny_dino_clip448_imagebind':{
            'data_dir': './datasets/blip3o/parquet_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global_clip448_imagebind',
            'num_files': 2891,
            'num_total_samples': 25_000_000,
            'parquet_info_path': './datasets/blip3o/parquet_info/blip3o_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global_clip448_imagebind.json',
        },
        'blip3o_rgb_caption_depth_normal_det_seg_grounding_canny_dino_clip448_imagebind_aligned':{
            'data_dir': './datasets/blip3o/parquet_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global_clip448_imagebind_aligned',
            'num_files': 1000,
            'num_total_samples': 3_899_886,
            'parquet_info_path': './datasets/blip3o/parquet_info/parquet_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global_clip448_imagebind_aligned.json',
        }
    },
    'unified_any2imagebind': {
        'blip3o_example_toydataset_13modality_clip448_imagebind': {
            'data_dir': './datasets/blip3o/parquet_toy_13modality_clip448_imagebind/train',
            'num_files': 1,
            'num_total_samples': 1000,
            'parquet_info_path': './datasets/blip3o/parquet_toy_13modality_clip448_imagebind/train/blip3o_example_toydataset.json',
        },
        'blip3o_rgb_caption_depth_normal_det_seg_grounding_canny_dino_clip448_imagebind':{
            'data_dir': './datasets/blip3o/parquet_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global_clip448_imagebind',
            'num_files': 2891,
            'num_total_samples': 25_000_000,
            'parquet_info_path': './datasets/blip3o/parquet_info/blip3o_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global_clip448_imagebind.json',
        },
        'blip3o_rgb_caption_depth_normal_det_seg_grounding_canny_dino_clip448_imagebind_aligned':{
            'data_dir': './datasets/blip3o/parquet_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global_clip448_imagebind_aligned',
            'num_files': 1000,
            'num_total_samples': 3_899_886,
            'parquet_info_path': './datasets/blip3o/parquet_info/parquet_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global_clip448_imagebind_aligned.json',
        }
    },
    'unified_any2imagebindlocal': {
        'blip3o_example_toydataset_13modality_clip448_imagebind': {
            'data_dir': './datasets/blip3o/parquet_toy_13modality_clip448_imagebind/train',
            'num_files': 1,
            'num_total_samples': 1000,
            'parquet_info_path': './datasets/blip3o/parquet_toy_13modality_clip448_imagebind/train/blip3o_example_toydataset.json',
        },
        'blip3o_rgb_caption_depth_normal_det_seg_grounding_canny_dino_clip448_imagebind':{
            'data_dir': './datasets/blip3o/parquet_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global_clip448_imagebind',
            'num_files': 2891,
            'num_total_samples': 25_000_000,
            'parquet_info_path': './datasets/blip3o/parquet_info/blip3o_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global_clip448_imagebind.json',
        },
        'blip3o_rgb_caption_depth_normal_det_seg_grounding_canny_dino_clip448_imagebind_aligned':{
            'data_dir': './datasets/blip3o/parquet_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global_clip448_imagebind_aligned',
            'num_files': 1000,
            'num_total_samples': 3_899_886,
            'parquet_info_path': './datasets/blip3o/parquet_info/parquet_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global_clip448_imagebind_aligned.json',
        }
    }
}


# ── 16-modality parquet (13 old + samseg/samedge/cocodet). Superset of the old
# parquet → registered under every modality key the new run uses. Additive: does
# not touch existing dataset entries. ────────────────────────────────────────
_PARQUET_16MOD_NAME = (
    'blip3o_rgb_caption_depth_normal_det_seg_grounding_canny_dino_clip448_imagebind'
    '_samseg_samedge_cocodet'
)
_PARQUET_16MOD_INFO = {
    'data_dir': (
        './datasets/blip3o/parquet_rgb_caption_depth_normal_det_seg_grounding'
        '_canny_dino_global_clip448_imagebind_samseg_samedge_cocodet'
    ),
    'num_files': 2888,   # 2891 minus 3 held-out val files (parquet_16mod_VAL_heldout)
    'num_total_samples': 25_000_000,
    'parquet_info_path': (
        './datasets/blip3o/parquet_info/'
        'blip3o_rgb_caption_depth_normal_det_seg_grounding_canny_dino_global'
        '_clip448_imagebind_samseg_samedge_cocodet.json'
    ),
}
for _mod_key in [
    'unified_any2rgb', 'unified_any2caption', 'unified_any2depth',
    'unified_any2normal', 'unified_any2seg', 'unified_any2canny',
    'unified_any2grounding', 'unified_any2dino', 'unified_any2dinolocal',
    'unified_any2clip', 'unified_any2imagebind', 'unified_any2imagebindlocal',
    'unified_any2cocodet', 'unified_any2samseg', 'unified_any2samedge',
]:
    DATASET_INFO.setdefault(_mod_key, {})[_PARQUET_16MOD_NAME] = dict(_PARQUET_16MOD_INFO)


# MODALITY_STATS = {
#     "rgb": {
#         "mean": [
#             0.09998548775911331, -0.25365808606147766, 0.02537657506763935, 0.15525108575820923,
#             -0.13307523727416992, 0.18412069976329803, -0.3696898818016052, 0.1216680109500885,
#             0.07324755936861038, 0.011612555012106895, 0.17749321460723877, 0.009583432227373123,
#             0.006619213614612818, -0.1372869461774826, -0.34581223130226135, -0.08381273597478867
#         ],
#         "std": [
#             0.6712213754653931, 1.1496732234954834, 0.6890255212783813, 0.48065587878227234,
#             0.5058963894844055, 0.5045602917671204, 0.964796781539917, 0.7499234676361084,
#             0.60835862159729, 0.8583494424819946, 0.5590784549713135, 0.6568386554718018,
#             0.8276281356811523, 0.8293890357017517, 0.7076113820075989, 0.8054326176643372
#         ]
#     },
#     "depth": {
#         "mean": [
#             0.3062567710876465, -0.9071211814880371, -0.5659458041191101, -0.027933871373534203,
#             0.13497579097747803, 1.6215254068374634, -1.7268935441970825, 1.1100306510925293,
#             0.06933767348527908, -0.5322779417037964, -0.2097727358341217, 0.07714101672172546,
#             -1.3442918062210083, -0.46783745288848877, -0.31114235520362854, 0.9852880239486694
#         ],
#         "std": [
#             0.4548855125904083, 2.0455949306488037, 0.40026405453681946, 0.34950074553489685,
#             0.40104442834854126, 0.47336098551750183, 1.658141851425171, 1.2504311800003052,
#             0.3990429639816284, 1.1091442108154297, 0.724875271320343, 0.8045597672462463,
#             1.4909602403640747, 1.501410961151123, 1.1055831909179688, 1.3423223495483398
#         ]
#     },
#     "normal": {
#         "mean": [
#             1.1839842796325684, 0.8041657209396362, -0.391467422246933, 0.5322199463844299,
#             -0.5196242928504944, 0.537630021572113, 0.4950917363166809, -0.8294451236724854,
#             1.2865995168685913, -1.2402032613754272, 0.2119874656200409, 0.4859861135482788,
#             0.43453630805015564, -1.550497055053711, -0.6797669529914856, -0.09787978231906891
#         ],
#         "std": [
#             0.6379154920578003, 0.5921596884727478, 1.1566283702850342, 0.6052563786506653,
#             0.6098771095275879, 0.5771185755729675, 0.5548882484436035, 0.42952394485473633,
#             0.7510039806365967, 1.1233150959014893, 0.7662065625190735, 0.6668922305107117,
#             0.5062761306762695, 1.2029993534088135, 0.9624990820884705, 0.5272090435028076
#         ]
#     }
# }


MODALITY_STATS = {
    "depth": {
        "mean": [
            0.28762707114219666, -1.011340856552124, -0.5465618371963501, -0.01756225898861885,
            0.12532611191272736, 1.56802499294281, -1.7659778594970703, 1.1437808275222778,
            0.0657510980963707, -0.5864744186401367, -0.2622603476047516, 0.04382546991109848,
            -1.4009149074554443, -0.35514819622039795, -0.2363666445016861, 1.04072904586792
        ],
        "std": [
            0.4532972276210785, 2.055128812789917, 0.397210031747818, 0.34771019220352173,
            0.4028947949409485, 0.47736671566963196, 1.6445382833480835, 1.2453844547271729,
            0.4030587077140808, 1.1071559190750122, 0.7346603870391846, 0.809998095035553,
            1.4902814626693726, 1.511523723602295, 1.1147352457046509, 1.346556544303894
        ]
    },
    "normal": {
        "mean": [
            1.1103582382202148, 0.7585429549217224, -0.4633699059486389, 0.4460155665874481,
            -0.4614908993244171, 0.5055513978004456, 0.5129532217979431, -0.8317145705223083,
            1.1523557901382446, -1.0965044498443604, 0.29184550046920776, 0.3881677985191345,
            0.41409650444984436, -1.3409103155136108, -0.7599201202392578, -0.14071156084537506
        ],
        "std": [
            0.658498227596283, 0.6019635200500488, 1.2136141061782837, 0.644258439540863,
            0.656324028968811, 0.5987396836280823, 0.5700253844261169, 0.4413583278656006,
            0.8087592124938965, 1.2170354127883911, 0.7942348718643188, 0.6947795748710632,
            0.5312002897262573, 1.2880762815475464, 1.0215877294540405, 0.5333302021026611
        ]
    }
}


def normalize_latents_by_modality(latents, modality_types, device):
    """
    Normalize VAE latents by modality type.
    Supports rgb, depth, and normal modalities.
    
    Args:
        latents: torch.Tensor of shape [B, C, H, W]
        modality_types: List[str] of length B indicating modality for each sample
        device: torch device
    
    Returns:
        Normalized latents tensor
    """
    if not MODALITY_STATS:
        return latents
    
    normalized_latents = latents.clone()
    
    for i, modality in enumerate(modality_types):
        if modality in MODALITY_STATS:
            # Get mean and std for this modality
            mean = torch.tensor(MODALITY_STATS[modality]["mean"], device=device, dtype=latents.dtype)
            std = torch.tensor(MODALITY_STATS[modality]["std"], device=device, dtype=latents.dtype)
            
            # Reshape to [C, 1, 1] for broadcasting
            mean = mean.view(-1, 1, 1)
            std = std.view(-1, 1, 1)
            
            # Normalize: (x - mean) / std
            normalized_latents[i] = (latents[i] - mean) / std
    
    return normalized_latents

def denormalize_latents_by_modality(latents, modality_type, device):
    """
    Denormalize VAE latents by modality type.
    Reverses the normalization applied during training for rgb, depth, and normal.
    
    Args:
        latents: torch.Tensor of shape [B, C, H, W] - normalized latents from model
        modality_type: str indicating modality for the sample
        device: torch device
    
    Returns:
        Denormalized latents tensor ready for VAE decoder
    """
    if not MODALITY_STATS or modality_type not in MODALITY_STATS:
        return latents
    
    # Get mean and std for this modality
    mean = torch.tensor(MODALITY_STATS[modality_type]["mean"], device=device, dtype=latents.dtype)
    std = torch.tensor(MODALITY_STATS[modality_type]["std"], device=device, dtype=latents.dtype)
    
    # Reshape to [C, 1, 1] for broadcasting
    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)
    
    # Denormalize: x * std + mean (reverse of normalization)
    denormalized_latents = latents * std + mean
    
    return denormalized_latents
