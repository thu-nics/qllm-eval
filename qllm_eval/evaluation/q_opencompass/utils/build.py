import copy
import argparse

import torch
from mmengine.config import ConfigDict
from opencompass.registry import LOAD_DATASET, MODELS

# self import
from qllm_eval.methods.rep.apply_rep import apply_awq
from qllm_eval.quantization.quant_wrapper import quantize_model



def build_dataset_from_cfg(dataset_cfg: ConfigDict):
    dataset_cfg = copy.deepcopy(dataset_cfg)
    dataset_cfg.pop('infer_cfg', None)
    dataset_cfg.pop('eval_cfg', None)
    dataset_cfg.pop('abbr', None)
    return LOAD_DATASET.build(dataset_cfg)


def build_model_from_cfg(model_cfg: ConfigDict):
    model_cfg = copy.deepcopy(model_cfg)
    model_cfg.pop('run_cfg', None)
    model_cfg.pop('max_out_len', None)
    model_cfg.pop('batch_size', None)
    model_cfg.pop('abbr', None)
    model_cfg.pop('pred_postprocessor', None)
    # dump the quantization configs
    quant_cfg = {}
    quant_cfg['w_group_size'] = model_cfg.pop('w_group_size', 128)
    quant_cfg['w_bit'] = model_cfg.pop('w_bit', None)
    quant_cfg['a_group_size'] = model_cfg.pop('a_group_size', 128)
    quant_cfg['a_bit'] = model_cfg.pop('a_bit', None)
    quant_cfg['kv_group_size'] = model_cfg.pop('kv_group_size', 64)
    quant_cfg['kv_bit'] = model_cfg.pop('kv_bit', None)
    use_flash_attn = model_cfg.pop('use_flash_attn', False)
    rep_file = model_cfg.pop('rep_file', None)

    # # if kv cache quantization is specified, we should add the config to the model.
    # if quant_cfg['kv_bit'] is not None:
    #     model_cfg['model_kwargs'].update({
    #         'kv_bit': quant_cfg['kv_bit'],
    #         'kv_group_size': quant_cfg['kv_group_size'],
    #         'use_flash_attn': use_flash_attn,
    #     })

    # build the original llm
    lm_model = MODELS.build(model_cfg)
    '''
    Implement Your Quantization Code Here.
    Below is a demo quantizing weights only.
    '''
    raw_model = lm_model.model # shallow copy

    if rep_file is not None:
        rep_results = torch.load(rep_file, map_location="cpu")
        apply_awq(raw_model, rep_results)

    lm_model.model = quantize_model(raw_model, argparse.Namespace(**quant_cfg))
    return lm_model
