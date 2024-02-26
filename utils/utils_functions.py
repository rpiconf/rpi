import gc
import time

import torch
from pytorch_lightning import seed_everything
from torch import Tensor

from config.config import ExpArgs
from config.types_enums import ModelBackboneTypes


def conv_class_to_dict(item):
    obj = {}
    for key in item.keys():
        obj[key] = item[key]
    return obj


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_current_time():
    return int(round(time.time()))


def run_model(model, input_ids: Tensor = None, attention_mask: Tensor = None, inputs_embeds: Tensor = None,
              is_return_logits: bool = False, is_return_output: bool = False):
    model_backbone = ExpArgs.explained_model_backbone
    if model_backbone == ModelBackboneTypes.LLAMA.value:
        # max_new_tokens = ExpArgs.llama_new_generate_tokens
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.to(torch.float16)
            model_output = model(inputs_embeds = inputs_embeds).logits[:, -1, :]
        else:
            model_output = model(input_ids = input_ids).logits[:, -1, :]
        logits = model_output[:, ExpArgs.labels_tokens_opt]  # logits = torch.clip(logits, -10e10, 10e10)
    elif model_backbone in [ModelBackboneTypes.ROBERTA.value, ModelBackboneTypes.BERT.value, ]:
        model_output = model(input_ids = input_ids, attention_mask = attention_mask, inputs_embeds = inputs_embeds)
        logits = model_output.logits
    else:
        raise ValueError("model_backbone is unsupported")

    if is_return_logits and is_return_output:
        return model_output, logits

    elif is_return_logits:
        return logits

    elif is_return_output:
        return model_output

    else:
        raise ValueError(f"must choose model output option")


def model_seq_cls_merge_inputs(inputs, task_prompt_embeds, label_prompt_embeds):
    if ExpArgs.explained_model_backbone != ModelBackboneTypes.LLAMA.value:
        if type(inputs) == list:
            return torch.stack(inputs)
        return inputs

    merged_inputs = []
    for item_idx in range(len(task_prompt_embeds)):
        merged_inputs.append(
            torch.concat([task_prompt_embeds[item_idx], inputs[item_idx], label_prompt_embeds], axis = 0))
    return torch.stack(merged_inputs)


def init_experiment():
    seed = ExpArgs.seed
    gc.collect()
    torch.cuda.empty_cache()
    seed_everything(seed)
