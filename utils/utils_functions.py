import gc
import time
from typing import Tuple

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


def is_model_encoder_only(model = None):
    if model is None:
        model = ExpArgs.explained_model_backbone
    if model in [ModelBackboneTypes.LLAMA.value, ModelBackboneTypes.MISTRAL.value]:
        return False
    if model in [ModelBackboneTypes.ROBERTA.value, ModelBackboneTypes.BERT.value, ModelBackboneTypes.DISTILBERT.value]:
        return True

    raise ValueError(f"unsupported model: {model}")

def get_model_special_tokens(model, tokenizer):
    if is_model_encoder_only(model):
        return [  #
            getattr(tokenizer, "cls_token_id"),  #
            getattr(tokenizer, "pad_token_id"),  #
            getattr(tokenizer, "sep_token_id")  #
        ]
    else:
        return [  #
            getattr(tokenizer, "bos_token_id"),  #
            # getattr(tokenizer, "pad_token_id"),  #
            getattr(tokenizer, "eos_token_id")  #
        ]


def is_use_prompt():
    return (not is_model_encoder_only()) and (not ExpArgs.task.is_finetuned_with_lora)


def get_current_time():
    return int(round(time.time()))


def run_model(model, input_ids: Tensor = None, attention_mask: Tensor = None, inputs_embeds: Tensor = None):
    if is_model_encoder_only():
        model_output = model(input_ids = input_ids, attention_mask = attention_mask, inputs_embeds = inputs_embeds)
        logits = model_output.logits
    else:
        if inputs_embeds is not None:
            logits = model(inputs_embeds = inputs_embeds, attention_mask = attention_mask).logits
        else:
            logits = model(input_ids = input_ids, attention_mask = attention_mask).logits

        if not ExpArgs.task.is_finetuned_with_lora:
            logits = logits[:, -1, :][:, ExpArgs.label_vocab_tokens]

    return logits


def merge_prompts(inputs, attention_mask, task_prompt_input_ids: Tensor = None, label_prompt_input_ids: Tensor = None,
                  task_prompt_attention_mask: Tensor = None, label_prompt_attention_mask: Tensor = None) -> Tuple[
    Tensor, Tensor]:
    if not is_use_prompt():
        return inputs, attention_mask

    if any(item is None for item in [task_prompt_input_ids, inputs, label_prompt_input_ids]):
        raise ValueError("can not be None")
    merged_inputs = torch.cat([task_prompt_input_ids, inputs, label_prompt_input_ids], dim = 1)
    merged_attention_mask = torch.cat([task_prompt_attention_mask, attention_mask, label_prompt_attention_mask],
                                      dim = 1)

    return merged_inputs, merged_attention_mask


def init_experiment():
    seed = ExpArgs.seed
    gc.collect()
    torch.cuda.empty_cache()
    seed_everything(seed)
