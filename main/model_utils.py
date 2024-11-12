from enum import Enum

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoTokenizer

from config.config import ExpArgs
from config.constants import HF_CACHE
from config.types_enums import ModelBackboneTypes, RefTokenNameTypes
from utils.dataclasses import Task
from utils.utils_functions import is_model_encoder_only


def get_model_tokenizer(task: Task):
    if ExpArgs.explained_model_backbone == ModelBackboneTypes.ROBERTA.value:
        from models.roberta import RobertaForSequenceClassification
        model_url = task.roberta_fine_tuned_model
        model = RobertaForSequenceClassification.from_pretrained(model_url, cache_dir = HF_CACHE)
    elif ExpArgs.explained_model_backbone == ModelBackboneTypes.BERT.value:
        from models.bert import BertForSequenceClassification
        model_url = task.bert_fine_tuned_model
        model = BertForSequenceClassification.from_pretrained(model_url, cache_dir = HF_CACHE)
    elif ExpArgs.explained_model_backbone == ModelBackboneTypes.DISTILBERT.value:
        from models.distilbert import DistilBertForSequenceClassification
        model_url = task.distilbert_fine_tuned_model
        model = DistilBertForSequenceClassification.from_pretrained(model_url, cache_dir = HF_CACHE)
    elif ExpArgs.explained_model_backbone == ModelBackboneTypes.LLAMA.value:
        from models.llama import LlamaForCausalLM, LlamaForSequenceClassification
        model_url = task.llama_model
        if task.is_finetuned_with_lora:
            model = LlamaForSequenceClassification.from_pretrained(model_url, torch_dtype = torch.bfloat16,
                                                                   cache_dir = HF_CACHE,
                                                                   num_labels = len(task.labels_int_str_maps.keys()))
            model = PeftModel.from_pretrained(model, task.llama_adapter)
            model = model.merge_and_unload()

        else:
            model = LlamaForCausalLM.from_pretrained(model_url, torch_dtype = torch.bfloat16, cache_dir = HF_CACHE)
    elif ExpArgs.explained_model_backbone == ModelBackboneTypes.MISTRAL.value:
        from models.mistral import MistralForCausalLM, MistralForSequenceClassification
        model_url = task.mistral_model
        if task.is_finetuned_with_lora:
            model = MistralForSequenceClassification.from_pretrained(model_url, torch_dtype = torch.bfloat16,
                                                                     cache_dir = HF_CACHE,
                                                                     num_labels = len(task.labels_int_str_maps.keys()))
            model = PeftModel.from_pretrained(model, task.mistral_adapter)
            model = model.merge_and_unload()
        else:
            model = MistralForCausalLM.from_pretrained(model_url, torch_dtype = torch.bfloat16, cache_dir = HF_CACHE)

    else:
        raise ValueError(f"unsupported model")

    tokenizer = AutoTokenizer.from_pretrained(model_url)

    if (not is_model_encoder_only()) and ExpArgs.task.is_finetuned_with_lora:
        tokenizer.pad_token_id = model.config.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id

    # SET MAX LENGTH
    if task.is_llm_set_max_len and (not is_model_encoder_only()):
        tokenizer.model_max_length = task.llm_explained_tokenizer_max_length

    return model, tokenizer


def get_ref_token_name(tokenizer):
    if ExpArgs.ref_token_name == RefTokenNameTypes.MASK.value:
        return tokenizer.mask_token_id
    elif ExpArgs.ref_token_name == RefTokenNameTypes.PAD.value:
        return tokenizer.pad_token_id
    elif ExpArgs.ref_token_name == RefTokenNameTypes.UNK.value:
        return tokenizer.unk_token_id
    else:
        raise NotImplementedError


def get_dataset(task: Task):
    ds = load_dataset(task.dataset_name)[task.dataset_test].shuffle(seed = ExpArgs.seed)
    if task.test_sample:
        ds = ds.train_test_split(train_size = task.test_sample, seed = ExpArgs.seed,
                                 stratify_by_column = task.dataset_column_label)
        ds = ds["train"]
    idx_col = ds["idx"] if "idx" in ds.features.keys() else list(range(len(ds[task.dataset_column_label])))
    ds = list(zip(ds[task.dataset_column_text], ds[task.dataset_column_label], idx_col))
    return ds


def get_folder_name(exp_name: str, metric: Enum):
    return f"{ExpArgs.default_root_dir}/{exp_name}/metric_{metric.value}"
