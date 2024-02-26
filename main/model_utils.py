from enum import Enum

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from config.config import ExpArgs
from config.constants import HF_CACHE
from config.types_enums import ModelBackboneTypes, RefTokenNameTypes
from models.bert import BertForSequenceClassification
from models.llama import LlamaForCausalLM
from models.roberta import RobertaForSequenceClassification
from utils.dataclasses import Task


def get_model_tokenizer(task: Task):
    if ExpArgs.explained_model_backbone == ModelBackboneTypes.ROBERTA.value:
        model_url = task.roberta_fine_tuned_model
        model = RobertaForSequenceClassification.from_pretrained(model_url, cache_dir = HF_CACHE)
    elif ExpArgs.explained_model_backbone == ModelBackboneTypes.BERT.value:
        model_url = task.bert_fine_tuned_model
        model = BertForSequenceClassification.from_pretrained(model_url, cache_dir = HF_CACHE)
    elif ExpArgs.explained_model_backbone == ModelBackboneTypes.LLAMA.value:
        model_url = task.llama_model
        model = LlamaForCausalLM.from_pretrained(model_url, torch_dtype = torch.float16, cache_dir = HF_CACHE)
    else:
        raise ValueError(f"unsupported model")

    tokenizer = AutoTokenizer.from_pretrained(model_url)

    # SET MAX LENGTH
    if task.is_llama_set_max_len and ExpArgs.explained_model_backbone == ModelBackboneTypes.LLAMA.value:
        tokenizer.model_max_length = task.llama_explained_tokenizer_max_length

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


def get_folder_name(exp_name: str, explainer: str, metric: Enum):
    return f"{ExpArgs.default_root_dir}/{exp_name}/exp_{explainer}_metric_{metric.value}"
