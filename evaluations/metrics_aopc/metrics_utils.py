import copy

import torch
from transformers import AutoTokenizer

from config.config import ExpArgs
from config.constants import INPUT_IDS_NAME, ATTENTION_MASK_NAME, TASK_PROMPT_KEY, LABEL_PROMPT_KEY
from utils.dataclasses.metrics_args import MetricArgsItem
from utils.utils_functions import get_device, run_model, model_seq_cls_merge_inputs


class MetricsFunctions:

    def __init__(self, model, explained_tokenizer: AutoTokenizer, ref_token_id):
        self.model = model
        self.explained_tokenizer = explained_tokenizer
        self.ref_token_id = ref_token_id
        self.perturbation_steps = torch.arange(10, 100, 10)
        self.device = get_device()

        self.labels_tokens = None

    def log_odds(self, metric_args: MetricArgsItem):
        prob_original = torch.softmax(metric_args.model_pred_origin_logits, dim=0)

        topk_indices = torch.topk(metric_args.tokens_attr, int(metric_args.tokens_attr.shape[-1] * metric_args.k / 100),
                                  sorted=False).indices

        inputs = copy.deepcopy(metric_args.item_data)

        inputs[INPUT_IDS_NAME][0][topk_indices] = self.ref_token_id

        inputs_ids = inputs[INPUT_IDS_NAME]
        inputs_ids = model_seq_cls_merge_inputs(inputs_ids, inputs[TASK_PROMPT_KEY], inputs[LABEL_PROMPT_KEY]).to(self.device)
        logits_perturbed = run_model(model=self.model, input_ids=inputs_ids,
                                     attention_mask=inputs[ATTENTION_MASK_NAME], is_return_logits=True).squeeze()
        prob_perturbed = torch.softmax(logits_perturbed, dim=0)
        result = (torch.log(prob_perturbed[metric_args.model_pred_origin]) - torch.log(
            prob_original[metric_args.model_pred_origin])).item()

        return result

    def sufficiency(self, metric_args: MetricArgsItem):
        prob_original = torch.softmax(metric_args.model_pred_origin_logits, dim=0)

        topk_indices = torch.topk(metric_args.tokens_attr, int(metric_args.tokens_attr.shape[-1] * metric_args.k / 100),
                                  sorted=False).indices

        # if len(topk_indices) == 0:
        if topk_indices.shape[-1] == 0:
            # topk% is too less to select even word - so no masking will happen.
            return 0

        inputs = copy.deepcopy(metric_args.item_data)
        mask = torch.zeros_like(inputs[INPUT_IDS_NAME][0]).bool()
        mask[topk_indices] = 1
        masked_input_ids = inputs[INPUT_IDS_NAME][0][mask].unsqueeze(0)
        masked_attention_mask = inputs[ATTENTION_MASK_NAME][0][mask].unsqueeze(0)

        masked_input_ids = model_seq_cls_merge_inputs(masked_input_ids, inputs[TASK_PROMPT_KEY], inputs[LABEL_PROMPT_KEY]).to(self.device)
        logits_perturbed = run_model(model=self.model, input_ids=masked_input_ids, attention_mask=masked_attention_mask,
                                     is_return_logits=True).squeeze()
        prob_perturbed = torch.softmax(logits_perturbed, dim=0)

        result = (prob_original[metric_args.model_pred_origin] - prob_perturbed[metric_args.model_pred_origin]).item()
        return result

    def comprehensiveness(self, metric_args: MetricArgsItem):
        prob_original = torch.softmax(metric_args.model_pred_origin_logits, dim=0)

        topk_indices = torch.topk(metric_args.tokens_attr, int(metric_args.tokens_attr.shape[-1] * metric_args.k / 100),
                                  sorted=False).indices

        inputs = copy.deepcopy(metric_args.item_data)
        mask = torch.ones_like(inputs[INPUT_IDS_NAME][0]).bool()
        mask[topk_indices] = 0

        masked_input_ids = inputs[INPUT_IDS_NAME][0][mask].unsqueeze(0)
        masked_attention_mask = inputs[ATTENTION_MASK_NAME][0][mask].unsqueeze(0)

        masked_input_ids = model_seq_cls_merge_inputs(masked_input_ids, inputs[TASK_PROMPT_KEY], inputs[LABEL_PROMPT_KEY]).to(self.device)
        logits_perturbed = run_model(model=self.model, input_ids=masked_input_ids, attention_mask=masked_attention_mask,
                                     is_return_logits=True).squeeze()
        prob_perturbed = torch.softmax(logits_perturbed, dim=0)

        result = (prob_original[metric_args.model_pred_origin] - prob_perturbed[metric_args.model_pred_origin]).item()
        return result
