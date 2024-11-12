import copy
from typing import Tuple, Union

import torch
from torch import Tensor
from transformers import AutoTokenizer

from config.config import ExpArgs
from config.types_enums import TokenEvaluationOptions
from utils.dataclasses.evaluations import DataForEvaluation
from utils.utils_functions import get_device, run_model, merge_prompts


class MetricsFunctions:

    def __init__(self, model, explained_tokenizer: AutoTokenizer, ref_token_id, special_tokens: Tensor):
        self.model = model
        self.explained_tokenizer = explained_tokenizer
        self.ref_token_id = ref_token_id
        self.special_tokens = special_tokens
        self.perturbation_steps = torch.arange(10, 100, 10)
        self.device = get_device()

        self.labels_tokens = None

    def log_odds(self, item_args: DataForEvaluation):
        topk_indices, required_tokens = self.preprocess_and_get_indices(item_args)
        original_probabilities = torch.softmax(item_args.explained_model_predicted_logits, dim = 0)

        inputs = copy.deepcopy(item_args.input)

        origin_input_ids = inputs.input_ids
        origin_input_ids[0][topk_indices] = self.ref_token_id
        origin_attention_mask = inputs.attention_mask

        merged_inputs_ids, merged_attention_mask = merge_prompts(  #
            inputs = origin_input_ids, attention_mask = origin_attention_mask,
            task_prompt_input_ids = inputs.task_prompt_input_ids,
            label_prompt_input_ids = inputs.label_prompt_input_ids,
            task_prompt_attention_mask = inputs.task_prompt_attention_mask,
            label_prompt_attention_mask = inputs.label_prompt_attention_mask  #
        )

        perturbed_logits = run_model(model = self.model, input_ids = merged_inputs_ids.cuda(),
                                     attention_mask = merged_attention_mask.cuda()).squeeze()
        perturbed_probabilities = torch.softmax(perturbed_logits, dim = 0)
        result = (torch.log(perturbed_probabilities[item_args.explained_model_predicted_class]) - torch.log(
            original_probabilities[item_args.explained_model_predicted_class])).item()

        return result

    def sufficiency(self, item_args: DataForEvaluation):
        topk_indices, required_tokens = self.preprocess_and_get_indices(item_args)
        original_probabilities = torch.softmax(item_args.explained_model_predicted_logits, dim = 0)

        # If no top-k indices are selected, no masking will occur
        if topk_indices.shape[-1] == 0:
            return 0

        inputs = copy.deepcopy(item_args.input)

        mask = torch.zeros_like(inputs.input_ids[0]).bool()
        mask[topk_indices] = 1
        if required_tokens is not None:
            mask[required_tokens] = 1
        origin_masked_input_ids = inputs.input_ids[0][mask].unsqueeze(0)
        origin_masked_attention_mask = inputs.attention_mask[0][mask].unsqueeze(0)

        print(f"EVAL - before merge: {origin_masked_input_ids}")
        merged_masked_input_ids, merged_masked_attention_mask = merge_prompts(  #
            inputs = origin_masked_input_ids, attention_mask = origin_masked_attention_mask,
            task_prompt_input_ids = inputs.task_prompt_input_ids,
            label_prompt_input_ids = inputs.label_prompt_input_ids,
            task_prompt_attention_mask = inputs.task_prompt_attention_mask,
            label_prompt_attention_mask = inputs.label_prompt_attention_mask  #
        )
        print(f"EVAL - before merge: {merged_masked_input_ids}")

        perturbed_logits = run_model(model = self.model, input_ids = merged_masked_input_ids.cuda(),
                                     attention_mask = merged_masked_attention_mask.cuda()).squeeze()
        perturbed_probabilities = torch.softmax(perturbed_logits, dim = 0)

        result = (original_probabilities[item_args.explained_model_predicted_class] - perturbed_probabilities[
            item_args.explained_model_predicted_class]).item()
        return result

    def comprehensiveness(self, item_args: DataForEvaluation):
        topk_indices, required_tokens = self.preprocess_and_get_indices(item_args)
        original_probabilities = torch.softmax(item_args.explained_model_predicted_logits, dim = 0)

        inputs = copy.deepcopy(item_args.input)
        mask = torch.ones_like(inputs.input_ids[0]).bool()
        mask[topk_indices] = 0
        if required_tokens is not None:
            mask[required_tokens] = 1

        origin_masked_input_ids = inputs.input_ids[0][mask].unsqueeze(0)
        origin_masked_attention_mask = inputs.attention_mask[0][mask].unsqueeze(0)

        merged_masked_input_ids, merged_masked_attention_mask = merge_prompts(  #
            inputs = origin_masked_input_ids, attention_mask = origin_masked_attention_mask,
            task_prompt_input_ids = inputs.task_prompt_input_ids,
            label_prompt_input_ids = inputs.label_prompt_input_ids,
            task_prompt_attention_mask = inputs.task_prompt_attention_mask,
            label_prompt_attention_mask = inputs.label_prompt_attention_mask)
        perturbed_logits = run_model(model = self.model, input_ids = merged_masked_input_ids.cuda(),
                                     attention_mask = merged_masked_attention_mask.cuda()).squeeze()
        perturbed_probabilities = torch.softmax(perturbed_logits, dim = 0)

        result = (original_probabilities[item_args.explained_model_predicted_class] - perturbed_probabilities[
            item_args.explained_model_predicted_class]).item()
        return result

    def preprocess_and_get_indices(self, item_args: DataForEvaluation) -> Tuple[Tensor, Tensor]:
        tokens_attr, n_attr, required_tokens = self.handle_evaluation_tokens(item_args)
        k = int(n_attr * item_args.k / 100)
        topk_indices = torch.topk(tokens_attr, k, sorted = False).indices

        if required_tokens is not None:
            overlap = bool(set(topk_indices.tolist()).intersection(set(required_tokens.tolist())))
            if overlap:
                raise ValueError(f"required_tokens souled not be in the topk_indices")
        return topk_indices, required_tokens

    def handle_evaluation_tokens(self, item_args: DataForEvaluation) -> Tuple[Tensor, Tensor, Union[Tensor, None]]:
        invalid_value = float('-inf')
        tokens_attributions: Tensor = copy.deepcopy(item_args.tokens_attributions)
        input_ids: Tensor = item_args.input.input_ids.squeeze()
        num_attributes = tokens_attributions.shape[-1]
        required_tokens = None

        # if ExpArgs.eval_tokens == EvalTokens.ALL_TOKENS.value:
        #     return tokens_attributions, num_attributes , required_tokens
        # elif ExpArgs.eval_tokens == EvalTokens.NO_CLS.value:
        #     if is_model_encoder_only():
        #         tokens_attributions[0] = invalid_value   # cls
        #         num_attributes  = num_attributes  - 1  # cls
        #         required_tokens = torch.tensor([0])  # cls
        #     else:
        #         raise ValueError("unsupported EvalTokens. NO_CLS.value for not encoders only models")
        #     return tokens_attributions, num_attributes , required_tokens
        if ExpArgs.token_evaluation_option == TokenEvaluationOptions.NO_SPECIAL_TOKENS.value:
            special_tokens_indices = torch.isin(input_ids, self.special_tokens)
            if special_tokens_indices.sum() == 0:
                return tokens_attributions, num_attributes, required_tokens
            tokens_attributions[special_tokens_indices] = invalid_value
            required_tokens = torch.nonzero(special_tokens_indices).squeeze()
            if required_tokens.dim() == 0:
                required_tokens = required_tokens.unsqueeze(0)
            num_attributes = num_attributes - required_tokens.shape[-1]
            return tokens_attributions, num_attributes, required_tokens
        else:
            raise ValueError("unsupported ExpArgs.eval_tokens selected")
