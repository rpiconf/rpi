import gc
import os
from pathlib import Path
from typing import List

import torch

from config.config import ExpArgs, MetricsMetaData
from config.constants import (TEXT_PROMPT, LABEL_PROMPT_NEW_LINE)
from config.types_enums import EvalMetric, DirectionTypes
from evaluations.evaluations import evaluate_tokens_attributions
from main.model_utils import get_dataset, get_model_tokenizer, get_ref_token_name, get_folder_name
from main.rpi_utils import run_rpi
from utils.dataclasses.evaluations import DataForEvaluation, DataForEvaluationInputs
from utils.utils_functions import (run_model, init_experiment, get_device, merge_prompts, is_use_prompt)


class Rpi:
    def __init__(self, exp_name: str, metrics: List[EvalMetric]):
        init_experiment()
        self.task = ExpArgs.task
        self.metrics = metrics
        self.exp_name = exp_name
        self.model, self.tokenizer = get_model_tokenizer(self.task)
        self.task_prompt_input_ids = None
        self.label_prompt_input_ids = None
        self.label_prompt_attention_mask = None
        self.task_prompt_attention_mask = None
        self.set_prompts()

    def run(self):
        device = get_device()
        self.model = self.model.to(device)
        self.model.eval()

        data = get_dataset(self.task)
        ref_token = get_ref_token_name(self.tokenizer)

        for metric in self.metrics:
            output_directory = get_folder_name(self.exp_name, metric)
            os.makedirs(output_directory, exist_ok = True)

        # Compute attributions
        for i, row in enumerate(data):
            item_id = row[2]
            txt = row[0]

            # No special tokens needed for the input, as it will be combined with the prompt.
            is_add_special_tokens = not is_use_prompt()
            origin_input_ids, origin_attention_mask = self.encode(txt, is_add_special_tokens = is_add_special_tokens,
                                                                  is_truncate = True)
            merged_input_ids, merged_attention_mask = merge_prompts(inputs = origin_input_ids,
                                                                    attention_mask = origin_attention_mask,
                                                                    task_prompt_input_ids = self.task_prompt_input_ids,
                                                                    label_prompt_input_ids = self.label_prompt_input_ids,
                                                                    task_prompt_attention_mask = self.task_prompt_attention_mask,
                                                                    label_prompt_attention_mask = self.label_prompt_attention_mask)

            with torch.no_grad():
                explained_model_predicted_logits = run_model(model = self.model, input_ids = merged_input_ids,
                                                             attention_mask = merged_attention_mask)
                explained_model_predicted_class = torch.argmax(explained_model_predicted_logits, dim = 1)

            self.model.zero_grad()
            all_samples_attribution_scores = run_rpi(self.model, merged_input_ids, merged_attention_mask,
                                                     task_prompt_input_ids = self.task_prompt_input_ids,
                                                     label_prompt_input_ids = self.label_prompt_input_ids,
                                                     tokenizer=self.tokenizer, origin_input_ids=origin_input_ids
                                                     )

            for metric in EvalMetric:
                experiment_path = get_folder_name(self.exp_name, metric)
                ExpArgs.evaluation_metric = metric.value

                best_metric_result, best_metric_result_item = None, None
                for sample_attribution_scores in all_samples_attribution_scores:
                    if is_use_prompt():
                        sample_attribution_scores = sample_attribution_scores[self.task_prompt_input_ids.shape[-1]:-
                        self.label_prompt_input_ids.shape[-1]].detach()

                    data_for_evaluation: DataForEvaluation = DataForEvaluation(  #
                        tokens_attributions = sample_attribution_scores,  #
                        input = DataForEvaluationInputs(  #
                            input_ids = origin_input_ids,  #
                            attention_mask = origin_attention_mask,  #
                            task_prompt_input_ids = self.task_prompt_input_ids,  #
                            label_prompt_input_ids = self.label_prompt_input_ids,  #
                            task_prompt_attention_mask = self.task_prompt_attention_mask,  #
                            label_prompt_attention_mask = self.label_prompt_attention_mask  #
                        ),  #
                        explained_model_predicted_class = explained_model_predicted_class.squeeze(),  #
                        explained_model_predicted_logits = explained_model_predicted_logits.squeeze(),  #
                    )

                    evaluation_result, evaluation_item = evaluate_tokens_attributions(self.model, self.tokenizer, ref_token,
                                                                                      data = data_for_evaluation,
                                                                                      experiment_path = experiment_path,
                                                                                      item_index = f"{i}_{item_id}", )

                    if best_metric_result is None:
                        best_metric_result, best_metric_result_item = evaluation_result, evaluation_item
                    elif (best_metric_result < evaluation_result) and (
                            MetricsMetaData.directions[ExpArgs.evaluation_metric] == DirectionTypes.MAX.value):
                        best_metric_result, best_metric_result_item = evaluation_result, evaluation_item
                    elif (best_metric_result > evaluation_result) and (
                            MetricsMetaData.directions[ExpArgs.evaluation_metric] == DirectionTypes.MIN.value):
                        best_metric_result, best_metric_result_item = evaluation_result, evaluation_item

                    gc.collect()
                    torch.cuda.empty_cache()

                if ExpArgs.is_save_results:
                    best_metric_result_item["__input_text__"] = txt
                    with open(Path(experiment_path, "results.csv"), 'a', newline = '', encoding = 'utf-8-sig') as f:
                        best_metric_result_item.to_csv(f, header = f.tell() == 0, index = False)

    def set_prompts(self):
        if is_use_prompt():
            task_prompt = "\n\n".join([self.task.llm_task_prompt, self.task.llm_few_shots_prompt, TEXT_PROMPT])
            self.task_prompt_input_ids, self.task_prompt_attention_mask = self.encode(task_prompt, True, False)
            self.label_prompt_input_ids, self.label_prompt_attention_mask = self.encode(LABEL_PROMPT_NEW_LINE, False,
                                                                                        False)
            labels_tokens = [self.tokenizer.encode(str(l), return_tensors = "pt", add_special_tokens = False) for l in
                             list(ExpArgs.task.labels_int_str_maps.keys())]
            ExpArgs.label_vocab_tokens = torch.stack(labels_tokens).squeeze()
            if ExpArgs.label_vocab_tokens.ndim != 1:
                raise ValueError("label_vocab_tokens must work with one token only")

    def encode(self, new_txt, is_add_special_tokens, is_truncate):
        tokenized = self.tokenizer.encode_plus(new_txt, truncation = is_truncate,
                                               add_special_tokens = is_add_special_tokens, return_tensors = "pt")
        input_ids = tokenized.input_ids.cuda()
        attention_mask = tokenized.attention_mask.cuda()
        return input_ids, attention_mask
