from pathlib import Path
from typing import List

import pandas as pd
import torch
from transformers import AutoTokenizer

from config.config import ExpArgs, MetricsMetaData
from config.types_enums import EvalMetric
from evaluations.metrics.metrics_utils import MetricsFunctions
from utils.dataclasses.evaluations import DataForEvaluation
from utils.dataclasses.metric_results import MetricResults
from utils.utils_functions import get_device, get_model_special_tokens


class Metrics:

    def __init__(self, model, explained_tokenizer: AutoTokenizer, ref_token_id, data: DataForEvaluation,
                 item_index: str, experiment_path: str):
        self.model = model
        self.explained_tokenizer = explained_tokenizer
        self.ref_token_id = ref_token_id
        self.device = get_device()
        self.special_tokens = torch.tensor(
            get_model_special_tokens(ExpArgs.explained_model_backbone, self.explained_tokenizer)).to(self.device)
        self.data: DataForEvaluation = data
        self.item_index = item_index
        self.experiment_path = experiment_path
        self.metric_functions = MetricsFunctions(model, explained_tokenizer, ref_token_id, self.special_tokens)
        self.pretu_steps = MetricsMetaData.top_k[ExpArgs.evaluation_metric]
        self.output_path = Path(experiment_path, "support_results_df.csv")

    def run_perturbation_test(self):
        results_steps: List[float] = []
        for idx, k in enumerate(self.pretu_steps):
            self.data.k = k
            step_metric_result = self.run_metric(self.data)
            results_steps.append(step_metric_result)

        # AOPC or one step only
        if ExpArgs.evaluation_metric in [EvalMetric.AOPC_SUFFICIENCY.value, EvalMetric.AOPC_COMPREHENSIVENESS.value]:
            metric_res = sum(results_steps) / (len(self.pretu_steps) + 1)
        elif ExpArgs.evaluation_metric in [EvalMetric.SUFFICIENCY.value, EvalMetric.COMPREHENSIVENESS.value,
                                           EvalMetric.EVAL_LOG_ODDS.value]:
            if len(results_steps) > 1:
                raise ValueError("has more than 1 value without AOPC calc")
            metric_res = results_steps[0]
        else:
            raise ValueError("unsupported ExpArgs.eval_metric selected - run_perturbation_test")

        results_item = self.transform_results(metric_res)
        self.save_results(results_item)
        return metric_res, results_item

    def run_metric(self, item_args):
        if ExpArgs.evaluation_metric in [EvalMetric.SUFFICIENCY.value, EvalMetric.AOPC_SUFFICIENCY.value]:
            return self.metric_functions.sufficiency(item_args)

        elif ExpArgs.evaluation_metric in [EvalMetric.COMPREHENSIVENESS.value, EvalMetric.AOPC_COMPREHENSIVENESS.value]:
            return self.metric_functions.comprehensiveness(item_args)

        elif ExpArgs.evaluation_metric == EvalMetric.EVAL_LOG_ODDS.value:
            return self.metric_functions.log_odds(item_args)
        else:
            raise ValueError("unsupported metric_functions selected")

    def save_results(self, results_item):
        if ExpArgs.is_save_support_results:
            with open(self.output_path, 'a', newline = '', encoding = 'utf-8-sig') as f:
                results_item.to_csv(f, header = f.tell() == 0, index = False)

    def transform_results(self, metric_result):
        item = MetricResults(item_index = self.item_index, task = ExpArgs.task.name,
                             evaluation_metric = ExpArgs.evaluation_metric,
                             explained_model_backbone = ExpArgs.explained_model_backbone,
                             metric_result = metric_result, metric_result_str = "{:.6f}".format(metric_result),
                             metric_steps_result = None, steps_k = self.pretu_steps,
                             explained_model_predicted_class = self.data.explained_model_predicted_class.squeeze().item(),
                             token_evaluation_option = ExpArgs.token_evaluation_option,
                             reverse_noise_layer_index = ExpArgs.reverse_noise_layer_index,
                             num_interpolation = ExpArgs.num_interpolation,
                             num_samples = ExpArgs.num_samples)
        return pd.DataFrame([item])
