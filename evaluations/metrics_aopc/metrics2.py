import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from config.config import ExpArgs, MetricsMetaData
from config.constants import INPUT_IDS_NAME
from config.types_enums import EvalMetric
from evaluations.evaluation_utils import get_input_data
from evaluations.metrics_aopc.metrics_utils import MetricsFunctions
from utils.dataclasses.metrics_args import MetricArgsItem
from utils.dataclasses.trainer_outputs import DataForEval
from utils.utils_functions import get_device


class Metrics2:

    def __init__(self, model, explained_tokenizer: AutoTokenizer, ref_token_id,
                 outputs: List[DataForEval], stage: str, item_index: int, experiment_path: str,
                 verbose: bool):
        self.model = model
        self.explained_tokenizer = explained_tokenizer
        self.ref_token_id = ref_token_id
        self.device = get_device()
        self.special_tokens = torch.tensor(self.explained_tokenizer.all_special_ids).to(self.device)
        self.outputs: List[DataForEval] = outputs
        self.stage = stage
        self.item_index = item_index
        self.experiment_path = experiment_path
        self.metric_functions = MetricsFunctions(model, explained_tokenizer, ref_token_id)
        self.pretu_steps = MetricsMetaData.top_k[ExpArgs.eval_metric]
        self.output_pkl_path = Path(experiment_path, f"{self.stage}_results_df.pkl")

    def run_perturbation_test(self):
        results = []
        for batch_idx, batch in enumerate(self.outputs):
            for i in range(len(batch.gt_target)):

                metric_args: MetricArgsItem = get_input_data(self.device, batch, idx=i)
                if len(metric_args.item_data[INPUT_IDS_NAME]) != 1:
                    raise ValueError("eval - loop over items issue")

                results_steps_vec = np.zeros([len(self.pretu_steps)], dtype=np.float64)
                for idx, k in enumerate(self.pretu_steps):
                    metric_args.k = k
                    results_steps_vec[idx] = self.run_metric(metric_args)

                # AOPC or just one step
                if ExpArgs.eval_metric in [EvalMetric.AOPC_SUFFICIENCY.value, EvalMetric.AOPC_COMPREHENSIVENESS.value]:
                    metric_res = results_steps_vec.sum() / (len(self.pretu_steps) + 1)
                elif ExpArgs.eval_metric in [EvalMetric.SUFFICIENCY.value, EvalMetric.COMPREHENSIVENESS.value,
                                             EvalMetric.EVAL_LOG_ODDS.value]:
                    if len(results_steps_vec) > 1:
                        raise ValueError("METRIC2 has more than 1 value without AOPC calc")
                    metric_res = results_steps_vec[0]
                else:
                    raise ValueError("unsupported ExpArgs.eval_metric selected - run_perturbation_test")
                results.append(metric_res)
        results_mean = torch.tensor(results).mean()
        self.save_results(results_mean)
        return results_mean

    def run_metric(self, metric_args):
        if ExpArgs.eval_metric in [EvalMetric.SUFFICIENCY.value, EvalMetric.AOPC_SUFFICIENCY.value]:
            return self.metric_functions.sufficiency(metric_args)

        elif ExpArgs.eval_metric in [EvalMetric.COMPREHENSIVENESS.value, EvalMetric.AOPC_COMPREHENSIVENESS.value]:
            return self.metric_functions.comprehensiveness(metric_args)

        elif ExpArgs.eval_metric == EvalMetric.EVAL_LOG_ODDS.value:
            return self.metric_functions.log_odds(metric_args)
        else:
            raise ValueError("unsupported metric_functions selected")

    def get_spec_tokens_indices(self, item_data):
        input_ids = item_data[INPUT_IDS_NAME].squeeze(0)
        spec_tokens_indices = torch.where(torch.isin(input_ids, self.special_tokens))
        if len(spec_tokens_indices) != 1:
            raise ValueError("spec_tokens_indices length is not 1")
        spec_tokens_indices = spec_tokens_indices[0]
        return spec_tokens_indices

    def save_results(self, metric_result):
        if ExpArgs.is_save_results:
            if not os.path.exists(self.experiment_path):
                os.makedirs(self.experiment_path, exist_ok=True)
            if os.path.exists(self.output_pkl_path):
                results_df = pd.read_pickle(self.output_pkl_path)
            else:
                results_df = pd.DataFrame(columns=["stage"])
            results_df = self.update_results_df(results_df, metric_result.item())
            results_df.to_pickle(self.output_pkl_path)

    def update_results_df(self, results_df: pd.DataFrame, metric_result):
        return pd.concat([results_df, pd.DataFrame([
            {"attr_score_function": self.stage,  "item_index": self.item_index,
             "task": ExpArgs.task.name, "eval_metric": ExpArgs.eval_metric,
             "explained_model_backbone": ExpArgs.explained_model_backbone,
             "validation_type": ExpArgs.validation_type, "metric_result": metric_result, "metric_steps_result": None,
             "steps_k": self.pretu_steps}])], ignore_index=True)
