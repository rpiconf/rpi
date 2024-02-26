from pathlib import Path
from typing import Union

import pandas as pd
from torch import Tensor

from config.config import ExpArgs
from config.constants import INPUT_IDS_NAME
from evaluations.evaluation_utils import calculate_auc, get_input_data
from evaluations.metrics_auc.metrics1 import Metrics1
from utils.dataclasses.metrics_args import MetricArgsItem


# Always works with batch_size=1
class Metrics1Sequel(Metrics1):

    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.output_pkl_path = Path(self.experiment_path, f"{self.item_index}_results_df_perturbation_tests.pkl")
        self.output_examples_path = Path(self.experiment_path, f"{self.item_index}_examples_perturbation_tests.pkl")

    def all_perturbation_tests(self) -> (Tensor, Tensor):
        if len(self.outputs) != 1:
            raise ValueError("This class works with batch_size 1 only")
        metric_args: MetricArgsItem = get_input_data(self.device, self.outputs[0], idx=0)
        if self.verbose:
            if len(metric_args.item_data[INPUT_IDS_NAME]) != 1:
                raise ValueError("eval - loop over items issue")
            example_txt = self.conv_input_ids_to_txt(metric_args.item_data[INPUT_IDS_NAME][0])
            self.examples.append(dict(stage=self.stage, txt=example_txt, gt_target=metric_args.gt_target.item(),
                                      model_pred_origin=metric_args.model_pred_origin.item(), type=None,
                                     item_index=self.item_index))

        model_pred, model_pred_hit = self.perturbation_test_item(item_data=metric_args.item_data,
                                                                 tokens_attr=metric_args.tokens_attr,
                                                                 target=metric_args.model_pred_origin,
                                                                 gt_target=metric_args.gt_target,
                                                                 model_pred_origin=metric_args.model_pred_origin)

        auc_res, steps_res = self.get_auc(num_correct_pertub=model_pred_hit, num_correct_model=metric_args.gt_target)

        return auc_res, steps_res, metric_args

    def get_auc(self, num_correct_pertub, num_correct_model):
        auc = calculate_auc(mean_accuracy_by_step=num_correct_pertub) * 100
        return (auc, num_correct_pertub)

    def update_results_df(self, results_df: pd.DataFrame, auc_res: Tensor, steps_res: Tensor,
                          metric_args: Union[None, MetricArgsItem]):
        return pd.concat([results_df, pd.DataFrame([
            {"attr_score_function": self.stage, "item_index": self.item_index,
             "task": ExpArgs.task.name, "gt_target": metric_args.gt_target, "eval_metric": ExpArgs.eval_metric,
             "explained_model_backbone": ExpArgs.explained_model_backbone,
             "model_pred_origin": metric_args.model_pred_origin,
             "validation_type": ExpArgs.validation_type, "metric_result": auc_res, "metric_steps_result": steps_res,
             "steps_k": self.pertu_steps}])], ignore_index=True)
