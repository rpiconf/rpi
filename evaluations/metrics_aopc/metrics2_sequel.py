from pathlib import Path

import pandas as pd

from config.config import ExpArgs
from evaluations.metrics_aopc.metrics2 import Metrics2


class Metrics2Sequel(Metrics2):

    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.output_pkl_path = Path(self.experiment_path, f"{self.item_index}_results_df.pkl")

    def update_results_df(self, results_df: pd.DataFrame, metric_result):
        gt_target = self.outputs[0].gt_target
        if type(gt_target) == list:
            if len(gt_target) != 1:
                raise ValueError(f"update_results_df")
            gt_target = gt_target[0]
        gt_target = gt_target.item()
        return pd.concat([results_df, pd.DataFrame([
            {"attr_score_function": self.stage, "item_index": self.item_index,
             "task": ExpArgs.task.name, "eval_metric": ExpArgs.eval_metric,
             "explained_model_backbone": ExpArgs.explained_model_backbone,
             "validation_type": ExpArgs.validation_type, "metric_result": metric_result, "metric_steps_result": None,
             "steps_k": self.pretu_steps, "model_pred_origin": self.outputs[0].pred_origin.squeeze().item(),
             "gt_target": gt_target}])], ignore_index=True)
