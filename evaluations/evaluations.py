from typing import List, Union

import torch
from transformers import AutoTokenizer

from config.config import ExpArgs, EvalMetric
from evaluations.metrics_aopc.metrics2 import Metrics2
from evaluations.metrics_aopc.metrics2_sequel import Metrics2Sequel
from evaluations.metrics_auc.metrics1 import Metrics1
from evaluations.metrics_auc.metrics1_sequel import Metrics1Sequel
from utils.dataclasses.trainer_outputs import DataForEval


def evaluate_tokens_attr(model, explained_tokenizer: AutoTokenizer, ref_token_id,
                         outputs: List[DataForEval], stage: str, experiment_path: str, verbose: bool, item_index: int,
                         is_sequel: bool = False):
    with torch.no_grad():

        if ExpArgs.eval_metric in [EvalMetric.NEG_AUC_WITH_REFERENCE_TOKEN.value,
                                   EvalMetric.POS_AUC_WITH_REFERENCE_TOKEN.value ]:
            if is_sequel:
                eval_class = Metrics1Sequel(model=model, explained_tokenizer=explained_tokenizer,
                                            ref_token_id=ref_token_id,
                                            outputs=outputs, stage=stage, experiment_path=experiment_path,
                                            verbose=verbose, item_index=item_index)
            else:
                eval_class = Metrics1(model=model, explained_tokenizer=explained_tokenizer,
                                      ref_token_id=ref_token_id,
                                      outputs=outputs, stage=stage, experiment_path=experiment_path, verbose=verbose,
                                      item_index=item_index)
            return eval_class.run_perturbation_test()

        elif ExpArgs.eval_metric in [EvalMetric.SUFFICIENCY.value, EvalMetric.COMPREHENSIVENESS.value,
                                     EvalMetric.EVAL_LOG_ODDS.value, EvalMetric.AOPC_SUFFICIENCY.value,
                                     EvalMetric.AOPC_COMPREHENSIVENESS.value]:
            if is_sequel:
                eval_class = Metrics2Sequel(model=model, explained_tokenizer=explained_tokenizer,
                                            ref_token_id=ref_token_id,
                                            outputs=outputs, stage=stage, experiment_path=experiment_path,
                                            verbose=verbose, item_index=item_index)
            else:
                eval_class = Metrics2(model=model, explained_tokenizer=explained_tokenizer,
                                      ref_token_id=ref_token_id,
                                      outputs=outputs, stage=stage, experiment_path=experiment_path, verbose=verbose,
                                      item_index=item_index)
            return eval_class.run_perturbation_test()

        else:
            raise ValueError("unsupported ExpArgs.eval_metric selected")
