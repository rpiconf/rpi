import torch
from transformers import AutoTokenizer

from config.config import ExpArgs, EvalMetric
from evaluations.metrics.metrics import Metrics
from utils.dataclasses.evaluations import DataForEvaluation


def evaluate_tokens_attributions(model, explained_tokenizer: AutoTokenizer, ref_token_id, data: DataForEvaluation,
                                 experiment_path: str, item_index: str):
    with torch.no_grad():
        if (data.input.input_ids.squeeze().ndim != 1) or (data.tokens_attributions.squeeze().ndim != 1):
            raise ValueError("Unsupported input: Both input IDs and token attributions must have a batch size of 1.")
        if ExpArgs.evaluation_metric in [EvalMetric.SUFFICIENCY.value, EvalMetric.COMPREHENSIVENESS.value,
                                         EvalMetric.EVAL_LOG_ODDS.value, EvalMetric.AOPC_SUFFICIENCY.value,
                                         EvalMetric.AOPC_COMPREHENSIVENESS.value]:
            evaluation_class = Metrics(model = model, explained_tokenizer = explained_tokenizer, ref_token_id = ref_token_id,
                                 data = data, experiment_path = experiment_path, item_index = item_index)
            return evaluation_class.run_perturbation_test()
        else:
            raise ValueError("unsupported ExpArgs.eval_metric selected")
