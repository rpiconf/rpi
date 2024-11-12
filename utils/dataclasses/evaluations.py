from dataclasses import dataclass

from torch import Tensor


@dataclass
class DataForEvaluationInputs:
    input_ids: Tensor
    attention_mask: Tensor
    task_prompt_input_ids: Tensor
    label_prompt_input_ids: Tensor
    task_prompt_attention_mask: Tensor
    label_prompt_attention_mask: Tensor


@dataclass
class DataForEvaluation:
    tokens_attributions: Tensor
    input: DataForEvaluationInputs
    explained_model_predicted_class: Tensor
    explained_model_predicted_logits: Tensor
    k: float = 0.0
