from dataclasses import dataclass

from torch import Tensor


@dataclass
class MetricArgsItem:
    item_data: dict
    tokens_attr: Tensor
    model_pred_origin: Tensor
    model_pred_origin_logits: Tensor
    gt_target: Tensor
    k: float=0.0
