from dataclasses import dataclass
from typing import Any, Union

from torch import Tensor


@dataclass
class DataForEval:
    tokens_attr: Tensor
    input: Any
    gt_target: Tensor
    pred_origin: Tensor
    pred_origin_logits: Tensor

    pred_loss: Union[Tensor, None] = None
    pred_loss_mul: Union[Tensor, None] = None
    tokens_attr_sparse_loss: Union[Tensor, None] = None
    tokens_attr_sparse_loss_mul: Union[Tensor, None] = None
    loss: Union[Tensor, None] = None
