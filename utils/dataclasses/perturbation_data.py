from dataclasses import dataclass
from typing import Optional

import numpy as np
from torch import Tensor


@dataclass
class PerturbationData:
    input_ids_mat: Tensor
    attention_mask_mat: Tensor


@dataclass
class PerturbationResult:
    pos_auc_no_reference_token: np.float64 = None
    pos_step_pred_no_reference_token: Optional[np.ndarray] = None
    pos_step_acc_no_reference_token: Optional[np.ndarray] = None

    neg_auc_no_reference_token: np.float64 = None
    neg_step_pred_no_reference_token: Optional[np.ndarray] = None
    neg_step_acc_no_reference_token: Optional[np.ndarray] = None

    pos_auc_reference_token: np.float64 = None
    pos_step_pred_reference_token: Optional[np.ndarray] = None
    pos_step_acc_reference_token: Optional[np.ndarray] = None

    neg_auc_reference_token: np.float64 = None
    neg_step_pred_reference_token: Optional[np.ndarray] = None
    neg_step_acc_reference_token: Optional[np.ndarray] = None

    model_gt_target_vec: np.ndarray = None
    model_pred_origin_vec: np.ndarray = None
