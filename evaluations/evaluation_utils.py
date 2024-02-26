import numpy as np
from sklearn.metrics import auc

from config.constants import INPUT_IDS_NAME, ATTENTION_MASK_NAME, TASK_PROMPT_KEY, LABEL_PROMPT_KEY
from utils.dataclasses.metrics_args import MetricArgsItem
from utils.dataclasses.trainer_outputs import DataForEval
from utils.utils_functions import conv_class_to_dict


def is_key_exists(k, item):
    return (k in item) and (item[k] is not None)


def calculate_auc(mean_accuracy_by_step: np.ndarray) -> float:
    x = np.array([0.0, 0.33, 0.66, 1.0])
    return auc(x = x, y = mean_accuracy_by_step)


def get_input_data(device, batch: DataForEval, idx: int) -> MetricArgsItem:
    item_data_obj = conv_class_to_dict(batch.input)

    keys = [INPUT_IDS_NAME, ATTENTION_MASK_NAME, TASK_PROMPT_KEY]
    item_data = {key: item_data_obj[key][idx].unsqueeze(0).to(device) if is_key_exists(key, item_data_obj) else None for
                 key in keys}
    item_data[LABEL_PROMPT_KEY] = item_data_obj[LABEL_PROMPT_KEY].to(device) if is_key_exists(LABEL_PROMPT_KEY,
                                                                                              item_data_obj) else None
    tokens_attr = batch.tokens_attr[idx].to(device)
    model_pred_origin = batch.pred_origin[idx]
    model_pred_origin_logits = batch.pred_origin_logits[idx]
    gt_target = batch.gt_target[idx]
    return MetricArgsItem(item_data = item_data, tokens_attr = tokens_attr, model_pred_origin = model_pred_origin,
                          model_pred_origin_logits = model_pred_origin_logits, gt_target = gt_target)
