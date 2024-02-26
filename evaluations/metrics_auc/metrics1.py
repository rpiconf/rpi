import copy
import os
from pathlib import Path
from typing import Union, List

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from transformers import AutoTokenizer

from config.config import ExpArgs, MetricsMetaData
from config.constants import INPUT_IDS_NAME, ATTENTION_MASK_NAME, TASK_PROMPT_KEY, LABEL_PROMPT_KEY
from config.types_enums import EvalMetric, ModelBackboneTypes
from evaluations.evaluation_utils import calculate_auc, get_input_data
from utils.dataclasses.metrics_args import MetricArgsItem
from utils.dataclasses.perturbation_data import PerturbationData
from utils.dataclasses.trainer_outputs import DataForEval
from utils.utils_functions import get_device, run_model


class Metrics1:

    def __init__(self, model, explained_tokenizer: AutoTokenizer, ref_token_id,
                 outputs: List[DataForEval], stage: str, item_index: int, experiment_path: str,
                 verbose: bool):
        self.model = model
        self.explained_tokenizer = explained_tokenizer
        self.ref_token_id = ref_token_id
        self.pertu_steps = MetricsMetaData.top_k[ExpArgs.eval_metric]
        self.device = get_device()
        self.special_tokens = torch.tensor(self.explained_tokenizer.all_special_ids).to(self.device)
        self.outputs: List[DataForEval] = outputs
        self.stage = stage
        self.item_index = item_index
        self.experiment_path = experiment_path
        self.verbose = verbose
        self.output_pkl_path = Path(experiment_path, f"{stage}_results_df_perturbation_tests.pkl")
        self.output_examples_path = Path(experiment_path, f"{stage}_examples_perturbation_tests.pkl")
        self.examples = []
        self.MAX_EXAMPLES_TO_PRINT = 5
        self.direction = MetricsMetaData.directions[ExpArgs.eval_metric]

    def all_perturbation_tests(self) -> (Tensor, Tensor, Union[MetricArgsItem, None]):
        n_samples = sum(output.pred_origin.shape[0] for output in self.outputs)

        item_index = 0
        model_acc_vec = np.zeros(n_samples)
        pertu_acc_mat = np.zeros((n_samples, len(self.pertu_steps)))
        for batch_idx, batch in enumerate(self.outputs):
            for i in range(len(batch.gt_target)):
                if item_index > self.MAX_EXAMPLES_TO_PRINT:
                    self.verbose = False
                metric_args: MetricArgsItem = get_input_data(self.device, batch, idx = i)

                model_acc_vec[item_index] = (metric_args.gt_target == metric_args.model_pred_origin).int().cpu().item()

                if self.verbose:
                    if len(metric_args.item_data[INPUT_IDS_NAME]) != 1:
                        raise ValueError("eval - loop over items issue")
                    example_txt = self.conv_input_ids_to_txt(metric_args.item_data[INPUT_IDS_NAME][0])
                    self.examples.append(
                        dict(stage = self.stage, txt = example_txt, gt_target = metric_args.gt_target.item(),
                             model_pred_origin = metric_args.model_pred_origin.item(), type = None,
                             perturbation_steps = -1,
                             item_index = self.item_index))

                model_preds, model_pred_hit = self.perturbation_test_item(item_data = metric_args.item_data,
                                                                          tokens_attr = metric_args.tokens_attr,
                                                                          target = metric_args.model_pred_origin,
                                                                          gt_target = metric_args.gt_target,
                                                                          model_pred_origin = metric_args.model_pred_origin)
                pertu_acc_mat[item_index] = model_pred_hit
                item_index += 1

        auc_res, steps_res = self.get_auc(num_correct_pertub = pertu_acc_mat, num_correct_model = model_acc_vec)
        # return None as item for the non fine tune case
        return auc_res, steps_res, None

    def perturbation_test_item(self, item_data, tokens_attr, target, gt_target, model_pred_origin):
        spec_tokens_indices = self.get_spec_tokens_indices(item_data)

        pertu_data = self.get_perturbated_data(tokens_attr = tokens_attr, item_data = item_data,
                                               spec_tokens_indices = spec_tokens_indices)

        if self.verbose:
            for example_idx, input_ids_example in enumerate(pertu_data.input_ids_mat):
                example_txt = self.conv_input_ids_to_txt(input_ids_example)
                self.examples.append(dict(stage = self.stage, txt = example_txt, gt_target = gt_target,
                                          model_pred_origin = model_pred_origin.item(),
                                          perturbation_steps = self.pertu_steps[example_idx].item(),
                                          type = ExpArgs.eval_metric,
                                          item_index = self.item_index))

        masked_input_ids = self.model_seq_cls_merge_inputs_mat(pertu_data.input_ids_mat, item_data[TASK_PROMPT_KEY],
                                                               item_data[LABEL_PROMPT_KEY]).to(self.device)
        out_logits = run_model(model = self.model,
                               input_ids = masked_input_ids, attention_mask = pertu_data.attention_mask_mat,
                               is_return_logits = True).squeeze()
        model_preds = torch.argmax(out_logits, dim = -1)
        model_pred_hit = (model_preds == target).int().cpu().tolist()
        model_preds = model_preds.int().cpu().tolist()
        return model_preds, model_pred_hit

    def get_auc(self, num_correct_pertub, num_correct_model) -> Union[np.float64, np.ndarray]:
        mean_accuracy_by_step = np.mean(num_correct_pertub, axis = 0)
        # mean_accuracy_by_step = np.insert(mean_accuracy_by_step, 0, np.mean(num_correct_model))
        auc = calculate_auc(mean_accuracy_by_step = mean_accuracy_by_step) * 100
        return auc, mean_accuracy_by_step

    def get_spec_tokens_indices(self, item_data):
        input_ids = item_data[INPUT_IDS_NAME].squeeze(0)
        spec_tokens_indices = torch.where(torch.isin(input_ids, self.special_tokens))
        if len(spec_tokens_indices) != 1:
            raise ValueError("spec_tokens_indices length is not 1")
        spec_tokens_indices = spec_tokens_indices[0]
        return spec_tokens_indices

    def get_perturbated_data(self, tokens_attr: Tensor, item_data: Tensor, spec_tokens_indices: Tensor):
        # POS -> remove the most important tokens first, so change the not relevant tokens to very small number
        if ExpArgs.eval_metric in [EvalMetric.NEG_AUC_WITH_REFERENCE_TOKEN.value]:
            large_num = 100
            descending = False
        elif ExpArgs.eval_metric in [EvalMetric.POS_AUC_WITH_REFERENCE_TOKEN.value]:
            large_num = -100
            descending = True
        else:
            raise ValueError("unsupported perturbation_type selected")

        item = copy.deepcopy(item_data)
        input_ids = item[INPUT_IDS_NAME].squeeze(0)
        attention_mask_mat = item[ATTENTION_MASK_NAME].tile(len(self.pertu_steps), 1)
        # token_type_ids_mat = item["token_type_ids"].tile(len(self.perturbation_steps), 1)
        special_tokens_vec = torch.zeros_like(input_ids)
        special_tokens_vec[spec_tokens_indices] = large_num
        tokens_attr_with_special_tokens_handler = special_tokens_vec + tokens_attr
        sorted_indices = torch.argsort(tokens_attr_with_special_tokens_handler, descending = descending)

        attr_shape = tokens_attr.shape[-1] - spec_tokens_indices.shape[-1]
        input_ids_mat = input_ids.tile(len(self.pertu_steps), 1)
        max_ind = int(attr_shape - 1)
        indices_to_update = [
            [i, sorted_indices[:torch.clamp(attr_shape * self.pertu_steps[i] // 100, 1, max_ind)].tolist()] for i in
            range(len(self.pertu_steps))]

        for indices in indices_to_update:
            input_ids_mat[indices] = self.ref_token_id
            if ExpArgs.eval_metric in [EvalMetric.NEG_AUC_WITH_REFERENCE_TOKEN.value,
                                       EvalMetric.POS_AUC_WITH_REFERENCE_TOKEN.value]:
                attention_mask_mat[indices] = 0


        return PerturbationData(input_ids_mat = input_ids_mat, attention_mask_mat = attention_mask_mat)

    def update_results_df(self, results_df: pd.DataFrame, auc_res: Tensor, steps_res: Tensor,
                          metric_args: Union[None, MetricArgsItem]):
        return pd.concat([results_df, pd.DataFrame([{"attr_score_function": self.stage,
                                                     "item_index": self.item_index, "task": ExpArgs.task.name,
                                                     "eval_metric": ExpArgs.eval_metric,
                                                     "explained_model_backbone": ExpArgs.explained_model_backbone,
                                                     "validation_type": ExpArgs.validation_type,
                                                     "metric_result": auc_res, "metric_steps_result": steps_res,
                                                     "steps_k": self.pertu_steps}])], ignore_index = True)

    def conv_input_ids_to_txt(self, input_ids) -> str:
        return self.explained_tokenizer.decode(input_ids)

    def run_perturbation_test(self):
        if ExpArgs.is_save_results:
            if not os.path.exists(self.experiment_path):
                os.makedirs(self.experiment_path, exist_ok = True)
            if os.path.exists(self.output_pkl_path):
                results_df = pd.read_pickle(self.output_pkl_path)
            else:
                results_df = pd.DataFrame(columns = ["stage"])
        self.model.eval()

        auc_res, steps_res, metric_args = self.all_perturbation_tests()

        if ExpArgs.is_save_results:
            results_df = self.update_results_df(results_df = results_df, auc_res = auc_res, steps_res = steps_res,
                                                metric_args = metric_args)
            results_df.to_pickle(self.output_pkl_path)
            if len(self.examples) > 0:
                examples_df = pd.DataFrame(self.examples)
                if os.path.exists(self.output_examples_path):
                    examples_old = pd.read_pickle(self.output_examples_path)
                    examples_df = pd.concat([examples_old, examples_df]).reset_index(drop = True)
                examples_df.to_pickle(self.output_examples_path)

        return auc_res

    def model_seq_cls_merge_inputs_mat(self, inputs, task_prompt_embeds, label_prompt_embeds):
        if ExpArgs.explained_model_backbone != ModelBackboneTypes.LLAMA.value:
            return inputs

        return torch.concat([task_prompt_embeds.tile(len(inputs), 1), inputs, label_prompt_embeds.tile(len(inputs), 1)],
                            dim = -1)
