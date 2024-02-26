import gc
import os

import torch

from config.config import BackbonesMetaData, ExpArgs
from config.constants import (LABELS_NAME, INPUT_IDS_NAME, ATTENTION_MASK_NAME, TASK_PROMPT_KEY, LABEL_PROMPT_KEY,
                              LABEL_PROMPT, TEXT_PROMPT)
from config.types_enums import EvalMetric, ModelBackboneTypes
from evaluations.evaluations import evaluate_tokens_attr
from main.model_utils import get_dataset, get_model_tokenizer, get_ref_token_name, get_folder_name
from main.rpi_utils import run_rpi
from utils.consts import AttrScoresFunctionsNames
from utils.dataclasses.trainer_outputs import DataForEval
from utils.utils_functions import run_model, init_experiment, get_device


class Rpi:
    def __init__(self, exp_name: str):
        init_experiment()
        self.task = ExpArgs.task
        self.exp_name = exp_name
        self.target = -1
        self.model, self.tokenizer = get_model_tokenizer(self.task)
        self.explainer = AttrScoresFunctionsNames.rpi
        self.task_prompt_input_ids = None
        self.label_prompt_input_ids = None
        self.label_prompt_input_ids_squeezed = None
        self.set_prompts()

    def run(self):
        device = get_device()
        self.model = self.model.to(device)
        self.model.eval()

        data = get_dataset(self.task)
        ref_token = get_ref_token_name(self.tokenizer)

        for metric in EvalMetric:
            result_path = get_folder_name(self.exp_name, self.explainer.name, metric)
            os.makedirs(result_path, exist_ok = True)

        # Compute attributions
        for i, row in enumerate(data):
            label = row[1]
            txt = row[0]

            if ExpArgs.explained_model_backbone == ModelBackboneTypes.LLAMA.value:
                input_ids, attention_mask = self.encode(txt, False)
                origin_inputs_ids = torch.concat([self.task_prompt_input_ids, input_ids, self.label_prompt_input_ids],
                                                 dim = 1)
                attention_mask = None
            else:
                input_ids, attention_mask = self.encode(txt, True)
                origin_inputs_ids = input_ids
            pred_origin_logits = run_model(model = self.model, input_ids = origin_inputs_ids,
                                           attention_mask = attention_mask, is_return_logits = True)
            model_pred_origin = torch.argmax(pred_origin_logits, dim = 1)
            self.target = model_pred_origin
            self.model.zero_grad()

            all_attr_scores = run_rpi(self.model, origin_inputs_ids, attention_mask)
            attr_model = AttrScoresFunctionsNames.rpi.name

            default_val = torch.tensor(0)

            if ExpArgs.explained_model_backbone == ModelBackboneTypes.LLAMA.value:
                attention_mask = torch.full(input_ids.shape, float('nan'))
            for attr_score in all_attr_scores:
                eval_attr_score = attr_score
                if ExpArgs.explained_model_backbone == ModelBackboneTypes.LLAMA.value:
                    eval_attr_score = attr_score[
                                      self.task_prompt_input_ids.shape[-1]:-self.label_prompt_input_ids.shape[
                                          -1]].detach()
                outputs: DataForEval = DataForEval(tokens_attr = eval_attr_score.unsqueeze(0), input = {  #
                    INPUT_IDS_NAME: input_ids,  #
                    ATTENTION_MASK_NAME: attention_mask,  #
                    LABELS_NAME: torch.tensor([label]),  #
                    TASK_PROMPT_KEY: self.task_prompt_input_ids,  #
                    LABEL_PROMPT_KEY: self.label_prompt_input_ids_squeezed  #
                }, loss = default_val, pred_loss = default_val, pred_loss_mul = default_val,
                                                   tokens_attr_sparse_loss = default_val,
                                                   pred_origin = model_pred_origin,
                                                   pred_origin_logits = pred_origin_logits,
                                                   tokens_attr_sparse_loss_mul = default_val,
                                                   gt_target = torch.tensor([label]))

                for metric in EvalMetric:
                    experiment_path = get_folder_name(self.exp_name, attr_model, metric)
                    ExpArgs.eval_metric = metric.value
                    evaluate_tokens_attr(self.model, self.tokenizer, ref_token, [outputs], stage = attr_model,
                                         experiment_path = experiment_path, verbose = ExpArgs.verbose, item_index = i,
                                         is_sequel = True)
                    gc.collect()
                    torch.cuda.empty_cache()

    def set_prompts(self):
        if ExpArgs.explained_model_backbone == ModelBackboneTypes.LLAMA.value:
            task_prompt = "\n\n".join([self.task.llama_task_prompt, self.task.llama_few_shots_prompt, TEXT_PROMPT])
            self.task_prompt_input_ids, task_prompt_attention_mask = self.encode(task_prompt, True)
            self.label_prompt_input_ids, label_prompt_attention_mask = self.encode(LABEL_PROMPT, False)
            self.label_prompt_input_ids_squeezed = self.label_prompt_input_ids.squeeze()
            labels_tokens = [self.tokenizer.encode(str(l), return_tensors = "pt", add_special_tokens = False) for l in
                             list(ExpArgs.task.labels_int_str_maps.keys())]

            ExpArgs.labels_tokens_opt = torch.stack(labels_tokens).squeeze()[:, -1]

    def encode(self, new_txt, is_add_special_tokens):
        tokenized = self.tokenizer.encode_plus(new_txt, truncation = True, add_special_tokens = is_add_special_tokens,
                                               return_tensors = "pt")
        input_ids = tokenized.input_ids.cuda()
        attention_mask = tokenized.attention_mask.cuda()
        return input_ids, attention_mask
