from typing import List

import numpy as np
import torch
from torch import Tensor

from config.config import ExpArgs
from utils.utils_functions import is_model_encoder_only, is_use_prompt


def compute_attention_gradients(model, input_ids, attention_mask, attention_probabilities, orig_explained_model_logits):
    model.zero_grad()
    attention_probabilities.requires_grad = True
    attention_probabilities.retain_grad()
    model_output = model(input_ids = input_ids, attention_mask = attention_mask, output_attentions = True,
                         rpi_attn_prob = attention_probabilities)

    model_logits = model_output.logits
    if (not is_model_encoder_only()) and (not ExpArgs.task.is_finetuned_with_lora):
        model_logits = model_output.logits[:, -1, :][:, ExpArgs.label_vocab_tokens]
    model_logits = model_logits.squeeze()

    one_hot_target = torch.zeros_like(orig_explained_model_logits)
    one_hot_target[orig_explained_model_logits.argmax().item()] = 1
    score = torch.sum(one_hot_target * model_logits)
    score.backward(retain_graph = True)
    gradients = attention_probabilities.grad

    del model_output

    attention_probabilities.requires_grad = False
    model.zero_grad()
    torch.cuda.empty_cache()

    return gradients, model_logits


def compute_alphas_from_time_steps(model, num_time_steps):
    beta1 = 1e-4
    beta2 = 0.02

    beta_t = (beta2 - beta1) * torch.linspace(0, 1, num_time_steps + 1, device = model.device) + beta1
    alpha_t = 1 - beta_t
    alpha_cumulative = torch.cumsum(alpha_t.log(), dim = 0).exp()
    alpha_cumulative[0] = 1
    return alpha_cumulative


def apply_perturbation(inputs, step_index, noise_tensor, alpha_tensor):
    return (alpha_tensor.sqrt()[step_index, None, None, None] * inputs.cuda() + (
            1 - alpha_tensor[step_index, None, None, None]) * noise_tensor).cuda()


def interpolate_values(baseline, target, num_steps):
    delta = target - baseline
    scales = None
    if baseline.ndim == 4:
        scales = np.linspace(0, 1, num_steps + 1, dtype = np.float32)[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    if scales is None:
        raise ValueError("Error: interpolate_values function. scales is none")
    shape = (num_steps + 1,) + delta.shape
    deltas = scales * np.broadcast_to(delta.detach().numpy(), shape)
    delta_tensor = torch.tensor(deltas)

    interpolated_activations = baseline.unsqueeze(0) + delta_tensor

    if not is_model_encoder_only():
        interpolated_activations = interpolated_activations.bfloat16()

    return interpolated_activations


def run_rpi(model, input_ids, attention_mask, task_prompt_input_ids = None, label_prompt_input_ids = None,
            tokenizer = None, origin_input_ids = None):
    model.zero_grad()

    origin_explained_model_output = model(input_ids = input_ids, attention_mask = attention_mask,
                                          output_attentions = True)
    attention_probabilities = origin_explained_model_output.attentions[-ExpArgs.reverse_noise_layer_index]
    if is_model_encoder_only() or ExpArgs.task.is_finetuned_with_lora:
        orig_explained_model_logits = origin_explained_model_output.logits.squeeze().detach()
    else:
        orig_explained_model_logits = origin_explained_model_output.logits[:, -1, :][:,
                                      ExpArgs.label_vocab_tokens].squeeze().detach()
    del origin_explained_model_output

    attention_probabilities = attention_probabilities.detach()
    torch.cuda.empty_cache()
    attention_gradients, origin_explained_model_output = compute_attention_gradients(model, input_ids, attention_mask,
                                                                                     attention_probabilities,
                                                                                     orig_explained_model_logits)
    alpha_cumulative = compute_alphas_from_time_steps(model, ExpArgs.time_steps)

    all_samples_attribution_scores = []
    model.zero_grad()
    for i in range(ExpArgs.num_samples):
        if is_use_prompt():

            start_input = task_prompt_input_ids.shape[-1]
            end_input = -label_prompt_input_ids.shape[-1]

            input_noise_tensor = torch.randn_like(attention_probabilities[:, :, start_input:end_input, :])

            noise_tensor = attention_probabilities.clone().detach()

            noise_tensor[:, :, start_input:end_input, :] = input_noise_tensor


        else:
            noise_tensor = torch.randn_like(attention_probabilities)

        baseline_attention = apply_perturbation(attention_probabilities, ExpArgs.time_steps, noise_tensor,
                                                alpha_cumulative)
        integrated_attention = interpolate_values(baseline = baseline_attention.cpu(),
                                                  target = attention_probabilities.cpu(),
                                                  num_steps = ExpArgs.num_interpolation)

        gradients_list: List[Tensor] = []
        batches = np.array_split(integrated_attention, np.ceil(len(integrated_attention) / ExpArgs.batch_size))

        for batch in batches:
            squeezed_batch = batch.squeeze()
            if squeezed_batch.ndim < 4:  # batch, layers, seq, seq
                squeezed_batch = squeezed_batch.unsqueeze(0)

            batch_gradients, _ = compute_attention_gradients(model, input_ids.repeat(batch.shape[0], 1),
                                                             attention_mask.repeat(batch.shape[0], 1),
                                                             squeezed_batch.cuda(), origin_explained_model_output)
            if ExpArgs.is_gradient_x_input:
                batch_gradients = squeezed_batch.cuda() * batch_gradients
            gradients_list += batch_gradients.detach().cpu()

            torch.cuda.empty_cache()

        # TODO: rollout()
        gradients: Tensor = torch.stack(gradients_list)
        mean_gradient: Tensor = torch.mean(gradients, dim = 0)
        attribution_scores = mean_gradient * attention_probabilities.cpu()

        attribution_scores = attribution_scores.squeeze().mean(dim = 0)
        if is_model_encoder_only():
            attribution_scores = attribution_scores[0]  # cls
            attribution_scores[0] = 1  # cls
        else:
            attribution_scores = attribution_scores[-1]  # last token

        all_samples_attribution_scores.append(attribution_scores.cuda().detach())
        model.zero_grad()

    return all_samples_attribution_scores
