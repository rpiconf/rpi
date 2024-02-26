from typing import List

import numpy as np
import torch
from torch import Tensor

from config.config import ExpArgs
from config.types_enums import ModelBackboneTypes


def get_attn_grad(model, input_ids, attention_mask, attn_prob, orig_output_logits):
    # model.zero_grad()
    attn_prob.retain_grad()
    out = model(input_ids = input_ids, attention_mask = attention_mask, output_attentions = True,
                rpi_attn_prob = attn_prob)

    out_logits = out.logits
    if ExpArgs.explained_model_backbone == ModelBackboneTypes.LLAMA.value:
        out_logits = out.logits[:, -1, :][:, ExpArgs.labels_tokens_opt]
    out_logits = out_logits.squeeze()

    one_hot = torch.zeros_like(orig_output_logits)
    one_hot[orig_output_logits.argmax().item()] = 1
    score = torch.sum(one_hot * out_logits)
    score.backward(retain_graph = True)
    g = attn_prob.grad

    model.zero_grad()
    del out
    torch.cuda.empty_cache()

    return g, out_logits


def get_alphas_from_timestamp(model, tsteps):
    beta1 = 1e-4
    beta2 = 0.02

    b_t = (beta2 - beta1) * torch.linspace(0, 1, tsteps + 1, device = model.device) + beta1
    a_t = 1 - b_t
    ab_t = torch.cumsum(a_t.log(), dim = 0).exp()
    ab_t[0] = 1
    return ab_t


def perturb_input(x, step, noise, ab_t):
    return (ab_t.sqrt()[step, None, None, None] * x.cuda() + (1 - ab_t[step, None, None, None]) * noise).cuda()


def get_interpolated_values(baseline, target, num_steps):
    """this function returns a list of all the images interpolation steps."""
    if num_steps <= 0:
        return np.array([])
    if num_steps == 1:
        return torch.stack([baseline, target])

    delta = target - baseline

    if baseline.ndim == 3:
        scales = np.linspace(0, 1, num_steps + 1, dtype = np.float32)[:, np.newaxis, np.newaxis,
                 np.newaxis]  # newaxis = unsqueeze
    elif baseline.ndim == 4:
        scales = np.linspace(0, 1, num_steps + 1, dtype = np.float32)[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    elif baseline.ndim == 5:
        scales = np.linspace(0, 1, num_steps + 1, dtype = np.float32)[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis,
                 np.newaxis]
    elif baseline.ndim == 6:
        scales = np.linspace(0, 1, num_steps + 1, dtype = np.float32)[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis,
                 np.newaxis, np.newaxis]

    shape = (num_steps + 1,) + delta.shape
    deltas = scales * np.broadcast_to(delta.detach().numpy(), shape)
    interpolated_activations = baseline.unsqueeze(0) + torch.tensor(deltas)

    # interpolated_activations.detach_()
    return interpolated_activations


def run_rpi(model, input_ids, attention_mask):
    model.zero_grad()
    # torch.cuda.empty_cache()
    orig_output = model(input_ids = input_ids, attention_mask = attention_mask, output_attentions = True)
    attn_prob = orig_output.attentions[-1]
    if ExpArgs.explained_model_backbone == ModelBackboneTypes.LLAMA.value:
        orig_output_logits = orig_output.logits[:, -1, :][:, ExpArgs.labels_tokens_opt].squeeze()
    else:
        orig_output_logits = orig_output.logits.squeeze()
    del orig_output

    attn_grad, orig_output = get_attn_grad(model, input_ids, attention_mask, attn_prob, orig_output_logits)
    ab_t = get_alphas_from_timestamp(model, ExpArgs.time_steps)

    attr_scores = []
    model.zero_grad()
    for i in range(ExpArgs.n_samples):

        noise_tensor = torch.randn_like(attn_prob)
        baseline_attention = perturb_input(attn_prob, ExpArgs.time_steps, noise_tensor, ab_t)
        integrated_attn = get_interpolated_values(baseline_attention.cpu(), attn_prob.cpu(),
                                                  num_steps = ExpArgs.n_interpolation)
        if (ExpArgs.explained_model_backbone == ModelBackboneTypes.LLAMA.value) and ExpArgs.llama_f16:
            integrated_attn = integrated_attn.half()

        gradients: List[Tensor] = []
        for ia in integrated_attn:
            g, _ = get_attn_grad(model, input_ids, attention_mask, ia.cuda(), orig_output)
            if ExpArgs.is_g_x_inp:
                g = ia.cuda() * g
            gradients.append(g.detach().cpu())

        # rollout()
        mean_grad = torch.mean(torch.stack(gradients), axis = 0)
        attr_score = mean_grad * attn_prob.cpu()

        attr_score = attr_score.squeeze().mean(axis = 0)
        if ExpArgs.explained_model_backbone == ModelBackboneTypes.LLAMA.value:
            attr_score = attr_score[-1]  # decoder first token
        elif ExpArgs.explained_model_backbone in [ModelBackboneTypes.BERT.value, ModelBackboneTypes.ROBERTA.value]:
            attr_score = attr_score[0]  # cls
            attr_score[0] = 1  # cls
        else:
            raise ValueError(f"unsupported model backbone")
        attr_scores.append(attr_score.detach())
        model.zero_grad()

    return torch.stack(attr_scores)
