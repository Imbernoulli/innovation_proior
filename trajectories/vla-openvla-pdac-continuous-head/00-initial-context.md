## Research question

Design a better **action method** for the OpenVLA-OFT PD&AC scaffold on LIBERO-Goal. An action method is the local decoder that reads the trunk's action-slot hidden states and emits a normalized chunk of robot actions. Everything outside it is frozen: the OpenVLA-7B trunk, parallel decoding and action chunking (PD&AC), LoRA fine-tuning, and the LIBERO-Goal protocol. The score is the geometric mean of `success_rate` over three disjoint LIBERO-Goal subsets, evaluated at a fixed 6000-step budget.

## Prior art / Background / Baselines

- **End-to-end imitation policies (e.g. ACT).** A from-scratch transformer predicts a chunk of future actions non-causally from per-timestep query embeddings, trained with L1 reconstruction. Gap: trained on a few hundred trajectories per task, it lacks an Internet-scale prior and generalizes poorly to novel objects, distractors, and rephrased instructions.
- **Diffusion Policy.** Replaces the point-estimate head with a conditional denoising-diffusion model over the action chunk. Gap: many sequential denoising steps at inference, slow training convergence, and still no web prior.
- **Vision-language-action models (RT-2 / OpenVLA).** Fine-tune a web-pretrained vision-language model to emit actions as discrete tokens by quantizing each action dimension into 256 bins. Gap: autoregressive decoding produces K·D sequential passes per chunk at 3-5 Hz, and the 256-bin grid limits action precision.
- **Parallel decoding + action chunking (PD&AC).** Fills the K·D action positions with positionally-marked slots and applies bidirectional attention to predict the whole chunk in one forward pass. This collapses the K·D sequential factor and is the fixed substrate here.

## Fixed substrate / Code framework

A PD&AC OpenVLA-OFT loop is frozen and must not be touched. Base model `/models/openvla-7b`; LoRA rank 32 on the trunk; train data `/data/rlds/libero_goal_no_noops`; two camera views (third-person + wrist) and proprioception enabled; FiLM disabled; batch size 8, gradient accumulation 4, learning rate `5e-4`; max 6000 steps. The trunk fuses vision, language, and proprioception under bidirectional attention over the K·D action slots and returns final-layer hidden states.

Action-method constants: `ACTION_DIM = 7`, `NUM_ACTIONS_CHUNK = 8`, LLM hidden width `4096`.

Available runtime helpers include: `runtime.forward(...)` (trunk pass, optionally given `action_token_features` and `diffusion_timestep_embeddings`); `runtime.phase` (`"train"`/`"eval"`); `runtime.batch_size`, `runtime.device`, `runtime.hidden_dtype`; at train time `runtime.ground_truth_actions`, `runtime.current_action_mask`, `runtime.next_actions_mask`, `runtime.should_compute_aux_metrics`, `runtime.compute_discrete_objective(...)`, `runtime.compute_action_l1_metrics(...)`; at eval time `runtime.num_prompt_tokens`; and `runtime.model.{vocab_size, bin_centers, llm_dim}`.

## Editable interface

Only one region is editable: the `CustomActionMethod` class (plus helper modules and a narrow `CONFIG_OVERRIDES`) in `openvla-oft/mlsbench/custom_pdac_method.py`. The contract is `training_step(runtime)` returning `(loss, metrics)`, and `predict_actions(runtime)` returning `(normalized_actions, aux)`. Hooks for `extract_action_hidden_states(...)`, `build_action_token_features(...)`, and `decode_discrete_actions(...)` also live in the editable region. `CONFIG_OVERRIDES` may only set `learning_rate`, `lr_warmup_steps`, and `num_steps_before_decay`. The harness instantiates `CustomActionMethod`, reads `(normalized_actions, _)`, and un-normalizes outside the method.

The starting point is the scaffold default: a temporal-residual L1-style example.

```python
# EDITABLE region of custom_pdac_method.py - default scaffold fill
import numpy as np
import torch
import torch.nn as nn

from mlsbench.action_method_runtime import (
    EvalActionMethodRuntime,
    ForwardPassResult,
    TrainActionMethodRuntime,
)
from prismatic.models.action_heads import ACTION_DIM, NUM_ACTIONS_CHUNK, MLPResNet


CONFIG_OVERRIDES = {}


class TemporalResidualActionMethod(nn.Module):
    """Default example: a temporal mixer + MLP-ResNet regressor trained by L1."""

    def __init__(self, input_dim=4096, hidden_dim=4096, action_dim=7):
        super().__init__()
        self.action_dim = action_dim
        self.temporal_mixer = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.decoder = MLPResNet(
            num_blocks=2,
            input_dim=input_dim * ACTION_DIM,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
        )

    def extract_action_hidden_states(self, runtime, forward: ForwardPassResult) -> torch.Tensor:
        if runtime.phase == "train":
            mask = runtime.current_action_mask | runtime.next_actions_mask
            return (
                forward.text_hidden_states[mask]
                .reshape(runtime.batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
                .to(runtime.hidden_dtype)
            )
        start = runtime.num_prompt_tokens
        end = start + NUM_ACTIONS_CHUNK * ACTION_DIM
        return forward.text_hidden_states[:, start:end, :].to(runtime.hidden_dtype)

    def _predict_normalized_actions(self, actions_hidden_states):
        batch_size = actions_hidden_states.shape[0]
        mixed = actions_hidden_states + self.temporal_mixer(actions_hidden_states)
        mixed = mixed.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
        return self.decoder(mixed)

    def training_step(self, runtime: TrainActionMethodRuntime):
        forward = runtime.forward()
        actions_hidden_states = self.extract_action_hidden_states(runtime, forward)
        predicted_actions = self._predict_normalized_actions(actions_hidden_states)
        loss = torch.nn.L1Loss()(runtime.ground_truth_actions, predicted_actions)
        metrics = {"loss_value": float(loss.item())}
        metrics.update(runtime.compute_action_l1_metrics(predicted_actions))
        return loss, metrics

    def predict_actions(self, runtime: EvalActionMethodRuntime):
        forward = runtime.forward()
        actions_hidden_states = self.extract_action_hidden_states(runtime, forward)
        normalized_actions = self._predict_normalized_actions(actions_hidden_states)
        return normalized_actions, actions_hidden_states


CustomActionMethod = TemporalResidualActionMethod
```

## Evaluation settings

The trained policy is evaluated on three disjoint LIBERO-Goal subsets — `goal_a` (task ids 0,1,2), `goal_b` (3,4,5), and `goal_c` (6,7,8,9, the hidden split) — at the fixed 6000-step checkpoint, with 3 eval trials per task and seed 42. The metric on each subset is `success_rate` (higher is better); the final score is the geometric mean over the three subset success rates. Each baseline is also tagged by training wall-clock (`elapsed_train`) and per-eval elapsed time, since staying inside the 5 H200h budget validity constraint is part of the comparison.
