## Research question

Design a better **action method** for the OpenVLA-OFT PD&AC scaffold on LIBERO-Goal. An "action
method" here is the full local algorithm that turns PD&AC action-token hidden states into a chunk of
robot actions: a continuous regression head, a diffusion noise predictor with its noisy-action
projector and sampler, a discrete action-token mapper, or any other local decoding logic. Everything
around the action method is frozen — the OpenVLA-7B trunk, parallel decoding and action chunking
(PD&AC), LoRA fine-tuning, and the paper-aligned LIBERO-Goal protocol — so the only thing being
designed is the localized decoder that reads the trunk's action-slot hidden states and emits a
normalized action chunk. The score is the geometric mean of `success_rate` over three disjoint
LIBERO-Goal subsets, evaluated at a fixed 6000-step budget chosen to keep the run inside the 5 H200h
MLS-Bench validity constraint.

## Prior art before the first rung (vision-language-action lineage)

The trunk every rung shares — a 7B vision-language model fine-tuned to emit robot actions — is itself
the resolution of a line of work. These are the ancestors the ladder reacts to; the fixed substrate
below is what they converged to.

- **End-to-end imitation policies (e.g. ACT, Zhao et al. 2023).** A from-scratch transformer reads a
  handful of fixed per-timestep query embeddings and emits a *chunk* of K future actions
  non-causally in one pass, trained with an L1 reconstruction loss. Action chunking shrinks the
  effective horizon and curbs compounding error; L1 (median) is more precise than L2 (mean) on noisy
  demonstrations. Gap: trained on a few hundred trajectories per task, it has no Internet-scale prior
  and generalizes poorly over novel objects, distractors, and rephrased instructions.
- **Diffusion Policy (Chi et al. 2023).** Replaces the point-estimate head with a conditional
  denoising-diffusion model over the action chunk: sample from a multimodal action distribution by
  learning to reverse a fixed Gaussian corruption, conditioned on the observation. Captures genuinely
  multimodal behaviour a point estimator averages away. Gap: many sequential denoising steps at
  inference, slow convergence in training, and still a from-scratch policy with no web prior.
- **Vision-language-action models (RT-2 / OpenVLA, Brohan et al. 2023; Kim et al. 2024).** Take a
  web-pretrained vision-language model and teach it to *speak actions* as discrete tokens — quantize
  each of the 7 action dimensions into 256 bins, name each bin with one of the model's least-used
  vocabulary tokens, and fine-tune under the model's native next-token cross-entropy. Inherits the
  Internet-scale generalization the from-scratch policies lacked. Gap: autoregressive, one token at a
  time under a causal mask — K·D sequential decoder passes for a chunk, 3-5 Hz, far below a real-time
  controller, and the 256-bin grid caps action precision.
- **Parallel decoding + action chunking (PD&AC).** The fixed substrate's central move: action
  coordinates have no intrinsic left-to-right order, so fill the K·D action positions with empty,
  positionally-marked slots, swap the causal mask for bidirectional attention over them, and predict
  the whole chunk in a *single* forward pass. This collapses the K·D factor to one, makes chunking
  essentially free, and opens the door to continuous heads on top of the action-slot hidden states.

## The fixed substrate

A PD&AC OpenVLA-OFT loop is frozen and must not be touched. Base model `/models/openvla-7b`; LoRA
rank 32 on the trunk; train data `/data/rlds/libero_goal_no_noops`; two camera views (third-person +
wrist) and proprioception enabled; FiLM disabled; batch size 8, gradient accumulation 4, learning
rate `5e-4`; max 6000 steps. The trunk fuses vision, language, and proprioception under bidirectional
attention over the K·D action slots and returns final-layer hidden states; the runtime owns prompt
and image preparation, the multimodal forward pass, dataset action un-normalization, and the metric
helpers. The action-method constants are fixed: `ACTION_DIM = 7` (the 7-DoF delta end-effector
command), `NUM_ACTIONS_CHUNK = 8` (timesteps decoded per query and executed open-loop), and the LLM
hidden width `4096`.

The loop also provides, through the two runtime objects, the helpers a method may use:
`runtime.forward(...)` (the trunk pass, optionally given `action_token_features` and
`diffusion_timestep_embeddings`); `runtime.phase` (`"train"`/`"eval"`); `runtime.batch_size`,
`runtime.device`, `runtime.hidden_dtype`; at train time `runtime.ground_truth_actions`,
`runtime.current_action_mask`, `runtime.next_actions_mask`, `runtime.should_compute_aux_metrics`,
`runtime.compute_discrete_objective(...)`, `runtime.compute_action_l1_metrics(...)`; at eval time
`runtime.num_prompt_tokens`; and `runtime.model.{vocab_size, bin_centers, llm_dim}`.

## The editable interface

Exactly one region is editable — the `CustomActionMethod` class (plus its helper modules and a narrow
`CONFIG_OVERRIDES`) in `openvla-oft/mlsbench/custom_pdac_method.py`. Every method on the ladder is a
fill of this same contract: `training_step(runtime)` returns `(loss, metrics)`;
`predict_actions(runtime)` returns `(normalized_actions, aux)`; and the method-adjacent plumbing hooks
`extract_action_hidden_states(...)`, `build_action_token_features(...)`, and
`decode_discrete_actions(...)` live inside the editable region too, so a method can change how
action-slot hidden states are selected, how action-conditioned inputs are injected, and how
logits/hidden states map back to normalized actions. `CONFIG_OVERRIDES` may only set
`learning_rate`, `lr_warmup_steps`, `num_steps_before_decay`; everything else is fixed protocol. The
harness instantiates `CustomActionMethod` and reads `(normalized_actions, _)` back, then
un-normalizes outside the method.

The starting point is the scaffold default: a temporal-residual L1-style example. Each rung below
replaces this whole editable region with its own method.

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

The trained policy is evaluated on three disjoint LIBERO-Goal task subsets — `goal_a` (task ids
0,1,2), `goal_b` (3,4,5), and `goal_c` (6,7,8,9, the hidden split) — at the fixed 6000-step
checkpoint, with 3 eval trials per task and a single seed (42). The metric on each subset is
`success_rate` (higher is better); the final score is the geometric mean over the three subset
success rates. Each baseline is also tagged by its training wall-clock (`elapsed_train`) and per-eval
elapsed time, since under the 5 H200h budget validity is part of the comparison.
