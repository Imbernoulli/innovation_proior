**Problem.** Discrete cleared the diffusion floor (fit-efficient, single-pass) but its 256-bin grid and
unordered cross-entropy cap precision — goal_c stayed empty, geometric mean still zero. Remove the grid:
map the action-slot hidden states straight to continuous actions with a distance-respecting loss.

**Key idea (PD&AC + Cont-L1).** Keep the single-pass parallel decode and the pretrained trunk; replace
the bin-logit decode with a small MLP-ResNet head that regresses real-valued normalized actions, trained
by `L1Loss(ground_truth, predicted)`. L1 (conditional median) is robust to noisy demonstrations and more
precise than L2 (mean), and unlike discrete cross-entropy it makes "nearly right" the loss itself, so it
recovers sub-grid precision. The head consumes each timestep's D action-slot hidden states jointly:
`(B, K·D, hidden)` → `(B, K, D·hidden)` → `MLPResNet(num_blocks=2, input_dim·ACTION_DIM →
action_dim)` → `(B, K, action_dim)`. Action slots are selected by `current_action_mask |
next_actions_mask` at train time and by the contiguous post-prompt span at eval time.

**Why it is the strongest rung.** It inherits discrete's fit advantage (single trunk pass, pretrained
trunk, converges in 6000 steps — diffusion's downfall) and adds continuous precision. The 7B trunk's
capacity makes simple regression competitive with diffusion's multimodality without diffusion's training
cost or 50-step rollout. Limitation: L1 learns the median mode, so genuinely multimodal demonstrations
would favor diffusion — but LIBERO-Goal demonstrations are focused and consistent.

**Implementation note.** The literal baseline is `edits/cont_l1.edit.py`: it forwards with *zeroed*
`action_token_features` (a placeholder, no action-conditioned injection — unlike diffusion's projected
noisy chunk), reuses the scaffold's `MLPResNet`, and reads slots via the `extract_action_hidden_states`
hook — the parallel-decode / chunking machinery is the fixed trunk, not part of this edit.

**Hyperparameters.** `CONFIG_OVERRIDES`: `lr_warmup_steps = 400`, `num_steps_before_decay = 6000`.
`MLPResNet(num_blocks=2, input_dim = 4096·7, hidden_dim = 4096, output_dim = 7)`; `ACTION_DIM = 7`,
`NUM_ACTIONS_CHUNK = 8`.

```python
"""MLS-Bench baseline: official OpenVLA-OFT PD&AC + Cont-L1 path."""

from __future__ import annotations

import torch
import torch.nn as nn

from mlsbench.action_method_runtime import (
    EvalActionMethodRuntime,
    ForwardPassResult,
    TrainActionMethodRuntime,
)
from prismatic.models.action_heads import ACTION_DIM, NUM_ACTIONS_CHUNK, MLPResNet


CONFIG_OVERRIDES = {
    "lr_warmup_steps": 400,
    "num_steps_before_decay": 6000,
}


class OfficialL1ActionMethod(nn.Module):
    """Official Cont-L1 baseline under the unified action-method API."""

    def __init__(self, input_dim=4096, hidden_dim=4096, action_dim=7):
        super().__init__()
        self.action_dim = action_dim
        self.model = MLPResNet(
            num_blocks=2,
            input_dim=input_dim * ACTION_DIM,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
        )

    def extract_action_hidden_states(
        self,
        runtime: TrainActionMethodRuntime | EvalActionMethodRuntime,
        forward: ForwardPassResult,
    ) -> torch.Tensor:
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

    def predict_action(self, actions_hidden_states):
        batch_size = actions_hidden_states.shape[0]
        rearranged_hidden_states = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
        return self.model(rearranged_hidden_states)

    def zero_action_token_features(
        self,
        runtime: TrainActionMethodRuntime | EvalActionMethodRuntime,
    ) -> torch.Tensor:
        return torch.zeros(
            runtime.batch_size,
            NUM_ACTIONS_CHUNK * ACTION_DIM,
            runtime.model.llm_dim,
            device=runtime.device,
            dtype=runtime.hidden_dtype,
        )

    def training_step(self, runtime: TrainActionMethodRuntime):
        forward = runtime.forward(action_token_features=self.zero_action_token_features(runtime))
        actions_hidden_states = self.extract_action_hidden_states(runtime, forward)
        predicted_actions = self.predict_action(actions_hidden_states)
        loss = torch.nn.L1Loss()(runtime.ground_truth_actions, predicted_actions)
        metrics = {"loss_value": float(loss.item())}
        metrics.update(runtime.compute_action_l1_metrics(predicted_actions))
        return loss, metrics

    def predict_actions(self, runtime: EvalActionMethodRuntime):
        forward = runtime.forward(action_token_features=self.zero_action_token_features(runtime))
        actions_hidden_states = self.extract_action_hidden_states(runtime, forward)
        normalized_actions = self.predict_action(actions_hidden_states)
        return normalized_actions, actions_hidden_states


CustomActionMethod = OfficialL1ActionMethod
```
