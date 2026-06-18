**Problem.** In the 6000-step budget the deciding factor is whether the head can be *fit*, not how
expressive it is — diffusion's denoiser was under-trained and sat near the floor. The most
data-efficient signal available on this trunk is the model's own pretrained objective. Keep the trunk's
mouth and its next-token cross-entropy; only teach it to speak actions as tokens.

**Key idea (PD&AC discrete).** Quantize each of the 7 action dimensions into 256 bins (1st/99th-quantile
clipped, uniform within), name each bin with one of the model's last 256 (least-used) vocabulary
tokens via the self-inverse map `token_id = vocab_size − bin_index` — zero new parameters — and train
by the trunk's native cross-entropy masked to the action positions (`runtime.compute_discrete_objective`).
Cross-entropy over unordered bins represents multimodal actions natively, but rides the *pretrained*
objective, so it has a head start diffusion lacked. Decode greedily: restrict the argmax to the action
sub-vocabulary `[vocab_size − num_bins, vocab_size)`, add the offset back, invert the map, clip the top
index into the last real interval (255 centers for 256 edges), and read off `bin_centers` — the
minimum-distortion reconstruction. The method holds no learnable parameters; the action-feature slots
are zeros.

**Why above diffusion, below continuous-L1.** One trunk pass per chunk (no 50-step rollout) and a
pretrained objective fit in 6000 steps, so success clears the diffusion floor and eval times collapse
to the single-pass range. But the 8-bit grid caps precision and cross-entropy ignores the ordinal
structure of a continuous action, leaving a ceiling a direct continuous regressor should close.

**Same-named-vs-paper note.** This task's `decode_discrete_actions` argmaxes only over the action
sub-vocabulary and validates the vocab bounds — unlike the scaffold default's full-vocabulary argmax,
which could select a non-action token. The literal baseline below is `edits/pdac_discrete.edit.py`.

**Hyperparameters.** `CONFIG_OVERRIDES`: `lr_warmup_steps = 400`, `num_steps_before_decay = 6000`.
`num_action_bins = bin_centers.shape[0] + 1 = 256`; `ACTION_DIM = 7`, `NUM_ACTIONS_CHUNK = 8`.

```python
"""MLS-Bench baseline: official OpenVLA-OFT PD&AC discrete path."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from mlsbench.action_method_runtime import EvalActionMethodRuntime, ForwardPassResult, TrainActionMethodRuntime
from prismatic.models.action_heads import ACTION_DIM, NUM_ACTIONS_CHUNK


CONFIG_OVERRIDES = {
    "lr_warmup_steps": 400,
    "num_steps_before_decay": 6000,
}


class OfficialDiscreteActionMethod(nn.Module):
    """Official discrete PD&AC baseline under the unified action-method API."""

    def __init__(self, input_dim=4096, hidden_dim=4096, action_dim=7):
        super().__init__()

    def decode_discrete_actions(
        self,
        runtime: EvalActionMethodRuntime,
        forward: ForwardPassResult,
    ) -> torch.Tensor:
        start = runtime.num_prompt_tokens
        end = start + ACTION_DIM * NUM_ACTIONS_CHUNK
        action_logits = forward.text_logits[:, start:end]
        num_action_bins = int(runtime.model.bin_centers.shape[0] + 1)
        action_vocab_end = int(runtime.model.vocab_size)
        action_vocab_start = action_vocab_end - num_action_bins
        if action_vocab_start < 0 or action_vocab_end > action_logits.shape[-1]:
            raise ValueError(
                f"Invalid action vocab bounds [{action_vocab_start}, {action_vocab_end}) "
                f"for logits with vocab size {action_logits.shape[-1]}"
            )
        predicted_token_ids = (
            action_logits[..., action_vocab_start:action_vocab_end].argmax(dim=2)
            + action_vocab_start
        ).cpu().numpy()
        discretized_actions = runtime.model.vocab_size - predicted_token_ids
        discretized_actions = np.clip(
            discretized_actions - 1,
            a_min=0,
            a_max=runtime.model.bin_centers.shape[0] - 1,
        )
        normalized_actions = runtime.model.bin_centers[discretized_actions].reshape(
            runtime.batch_size,
            NUM_ACTIONS_CHUNK,
            ACTION_DIM,
        )
        return torch.from_numpy(normalized_actions).to(runtime.device, dtype=torch.float32)

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
        return runtime.compute_discrete_objective(forward)

    def predict_actions(self, runtime: EvalActionMethodRuntime):
        forward = runtime.forward(action_token_features=self.zero_action_token_features(runtime))
        normalized_actions = self.decode_discrete_actions(runtime, forward)
        return normalized_actions, None


CustomActionMethod = OfficialDiscreteActionMethod
```
