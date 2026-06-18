# PD&AC Discrete Action-Token Fill

## Core Idea

Use the VLA's existing token output as the action decoder. Each normalized action coordinate is discretized with the OpenVLA action-token convention, predicted by next-token cross-entropy at the PD&AC action slots, then converted back to a normalized action value by the tokenizer's bin-center lookup. The method adds no learned action head.

The fixed PD&AC scaffold supplies `K * 7` action slots, where `K = NUM_ACTIONS_CHUNK`. The same token arithmetic applies to every slot, then the result is reshaped to `(batch, K, 7)`.

## Exact Arithmetic

Let `N = 256` and let `V` be the effective action vocabulary size used by the model for de-tokenization.

Encoding follows the tokenizer:

1. Normalize by dataset quantile statistics outside the local method, then clip to `[-1, 1]`.
2. Construct `edges = np.linspace(-1, 1, N)` and `centers = (edges[:-1] + edges[1:]) / 2`, so `len(centers) = N - 1 = 255`.
3. Compute `d = np.digitize(action, edges)`, giving `d in [1, N]` after clipping.
4. Map to a token id with `token_id = V - d`.

Valid emitted action tokens are therefore `[V - N, V)`. The tokenizer's `action_token_begin_idx = V - (N + 1)` is a mask threshold; training uses `token_id > action_token_begin_idx`, which selects `[V - N, V)`.

Decoding inverts the same map:

```python
d = V - token_id
idx = np.clip(d - 1, 0, centers.shape[0] - 1)
action = centers[idx]
```

The endpoint cases are:

- `token_id = V - 1` -> `d = 1` -> `idx = 0`, the first center.
- `token_id = V - N` -> `d = N` -> clipped to `idx = N - 2`, the last center.
- `token_id = V - (N - 1)` also maps to `idx = N - 2`, so the exact upper endpoint collapses onto the last interval center, matching the canonical tokenizer.

At inference, argmax must be restricted to `[V - N, V)`. A full-vocabulary argmax can choose non-action language tokens and corrupt the de-tokenization.

## Scaffold Code

```python
"""MLS-Bench baseline: OpenVLA discrete action tokens under the PD&AC API."""

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
    """Discrete action-token path with no learned action-side parameters."""

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

## Tokenizer Arithmetic Mirrored Above

```python
class ActionTokenizer:
    def __init__(self, tokenizer, bins=256, min_action=-1, max_action=1):
        self.tokenizer, self.n_bins = tokenizer, bins
        self.bins = np.linspace(min_action, max_action, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        self.action_token_begin_idx = int(tokenizer.vocab_size - (bins + 1))

    def __call__(self, action):
        action = np.clip(action, self.bins[0], self.bins[-1])
        discretized = np.digitize(action, self.bins)
        return self.tokenizer.decode(list(self.tokenizer.vocab_size - discretized))

    def decode_token_ids_to_actions(self, action_token_ids):
        idx = self.tokenizer.vocab_size - action_token_ids
        idx = np.clip(idx - 1, 0, self.bin_centers.shape[0] - 1)
        return self.bin_centers[idx]
```

The scaffold code keeps the PD&AC machinery outside the method: prompt/image processing, bidirectional action-slot attention, LoRA trunk updates, and physical-unit un-normalization remain runtime responsibilities.
