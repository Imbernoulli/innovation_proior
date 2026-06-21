The slot I have to fill is narrow: inside an otherwise fixed OpenVLA-OFT parallel-decoding-and-action-chunking stack, I must supply only the local decoder/training rule that turns the trunk's action-slot outputs into a normalized action chunk of shape $(\text{batch}, K, D)$, with $D = 7$ end-effector delta dimensions and $K = 8$ future timesteps. The trunk is a fixed 7B vision-language model that already emits language-model hidden states and logits at the empty action positions; the runtime around it already owns prompt construction, image processing, the forward pass, un-normalization, and metric logging. So the real question is what to do with logits at $K \times 7$ action slots, under a tight LIBERO-Goal fine-tuning budget, without disturbing anything outside the method. The obvious richer options all fight the substrate. A diffusion action head models continuous distributions well but adds a denoising loop and a separate objective. A direct continuous regression head changes the output interface and introduces fresh learned decoder parameters that must be trained from scratch under a single seed and a limited step count. Both of these throw away the very thing the runtime hands me for free: logits over the language vocabulary at every action slot, plus a helper that computes next-token cross-entropy. The web-pretrained VLM already has a mouth; the right fill uses it rather than bolting on a second one.

The method I propose is the PD&AC discrete action-token fill: treat each normalized action coordinate as a categorical symbol the trunk already knows how to predict, train it with the native next-token cross-entropy at the action slots, and reconstruct the continuous value by the tokenizer's bin-center lookup — adding no learned action head at all. What makes this work, and what makes it fragile if done carelessly, is that the quantization must mirror the released OpenVLA action tokenizer exactly, off-by-one and all. Each action dimension is first normalized through stored quantile statistics into $[-1, 1]$ outside this method, then clipped to that interval. With $N = 256$, the tokenizer builds $\text{edges} = \text{linspace}(-1, 1, N)$ and $\text{centers} = (\text{edges}_{:-1} + \text{edges}_{1:})/2$. This is the treacherous part: there are $256$ edge values but only $255$ centers. The symbol count stays $256$ because $\text{digitize}$ can return an endpoint index in addition to the interior interval indices. After clipping, the value $-1$ digitizes to $d = 1$ and the value $+1$ digitizes to $d = N$, so $d \in [1, N]$, and the token id is

$$\text{token\_id} = V - d,$$

where $V$ is the effective action vocabulary size used for de-tokenization. The minimum endpoint maps to $V-1$, the maximum endpoint to $V-N$, and every valid action token lives in the half-open block $[V-N, V)$. The tokenizer's $\text{action\_token\_begin\_idx} = V - (N+1)$ is only a mask threshold: training selects labels by $\text{token\_id} > \text{action\_token\_begin\_idx}$, which is exactly $[V-N, V)$ and deliberately excludes the extra id $V-(N+1)$.

Decoding runs that same subtraction backwards. Given a predicted token $t$, the digitized value is $d = V - t$, and the zero-based center index is $d - 1$. The lower endpoint is clean: $t = V-1$ gives $d=1$ and index $0$, the first center. The upper endpoint would overflow: $t = V-N$ gives $d = N$ and index $N-1$, but the last valid center index is $N-2$. So the canonical decode clips $d-1$ into $[0, \text{centers.shape}[0]-1] = [0, N-2]$:

$$d = V - t, \qquad \text{idx} = \mathrm{clip}(d-1,\, 0,\, N-2), \qquad a = \text{centers}[\text{idx}].$$

The consequence is that the exact-max token $V-N$ and the last ordinary interval token $V-(N-1)$ both reconstruct to the last center — the two collapse onto one value. I am not inventing that collapse; it is the released tokenizer's own edge handling, and preserving it bit-for-bit is the entire point of a scaffold fill, because any cleverer rounding here would silently desynchronize from the dataset statistics and the trunk's pretraining.

The second load-bearing choice is where the argmax is allowed to look. The action slots produce logits over the whole language vocabulary, but only the tail block $[V-N, V)$ is a valid action symbol. A full-vocabulary argmax can let an ordinary language token win, after which the de-tokenizer treats an arbitrary id as an action bin and the reconstructed coordinate is garbage. So at inference I slice the logits to $[V-N, V)$, take the argmax inside that slice, and add the start offset back to recover the real token id. Greedy decoding is sufficient and correct here because the runtime wants one deterministic normalized chunk, not a sampled continuation. Training, by contrast, needs no such restriction: I pass zero action-slot features into the runtime — this method conditions the decoder on nothing on the action side, no noised actions and no previously predicted continuous values — and let the runtime's discrete helper compute next-token cross-entropy on the action labels while ignoring prompt and non-action positions. Because the method owns no parameters, all learning flows through the LoRA-adapted trunk and the existing token embedding and output head. Chunking is handled by the same per-slot arithmetic applied independently: the scaffold lays out $K \times 7$ slots, I take the logit slice from $\text{num\_prompt\_tokens}$ through $\text{num\_prompt\_tokens} + K \cdot 7$, restrict each slot to the valid block, invert ids to bin indices, look up centers, and reshape to $(\text{batch}, K, 7)$. The method stops at normalized actions; the runtime owns the quantile-based un-normalization into physical robot units.

The tradeoff is honest and worth stating: cross-entropy treats neighboring bins as unrelated classes, ignoring the ordinal geometry of the action space, and the edge convention yields only $255$ distinct centers for $256$ token ids. But those are precisely the tradeoffs of the canonical discrete action-token path, and the value of this fill is that it inherits the pretrained behavior with no new head, no custom loss, no off-by-one drift, no full-vocabulary decode, and no leakage of un-normalization or benchmark bookkeeping into the editable surface.

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

The tokenizer arithmetic that the fill mirrors, the source of truth for the bin edges and the de-tokenization edge cases, is:

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
