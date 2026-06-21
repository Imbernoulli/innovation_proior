The diffusion rung landed exactly where the budget said it would. On seed 42 it scored 0.1111 on goal_a, a flat 0.0000 on goal_b, and 0.0833 on goal_c — geometric mean zero, arithmetic mean barely off the floor at $\sim 0.065$ — while the eval timers ran 552/567/707 seconds per subset, the $\sim 50$-trunk-passes-per-chunk tax of the reverse rollout. That is the under-training I expected: the denoiser never learned in 6000 steps to walk noise back to coherent action chunks, so the sampled actions stay close to noise and the policy mostly fails. The lesson is sharp — in this regime the deciding factor is not the *expressivity* of the head but whether the head can be *fit* inside 6000 steps, and multimodality is worthless if the model representing it is still noise. So the next move is to stop modeling a distribution from scratch and instead ride the most data-efficient signal available on this trunk: the model's own pretrained objective.

I propose **PD&AC discrete**: keep the trunk's mouth and its native next-token cross-entropy, and only teach it to *speak actions* as tokens. The base trunk is a web-pretrained vision-language model whose single native skill is next-token prediction over a fixed vocabulary; the diffusion head grafted a from-scratch denoiser and a from-scratch MSE-on-noise objective onto it, whereas the discrete path does the opposite and keeps both the objective and the output head intact. The one tension is that the model emits discrete tokens while a robot action is a vector of continuous reals — so the only way through is to turn each real into a token, giving up the continuum and quantizing.

The quantization is a rate-distortion call, and the robot does not need infinite precision, only resolution below the controller's own repeatability. I discretize each of the 7 action dimensions into 256 bins. 256 levels is 8 bits per degree of freedom — finer than the controller's repeatability, so quantization error is dominated by other error sources and effectively free — while 256 classes is a tame classification problem for a 7B model and one bin index fits cleanly into one token. The bins are placed by *quantile*, not by raw range: demonstration data has heavy tails, and a single teleop outlier would stretch a min-to-max interval so that almost all 256 bins cover empty extremes and the dense bulk collapses into a handful. So I clip each axis to its 1st/99th percentile (on the normalized $[-1,+1]$ scale), lay 256 uniform bins across the clipped interval, and let outliers clamp to the end bins; uniform within the clip keeps encoding (a digitize) and decoding (a center lookup) trivial.

The load-bearing trick is making those integers be tokens the model *already has*, so I introduce zero new parameters. Adding 256 new vocabulary entries would mean new randomly-initialized embedding rows and output-projection columns — the very from-scratch cost I am trying to avoid. A BPE vocabulary is ordered by frequency and its tail is junk: token identities the pretrained model has barely produced, carrying almost no learned meaning. So I overwrite the last 256 entries and let them *be* the action bins, reusing existing embedding rows and projection columns and teaching them a new job during fine-tuning. I pin the index arithmetic to a self-inverse map: bin index $k$ of a dimension maps to $\text{token\_id}=\text{vocab\_size}-k$, so the action tokens occupy the contiguous range $[\text{vocab\_size}-256,\ \text{vocab\_size})$ and the bin$\leftrightarrow$token map is "subtract from vocab\_size," its own inverse.

Training is then *nothing new*, which is the whole point of starting from the trunk's own objective. Every demonstration action becomes 7 action tokens appended to the prompt; the model is already a teacher-forced next-token predictor with cross-entropy; I mask the loss to the action positions (the prompt is set to the ignore index, so I never train the model to "predict" the given image and instruction) and evaluate categorical cross-entropy on the action tokens only — the runtime exposes this directly as `compute_discrete_objective`. There is a real argument that cross-entropy over *unordered* bins is a good control loss, not merely a convenient one: it treats the 256 bins as unordered classes, so at a fork it can place mass on two separated bins and commit to one at decode, representing the same multimodality diffusion chased — natively, at the cost only of a smoothness prior the giant model can learn from data anyway. The difference from the diffusion rung is that this multimodal representation rides the trunk's *pretrained* objective, so it has a head start instead of starting from scratch.

Decode is where the off-by-ones live. At the action positions the model gives a distribution over the *entire* vocabulary, but an action token is only ever one of the last 256, and a natural-language token at an action slot is meaningless as control. So I must not argmax over the whole vocabulary: I slice the logits to the action sub-vocabulary $[\text{vocab\_size}-\text{num\_bins},\ \text{vocab\_size})$, argmax *within that slice*, then add the offset back to recover the actual token-id (and raise if the vocab bounds are inconsistent). This is the one place this method differs from the naive scaffold default, which argmaxes over all logits and can silently select a non-action token. The argmax is greedy, not sampled — for control I want the single most-likely bin per dimension, deterministic. Then I invert the encoding with the same subtraction, $\text{bin\_index}=\text{vocab\_size}-\text{token\_id}$, and handle the bin-vs-center care: 256 bin *edges* define only 255 intervals, so there are 255 centers; the digitize returned indices in $[1,256]$, so I subtract 1 to land in $[0,255]$ and clip the top index down into the last real interval — $\mathrm{clip}(\text{vocab\_size}-\text{token\_id}-1,\,0,\,\text{num\_centers}-1)$ with $\text{num\_centers}=\text{bin\_centers.shape}[0]$. The decoded value for a bin is its *center*, the minimum-distortion reconstruction for a uniform quantizer, looked up from the precomputed `runtime.model.bin_centers`, reshaped to $(B,\ \text{NUM\_ACTIONS\_CHUNK},\ \text{ACTION\_DIM})$ and handed back as normalized actions in $[-1,+1]$.

Two scaffold details complete the wiring, both following from the discrete path conditioning on no extra action-side inputs. The action tokens are predicted from the prompt context, so I fill the action-feature slots with zeros (`zero_action_token_features`, a $(B,K\cdot D,\text{llm\_dim})$ zero tensor passed to `runtime.forward`), and the method holds *no parameters of its own* — its `__init__` is empty, because all learning is fine-tuning the trunk under its own objective. The decode extends across the chunk unchanged: binning, the least-used-token map, the cross-entropy, and the center lookup are all agnostic to how many action positions there are, so $K\cdot D$ slots decode exactly like one.

The expectation against the diffusion numbers is asymmetric. The discrete head shares diffusion's multimodality but pays none of its training cost — it reuses the pretrained objective, adds zero parameters, and decodes in a single trunk pass instead of 50 — so eval times should collapse from $\sim 550$–$700$ s/subset back to the single-pass range, and success should clear the diffusion floor with non-trivial signal on more than one subset. But I do not expect the ceiling, because of the 256-bin grid: cross-entropy over unordered bins throws away the ordinal structure of a continuous action (it never learns that bin 130 is closer to the truth than bin 12 when the answer is 131), and the 8-bit grid caps precision exactly at the level fine manipulation lives below. So discrete should beat diffusion clearly in this budget yet leave a precision gap that a head regressing the continuous action directly — no grid, no ordinal blindness — should be able to close.

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
