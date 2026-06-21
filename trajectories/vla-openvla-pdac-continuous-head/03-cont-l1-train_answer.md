The discrete rung confirmed the diagnosis exactly. On seed 42 it scored 0.2222 on goal_a, 0.1111 on goal_b, and 0.0000 on goal_c, with eval timers collapsed to 190/126/165 seconds — back in the single-pass range, no 50-step rollout, just as predicted once the head stopped denoising from scratch. So "fit, not expressivity" was right: where the under-trained diffusion head sat at 0.1111/0.0000/0.0833 with a $\sim 0.065$ arithmetic mean, the discrete head — riding the trunk's pretrained next-token objective, zero new parameters — registered real success on two of three subsets and roughly doubled the off-floor signal. But discrete did not break clear: its geometric mean is still 0.0000, dragged down by the empty goal_c, and even its best subset is far from the regime where the policy actually completes tasks. That is the precision gap I set up in advance. The 256-bin grid is an 8-bit quantizer over $[-1,+1]$ and fine manipulation lives in the gaps between grid points, while cross-entropy over *unordered* bins throws away ordinal structure — it never learns that bin 130 is a near-miss when the truth is 131, only "right class / wrong class." On goal_c, the hardest four-task subset, that coarseness and ordinal blindness apparently cost just enough accumulated end-effector error per chunk that nothing completes. So the move is forced: stop discretizing.

I propose **PD&AC + Cont-L1**: keep the single-pass parallel decode and the pretrained trunk, but map the action-slot hidden states straight to real-valued normalized actions with a small regression head and train it with a distance-respecting loss. Parallel decoding already gives me, in one forward pass, a hidden state at each of the $K\cdot D$ action slots — shape $(B,K\cdot D,\text{hidden})$ — conditioned bidirectionally on the full observation. The discrete path took those hidden states to bin-logits; instead I map them to continuous actions, so the precision ceiling the 256 bins imposed simply disappears while every other property of the scaffold survives.

The loss is the load-bearing choice, because it interacts with the messy reality of human demonstrations. The obvious default is squared error, but L2 penalizes large residuals quadratically, so it is pulled hard toward outliers and its optimum is the conditional *mean* — and the mean of a spread of distinct valid motions can be a blended action valid under none of them, the same mode-averaging failure that motivated diffusion. L1, the mean absolute error, penalizes linearly, so it is far less swayed by the occasional large deviation and its optimum is the conditional *median*. For action prediction the median is the better summary: robust to noisy demonstration outliers, and more precise. So the objective is
$$L=\operatorname{mean}\bigl|\text{predicted}-\text{ground\_truth}\bigr|$$
over all $K$ timesteps and all $D$ dimensions on the normalized $[-1,+1]$ actions — one `torch.nn.L1Loss` call against `runtime.ground_truth_actions`. Because the actions are normalized, the loss sits on a clean comparable scale, part of why it converges fast and stably inside the budget.

It is worth sitting with the alternative I already tried, because it sharpens why L1 is the right call and not merely a fallback. Diffusion's selling point is genuine — it represents multimodal action distributions, capturing several distinct valid ways to act instead of collapsing to a median, an expressivity L1 lacks. But the usual reason simple regression loses to a generative head is *under-capacity*: a small network forced to output one vector hedges by averaging modes. The bet L1 makes is that a 7B trunk conditioned on a rich bidirectional observation has enormous capacity and can often pin down which mode the current situation calls for and regress it cleanly, so the multimodality diffusion buys may be largely redundant here. Since diffusion already lost to discrete on fit alone, L1 inherits discrete's fit advantage and adds continuous precision on top. The honest limitation stands: if the demonstrations are truly multimodal — several genuinely different sequences all correct for the same input — the median-seeking head cannot represent that and diffusion would have the edge. For the focused, consistent-strategy LIBERO-Goal demonstrations, the median is exactly what I want.

Now the head and its wiring. After the forward pass I have a hidden state at each of the $K\cdot D$ positions and want one $D$-dimensional action per timestep. The natural grouping is by timestep: the $D=7$ action positions belonging to timestep $k$ together determine that timestep's 7-dim action, so I gather those $D$ hidden states and feed them jointly — reshape $(B,K\cdot D,\text{hidden})\to(B,K,D\cdot\text{hidden})$, each of the $K$ rows the concatenation of that timestep's $D$ action-position hidden states, and map each row to $D$ outputs. The head's input width is $D\cdot\text{hidden}$ and its output is action\_dim, applied across the $K$ timesteps. It sits on top of the LoRA-adapted trunk, reading high-dimensional features and regressing a low-dimensional target on a few hundred demonstrations, so I want it expressive enough to extract the action but stable and not prone to overfitting. A plain linear+ReLU stack is finicky at this width; residual connections give gradients a clean path and layer normalization stabilizes activation scale, so an MLP-ResNet of a few pre-normalized residual feedforward blocks (LayerNorm $\to$ Linear $\to$ ReLU, added to the input) is the right shape — pre-norm keeps the residual stream's gradients well-behaved, the same reasoning that favors pre-norm feedforward blocks in transformers. I bracket the stack with an input projection lifting $D\cdot\text{hidden}$ to the working width, two residual blocks, a final LayerNorm, and a linear projection to action\_dim. The scaffold provides exactly this as `prismatic.models.action_heads.MLPResNet`, instantiated with `num_blocks=2`, `input_dim = input_dim * ACTION_DIM`, `hidden_dim`, `output_dim = action_dim`; it is tiny relative to the 7B trunk, adding negligible inference cost, which is the whole point of choosing a head over a heavier generative decoder.

The extraction has a subtlety the `extract_action_hidden_states` hook owns. I run the trunk forward with *zeroed* action-token features — unlike diffusion, the L1 head injects nothing action-conditioned back into the input; like discrete it hands a zero placeholder of shape $(B,K\cdot D,\text{llm\_dim})$ to `runtime.forward`, since it conditions only on the observation — under bidirectional attention. At train time I have the ground-truth labels, so I build the boolean mask `current_action_mask | next_actions_mask` over the response positions, select those hidden states, and reshape to $(B,K\cdot D,\text{hidden})$. At eval time there are no labels, but the layout is known: the action positions start right after the prompt and run for $K\cdot D$ contiguous slots, so I slice $[\text{num\_prompt\_tokens}:\text{num\_prompt\_tokens}+K\cdot D]$. Either way the head sees $(B,K\cdot D,\text{hidden})$, regroups to $(B,K,D\cdot\text{hidden})$, and predicts $(B,K,\text{action\_dim})$; the loss is the single L1 call against the ground-truth chunk, with `runtime.compute_action_l1_metrics` logging the auxiliary diagnostics. This is precisely the ordinal-respecting loss discrete's precision gap demanded — discrete's unordered cross-entropy could not learn that a near-miss bin is nearly right, whereas $|\text{predicted}-\text{truth}|$ shrinks smoothly as the prediction approaches the target, making "nearly right" the loss itself and recovering the sub-grid precision discrete threw away.

The expectation against the discrete numbers is sharp. L1 keeps everything that made discrete beat diffusion — single trunk pass, pretrained trunk, fits in 6000 steps — so eval times should stay in the single-pass range, not the diffusion rollout range. Removing the 256-bin grid and using a distance-respecting loss should lift *every* subset and, critically, unstick the one discrete left empty: I expect goal_c to go non-zero, which alone flips the geometric mean off zero where both discrete and diffusion scored a flat 0.0000. So the bar is unambiguous — beat discrete on every subset and produce a positive geometric mean. If L1 lifts goal_c off zero and dominates the other two, then the continuous regression head, fit-efficient like discrete, precise like no grid allows, and matching diffusion's success without its training cost, is the strongest action method this budget supports.

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
