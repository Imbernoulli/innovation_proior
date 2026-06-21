# Context: critic-free advantage estimation for LLM reasoning RL (circa early 2025)

## Research question

We are fine-tuning a base language model with reinforcement learning on math-reasoning
problems under a *verifiable* reward: sample a group of responses to each question, score
each response 1 if its final answer is correct and 0 otherwise, and push the policy toward the
behavior that earns reward. The policy is the autoregressive model itself, so the thing we
actually optimize is a per-token policy-gradient/PPO loss, and the single knob that decides
*how each sampled response is weighted* in that loss is the **advantage estimator**: given the
per-token rewards, the response mask, and the group identity of each sample (which responses
came from the same question), it must emit a per-token advantage that the actor loss
multiplies against the log-prob ratio.

The question is what an estimator should compute, from exactly the group of scored responses
it is handed, to turn returns into the per-token advantages that the actor loss consumes —
and how the resulting per-token loss should be reduced over a response's tokens.

## Background

**Language generation as a token-level MDP.** A response is a trajectory in an MDP whose state
at step `t` is the question concatenated with the tokens emitted so far, `s_t = q; o_{<t}`,
whose action is the next token, and whose transition is deterministic (append the token). The
return of a trajectory is `R(q, o) = Σ_t r(s_t, o_t)`. For verifiable math reasoning the reward
is *sparse and outcome-level*: it is zero at every token except the last, where it is the
correctness bit. We maximize the expected return `J(π_θ) = E_{q} E_{o ~ π_θ(·|q)}[R(q, o)]`.
In reinforcement learning from human feedback a KL-to-reference term is usually added to keep
the policy near the distribution where a learned reward model is trustworthy (Christiano et al.
2017; Stiennon et al. 2020), but with a *rule-based verifier* there is no reward-model
distribution to stay near, so that term can be dropped — saving the memory and compute of a
reference forward pass — and we take the coefficient to be zero throughout.

**The policy-gradient identity and the role of a baseline.** The Monte-Carlo policy gradient
(Williams 1992; Sutton & Barto 2018) is
`∇_θ J = E[ ∇_θ log π_θ(o|q) · R(q, o) ] = E[ Σ_t ∇_θ log π_θ(o_t|q, o_{<t}) · R(q, o) ]`.
A token's choice can only influence rewards that come *after* it, so the return multiplying the
`t`-th token can be narrowed to the reward-to-go `Σ_{t'≥t} r(s_{t'}, o_{t'})` without changing
the expectation. One may further subtract from the reward-to-go any **baseline**
`B(q, o_{<t})` that does not depend on the action `o_t`; this leaves the gradient unbiased,
because `E_{o_t}[∇_θ log π_θ(o_t|·) · B] = B · ∇_θ Σ_{o_t} π_θ(o_t|·) = B · ∇_θ 1 = 0`, while
reducing variance when `B` correlates with the return. The quantity reward-to-go minus
baseline is the **advantage**. Choosing `B` to be the expected future return — the state value
— is the textbook choice; in LLM RL the natural candidate is a second network the size of the
policy.

**Outcome reward collapses the per-token structure.** Because the reward is nonzero only at the
final token, the reward-to-go from *any* step equals the whole-trajectory return:
`Σ_{t'≥t} r(s_{t'}, o_{t'}) = R(q, o)` for every `t`. So with outcome reward the per-token
advantage is the same scalar for all tokens of a response — a single number per response,
broadcast over its tokens — and the entire design reduces to: pick a baseline, subtract it from
each response's return, broadcast.

**Advantage normalization.** Whitening advantages to zero mean and unit variance
is a standard stabilization trick in on-policy RL (Andrychowicz et al. 2021), normally applied
*across a whole batch*. How a normalization is *scoped* — batch-wide versus within each
question's group — sets whether it rescales the global step uniformly or changes the
*relative* weight of one question against another.

**Response-length dynamics.** Across open-source reproductions of the R1-Zero paradigm
(SimpleRL-Zero; Open-Reasoner-Zero), training exhibits a rise in response length over training,
tracked by correctness of the response. These projects use group-relative or PPO-style
objectives whose loss is normalized per response. Response length over training is a measured
quantity reported about the systems already in use.

## Baselines

**Value-model PPO with GAE (Schulman et al. 2015, 2017).** PPO maximizes a clipped surrogate
`E Σ_t min[ ρ_t Â_t, clip(ρ_t, 1-ε, 1+ε) Â_t ]`, where `ρ_t` is the new/old probability ratio
and `Â_t` is an advantage estimate, classically Generalized Advantage Estimation from a learned
value network `V_φ`, with a `λ` that trades bias against variance. Written this way the PPO
surrogate is a **sum over the response's tokens** — there is no division by response length.
`V_φ` is a second model of comparable size to the policy, trained alongside it.

**REINFORCE with a leave-one-out baseline — RLOO (Ahmadian et al. 2024; Kool et al. 2019).**
For a strongly pretrained LLM one can use plain REINFORCE on the full-sequence return with a
cheap, *parameter-free* baseline from multiple samples: for response `i` in a group of `k`,
use the mean reward of the *other* `k-1` responses,
`Â_i = R_i − (1/(k-1)) Σ_{j≠i} R_j`. Each response's baseline is an unbiased estimate of the
prompt's expected return, computed on the fly. It is presented as a sequence-level estimator.

**Group Relative Policy Optimization — GRPO (Shao et al. 2024).** A widely used critic-free
choice for reasoning RL. Sample `G` responses per question, score them
`R = {R_1, …, R_G}`, and set the advantage of *every* token of response `i` to the
group-normalized score
`Â_{i,t} = (R_i − mean(R)) / std(R)`. The objective is
`E (1/G) Σ_i (1/|o_i|) Σ_t min[ ρ_{i,t} Â_{i,t}, clip(ρ_{i,t}, 1-ε, 1+ε) Â_{i,t} ]`.
The group mean serves as the cheap baseline (no value network), the per-group standard
deviation rescales the centered reward, and the loss divides each response's
token-sum by its own length `|o_i|`.

**Length normalization inside open-source PPO loss code (von Werra et al. 2022 trl; Hu 2024
OpenRLHF; Sheng et al. 2024 verl).** Independently of GRPO, popular PPO implementations compute
the loss with `masked_mean(per_token_loss, mask, dim=-1)` — i.e. they divide each response's
token-sum by `mask.sum`, the response length. This convention traces to the pretraining stage,
where tokens are packed into a fixed context window and the loss is computed as
`loss.mean(-1)` over that (constant) length.

## Evaluation settings

The natural yardsticks for this regime, all pre-existing:

- **Policy / training data.** A small base model (e.g. Qwen2.5-0.5B, full-parameter), RL-tuned
  on competition-math training questions (MATH level 3–5, ~8K problems), with `G` responses
  sampled per prompt (8–16), batch size ~128, a short run (~100 steps), one GPU per experiment.
- **Reward.** Rule-based verifier: 1 if the response contains the correct final answer, else 0;
  no KL penalty (`β = 0`); PPO clip `ε = 0.2`; AdamW (`β` = 0.9/0.95), constant LR ~1e-6,
  grad-norm clip 1.0, one inner update epoch, max response length ~3000 tokens, temperature 1.0.
- **Metrics.** Math-reasoning accuracy (`mean@1`) on held-out benchmarks — GSM8K (grade-school,
  1,319 problems), MATH-500 (competition subset), and an AMC competition subset — higher is
  better. Alongside accuracy, the *response-length dynamics* (length of correct vs. incorrect
  responses over training) are tracked as a diagnostic.

## Code framework

The estimator plugs into an existing PPO-style training loop. Everything around the estimator
already exists: the rollout that samples a group of responses per prompt, the rule-based reward
manager that fills the per-token reward tensor (scalar at the last valid token), the response
mask, the group `index` array, the actor loss that consumes the advantages, and the reduction
utility for per-token losses. What is not settled is the body of the estimator itself: how, from
the group of scored responses, to turn returns into per-token advantages. That is the single
empty slot.

```python
from typing import Optional

import numpy as np
import torch

def compute_custom_advantage(
    token_level_rewards: torch.Tensor,   # (bs, response_length); outcome scalar at last valid token
    response_mask: torch.Tensor,         # (bs, response_length); 1 = valid response token
    index: np.ndarray = None,            # (bs,) group/prompt id; same id = same question
    epsilon: float = 1e-6,
    config: Optional[object] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:  # (advantages, returns), both (bs, response_length)
    """Turn per-token rewards + group identity into per-token advantages and returns.

    Recover the per-sequence score with token_level_rewards.sum(dim=-1); samples sharing
    an index came from the same prompt and may be compared. Both outputs must be masked
    by response_mask. Wrap the computation in torch.no_grad().
    """
    # TODO: design the estimator.
    raise NotImplementedError


# existing PPO actor loss the estimator feeds (already implemented elsewhere):
#   per_token_loss = -min(ratio * advantages, clip(ratio, 1-eps, 1+eps) * advantages)
#   loss = aggregate(per_token_loss, response_mask)   # how the token-sum is reduced
```

The training loop hands the estimator one group of scored responses at a time; the body is
where the per-token advantage rule will live, and the loss `aggregate(...)` is where the
per-response token-sum is reduced before averaging over the batch.
