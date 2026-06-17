# Context: scaling RL on a verifiable reward for long chain-of-thought reasoning (early 2025)

## Research question

Test-time scaling has just shown that letting a large language model think for far longer —
emitting a long chain of thought before its final answer — sharply raises its competition
math and coding ability, and that the engine producing this behavior is large-scale
reinforcement learning against a *verifiable* reward (does the final answer match the
ground truth?). The public reports that demonstrate it withhold the actual training recipe.
When the community runs the obvious open recipe — a critic-free, group-relative policy
gradient on a +1/-1 correctness reward — on a strong base model, the result lands well below
the reported frontier scores and training is fragile in characteristic ways: the policy's entropy
collapses early (the sampled group of answers for a prompt becomes nearly identical, so
exploration dies); a growing fraction of prompts produce a group of answers that are *all*
correct or *all* wrong, which carries no learning signal; and the average response length
drifts upward over training, often with the longest responses degenerating into repetition
and gibberish, while accuracy stalls.

The precise goal is a single optimization recipe for this setting — critic-free,
verifiable-reward, very long responses (tens of thousands of tokens), a group of rollouts
per prompt — that (1) keeps the policy exploring instead of collapsing to a deterministic,
low-entropy mode; (2) keeps every training batch full of prompts that actually carry
gradient; (3) does not let the loss reduction silently reward or excuse verbosity, so that
length grows only when longer reasoning is genuinely better; and (4) handles the practical
fact that generation is capped at a maximum length, so some sound-but-long responses get
truncated, without that truncation injecting noise that confuses the model about whether its
reasoning was valid. Each existing ingredient below handles part of this; none, used
naively, gives a stable long-CoT run. Closing that gap is the problem.

## Background

By this time the dominant way to RL-tune an LLM is a clipped-surrogate policy gradient.
The field's load-bearing concepts:

- **Policy-gradient with a clipped trust region.** The model is a token-level policy
  `π_θ(o_t | q, o_{<t})`. One maximizes a surrogate in the importance ratio
  `r_t(θ) = π_θ(o_t|q,o_{<t}) / π_{θ_old}(o_t|q,o_{<t})` against an advantage estimate `Â_t`,
  with the ratio clipped to `[1-ε, 1+ε]` so a single update cannot move the policy too far —
  a soft trust region that stabilizes training and lets data be reused for several gradient
  steps.

- **Group-relative, critic-free advantage.** Training a separate value network the size of
  the policy is expensive, and with only an end-of-sequence reward it is hard to learn a
  per-token value that is accurate. The prevailing alternative removes the critic entirely:
  for each prompt `q`, sample a *group* of `G` responses from the old policy, score each with
  the rule reward, and use the group's own statistics as the baseline. The advantage of every
  token in response `i` is set from the group-normalized scalar reward
  `Â_{i,t} = (R_i − mean({R_j})) / std({R_j})`, broadcast to all tokens of that response. The
  quality of this advantage depends entirely on the **scale and distribution of the scalar
  rewards going in**.

- **Rule-based, verifiable reward.** To avoid reward hacking, the reward is not a learned
  model but a rule: `R = +1` if the final answer is equivalent to the ground truth, else `−1`
  (or `1/0`). For math this requires answers that a parser can check reliably, which favors
  problems whose answer is a single integer.

- **The EMA / moment language is not the issue here; the loss-aggregation arithmetic is.**
  A subtlety that turns out to matter: how the per-token losses are *summed up* into one
  scalar loss. The standard group-relative objective averages within each response first and
  then across responses — it puts a factor `1/|o_i|` in front of response `i`'s token sum and
  a factor `1/G` across responses. So a token's weight in the final loss is
  `(1/G)·(1/|o_i|)`: it shrinks as the response it lives in gets longer.

Two diagnostic findings about these existing systems are already documented and set up the
problem:

- **Length drifts upward, especially on wrong answers.** Running the group-relative recipe,
  response length climbs throughout training; this is often read as the emergence of richer
  reasoning, but the *incorrect* responses grow fastest and longest, and a large share of the
  growth is low-quality (repetition, padding). A careful accounting of the per-token gradient
  weight under the `1/|o_i|` aggregation explains the direction of the drift: when the
  advantage is positive (a correct response), the per-token update is proportional to
  `1/|o_i|`, so a *shorter* correct response receives a *larger* per-token push — the recipe
  quietly prefers brevity when it is right; when the advantage is negative (an incorrect
  response), the per-token penalty is also proportional to `1/|o_i|`, so a *longer* wrong
  response is penalized *less* per token — the recipe quietly tolerates length when it is
  wrong. The two together push incorrect responses to grow without bound. This same
  per-sequence length normalization is present even in many open-source PPO loss
  implementations, which average each response by its own valid-token count. That
  formulation/implementation mismatch predates the group-relative method and likely descends
  from pretraining, where dividing by a fixed context length is a numerical-stability
  convenience; in RL-tuning the divisor varies with the generated response.

- **A second normalization, by the per-group reward standard deviation, re-weights prompts by
  difficulty.** Dividing the centered reward by `std({R_j})` gives groups with small spread
  (prompts that are nearly all-correct or all-wrong) disproportionately large weight in the
  objective. Advantage normalization is standard practice, but it is normally taken over a
  whole batch; doing it per prompt makes the effective objective weight prompts unevenly by
  how easy or hard they happen to be.

- **Truncation noise.** Generation is capped at a maximum length; responses that hit the cap
  are truncated. Assigning the usual punitive reward to a truncated sample penalizes a
  possibly-sound reasoning chain purely for being long, injecting noise that destabilizes
  training.

These are properties of the existing recipe, read off its objective and its training curves.

## Baselines

The prior methods a new recipe is measured against and reacts to.

**PPO (Schulman et al. 2017).** A clipped-surrogate actor-critic method:

```
J_PPO(θ) = E[ min( r_t(θ) Â_t,  clip(r_t(θ), 1-ε, 1+ε) Â_t ) ],
           r_t(θ) = π_θ(o_t|q,o_{<t}) / π_{θ_old}(o_t|q,o_{<t}),
```

with `Â_t` from Generalized Advantage Estimation on a learned value function `V`. The clip
forms a trust region; the value baseline reduces variance. **Gap in LLM RL:** the value model
is a second network of comparable size to the policy, a heavy memory and compute burden, and
because the reward typically arrives only at the last token, a per-token value is hard to fit
accurately — so the variance-reduction baseline the method depends on is itself unreliable in
this regime.

**GRPO (Shao et al. 2024, DeepSeekMath).** Remove the critic. For each prompt sample a group
of `G` responses, score them, and use the group as the baseline:

```
J_GRPO(θ) = E[ (1/G) Σ_{i=1}^G (1/|o_i|) Σ_{t=1}^{|o_i|}
              ( min( r_{i,t}(θ) Â_{i,t}, clip(r_{i,t}(θ), 1-ε, 1+ε) Â_{i,t} )
                − β·D_KL(π_θ || π_ref) ) ],
   Â_{i,t} = ( R_i − mean({R_j}) ) / std({R_j}).
```

It obviates the value network and aligns with the comparative nature of the reward. **Gaps,
all observable from the objective and the training curves:** (a) the symmetric clip with the
usual `ε = 0.2` lets a high-probability token rise easily (a token at `0.9` can go to `1.08`)
while it tightly bounds the increase of a low-probability token (a token at `0.01` can only
reach `0.012`), so as training proceeds the policy concentrates on a few high-probability
tokens and entropy collapses; (b) when a group is all-correct or all-wrong the centered
reward is zero for every member, so `Â = 0` and the group contributes *no* gradient — and the
fraction of such groups grows over training, shrinking the effective batch and raising
gradient variance; (c) the `1/|o_i|` sample-level aggregation makes a token's contribution
shrink with its response's length, which (per the diagnostic above) prefers brevity on
correct answers and tolerates length on incorrect ones; (d) the per-group `std` normalization
weights prompts by difficulty. The `β·D_KL` term, inherited from alignment RL where the goal
is to stay near the initial model, is a poor fit when a reasoning policy is *supposed* to move
far from its initialization — it then only holds the policy back.

The four gaps above are the prior art's stalls: where entropy dies, where the gradient
vanishes, where the loss arithmetic misweights tokens by length, and where the reference-KL
and the difficulty normalization fight the objective. How to get past each is what the
reasoning works out.

## Evaluation settings

The natural yardsticks already in use:

- **Base policy:** a strong open base model (Qwen2.5 family; in the small-scale setting
  Qwen2.5-0.5B, full-parameter), trained critic-free with a group of `n = 16` rollouts per
  prompt and the group-relative (GRPO) advantage estimator.
- **Training data:** competition-math problems with verifiable integer answers — e.g. MATH
  levels 3–5 and additional curated math problems — preprocessed so the answer is a single
  integer the rule can check.
- **Reward source:** the rule reward (+1 / −1 or 1 / 0 on answer equivalence); a maximum
  generation length (e.g. 16,384 tokens, or larger with a soft buffer) with truncation of
  overlong samples.
- **Metrics:** math-reasoning accuracy, `mean@1`, on held-out math benchmarks (GSM8K,
  MATH-500, AMC 23; in the large-scale setting, AIME 2024), reported against gradient-update
  steps; the headline score is the mean across benchmarks. The actor's entropy and the mean
  response length over training are tracked as stability diagnostics.
- Protocol: fixed model, optimizer, rollout count, KL setting and evaluation data across
  variants, so the comparison isolates the change being studied.

## Code framework

The recipe plugs into an existing critic-free group-relative RL trainer (verl-style). The
reward manager emits one scalar per response, placed at that response's last valid token,
giving a `(batch_size, response_length)` tensor `token_level_scores`. Downstream, the
group-relative advantage estimator subtracts the per-prompt group mean and divides by the
per-prompt group std, broadcasts the result across the response's tokens, and the clipped
surrogate is aggregated into a scalar loss and backpropagated. What is *not* settled is the
upstream transformation of the per-response reward tensor before it reaches the advantage
estimator — that transformation is exactly what is to be designed. So the substrate is only
the generic machinery that already exists: the per-token reward tensor and its response mask,
the masked-reduction utilities, and a single empty slot for the reward transformation.

```python
import torch
import numpy as np
from typing import Optional


def normalize_rewards(
    token_level_scores: torch.Tensor,   # (bs, response_length): outcome scalar at last valid token
    response_mask: torch.Tensor,        # (bs, response_length): 1 on valid response tokens
    index: np.ndarray = None,           # (bs,): group / prompt id (G rollouts share an id)
    epsilon: float = 1e-6,
    config: Optional[object] = None,    # algorithm config
    **kwargs,
) -> torch.Tensor:                      # (bs, response_length): transformed reward tensor
    """Transform the per-response reward tensor before the group-relative advantage
    estimator consumes it. The outcome scalar lives at the last valid token; recover it
    with .sum(dim=-1). Runs under no_grad. The downstream estimator will still subtract
    the group mean and divide by the group std on top of whatever this returns."""
    with torch.no_grad():
        # TODO: the reward transformation we will design.
        #       Given the per-response scalar reward and the response mask (and the
        #       group index if needed), reshape the reward tensor and return it with
        #       the same (bs, response_length) shape, preserving last-token semantics.
        pass


# existing group-relative RL step the transform plugs into (read-only downstream)
def grpo_advantage(token_level_scores, response_mask, index, epsilon=1e-6):
    """Per-prompt group-mean baseline and group-std normalization, broadcast to tokens.
    (Fixed; runs after normalize_rewards.)"""
    scores = token_level_scores.sum(dim=-1)                     # (bs,)
    # group by index: subtract per-prompt mean, divide by per-prompt std (+epsilon)
    ...                                                          # returns (bs, response_length) advantages
```

The transform supplies the reward tensor; `normalize_rewards` is where the upstream
reward transformation will live.
