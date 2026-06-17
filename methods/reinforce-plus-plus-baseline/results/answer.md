# REINFORCE++-baseline: critic-free advantage via group-mean baseline + global batch whitening

## Problem

Fine-tune an LLM with online RL when (a) a critic is too expensive and unreliable (reward only at
EOS), and (b) the critic-free alternatives estimate the advantage from a small per-prompt group of
responses. GRPO's per-group standardization `A_i = (r_i - mean(r)) / (std(r) + ε)` is statistically
biased at small group size because the centered numerator and denominator are computed from the same
few rewards; its local std is also a brittle prompt-sized scale. We want a critic-free, per-token
advantage estimator whose scale is set by a large pool rather than by one small group.

## Key idea

Split the normalization into the stable part and the dangerous part, and pool the dangerous part over a
large sample. **Subtract the per-prompt group mean** (local reward centering that reshapes rewards so
0/1 and -1/1 schemes both center at zero), then **whiten over the entire global batch of valid response
tokens** (not the small group). The local-std bias comes from estimating the denominator on a small
sample where numerator and denominator are coupled; computing the scale over a batch of roughly 1024
valid-response samples/tokens makes any one sample's leverage small, and the coupling vanishes in the
large-pool limit. This is PPO with the critic removed, GAE at `γ = λ = 1` (reward only at EOS, so the
advantage is the centered full return), and the value baseline replaced by group-mean-then-global-whiten.

## Why local-std normalization is biased (the result that motivates "go global")

Model a group's rewards as `r_i = θ + ε_i`, `ε_i ~ N(0, σ²)` i.i.d., `i = 1..N`; `ε̄ = (1/N) Σ ε_j`,
`D = sqrt((1/N) Σ (ε_j - ε̄)²)`, `A_i = (ε_i - ε̄)/D`. Then for any finite `N ≥ 2`,
`E[A_i | ε_i]` is not identically `ε_i`.

- Numerator: `E[ε_i - ε̄ | ε_i] = (1 - 1/N) ε_i`.
- Denominator depends on `ε_i`: `E[D² | ε_i] = ((N-1)²/N²) σ² + ((N-1)/N²) ε_i² =: α + β ε_i²`,
  `β > 0`. Taylor-expanding `f(x) = x^{-1/2}` about `μ = E[D²|ε_i]` gives
  `f(x)=μ^{-1/2}-(1/2)μ^{-3/2}(x-μ)+(3/8)μ^{-5/2}(x-μ)^2+O((x-μ)^3)`, hence
  `E[1/D | ε_i] ≈ μ^{-1/2} + (3/8) Var(D²|ε_i) μ^{-5/2}`; the leading scale already varies with
  `ε_i²`.
- For a rigorous bias check, set `z_j = ε_j - ε̄`. Since `Σ_j z_j = 0`,
  `Σ_{j≠i} z_j² ≥ z_i²/(N-1)`, so `D² = (1/N)Σ_j z_j² ≥ z_i²/(N-1)` and `|A_i| ≤ sqrt(N-1)`.
  The normalized local-std advantage is bounded, while `ε_i` is Gaussian and unbounded, so the
  conditional expectation cannot equal `ε_i` as a function. As `N → ∞`, `D → σ` and `(1-1/N) → 1`,
  so the bias vanishes — hence normalize over the large global batch. ∎

## KL loss: use k2 (reverse KL), reject k1 and k3

Sampling from `π_θ` constrains the reverse KL `D_KL(π_θ || π_ref)`. With
`ℓ = log π_θ - log π_ref` and `δ = π_ref/π_θ = exp(-ℓ)`, its reverse-KL gradient is
`E_{y~π_θ}[ ℓ(y) ∇_θ log π_θ(y) ]`. For a separate fixed-sample loss, the relevant question is the
autodiff gradient:
- **k1** `= -log δ = ℓ`: scalar-unbiased for reverse KL under `π_θ` samples, but as a fixed-sample loss
  `∇k1 = ∇log π_θ`, missing the log-ratio multiplier (k1 is fine inside the reward, not as a standalone
  loss).
- **k2** `= (1/2)(log δ)² = (1/2)ℓ²`: `∇k2 = ℓ ∇log π_θ` — exactly the reverse-KL loss gradient.
  **Correct and stable.**
- **k3** `= δ - 1 - log δ = exp(-ℓ) - 1 + ℓ`: scalar-unbiased for reverse KL, but as a fixed-sample
  loss `∇k3 = (1-exp(-ℓ))∇log π_θ`; after the score term vanishes in expectation, this leaves
  `-E_{y~π_θ}[(π_ref/π_θ)∇log π_θ] = -E_{y~π_ref}[∇log π_θ]`, the forward-KL gradient. The
  `π_ref/π_θ` coefficient explodes when `π_θ(y)` is tiny and can overflow numerically.

Final reasoning-task objective: `L = L_PPO(A^norm) - λ · J_{k2 as loss}(θ)`.

## Algorithm

For a global batch of sampled responses, grouped by prompt id:
1. `score_i = Σ_t r_{i,t}` (outcome scalar from the token-level rewards).
2. `A'_i = score_i - mean_{group(i)}(score)` (group mean `:= 0` for singleton groups).
3. Broadcast `A'_i` to all valid tokens; mask padding.
4. `A^norm = masked_whiten(A', mask) * mask`, whitening over all valid response tokens in the batch.
5. Return `(A^norm, A^norm)` because this critic-free API has no separate value target.

## Code

```python
from collections import defaultdict
from typing import Optional

import torch
import verl.utils.torch_functional as verl_F
from verl.trainer.config import AlgoConfig


@register_adv_est(
    AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE
)  # or simply: @register_adv_est("reinforce_plus_plus_baseline")
def compute_reinforce_plus_plus_baseline_outcome_advantage(
    token_level_rewards: torch.Tensor,   # (bs, response_length)
    response_mask: torch.Tensor,         # (bs, response_length)
    index: torch.Tensor,                 # (bs,) prompt group id
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """REINFORCE++-baseline: per-prompt group-mean baseline, then global token-level whitening."""
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)              # outcome scalar per response

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)         # no local baseline for a singleton group
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]     # step 1: group-mean centering

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask  # broadcast to tokens
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask       # step 2: global whiten

    return scores, scores                                 # API returns advantages and return tensor
```

The framework's `masked_whiten` computes the masked mean and Bessel-corrected masked variance over all
valid response tokens. The `k = 1` sibling skips the group-mean baseline and globally whitens
token-level returns; the **baseline** variant above adds per-prompt group-mean centering and uses the
`k2` KL loss gradient, which is the multi-sample reasoning-training form.
