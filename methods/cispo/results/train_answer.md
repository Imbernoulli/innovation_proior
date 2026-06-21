We train an LLM with reinforcement learning to lift its reasoning — competition mathematics, coding — by rolling out responses from a frozen snapshot $\pi_{\text{old}}$, scoring them with a verifier, partitioning the batch into mini-batches, and taking several gradient steps on the current policy $\pi_\theta$. Because the data is off-policy, each token carries an importance ratio $r_{i,t} = \pi_\theta(y_{i,t}\mid x,y_{i,<t}) / \pi_{\text{old}}(y_{i,t}\mid x,y_{i,<t})$, and the PPO/GRPO surrogate combines the unclipped and clipped terms pessimistically as $\min\!\big(r_{i,t}\hat A_{i,t},\ \mathrm{clip}(r_{i,t},1-\varepsilon,1+\varepsilon)\hat A_{i,t}\big)$. The clip exists to stop the optimizer from chasing surrogate value by driving a single stale token's ratio far from $1$, and that purpose is sound. What is not sound is the *price* it charges, because in reasoning RL that price is paid by exactly the tokens I least want to silence. Mechanically the clip is a hard gate: for a positive-advantage token, while $r_{i,t}$ is inside the band the gradient is $r_{i,t}\hat A\,\nabla_\theta\log\pi_\theta$, but the instant the policy raises that token's probability past $1+\varepsilon$ so that $r_{i,t}>1+\varepsilon$, the pessimistic $\min$ switches to the clipped branch $(1+\varepsilon)\hat A$, which has no dependence on $\theta$ and whose gradient is identically zero. The token is not down-weighted — it is *deleted* from the update.

The failure is that the deleted tokens are the ones that matter. The ratio moves most, in relative terms, for tokens that were *rare under $\pi_{\text{old}}$*: a token the old policy assigned $0.02$ that the new policy now likes at $0.10$ has ratio $5$, wildly outside any sane band, even though in absolute terms it moved only eight points, while a token already at $0.6$ can barely produce a large ratio at all. So the clip preferentially fires on low-probability tokens — and in a reasoning trace those low-probability, high-entropy tokens are the *fork* tokens, the "However", the "Wait", the "Let me recheck" that begins a self-correction or switches the line of reasoning. They are rare because branching is rare and high-leverage because they redirect everything downstream, and the clip zeroes their gradient on every update. Widening $\varepsilon$ does not save this: a wider band keeps the fork tokens a little longer but also loosens the leash on the genuinely-stale tokens the clip exists to catch. "Rare" and "stale" are different axes — a fork token can be perfectly trustworthy and still produce a huge ratio — so no single $\varepsilon$ separates them. The problem is not the *width* of the gate; the problem is that there is a *gate at all*.

I propose CISPO, Clipped IS-weight Policy Optimization. Step back to why a large ratio is dangerous: the danger is *variance*, and variance is a property of the *magnitude of the token's coefficient*, not of whether its gradient is gated. The PPO clip controls variance by capping a large-ratio token's influence at zero — it caps by deletion — but capping influence and deleting the gradient are not the same operation. What I actually want is for a large-ratio token to contribute a *bounded* amount, not zero. So I write the objective as the off-policy score-function estimator directly rather than as a clipped surrogate, $\mathbb{E}\big[w_{i,t}\,\hat A_{i,t}\,\nabla_\theta\log\pi_\theta(y_{i,t}\mid\cdot)\big]$, plain REINFORCE with an importance weight $w_{i,t} = \pi_\theta/\pi_{\text{old}}$. Here the weight is a scalar coefficient and the gradient flows through $\log\pi_\theta$ directly; there is no clipped surrogate whose slope can go to zero, so every token contributes no matter how large $w$ is, and the entire variance problem is isolated in the magnitude of $w$ — a quantity I can bound directly. I bound it by clipping: replace $w$ with $\mathrm{clip}(w,1-\varepsilon_{\text{low}},1+\varepsilon_{\text{high}})$, so each token's coefficient is at most $1+\varepsilon_{\text{high}}$ and no single stale token can dominate the gradient. The crucial difference from PPO is that the clip now acts on the *coefficient* while the gradient still runs through $\nabla\log\pi_\theta$: a fork token with $w=5$ contributes $(1+\varepsilon_{\text{high}})\hat A\,\nabla\log\pi_\theta$, a bounded but nonzero gradient in the right direction, instead of the zero the PPO clip gave it.

There is one trap to avoid. If I literally put $\mathrm{clip}(w)\cdot\hat A\cdot\log\pi_\theta$ into the loss and let autograd differentiate it, the product rule gives two terms: the one I want, $\mathrm{clip}(w)\cdot\hat A\cdot\nabla\log\pi_\theta$, plus a spurious $\hat A\cdot\log\pi_\theta\cdot\nabla\,\mathrm{clip}(w)$. In the flat region $\nabla\,\mathrm{clip}(w)=0$, which re-creates the gating I am trying to kill; in-band $\nabla\,\mathrm{clip}(w)=w\,\nabla\log\pi_\theta$, which double-counts. The fix is that the *value* of the coefficient should be the clipped weight while *no gradient* flows through it — which is exactly a stop-gradient. So I write the coefficient as $\mathrm{sg}\!\big[\mathrm{clip}(w,1-\varepsilon_{\text{low}},1+\varepsilon_{\text{high}})\big]$, where $\mathrm{sg}$ is detach. In the forward pass it is the clipped weight and bounds the coefficient as intended; in the backward pass it is a constant, so the optimizer cannot move $\theta$ to game the clip and there is no flat region to zero a gradient. The loss to minimize is

$$L = -\,\mathrm{sg}\!\big[\mathrm{clip}(w_{i,t},\,1-\varepsilon_{\text{low}},\,1+\varepsilon_{\text{high}})\big]\cdot \hat A_{i,t}\cdot \log\pi_\theta(y_{i,t}\mid\cdot),$$

aggregated over the masked tokens, and since the only $\theta$-dependent factor is $\log\pi_\theta$, its gradient is

$$\nabla_\theta L = -\,\mathrm{sg}[\mathrm{clip}(w)]\cdot\hat A\cdot\nabla_\theta\log\pi_\theta,$$

which is exactly bounded-coefficient REINFORCE: the off-policy policy gradient with the IS weight clipped to a bounded range, every token present. The clip caps the variance; the stop-gradient keeps the gate from ever deleting a gradient.

A few design choices make this train. The clip band here is not PPO's band even though I reuse the symbol — PPO's clip sits on the surrogate and defines a trust region around $\pi_{\text{old}}$, naturally symmetric and modest at $\varepsilon\approx 0.2$, whereas my clip sits on the IS weight purely for variance control of the coefficient. The asymmetry is real: the dangerous direction is $w$ becoming *large* (a rare token the new policy now loves), which inflates the coefficient and the variance, while $w$ becoming small merely shrinks a coefficient toward zero, which is harmless. So the upper edge $1+\varepsilon_{\text{high}}$ is the load-bearing knob — it is what caps the variance — and it can be set generously, because a token exceeding it is kept with its coefficient pinned rather than dropped; the lower edge $1-\varepsilon_{\text{low}}$ is nearly inert. I read both from config (`clip_ratio_low` / `clip_ratio_high`, each falling back to `clip_ratio`) so the band can be tuned without touching the loss. For numerics, $w = \exp(\log\pi_\theta - \log\pi_{\text{old}})$ can overflow, so I clamp the log-ratio to $[-20,20]$ before exponentiating; $\exp(20)$ is already far outside any band that matters, so the clamp defuses overflow without distorting a real update. For diagnostics, `actor/ppo_kl` is the cheap $k_1$ estimate $\text{masked\_mean}(\text{old\_log\_prob} - \text{log\_prob})$, and `actor/pg_clipfrac` here means the fraction of tokens whose IS weight was actually clipped — whose coefficient was pinned — not where a surrogate branch was active. There is no dual-clip floor: token-level PPO needed a $c\hat A$ floor ($c>1$) to bound the large-ratio / negative-advantage surrogate, but here the coefficient is already bounded by $\mathrm{clip}(w)$, so that runaway cannot happen, and I return `pg_clipfrac_lower` as $0.0$ for interface compatibility. Aggregation is the flat token-mean, summing per-token losses over the response mask and dividing by the token count, every valid token weighted equally — because the ratio, the clip, and the gradient are all per-token, this is the matching aggregation, not the sequence-level `seq-mean-token-mean` that exists to mirror a per-sequence ratio I do not have. And if the trainer hands me `rollout_is_weights`, an outer correction for the rollout-engine-vs-training-engine likelihood mismatch, I multiply them onto the per-token losses before aggregation, since that correction is orthogonal to the IS clipping. The single design decision — clip the importance-sampling *weight* under a stop-gradient instead of clipping the *surrogate* — is what lets a clipped, off-policy objective keep every token's gradient.

```python
from typing import Any, Optional

import torch

import verl.utils.torch_functional as verl_F
from verl.workers.config import ActorConfig
from verl.trainer.ppo.core_algos import agg_loss, register_policy_loss


@register_policy_loss("cispo")
def compute_policy_loss_cispo(
    old_log_prob: torch.Tensor,      # (bs, response_length): log π_old per token
    log_prob: torch.Tensor,          # (bs, response_length): log π_θ  per token
    advantages: torch.Tensor,        # (bs, response_length): advantage estimate
    response_mask: torch.Tensor,     # (bs, response_length): 1 on real response tokens
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """CISPO: clip the IS weight under a stop-gradient; never zero a token's gradient."""
    assert config is not None
    assert isinstance(config, ActorConfig)
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio

    # importance ratio w = π_θ / π_old, with an overflow guard
    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)  # k1 KL diagnostic

    # KEY: clip the IS weight, then stop-gradient it so it bounds the coefficient
    # without zeroing any token's gradient. Gradient flows only through log_prob.
    clipped_ratio = torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    clipped_ratio_sg = clipped_ratio.detach()

    # objective (maximize): J = sg(clip(w)) * A * log π_θ ; loss = -J
    pg_losses = -clipped_ratio_sg * advantages * log_prob

    # fraction of tokens whose IS weight was actually clipped (coefficient pinned)
    pg_clipfrac = verl_F.masked_mean((ratio != clipped_ratio).float(), response_mask)

    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode, **config.global_batch_info,
    )

    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)  # no dual-clip floor in CISPO
    return pg_loss, {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
```
