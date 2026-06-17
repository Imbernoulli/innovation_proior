# PPO (clipped surrogate objective), per-token granularity — distilled

Proximal Policy Optimization replaces the trust-region constraint of TRPO with a clipped
first-order surrogate that can be optimized for several epochs of minibatch SGD on a single
batch of rollouts. The per-token form maintains one importance ratio per token, clips each
token's ratio independently to a band around 1, takes the pessimistic (minimum) of the clipped
and unclipped surrogate, and adds a dual floor for negative-advantage tokens. This is the
"token-level" importance-sampling granularity: ratio and clip both live at the token.

## Problem it solves

Improve a neural-network policy `pi_theta` from expensive rollouts, reusing each batch for
multiple first-order gradient steps, while keeping the update inside the region where the
old-policy batch still gives a trustworthy improvement estimate — without the second-order
machinery (Fisher-vector products, conjugate gradient) of a hard KL constraint, and compatible
with dropout and shared policy/value parameters.

## Key idea

Sampling from `pi_old` but scoring `pi_theta` is importance sampling: with ratio
`r_t(theta) = pi_theta(a_t|s_t) / pi_old(a_t|s_t)`, the off-policy surrogate is
`L^CPI = E_t[r_t A_hat_t]`, whose gradient at `r=1` is the policy gradient. This surrogate is
only a *local* model of the return (it overestimates improvement with an `O(step^2)` penalty),
so it needs a leash. Instead of constraining the global KL, clip the ratio per token and take
the worse of clipped and unclipped:

```
L^CLIP(theta) = E_t[ min( r_t A_hat_t, clip(r_t, 1-eps, 1+eps) A_hat_t ) ],   eps ~ 0.2.
```

The `min` is a pessimistic lower bound on `L^CPI`, so the objective is never over-estimated.
By the sign of the advantage it acts asymmetrically:

- `A_t > 0`: `min(r_t, 1+eps) A_t` — full penalty if a good action's probability is pushed
  down (`r_t < 1`); ceilinged at `(1+eps) A_t` once `r_t > 1+eps`, so over-improving a good
  action from one stale sample stops paying.
- `A_t < 0`: `max(r_t, 1-eps) A_t` — full penalty if a bad action's probability is pushed up
  (`r_t > 1`); floored at `(1-eps) A_t` once `r_t < 1-eps`.

`eps` is the trust-region half-width, playing the role TRPO's `delta` played but per-token and
inside the gradient. To first order around `theta_old` (`r=1`) the clip is inactive and
`L^CLIP = L^CPI`.

Clipping restrains but does not *cap* the KL, so a cheap diagnostic using the k1 estimate
`E_t[old_log_prob - log_prob]` and the clip fraction are emitted for monitoring / early stop.

**Dual clip** (for `A_t < 0` only): the two-sided clip leaves the large-ratio, negative-advantage
corner unbounded — if `r_t` blows up while `A_t < 0`, the term `r_t A_t` is a large negative
surrogate (a loss `~ r_t |A_t|`). Floor the objective at `c A_t` with `c > 1`:

```
A_t < 0:  max( min(r_t A_t, clip(r_t, 1-eps, 1+eps) A_t), c A_t ),   c ~ 3.
```

This floors the maximized objective and caps the minimized per-token loss at `c |A_t|`, so a few
stale low-probability samples can't dominate the gradient. Positive advantages need no floor
(already ceilinged at `(1+eps)A`).

## Token-level granularity

Aggregation is `token-mean`: sum the per-token clipped losses over the response mask, divide by
the token count. Every valid token contributes its own ratio and its own clip decision, equally
weighted — the importance sampling and the clipping both operate at token granularity. (A
sequence-level granularity would instead collapse each response to a single scalar ratio; that
is a different aggregation, not used here.)

## Defaults and why

- `eps = 0.2` (clip half-width) — the no-incentive band around `r = 1`; wide enough for real
  progress per step, tight enough that one stale sample can't be milked. Natural sweep grid
  `{0.1, 0.2, 0.3}`; `0.2` is the working default. Asymmetric `eps_low` / `eps_high` are
  allowed (decoupled trust region), each falling back to `eps`.
- `c = 3` (dual-clip lower bound, `c > 1`) — the floor on the negative-advantage objective,
  equivalent after negation to a loss cap; `3` leaves headroom so it engages only on genuine
  large-ratio blow-ups.
- Log-ratio clamp `[-20, 20]` before `exp` — overflow guard so one pathological token (a ratio
  that would be `inf`/`0`) cannot poison the batch; the bounds are far outside any band that
  affects a real update.

## Full algorithm (PPO, actor-critic)

```
for iteration = 1, 2, ...:
    for actor = 1..N:
        run pi_old for T steps; compute advantage estimates A_hat_1..A_hat_T
    for epoch = 1..K:
        for minibatch in collected NT samples:
            recompute log_prob = log pi_theta(a) for stored actions
            ascend L^CLIP (clipped, dual-clipped) on the minibatch  (first-order, e.g. Adam)
    pi_old <- pi_theta
```

The advantage estimate is typically a truncated generalized advantage estimate
`A_hat_t = delta_t + (gamma lambda) delta_{t+1} + ... + (gamma lambda)^{T-t+1} delta_{T-1}`,
`delta_t = r_t + gamma V(s_{t+1}) - V(s_t)`; with a shared value head the total objective adds a
value-error term and an entropy bonus, `L^{CLIP} - c1 L^{VF} + c2 S[pi_theta]`. These are inputs
to / wrappers around the policy loss below, which is the contribution.

## Working code

Filling the policy-loss slot of the rollout-and-update loop (per-token ratio, per-token clip,
dual floor, token-mean aggregation), faithful to the verl `compute_policy_loss_vanilla`:

```python
from typing import Any, Optional
import torch


def masked_mean(values, mask):
    return (values * mask).sum() / mask.sum().clamp(min=1.0)


def agg_loss(loss_mat, loss_mask, loss_agg_mode="token-mean", **kwargs):
    if loss_agg_mode == "token-mean":  # per-token: average every valid token equally
        return (loss_mat * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)
    raise NotImplementedError(loss_agg_mode)


def compute_policy_loss_vanilla(
    old_log_prob: torch.Tensor,      # (bs, length) log pi_old(a) at collection time
    log_prob: torch.Tensor,          # (bs, length) log pi_theta(a) now
    advantages: torch.Tensor,        # (bs, length) advantage estimate (given)
    response_mask: torch.Tensor,     # (bs, length) 1 for valid tokens, 0 for padding
    loss_agg_mode: str = "token-mean",
    config: Optional[Any] = None,     # clip_ratio, optional clip_ratio_low/high, clip_ratio_c
    rollout_is_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Token-level clipped PPO surrogate (with dual clip for A < 0)."""
    assert config is not None
    clip_ratio = config.clip_ratio                              # eps
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.get("clip_ratio_c", 3.0)             # c > 1, dual-clip floor for A < 0
    assert clip_ratio_c > 1.0

    negative_approx_kl = log_prob - old_log_prob               # per-token log-ratio
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)  # overflow guard
    ratio = torch.exp(negative_approx_kl)                      # r_t = pi_theta / pi_old, per token
    ppo_kl = masked_mean(-negative_approx_kl, response_mask)   # k1 KL(pi_old, pi_theta) diagnostic

    # surrogate as a LOSS (= -objective)
    pg_losses1 = -advantages * ratio                                              # -(r * A)
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)  # -(clip(r) * A)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)    # min over objective == max over loss
    pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c                    # c*|A| when A < 0
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)   # floor objective at c*A == cap loss at c*|A|
    pg_clipfrac_lower = masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)  # dual floor only for A < 0

    if rollout_is_weights is not None:                        # outer rollout-vs-train IS correction
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info,
    )
    return pg_loss, {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
```

## Relation to prior methods

- **Vanilla policy gradient / actor-critic**: PPO's surrogate reduces to it at `r = 1`
  (`grad L^CLIP = grad L^PG` to first order around `theta_old`); PPO adds the clip so the *same*
  batch survives several SGD epochs without destructive updates.
- **TRPO**: same surrogate `E_t[r_t A_t]`, but TRPO leashes it with a hard KL constraint solved
  by conjugate gradient on a Fisher-vector approximation. PPO replaces that constraint with the
  per-token clip — first-order, no curvature, dropout- and parameter-sharing-compatible,
  multi-epoch-SGD-friendly.
- **KL-penalty form** `E_t[r_t A_t - beta KL]`: an alternative leash; a fixed `beta` does not
  transfer and an adaptive `beta` re-adds a tuning loop, which is why the clip is preferred.
- **Dual clip** (`clip_ratio_c`): floors the negative-advantage objective at `c A` (`c > 1`) to
  bound the large-ratio blow-up the two-sided clip leaves open in heavily off-policy training.
