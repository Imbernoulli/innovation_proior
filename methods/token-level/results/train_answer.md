We have a stochastic policy $\pi_\theta(a\mid s)$ — a neural net with millions of parameters — and we want to push up its expected return $\eta(\theta) = \mathbb{E}\big[\sum_t \gamma^t r_t\big]$. The only handle we have is the policy gradient $\hat g = \mathbb{E}_t\big[\nabla_\theta \log \pi_\theta(a_t\mid s_t)\,\hat A_t\big]$, the score function times a given advantage estimate, averaged over a batch of trajectories. The advantage estimator $\hat A_t$ is handed to us; the thing we control is what we do with $(\text{old\_log\_prob}, \text{log\_prob}, \text{advantage})$ once a batch is in hand. And the thing that hurts is sample cost: every trajectory is expensive — environment steps in control, or for a language model an entire generation scored by a reward model — so after paying for a batch we want to wring several gradient steps out of it rather than one. That single desire, reuse the batch, is where all the trouble lives. The estimator is an empirical average over samples drawn from the policy that collected the batch, $\pi_\text{old}$, but $\eta$ is over trajectories of the current policy; the instant we take a step, $\theta$ moves, $\pi_\theta \neq \pi_\text{old}$, and the batch is stale. Ascending $L^{PG} = \mathbb{E}_t[\log\pi_\theta(a_t\mid s_t)\,\hat A_t]$ on the same trajectories for several epochs is known to produce destructively large updates: $L^{PG}$ has no term that notices the policy has drifted out of the region where the batch still tells the truth.

The honest way to see what the batch says as $\theta$ moves is importance sampling. With $r_t(\theta) = \pi_\theta(a_t\mid s_t)/\pi_\text{old}(a_t\mid s_t)$, the new policy's expected advantage on old-visited states is $\mathbb{E}_{a\sim\pi_\text{old}}[r_t(\theta)\hat A_t]$, so the quantity estimable from the frozen batch with no resampling is $L^{CPI}(\theta) = \mathbb{E}_t[r_t(\theta)\hat A_t]$. At $\theta = \theta_\text{old}$ every $r_t = 1$, and differentiating recovers the policy gradient exactly, since $\nabla r_t\,\hat A_t = (\pi_\theta/\pi_\text{old})\nabla\log\pi_\theta\,\hat A_t$ equals $\nabla\log\pi_\theta\,\hat A_t$ at $\theta_\text{old}$. So $L^{CPI}$ is a function of $\theta$ we can keep differentiating without new data. But the conservative-policy-iteration analysis shows it is only a *local* model of the return: studying the mixture update $\pi_\text{new} = (1-\alpha)\pi + \alpha\pi'$, the return's first derivative is $\tfrac{1}{1-\gamma}A_{\pi,\mu}(\pi')$ — the surrogate is the linearization around the current policy — while the improvement guarantee carries an $O(\alpha^2)$ penalty $\eta_\mu(\pi_\text{new}) - \eta_\mu(\pi) \ge \tfrac{\alpha}{1-\gamma}\big(A - \tfrac{2\alpha\gamma\epsilon}{1-\gamma(1-\alpha)}\big)$ that grows with the step. Maximizing $L^{CPI}$ with no leash therefore chases phantom improvement, driving $r_t$ as far from $1$ as it can. The clean cure — keep the surrogate but constrain the KL drift $\mathbb{E}_t[\mathrm{KL}(\pi_\text{old},\pi_\theta)] \le \delta$, since a max-KL-penalized surrogate is a genuine lower bound on the return — is right in spirit, but its implementation is the wrong tool: linearizing the objective, taking a quadratic Fisher approximation of the constraint, and running conjugate gradient with Fisher-vector products and a line search per update is heavy and slow, assumes a clean curvature estimate that dropout wrecks, breaks when policy and value heads share parameters, and is one constrained solve per batch rather than the many cheap minibatch-SGD epochs we want. Softening the constraint into a fixed-$\beta$ penalty $\mathbb{E}_t[r_t\hat A_t - \beta\,\mathrm{KL}(\pi_\text{old},\pi_\theta)]$ gives the first-order, multi-epoch property but loses the reliability: a $\beta$ that barely restrains an early steep update over-freezes a late one, and it does not transfer across tasks; an adaptive $\beta$ controller lags and re-adds a schedule to tune. The constraint is reliable but heavy; the penalty is light but unreliable.

I propose PPO — Proximal Policy Optimization — in its per-token, clipped-surrogate form. Rather than constraining a global summary of the drift, I attack the drift where it actually does damage, per sample, through the ratio itself. The damage is always mediated by $r_t$ running away from $1$ to harvest surrogate value from a single stale sample, so I stop rewarding the optimizer for pushing $r_t$ past a band $[1-\epsilon, 1+\epsilon]$ around $1$: once $r_t$ leaves the band, $\mathrm{clip}(r_t, 1-\epsilon, 1+\epsilon)$ flatlines and its gradient is zero, so there is no incentive to move further. A naive symmetric clip is exploitable, though: clipping the low side of a positive-advantage sample would *hide* the penalty the optimizer should feel for moving a good action's probability the wrong way. The fix is to take the pessimistic minimum of the clipped and unclipped surrogate,
$$L^{CLIP}(\theta) = \mathbb{E}_t\big[\min\big(r_t\hat A_t,\ \mathrm{clip}(r_t, 1-\epsilon, 1+\epsilon)\,\hat A_t\big)\big],\qquad \epsilon \approx 0.2,$$
which is always $\le L^{CPI}$, so improvement is never over-estimated, and whose behavior is asymmetric by the sign of $\hat A_t$. For $\hat A_t > 0$ the term is $\min(r_t, 1+\epsilon)\,\hat A_t$: the full unclipped penalty if a good action's probability is pushed down ($r_t < 1$), but ceilinged at $(1+\epsilon)\hat A_t$ once $r_t > 1+\epsilon$, so over-improving a good action from one stale sample stops paying. For $\hat A_t < 0$ it is $\max(r_t, 1-\epsilon)\,\hat A_t$: the full penalty if a bad action's probability is pushed up ($r_t > 1$), floored at $(1-\epsilon)\hat A_t$ once $r_t < 1-\epsilon$. Here $\epsilon$ is the trust-region half-width, playing the role TRPO's $\delta$ played but per-token and inside the gradient — I take $\epsilon = 0.2$ as the working default (sweep grid $\{0.1, 0.2, 0.3\}$), and allow decoupled $\epsilon_\text{low}$, $\epsilon_\text{high}$ since the band need not be symmetric. To first order around $\theta_\text{old}$, where every $r_t = 1$, the clip is inactive and $L^{CLIP} = L^{CPI}$, so I have not changed the gradient at the start of each batch — only what happens as the policy drifts.

Clipping restrains but does not *cap* the KL: other samples still move the policy, and the unclipped "getting worse" branch still has gradient. So I emit two cheap diagnostics for monitoring and early stopping — the $k1$ estimate $\mathbb{E}_t[\text{old\_log\_prob} - \text{log\_prob}]$ of $\mathrm{KL}(\pi_\text{old},\pi_\theta)$, read straight off the log-ratio, and the fraction of tokens where the clip was active. One quadrant still slips through the two-sided clip and bites hard in heavily off-policy, distributed training: for $\hat A_t < 0$ the clipped objective is $\max(r_t, 1-\epsilon)\hat A_t$, which protects the low side of $r_t$ but leaves the high side wide open — if $\pi_\text{old}$ gave a sampled action a tiny probability and $\pi_\theta$ now gives it much more, $r_t$ can be huge, $\max(r_t, 1-\epsilon)\hat A_t = r_t\hat A_t$ is a large negative surrogate, and a handful of stale low-probability negative-advantage tokens can dominate the minibatch loss. So I add a dual clip: floor the objective for negative-advantage tokens at $c\hat A_t$ with $c > 1$,
$$\hat A_t < 0:\quad \max\big(\min(r_t\hat A_t,\ \mathrm{clip}(r_t, 1-\epsilon, 1+\epsilon)\,\hat A_t),\ c\,\hat A_t\big),\qquad c \approx 3,$$
which caps the per-token loss at $c|\hat A_t|$ no matter how large $r_t$ grows; I take $c = 3$ (any $c>1$ works, $3$ leaves headroom so it engages only on genuine blow-ups). Positive advantages need no floor — a large $r_t$ with $\hat A_t > 0$ is already ceilinged at $(1+\epsilon)\hat A_t$.

Translating to tensor operations requires care with sign conventions, because the optimizer minimizes a *loss* while everything above is an *objective* to maximize: $\text{loss} = -\text{objective}$, so $\min$ over objectives becomes $\max$ over losses and the objective floor becomes a loss ceiling. The per-token log-ratio $\text{negative\_approx\_kl} = \text{log\_prob} - \text{old\_log\_prob}$ is clamped to $[-20, 20]$ before exponentiating — $\exp$ of a runaway log-ratio overflows, and clamping caps $r_t$ to roughly $[2\times10^{-9}, 5\times10^8]$, far outside any band that matters, so it defuses overflow without distorting a real update. Then $r_t = \exp(\cdot)$, the unclipped loss is $-\hat A_t r_t$, the clipped loss is $-\hat A_t\,\mathrm{clamp}(r_t, 1-\epsilon_\text{low}, 1+\epsilon_\text{high})$, and the pessimistic combine is their elementwise $\max$. The dual-floor objective $c\hat A_t$ becomes the loss ceiling $-\hat A_t c = c|\hat A_t|$ (positive when $\hat A_t < 0$), applied by taking the $\min$ of that with the combined loss and selecting it only where $\hat A_t < 0$ via `where`. An optional outer correction $\text{rollout\_is\_weights}$ (for rollouts generated by a third distribution whose numerics differ from the training forward pass) multiplies onto the per-token loss before aggregation, since it is an importance correction orthogonal to the clipping. Finally the aggregation decides the *granularity*: `token-mean` sums the per-token losses over the response mask and divides by the token count, so every valid token contributes its own ratio and its own clip decision, equally weighted. This is the token-level reading of the method — one importance ratio per token, one clip per token, averaged across tokens; a sequence-level aggregation collapsing each response to a single scalar ratio would be a different granularity, not used here.

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
