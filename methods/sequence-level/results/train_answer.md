I have been training a large language model with reinforcement learning — roll out responses from a frozen old policy $\pi_{\text{old}}$, score them with a verifier, partition the rollout batch into mini-batches, and take several PPO-style clipped-surrogate steps on the current policy $\pi_\theta$, reweighting each off-policy contribution by an importance ratio and clipping it to a proximal band around $1$. At moderate scale this is fine; at large scale it falls apart. Big models, sparsely activated mixture-of-experts, long responses: the reward climbs for a while and then the whole run collapses, and the collapse is *irreversible*. Reverting to an earlier checkpoint, retuning the clip ranges, lengthening the generation budget, swapping the query set — none of it brings the model back. That irreversibility is the diagnostic tell. A merely noisy optimizer would wobble and recover; a collapse that repeats even after rewinding and changing the obvious knobs is a structural corruption of the gradient being computed step after step, not a bad learning rate. So the right question is not "what schedule fixes this" but "is the gradient I am computing even the gradient I think it is."

The recipe in use is GRPO, which removes PPO's expensive, fragile critic by reading the advantage off a group: sample $G$ responses to the same query, score each, and set $\hat A_i = (r(x,y_i) - \operatorname{mean}_j r(x,y_j)) / \operatorname{std}_j r(x,y_j)$, one scalar shared by every token of $y_i$. It then optimizes the PPO surrogate with the per-token ratio $w_{i,t}(\theta) = \pi_\theta(y_{i,t}\mid x,y_{i,<t}) / \pi_{\text{old}}(y_{i,t}\mid x,y_{i,<t})$, clipped per token. The one piece doing the off-policy correction, and the one I trust least, is that per-token ratio. Importance sampling is a precise statement, $\mathbb{E}_{z\sim\pi_{\text{tar}}}[f(z)] = \mathbb{E}_{z\sim\pi_{\text{beh}}}[(\pi_{\text{tar}}(z)/\pi_{\text{beh}}(z))\,f(z)]$, and the reweighting only realizes its distribution-correction role *in aggregate*: the right-hand side is an expectation over $\pi_{\text{beh}}$, recovered in practice by averaging the weighted function over many samples. A single weighted draw is unbiased but can have enormous variance. Now look at $w_{i,t}$: at each position there is exactly one realized token $y_{i,t}$ drawn from that position's old distribution, so $w_{i,t}$ is built from a single sample with no averaging over the next-token distribution. Inside the clipped objective it stops behaving like a distribution-correction device and behaves like a noisy per-token gradient multiplier. These weights do not cancel — for a positive-advantage token the active range is $(0, 1+\varepsilon]$ and for a negative-advantage token it is $[1-\varepsilon, \infty)$ — and they accumulate: a long response is a long run of noisy factors, so the injected variance grows with length. Per-token clipping does not damp this; it amplifies it, because the clip decides token by token whether to keep or zero each contribution based on whether that *noisy* ratio crossed a threshold. The longer the response and the larger the model, the more this manufactured noise dominates the learning signal, until the update walks the model somewhere it cannot recover from. That is the irreversible collapse.

The diagnosis is therefore sharper than "GRPO is unstable": GRPO applies its importance weight at a granularity where there is no averaging over the behavior distribution, one sample per next-token distribution, and the consequence is high-variance gradient noise that scales with length and is worsened by clipping. The fix cannot be a better clip range. It has to move the off-policy correction to a granularity where the weight is a coherent sample-level quantity. And there is an obvious such granularity hiding in plain sight: the reward. The verifier scores the *whole response* and the group-relative advantage is one scalar per response, so the unit of reward is the sequence, while the importance weighting and clipping live at the token. That is a mismatch of units. The principle I propose to enforce is that the unit of optimization should match the unit of reward — so do importance sampling per sequence.

The method is GSPO, Group Sequence Policy Optimization: importance sampling, clipping, and optimization performed at the sequence level. The natural per-sequence object is $\pi_\theta(y_i\mid x)/\pi_{\text{old}}(y_i\mid x)$, a single coherent measure of how off-policy the whole response is, aligned with the sequence-level reward and advantage. But I cannot drop it into the surrogate as-is, and the reason exposes the one real design choice. The response likelihood factorizes, $\pi_\theta(y_i\mid x) = \prod_t \pi_\theta(y_{i,t}\mid x,y_{i,<t})$, so the ratio is a *product* of $|y_i|$ per-token ratios and its log is a *sum* of per-token log-ratios. Summing hundreds or thousands of small fluctuating log-ratios lets a few tokens that shift appreciably lurch the total, and exponentiating gives a sequence ratio wildly far from $1$ — even more volatile in magnitude than the per-token version. Worse, the sum scales with $|y_i|$, so a 50-token and a 2000-token response live at different numerical scales and no single clip band serves both. Both problems point the same way: divide by the length. Define the per-token *average* log-ratio and exponentiate it, giving the geometric mean of the per-token ratios,

$$s_i(\theta) = \left(\frac{\pi_\theta(y_i\mid x)}{\pi_{\text{old}}(y_i\mid x)}\right)^{1/|y_i|} = \exp\!\left(\frac{1}{|y_i|}\sum_t \log\frac{\pi_\theta(y_{i,t}\mid x,y_{i,<t})}{\pi_{\text{old}}(y_{i,t}\mid x,y_{i,<t})}\right).$$

The $1/|y_i|$ is not cosmetic; it is what makes a sequence-level ratio usable at all. It does two jobs the raw product failed at. First, the scale is now per-token regardless of response length, so short and long responses produce $s_i$ in the same numerical range and one clip band $\varepsilon$ serves both. Second, it is a variance reducer: averaging $|y_i|$ fluctuating log-ratios is far more stable than summing them, since a few hard-swinging tokens can drag a sum dramatically but barely move a mean over hundreds of tokens. The objective is then the PPO clipped surrogate with $s_i$ in place of the per-token ratio, the group-relative advantage, and both the surrogate and the clip at the sequence level,

$$J_{\text{GSPO}}(\theta) = \mathbb{E}_{x\sim D,\,\{y_i\}\sim\pi_{\text{old}}(\cdot\mid x)}\left[\frac{1}{G}\sum_i \min\!\Big(s_i(\theta)\,\hat A_i,\ \operatorname{clip}\big(s_i(\theta), 1-\varepsilon, 1+\varepsilon\big)\,\hat A_i\Big)\right],$$

so the unit being clipped — an entire response that is too off-policy as a whole — matches the unit being rewarded. Because $s_i$ is the exponential of an *average* per-token log-ratio it hugs $1$ extremely closely (the spread is on the order of the per-token KL), so the clip band here must be far narrower than GRPO's: left/right edges of roughly $3\times10^{-4}/4\times10^{-4}$ for the sequence ratio versus $0.2/0.27$ for GRPO's token ratio, with asymmetric (decoupled) edges allowed. Same clipping idea, different natural scale, purely from how the ratio is defined; $\varepsilon$ is read from config, never hardcoded.

What makes me confident this attacks the actual failure is the gradient comparison. Dropping the clip (a region selector, not the heart of the gradient), the log-derivative trick gives $\nabla_\theta s_i = s_i \nabla_\theta \log s_i$, and since only the $\pi_\theta$ terms depend on $\theta$, $\nabla_\theta \log s_i = (1/|y_i|)\sum_t \nabla_\theta \log \pi_\theta(y_{i,t}\mid\cdot)$, so

$$\nabla_\theta J_{\text{GSPO}} = \mathbb{E}\!\left[\frac{1}{G}\sum_i s_i(\theta)\,\hat A_i \cdot \frac{1}{|y_i|}\sum_t \nabla_\theta \log \pi_\theta(y_{i,t}\mid\cdot)\right],$$

against GRPO's

$$\nabla_\theta J_{\text{GRPO}} = \mathbb{E}\!\left[\frac{1}{G}\sum_i \hat A_i \cdot \frac{1}{|y_i|}\sum_t w_{i,t}(\theta)\,\nabla_\theta \log \pi_\theta(y_{i,t}\mid\cdot)\right].$$

Both are, per response, $\hat A_i$ times a token-average of (weight) $\cdot \nabla_\theta \log \pi_\theta(y_{i,t}\mid\cdot)$. The *entire* difference is the weight on each token's score gradient: GRPO weights tokens unequally by their individual noisy $w_{i,t}$, GSPO weights every token of a response equally by the single $s_i$. The unequal active weights are exactly the instability factor I traced; collapsing them to one $s_i$ per response removes it. The same change explains the violent MoE failure. There, after a single gradient update around $10\%$ of the activated experts can change for the same response (measured on a 48-layer Qwen3-30B-A3B model), and since each token's likelihood depends on which experts fired, the per-token $w_{i,t}$ swing drastically across an update — numerator and denominator are effectively evaluated through different sub-networks, catastrophic for a scheme whose gradient weight *is* the per-token ratio. The usual remedy is routing replay: cache $\pi_{\text{old}}$'s routing and force $\pi_\theta$ to reuse it, at a cost of memory, communication, and usable capacity. But $s_i$ uses the length-normalized aggregate sequence likelihood, not any exposed per-token weight, so individual expert reshuffles are pooled in the averaged log-ratio and the workaround's reason largely disappears. The same averaging also makes the loss tolerant of small rollout-vs-training likelihood discrepancies, since they are averaged before they become the ratio — useful in partial-rollout, multi-turn, and disaggregated stacks. One honest consequence: sequence-level clipping drops *whole responses*, so it can mark far more token positions as clipped than per-token clipping; that is acceptable because the dropped tokens belong to a response that is off-policy as a whole, the coherent unit to judge, whereas tokens retained under per-token clipping can still carry the unreliable noisy weights.

The implementation needs care, because $s_i$ is one scalar per sequence but autodiff runs over per-token log-probs, and I want the forward value to be $s_i$ broadcast to every token (so the clip and surrogate use the sequence ratio) while the backward path flows through per-token $\log\pi_\theta(y_{i,t}\mid\cdot)$ with the equal weight $s_i$. The flexible way to get both — and to leave room for a per-token advantage later without changing the numbers — is a straight-through token-level surrogate, $s_{i,t}(\theta) = \operatorname{sg}[s_i(\theta)]\cdot \pi_\theta(y_{i,t}\mid\cdot)/\operatorname{sg}[\pi_\theta(y_{i,t}\mid\cdot)]$ with $\operatorname{sg}[\cdot]$ the stop-gradient (PyTorch `detach`). Numerically the second factor is $1$, so $s_{i,t} = s_i$ token-for-token in the forward pass; the gradient is carried only by the un-detached numerator, giving $s_i\,\nabla_\theta\log\pi_\theta(y_{i,t}\mid\cdot)$ at the evaluation point, which collapses to the GSPO gradient when $\hat A_{i,t} = \hat A_i$. In log space, which is what I code since I already have per-token log-ratios, $\log s_{i,t} = \operatorname{sg}[\log s_i] + \log\pi_\theta - \operatorname{sg}[\log\pi_\theta]$. With `negative_approx_kl` $= \log\pi_\theta - \log\pi_{\text{old}}$ per token and its masked sequence mean as $\operatorname{sg}[\log s_i]$, the per-token surrogate ratio's value is $s_i$ and its gradient is the per-token score function; I clamp the log at a safe ceiling of $10$ before exponentiating so a pathological row cannot overflow. The rest is the ordinary clipped surrogate — $-\hat A\cdot$ ratio versus $-\hat A\cdot$ clipped-ratio, elementwise max (max of negatives is min of surrogates) — with one non-negotiable aggregation choice: seq-mean-token-mean, taking each sequence's token-mean of its per-position losses and then the mean over sequences, mirroring the objective's two-level $(1/G)(1/|y_i|)$ average. A flat token-mean over the batch would let long responses dominate in proportion to their length and undo the very length normalization $s_i$ was built to have.

```python
from typing import Any, Optional

import torch
import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.core_algos import agg_loss, register_policy_loss
from verl.workers.config import ActorConfig


@register_policy_loss("gspo")
def compute_policy_loss_gspo(
    old_log_prob: torch.Tensor,      # (bs, response_length): log π_old per token
    log_prob: torch.Tensor,          # (bs, response_length): log π_θ  per token
    advantages: torch.Tensor,        # (bs, response_length): group-relative advantage
    response_mask: torch.Tensor,     # (bs, response_length): 1 on real response tokens
    loss_agg_mode: str = "seq-mean-token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute the clipped policy objective and related metrics for GSPO."""
    assert config is not None
    assert isinstance(config, ActorConfig)
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio

    # per-token log-ratio  log π_θ − log π_old  (the negative approximate KL)
    negative_approx_kl = log_prob - old_log_prob

    # length-normalized sequence log-ratio: (1/|y_i|) Σ_t (log π_θ − log π_old), masked mean
    seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
    negative_approx_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths

    # straight-through: forward value is the broadcast sequence ratio s_i;
    # backward path is the per-token log π_θ, so each token's score gradient is weighted by s_i.
    #   log s_{i,t} = sg[log s_i] + log π_θ − sg[log π_θ]
    log_seq_importance_ratio = (
        log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
    )
    log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10.0)
    seq_importance_ratio = torch.exp(log_seq_importance_ratio)

    pg_losses1 = -advantages * seq_importance_ratio
    pg_losses2 = -advantages * torch.clamp(seq_importance_ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)       # max of negatives = min of surrogates

    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    # (1/G) Σ_i (1/|y_i|) Σ_t [...]  → per-seq token-mean, then mean over sequences
    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask,
        loss_agg_mode="seq-mean-token-mean", **config.global_batch_info,
    )

    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics
```
