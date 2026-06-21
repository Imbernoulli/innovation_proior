The truncation rung came back at $\text{score\_mean} = -0.751$, the worst result on this ladder, and its per-benchmark breakdown was exactly the fingerprint of a length-dependent bias: GSM8K held up (0.4391), while MATH-500 (0.2875) and AMC (0.0828) sagged hardest. A 64-token prefix covers most of a short GSM8K answer but only a fraction of a multi-step MATH or AMC solution, so freezing the rest discards precisely the late-token gradient that earns those rewards. The late tokens were carrying *signal*, not drift-noise — at least in this small, dense 0.5B regime where the per-token ratios were never that volatile to begin with. So the lesson is sharp: stop discarding tokens. The question is no longer *which* tokens to trust but *how* to combine the per-token ratios so the combination is less noisy without throwing any away.

The thing to stare at is what the per-token weight actually is. Importance sampling is the identity $\mathbb{E}_{z\sim\pi_{\text{tar}}}[f(z)] = \mathbb{E}_{z\sim\pi_{\text{beh}}}\big[\tfrac{\pi_{\text{tar}}(z)}{\pi_{\text{beh}}(z)} f(z)\big]$, and the reweighting corrects the distribution mismatch only *in aggregate* — the right-hand side is an expectation realized by averaging the weighted function over many samples. A single weighted draw is unbiased but can have enormous variance; averaging is the mechanism, not decoration. Now look at the per-token weight $w_{i,t} = \pi_\theta(y_{i,t}\mid\cdot)/\pi_{\text{old}}(y_{i,t}\mid\cdot)$. At each position there is exactly *one* realized token drawn from that position's old next-token distribution, so $w_{i,t}$ is a one-sample IS estimate with no averaging over the next-token distribution — all variance, no aggregation. Inside the clipped objective it stops behaving like a distribution-correction and behaves like a noisy per-token gradient multiplier, and a long response stacks $|y_i|$ of these noisy factors into one gradient. Truncation cut the stack short and paid in bias; the better move is to keep the whole stack but *average* it into a single coherent weight, which is exactly the aggregation IS theory says is the real mechanism.

Where the average should live follows from what is being rewarded. The verifier scores the whole response and returns one number; the GRPO advantage $\hat A_i$ is one scalar shared across every token. The reward lives at the *sequence* level, yet the per-token clip does its correction and clipping at the *token* level — a unit mismatch, rewarding a sequence but correcting a token. If the off-policy correction is to be coherent, its unit should match the unit of the reward, so I do importance sampling per sequence.

I propose **GSPO**, sequence-level importance sampling. The natural per-sequence ratio is $\pi_\theta(y_i\mid x)/\pi_{\text{old}}(y_i\mid x)$, but it cannot go into the surrogate raw, and the reason is instructive. The response likelihood is a product over $|y_i|$ tokens, so the ratio is a product of per-token ratios and its log is a *sum* of per-token log-ratios; summing a few hundred small fluctuating numbers compounds tiny shifts into a huge swing, making the raw sequence ratio even more volatile than the per-token one, and its magnitude scales with $|y_i|$ so no single clip band fits both a 50-token answer and a 500-token solution. Both problems — volatility and length-scaling — point to dividing the summed log-ratio by the length. I define the length-normalized sequence ratio as the geometric mean of the per-token ratios,

$$s_i(\theta) = \left(\frac{\pi_\theta(y_i\mid x)}{\pi_{\text{old}}(y_i\mid x)}\right)^{1/|y_i|} = \exp\!\left(\frac{1}{|y_i|}\sum_t \log\frac{\pi_\theta(y_{i,t}\mid\cdot)}{\pi_{\text{old}}(y_{i,t}\mid\cdot)}\right).$$

The $1/|y_i|$ is not cosmetic; it does two jobs at once. It puts the scale on a per-token footing regardless of response length, so a 50-token and a 500-token response produce $s_i$ in the same range and one clip band serves both — the direct fix for the length-dependent failure truncation papered over by deletion. And it is a variance reducer: averaging $|y_i|$ fluctuating log-ratios is far more stable than summing them, so a few tokens swinging hard barely move a mean over hundreds. The clipped surrogate keeps the PPO shape but with ratio, clip, and aggregation all at the sequence level, $J = \mathbb{E}\big[\tfrac{1}{G}\sum_i \min(s_i \hat A_i,\, \mathrm{clip}(s_i, 1-\varepsilon, 1+\varepsilon)\hat A_i)\big]$. One caution about the band: because $s_i$ is the exponential of an *average* of small per-token log-ratios, it hugs 1 far more tightly than a single per-token ratio, so the sensible clip band here is much narrower than the token-level $\varepsilon$ — which is why I read it from config rather than bake it in.

What convinces me this is the right axis is the gradient comparison. Dropping the clip (a region selector, not the heart of the gradient), $\nabla_\theta s_i = s_i \nabla_\theta \log s_i$ with $\nabla_\theta \log s_i = \tfrac{1}{|y_i|}\sum_t \nabla_\theta \log\pi_\theta(y_{i,t}\mid\cdot)$, so $\nabla_\theta J_{\text{seq}} = \mathbb{E}\big[\tfrac{1}{G}\sum_i s_i \hat A_i \cdot \tfrac{1}{|y_i|}\sum_t \nabla_\theta\log\pi_\theta\big]$. The per-token (GRPO) gradient is $\mathbb{E}\big[\tfrac{1}{G}\sum_i \hat A_i \cdot \tfrac{1}{|y_i|}\sum_t w_{i,t}\nabla_\theta\log\pi_\theta\big]$. Both are $\hat A_i$ times a token-average of (weight)$\cdot$(score gradient); the *entire* difference is the weight on each token's score. The per-token method uses the individual noisy $w_{i,t}$, a different number per token; the sequence method uses the single $s_i$, the same number for every token in the response. The unequal per-token weights are the instability factor — they do not cancel and their position-by-position variation accumulates — and collapsing them to one $s_i$ removes that factor entirely while keeping every token in the gradient. This is the exact opposite remedy from truncation: truncation kept the noisy weights but deleted most tokens; GSPO keeps every token but deletes the weight disparity.

The implementation has one subtlety: $s_i$ is one scalar per sequence, but the autodiff graph is built over per-token log-probs. I want the forward value of the per-token ratio to be the broadcast $s_i$ (so the clip and surrogate use the sequence ratio) while the backward path flows through the per-token $\log\pi_\theta$ with equal weight $s_i$. The straight-through construction does exactly this. With $\text{negative\_approx\_kl} = \text{log\_prob} - \text{old\_log\_prob}$ the per-token log-ratio and $\text{neg\_kl\_seq}$ its masked per-sequence mean, write $\log s^{\text{eff}}_{i,t} = \text{log\_prob} - \mathrm{sg}[\text{log\_prob}] + \mathrm{sg}[\text{neg\_kl\_seq}]$ broadcast over the token axis. The detached terms set the forward value to $\mathrm{sg}[\log s_i]$; the live $\text{log\_prob} - \mathrm{sg}[\text{log\_prob}]$ is zero in value but carries the per-token gradient. Exponentiating gives a per-token ratio whose value is $s_i$ and whose gradient is the per-token score function — precisely the gradient derived above. I clamp $\log s^{\text{eff}}$ at a ceiling around 10 before `exp`, since legitimate values sit microscopically close to 0 in log space and only a pathological row could overflow. Then the ordinary clipped machinery on $\text{seq\_ratio} = \exp(\log s^{\text{eff}})$: $\text{pg\_losses1} = -\hat A\cdot\text{seq\_ratio}$, $\text{pg\_losses2} = -\hat A\cdot\mathrm{clamp}(\text{seq\_ratio}, 1-\varepsilon_{\text{low}}, 1+\varepsilon_{\text{high}})$, and $\max$ of the two (max of negatives = min of surrogates). I keep no dual-clip floor — like the truncation rung this is the minimal clipped variant for its granularity, and the sequence ratio's tight band around 1 already limits the runaway that floor guarded against.

The one piece that *must* follow the sequence structure is the aggregation, and getting it wrong would silently undo the method. The objective is $\tfrac{1}{G}\sum_i[\text{per-response token-mean}]$, a two-level average, so I aggregate with `seq-mean-token-mean`: per sequence, take the token-mean of its per-position losses; then mean over sequences. A flat `token-mean` over the whole batch would let long responses dominate in proportion to their length — re-introducing exactly the length bias the $1/|y_i|$ was built to remove. So I pass `loss_agg_mode="seq-mean-token-mean"` to `agg_loss`, even though the contract's default is `token-mean`.

The falsifiable expectation against rung 1: keeping every token should erase truncation's long-response bias, and replacing the noisy per-token weight with one stable per-sequence weight should lower gradient variance, so I expect to clear $-0.751$ comfortably and — the sharp test — to recover *exactly* the long-response benchmarks truncation crushed, with MATH-500 and AMC rising most while GSM8K holds. The honest caveat I carry forward is that the variance-reduction story for sequence-level IS was forged in the large-MoE, long-response regime; on a small dense 0.5B model where per-token noise was modest, GSPO may cleanly beat truncation yet not dominate a *plain* per-token clip — a tension the next rung resolves.

```python
# EDITABLE region of custom_policy_loss.py — rung 2: sequence-level IS (GSPO)
from typing import Any, Optional

import torch

import verl.utils.torch_functional as verl_F
from verl.workers.config import ActorConfig
from verl.trainer.ppo.core_algos import agg_loss, register_policy_loss


@register_policy_loss("custom")
def compute_custom_policy_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Sequence-level IS (GSPO): one scalar ratio per sequence."""
    assert config is not None
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio

    negative_approx_kl = log_prob - old_log_prob
    seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
    neg_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths

    # straight-through: keep per-token log_prob gradient, ratio value is per-sequence
    log_seq_ratio = log_prob - log_prob.detach() + neg_kl_seq.detach().unsqueeze(-1)
    log_seq_ratio = torch.clamp(log_seq_ratio, max=10.0)
    seq_ratio = torch.exp(log_seq_ratio)

    pg_losses1 = -advantages * seq_ratio
    pg_losses2 = -advantages * torch.clamp(seq_ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    # GSPO aggregates at the sequence level (seq-mean-token-mean)
    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask,
        loss_agg_mode="seq-mean-token-mean", **config.global_batch_info,
    )
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    return pg_loss, {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": 0.0,
    }
```
