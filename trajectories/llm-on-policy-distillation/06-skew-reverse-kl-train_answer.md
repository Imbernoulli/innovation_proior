OPD closed the ladder where its construction pointed, and the residual it left is a specific, fixable instability rather than a vague "try harder." `opd` reads GSM8K 0.4852, MATH-500 0.310: the GSM8K prediction landed — 0.4852 is the best on the ladder, clearing GKD's 0.4716 by the larger margin the divergence going all the way to reverse KL was supposed to buy, since full commitment to the teacher's preferred continuation pays most on short, near-deterministic arithmetic chains. But MATH-500 came in at 0.310, a hair *under* GKD's 0.312 — the mode-collapse tax I flagged: pure reverse KL overshot half-JSD's balanced compromise and committed to slightly too few of the teacher's reasoning modes on the hard set. The whole-ladder picture is now unambiguous — the divergence direction is the binding axis, more mode-seeking lifts GSM8K, but *unbounded* reverse KL pays a small MATH-500 tax by overshooting into collapse. The gap to close is precise: keep OPD's mode-seeking commitment, which won GSM8K, while stopping it from collapsing on the hard set. The way it overshoots tells me the mechanism, so read the gradient. OPD's reverse KL $\sum_v p_S(\log p_S - \log p_T)$ has parameters in both the outer weight $p_S$ and the inner $\log p_S$; collapsing onto the sampled sequence gives a per-token coefficient of the form $\log r + 1$ with $r = p_S/p_T$ the student-to-teacher ratio. That coefficient runs away when $p_T \to 0$ — when the teacher assigns near-zero probability to what the student produced — and the on-policy loop *guarantees* this is the typical case, not an edge case: the batches are the student's own generations, sequences the 7.6B teacher did not write, so on a self-generated math chain the teacher is frequently surprised. OPD's gradient detonates on its own training data precisely where the student is exploring, throwing huge noisy steps that yank it toward whatever narrow region the teacher endorses — the mechanism of the measured MATH-500 collapse.

I propose **Skewed Reverse KL (SRKL)**. The blow-up has one exact source: a *raw probability sits in a denominator and goes to zero* — the teacher $p_T$ in the ratio $p_S/p_T$. The structural cure is to not let the raw teacher distribution be the denominator; replace it with a *mixture* that always carries a sliver of the student, flooring the denominator away from zero. Take the reverse KL of the student against a skewed mixture $\tilde p = (1-\alpha)\,p_T + \alpha\,p_S$:
$$D_{\mathrm{SRKL}}^{\alpha}(p_T, p_S) = \mathrm{KL}\big(p_S \,\|\, (1-\alpha)\,p_T + \alpha\,p_S\big),\qquad \alpha = 0.1.$$
At $\alpha = 0$ this is exactly OPD's reverse KL; on a token where $p_T \approx 0$ the mixture still carries the $\alpha\,p_S$ leg, so $\tilde p \ge \alpha\,p_S$, the ratio $r = p_S/\tilde p \le 1/\alpha$ is *bounded*, and the log can no longer diverge.

What makes it work is that the skewed gradient does what I claim with no other moving part. Differentiating $\mathrm{KL}(p_S \,\|\, \tilde p)$, where the $\alpha\,p_S$ leg of $\tilde p$ also depends on the parameters, the reverse-KL product rule still gives the $\log r + 1$ structure but with $r = p_S/\tilde p$, and the $\theta$-dependence in the $\alpha\,p_S$ leg adds a term, leaving a coefficient
$$\log r + 1 - \alpha\,r,\qquad r = p_S/\tilde p.$$
Two things help at once. First, $r \le 1/\alpha$ is bounded, so $\log r$ cannot run to infinity the way $\log(p_S/p_T)$ did when the teacher vanished. Second, the new $-\alpha\,r$ term subtracts and grows with $r$, actively pulling the coefficient back down exactly where it would otherwise spike. So skewed reverse KL has a bounded, well-behaved gradient on precisely the teacher-surprised on-policy tokens that collapsed OPD on MATH-500 — and it keeps reverse KL's mode-seeking weighting, still weighted by $p_S$, still zero-forcing, still committing to the teacher's modes, which is what won GSM8K. It removes the detonation without removing the commitment.

The size of $\alpha$ is a genuine trade-off, not a free parameter, because a floor that is too high stops me distilling the teacher at all, and the loss is estimated on mini-batches so a high-variance estimator gives a noisy objective. The empirical $L_2$ error of the skewed estimator, under mild assumptions, carries inverse-$\alpha$ terms: moving $\alpha$ away from zero shrinks the estimation error from the unprotected raw denominator, so larger $\alpha$ both bounds the gradient and tightens the mini-batch estimate. That argues for $\alpha$ not too small. But pushing $\alpha$ toward one is wrong for a reason specific to the optimizer: an Adam-style optimizer normalizes by a running estimate of the gradient scale, so a uniformly smaller gradient coefficient is divided back out and the gradient-shrinkage "benefit" of large $\alpha$ is largely compensated away. Looking at the estimation error in the units the optimizer actually steps in — the $L_2$ norm normalized by the gradient scale — the inverse-$\alpha$ pieces want $\alpha$ large while inverse-$(1-\alpha)$ pieces want $\alpha$ small, a convex trade-off with an interior optimum: small enough that the target is still mostly the teacher (so I am distilling the teacher, not a teacher-student blend), large enough to floor the denominator and cut the variance. A mild $\alpha = 0.1$ sits in that interior — the comparison distribution is $\tilde p = 0.9\,p_T + 0.1\,p_S$, 90% faithful to the teacher with just enough student mixed in to bound the ratio.

This also tells me why skewing is not GKD's generalized JSD with a different number, so I am not re-running rung four. JSD is a *sum of two skewed KLs whose skew parameters are tied to one $\beta$* — schematically $\beta\cdot\mathrm{SKL}^{\beta} + (1-\beta)\cdot\mathrm{SRKL}^{1-\beta}$ — so the single $\beta$ that sets the skew of the forward leg forces the complementary skew on the reverse leg. I cannot make *both* legs mildly skewed: a mild $\alpha = 0.1$ on the reverse term forces $0.9$ on the forward term, slamming that leg into the high-skew regime. The estimation-vs-$\alpha$ analysis says I want a *mild* skew on the one term I am actually using, and a single freely-chosen-$\alpha$ skewed reverse KL reaches that interior optimum that JSD's coupled parameter structurally cannot. GKD's $\beta = 0.5$ balanced two *unskewed* KL directions; this skews *one* mode-seeking direction by a small, freely tuned amount — the knob the OPD failure mode actually calls for.

The arithmetic mirrors OPD's reverse-KL body with the denominator swapped for the mixture. Divide both logit tensors by the shared temperature, softmax to $p_T$ and $p_S$, form $\tilde p = (1-\alpha)\,p_T + \alpha\,p_S$ in probability space and take its log, then accumulate the per-token reverse KL $\sum_v p_S(\log p_S - \log\tilde p)$ — keeping *both* legs, because unlike a forward KL the student-entropy leg $\sum_v p_S\log p_S$ depends on the parameters and carries the $+1$ normalization gradient I derived, so I must not drop it. I guard any $\pm\infty$ logit positions so they contribute zero rather than `nan`, sum over the vocabulary to a per-token value, mask to completion tokens, and average per token for `batchmean`. The direction is pinned by construction: the student is the outer weight and the mixture is the comparison, so this is genuinely $\mathrm{KL}(p_S \,\|\, \tilde p)$, the skewed *reverse* direction, not its forward cousin.

One scope note: the full streamlined recipe pairs this skewed KL with two data-side mechanisms — an adaptive student-generated-output scheduler that ramps self-generation guided by validation loss, and an off-policy replay buffer with a decaying replay ratio to amortize generation cost — both of which need framework-level access to the data pipeline that this single-loss surface does not expose. So I drop them and land only the *loss*: the skewed reverse KL at $\alpha = 0.1$, computed on whatever batch the trainer's static $\lambda$ mixing produced, exactly as every prior rung consumed that mixing. The synergy is real — the skewed loss's fast, stable early movement is what would let an off-policy buffer work — but the per-token loss stands on its own.

```python
import torch
import torch.nn.functional as F


def compute_distill_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor = None,
    beta: float = 0.5,
    temperature: float = 1.0,
    reduction: str = "batchmean",
    step: int = 0,
    total_steps: int = 0,
    lmbda: float = 0.5,
) -> torch.Tensor:
    # Skewed Reverse KL (SRKL, Ko et al., ICML'24) — KL(p_S || (1-a)*p_T + a*p_S), a = 0.1.
    # Mode-seeking like OPD, but the a*p_S leg floors the teacher-side denominator so the
    # gradient coefficient log r + 1 - a*r (r = p_S/p~ <= 1/a) cannot explode where p_T ~= 0
    # (the typical case on student on-policy rollouts), removing OPD's mode-collapse tax.
    alpha = 0.1

    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(student_logits, dim=-1, dtype=torch.float32)
    mixed_probs = (1 - alpha) * teacher_probs + alpha * student_probs        # p~ = (1-a)p_T + a p_S

    student_log_probs = F.log_softmax(student_logits, dim=-1, dtype=torch.float32)
    mixed_log_probs = torch.log(mixed_probs.clamp_min(1e-9))

    # KL(p_S || p~) = Sum_v p_S * (log p_S - log p~). Keep BOTH legs: the p_S*log p_S leg
    # depends on theta and carries the reverse-KL +1 normalization gradient.
    inf_mask = torch.isinf(student_logits) | torch.isinf(teacher_logits)     # guard -inf logits
    prod = torch.masked_fill(student_probs * student_log_probs, inf_mask, 0)
    prod -= torch.masked_fill(student_probs * mixed_log_probs, inf_mask, 0)
    per_token = prod.sum(dim=-1)                                             # [B, T], = +KL(p_S || p~)

    if labels is not None:
        mask = labels != -100
        per_token = per_token[mask]
        denom = mask.sum().clamp_min(1)
    else:
        denom = torch.tensor(max(per_token.numel(), 1), device=per_token.device, dtype=per_token.dtype)

    if reduction == "batchmean":
        return per_token.sum() / denom
    elif reduction == "sum":
        return per_token.sum()
    elif reduction == "mean":
        return per_token.mean()
    return per_token
```
