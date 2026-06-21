The hard-target floor landed where its construction said it would: `dagger` reads GSM8K 0.4511, MATH-500 0.290, AMC 0.0813 — it functions, 595 of 1319 grade-school problems solved, but it sits at the bottom of the two reliable metrics, and MATH-500 is the tell. On the harder set, where chains are long and the teacher's per-token distribution is genuinely multi-modal, collapsing that distribution to its argmax and demanding one-hot cross-entropy left the student with the least signal of any loss I will try. So the diagnosis is clean: the hard target threw away the teacher's soft distribution, and on the problems that need it most that cost showed. The obvious move is to stop discarding the distribution — give the student soft per-token targets that preserve the relative mass on the alternatives. The wall is not modeling but cost: the faithful soft target is the teacher's full $\sim$152k-wide distribution, and materializing a forward KL over that width at every one of the millions of completion tokens in a 2000-step run is expensive and impossible to cache cheaply. I want the dagger floor's cheapness with the soft target's signal, which means a *sparse* summary of the teacher's distribution per token — a handful of numbers that nevertheless reproduces what full-distribution distillation would have done.

I propose **RS-KD** in its deterministic **top-$K$ + tail-bucket** realization. The obvious sparse summary is plain top-$K$ — keep the $K$ highest-probability teacher tokens and renormalize — and it is even $L_1$-optimal: if forced to keep $K$ entries and renormalize, the $L_1$ error to the true distribution is $2(1-a)$ where $a$ is the kept mass, minimized by keeping the $K$ largest. But $L_1$ reconstruction error is the wrong yardstick, and the gradient says why. The forward-KL logit gradient against a target $t$ is $\big(\sum_i t_i\big)\,p_j - t_j$, where I deliberately keep the leading sum un-simplified because the moment I truncate it stops being one. With a top-$K$ target the kept mass is $a = \sum_{i\in K} t_i < 1$, so the gradient is $a\,p_j - t_j$. For a token *outside* the kept set $t_j = 0$, leaving $a\,p_j$, which vanishes only at $p_j = 0$ — the loss actively drives every non-kept token, including rare ground-truth math tokens that fell outside the top-$K$, to zero. For a token *inside* the set the gradient vanishes at $p_j = t_j/a$, the teacher probability scaled *up* by $1/a > 1$. So naive top-$K$ trains the student to be over-confident on the head, inflated by $1/a$, and zeroed on the tail, and it gets worse the smaller $K$ is — exactly the known miscalibration of top-$K$ caches. This is the same over-sharpening disease dagger had, in a different costume.

What makes RS-KD work is to stop pretending the kept mass is the whole story and carry the leftover explicitly, as one extra "tail" bucket on *both* sides. I hold out the top-$K$ teacher log-probs and append a single bucket holding the residual log-mass $\log\big(1 - \sum_{i\in K} p_T(i)\big)$; I build the student's $(K{+}1)$-vector on the *same* $K$ support with its own residual $\log\big(1 - \sum_{i\in K} p_S(i)\big)$. Now both sides are proper $(K{+}1)$-element distributions, and the loss is the forward KL $\mathrm{KL}(p_T \,\|\, p_S)$ over those $K+1$ buckets. The tail bucket repairs the gradient exactly: its derivative enters only through the student residual, and on the kept tokens it contributes $(1-a)\,p_j$, which added to the top-$K$ gradient $a\,p_j - t_j$ recombines to the full-distribution gradient $p_j - t_j$. The $1/a$ up-scaling is gone and the head is supervised faithfully; on the tail tokens the combined gradient becomes a residual correction proportional to the student's own tail mass rather than a hard push to zero. Because the target is a genuine probability vector again, the leading $\sum t$ factor is one and there is no inflation. The direction stays forward KL deliberately — mass-covering is the right choice when the goal is to reproduce the teacher's *whole* distribution including its tail.

Two numerical details decide whether this is real or a `nan` factory. The residual must be computed in log space without ever exponentiating the head: $\log(1 - \sum \exp(\log p_{\text{topk}})) = \log(-\mathrm{expm1}(\mathrm{logsumexp}(\log p_{\text{topk}})))$, and the inner `logsumexp` must be clamped strictly below zero (a tiny negative ceiling) so the $1 - \sum$ inside the log can never hit zero or go negative from floating-point drift — otherwise $\log(0)$ blows up on exactly the confident positions where the head mass rounds to one. Second, the top-$K$ + tail decomposition is exact in expectation but numerical error can push the $(K{+}1)$-vector slightly off a unit sum, so I renormalize both sides in log space (subtract their own `logsumexp`) before the divergence. I pin $K = 128$: large enough that on these peaked math distributions the head plus tail bucket captures almost all the teacher's mass, small enough to keep memory tame against the 152k vocabulary, and close to the open-source RS-KD default.

I should be explicit about how this differs from the unbiased-sampling story the name evokes. The sampling view says the cure for top-$K$'s bias is to treat the summary as an importance-sampling estimate whose proposal must have support everywhere the teacher does — sample token ids from the teacher, cache the empirical $\text{counts}/N$, get a target whose gradient equals full distillation in expectation, tail included — and it is the cleanest derivation of *why* truncation is biased (a proposal that is zero on the entire tail violates the support condition). The loss I fill here is instead the deterministic top-$K$ + explicit tail bucket (the `_add_tail_bucket` trick): it addresses the *same* root cause — the sub-stochastic target and its $1/a$ over-scaling — by a fixed support with the leftover carried, rather than an unbiased random support. It is fully deterministic and cache-friendly, which is what the cost constraint demanded; it recovers the *head* gradient faithfully and gets the *total* tail mass right, but it lumps the tail into one undifferentiated bucket, so it does not reconstruct the teacher's tail *distribution* the way the sampling estimator does in expectation. For peaked math-teacher distributions, where almost all mass sits in the top tokens, that is a small price.

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
    # RS-KD (Anshumann et al., ACL'25) — sparse top-K KL with explicit tail bucket.
    top_k = 128
    eps = 1e-9
    log_one_minus_eps = -1e-7  # ensures log(1 - sum(exp(top_k))) is finite

    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    # Select teacher's top-K indices per position; gather student log-probs on the same support.
    K = min(int(top_k), teacher_log_probs.size(-1))
    teacher_topk_logp, topk_idx = torch.topk(teacher_log_probs, k=K, dim=-1)  # [B, T, K]
    student_topk_logp = torch.gather(student_log_probs, dim=-1, index=topk_idx)

    # Tail buckets: log(1 - sum(exp(top_k_logp))). Clamp the inner sum < 1 for numerical safety.
    def _tail(logp_topk):
        log_sum = torch.logsumexp(logp_topk, dim=-1, keepdim=True).clamp(max=log_one_minus_eps)
        # log(1 - exp(log_sum))  via log(-expm1(log_sum))
        return torch.log(-torch.expm1(log_sum))

    teacher_full = torch.cat([teacher_topk_logp, _tail(teacher_topk_logp)], dim=-1)  # [B, T, K+1]
    student_full = torch.cat([student_topk_logp, _tail(student_topk_logp)], dim=-1)

    # Renormalise (the topk + tail decomposition is exact in expectation but
    # numerical errors can push it slightly off; a small renorm keeps the
    # divergence well-defined).
    teacher_full = teacher_full - torch.logsumexp(teacher_full, dim=-1, keepdim=True)
    student_full = student_full - torch.logsumexp(student_full, dim=-1, keepdim=True)

    # KL(p_T || p_S) over the K+1 support.
    per_token = F.kl_div(
        student_full, teacher_full, reduction="none", log_target=True
    ).sum(dim=-1)  # [B, T]

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
