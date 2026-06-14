**Problem (from step 1).** DAgger's hard target collapsed the teacher's distribution to its argmax
and landed bottom on the reliable metrics (GSM8K 0.4511, MATH-500 0.290), worst on MATH-500 where
chains are long and the teacher is multi-modal. Give the student *soft* per-token targets — but the
full 152k-wide teacher distribution is too expensive to carry/cache per token, so the target must be
a sparse summary that reproduces full-distribution distillation.

**Key idea (RS-KD, top-K + tail bucket).** Keep the teacher's top-`K` tokens, gather the student's
log-probs on the *same* support, and append one explicit "tail" bucket on both sides holding the
residual mass `log(1 − Σ_{i∈K} p)`. Take forward KL `KL(p_T ‖ p_S)` over the resulting `K+1`-element
distributions. `K = 128`.

**Why the tail bucket.** Naive top-`K` makes the target sub-stochastic (kept mass `a < 1`), so the
forward-KL gradient `a·p_j − t_j` inflates the head by `1/a` (over-confidence) and drives the tail to
zero — the same over-sharpening disease dagger had, in a different costume. The tail bucket restores
the head gradient to the full-distribution `p_j − t_j` exactly (the residual term contributes
`(1−a)·p_j`, recombining to `p_j − t_j`) and replaces the hard tail push with a residual correction.
The target is a proper distribution again, so there is no `1/a` inflation.

**Paper-vs-task note.** The name suggests *sampling* token ids from the teacher and caching unbiased
`counts/N` (the cleanest derivation of why truncation is biased: a proposal that is zero on the tail).
This task's loss is instead the *deterministic top-K + explicit tail bucket* (`_add_tail_bucket`)
realization — same root cause (sub-stochastic target, `1/a` over-scaling) fixed by a fixed support
with carried leftover, not an unbiased random support. It recovers the *head* gradient and the *total*
tail mass, but lumps the tail into one bucket (no per-token tail signal). Cache-friendly and
deterministic, which the cost constraint demanded.

**Numerics.** Residual computed in log space as `log(−expm1(logsumexp(logp_topk)))`, with the inner
`logsumexp` clamped strictly below 0 so `1 − Σ` cannot hit zero; both `K+1` vectors renormalized in
log space before the divergence.

**Hyperparameters.** `top_k = 128`; forward KL direction (mass-covering); `batchmean` over valid
completion tokens; temperature from the signature applied to both sides.

**What to watch.** A *modest* lift over dagger (head-faithful soft targets help), but it will not
reach the reverse-KL methods: forward KL keeps it mass-covering exactly where a small student should
be mode-seeking, and the lumped tail denies it rare-token signal long competition chains need. If it
lands just above dagger on GSM8K and roughly level on MATH-500, the next moves are clear — adapt the
target to the student's reach, then flip the divergence to mode-seeking. AMC stays noise.

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
