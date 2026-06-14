**Problem (from step 4).** GKD's symmetric JSD recovered MATH-500 to 0.312 (the best so far,
reversing TAID's 0.280 collapse) and nudged GSM8K to 0.4716 — confirming the *divergence direction* is
the binding axis and mode-seeking is what pays. But `β = 0.5` is only half mode-seeking; the modest
GSM8K move suggests value left on the table. Push the divergence all the way to the reverse-KL
endpoint.

**Key idea (OPD, pure reverse KL).** Per-token reverse KL `KL(p_S ‖ p_T) = Σ_v p_S(v)·(log p_S(v) −
log p_T(v))`, summed over the vocabulary, on the student's own (on-policy) tokens. Zero-forcing /
mode-seeking: the student withdraws mass from anything the teacher dislikes and commits to the
teacher's favored modes. This is the `β = 1`, on-policy corner of the (divergence × data) family the
whole ladder lives in.

**Why analytic, not policy-gradient.** Sequence reverse KL is an RL objective; its naive policy
gradient has a high-variance reward-to-go term that invites reward-hacking and a length bias toward
empty answers. But decomposing the gradient, the *immediate* term equals the per-token reverse KL,
computable *analytically* over the vocabulary from the two logit tensors with zero sampling noise; all
pathologies live in the *long-horizon* term. The teacher's signal is dense at every token, so
long-horizon credit assignment (the only reason to keep that term) is unnecessary — set the discount
to zero, drop it, and what remains is a supervised-style loss on student-visited prefixes, with none
of the policy-gradient stabilizers.

**Direction trap.** `F.kl_div(input=log_q, target=log_p, log_target=True) = KL(p ‖ q)`. For
`KL(p_S ‖ p_T)`: student log-probs are the *target*, teacher log-probs the *input* —
`kl_div(teacher_log_probs, student_log_probs, log_target=True)`. Swapping gives forward KL, the exact
mass-covering objective the ladder climbed away from.

**Paper-vs-task note.** The full recipe is sometimes written with a sampled-advantage
(`(r_t − 1)·∇log p_S`) realization and pins `lmbda = 1.0` (always on-policy). This task's body is the
*analytic vocabulary-sum* reverse KL (preferred when teacher logits are available) and does *not*
change `lmbda` — it relies on the trainer's default `lmbda = 0.5` mixing and applies reverse KL
regardless of batch source. What makes it OPD is the reverse-KL formulation, not the on-policy
fraction.

**Hyperparameters.** None beyond the signature; reverse-KL direction; shared temperature on both
sides; `batchmean` per-token mean over completion tokens (length-scale-free, matching the discount-0
design).

**What to watch.** GSM8K should clear GKD's 0.4716 by a larger margin (high-0.48s — full commitment
pays most on short near-deterministic chains). MATH-500 is the risk: full (unbounded) reverse KL could
overshoot into mode-collapse and sit roughly level with GKD (low-0.31s), plausibly a hair under. If so,
OPD is the strongest baseline on the headline metric, and the residual it leaves is the mode-collapse
risk of an *unbounded* reverse KL — motivating a reverse-KL objective whose gradient is bounded where
the teacher mass vanishes. AMC stays noise.

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
    # OPD (Lu et al. 2025 / Qwen3 report) — per-token reverse KL: KL(p_S || p_T).
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    # KL(p_S || p_T) per token, summed over vocab.
    # F.kl_div(input=log_q, target=log_p, log_target=True) = Σ_v p · (log p - log q).
    # With input=teacher_log_probs and target=student_log_probs this gives KL(p_S || p_T).
    per_token = F.kl_div(
        teacher_log_probs, student_log_probs, reduction="none", log_target=True
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
