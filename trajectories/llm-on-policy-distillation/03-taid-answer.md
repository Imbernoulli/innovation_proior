**Problem (from step 2).** RS-KD's head-faithful soft target barely moved the needle over dagger
(GSM8K 0.4541 vs 0.4511; MATH-500 0.292 vs 0.290), confirming the bottleneck is not "hard vs soft
head." Two things are still wrong: forward KL is mass-covering (bad under the capacity gap), and —
the target attacked here — the target is *fixed at the teacher* from step zero, which may be out of
reach for a 0.5B student the whole run.

**Key idea (TAID, moving target).** Replace the fixed teacher target with an *intermediate teacher*
`p_t` that is the student at `t=0` and the teacher at `t=1`, ramping `t` over training. Build it by
**logit-space** interpolation `target_logits = (1−t)·student.detach() + t·teacher`, softmax, and
minimize forward KL `KL(p_t ‖ student)`. The target stays just ahead of the student's reach the
whole way up.

**Why.** Logit-space (not probability-space) interpolation is a normalized geometric blend that
preserves relative confidence rather than a bimodal two-humped average. Early `t` ≈ self-distillation
(consolidate the student's own modes — the regime RS-KD never entered), late `t` ≈ ordinary teacher
distillation but from a student walked up to the teacher's neighborhood instead of dropped in cold —
the medicine for the capacity gap. The `t·teacher` leg is a constant signal injection that prevents
the self-distillation collapse a pure student-target recursion would suffer.

**The detach is load-bearing.** `student.detach()` inside the target stops gradient flowing through
the target's student leg; without it the student chases a target made partly of itself (a degenerate
"move the goalposts toward me" channel).

**Paper-vs-task note.** The full method keys `t`'s speed to a momentum-EMA of the student's relative
loss progress (adaptive schedule), which needs scalar state carried across steps. The edit surface is
a *stateless* loss body (only `step`/`total_steps`, nothing persistent), so the adaptive momentum is
dropped and the plain **linear** schedule is used — the non-adaptive variant the method compares
against and the one its non-collapse proof is written for.

**Hyperparameters.** `t_start = 0.4` (step-zero target ≈ 60% student / 40% teacher — skips the
trivially-easy near-student phase that would give near-zero signal, as `t_start=0.10` once did),
`t_end = 1.0`, linear ramp `t = t_start + (1−t_start)·min(1, step/total_steps)`; fixed mid-value
fallback when `total_steps` is unavailable; temperature on both sides; `batchmean` over completion
tokens.

**What to watch.** Should finally break out of the dagger/RS-KD cluster: GSM8K into the high-0.46s as
the walk-up lets the student fit teacher-like structure it could not reach cold. But it ends via
*forward* KL (mass-covering), so MATH-500 may stay flat-to-down (low-0.28s) even as GSM8K jumps — the
curriculum fixes *reachability*, not *direction*. That split is the signal the next rung must flip the
divergence to mode-seeking. AMC stays noise.

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
    # TAID (Shing et al., ICLR'25 Spotlight) — logit-space interpolation, linear schedule.
    t_start = 0.4
    if total_steps and total_steps > 0:
        t_val = t_start + (1.0 - t_start) * min(1.0, max(0.0, step / float(total_steps)))
    else:
        t_val = 0.7

    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature

    # Target logits = (1-t) * student.detach() + t * teacher.  Detach is critical so
    # the target distribution carries no gradient back through the student branch.
    t = torch.tensor(t_val, dtype=student_logits.dtype, device=student_logits.device)
    target_logits = (1 - t) * student_logits.detach() + t * teacher_logits
    target_log_probs = F.log_softmax(target_logits, dim=-1)
    student_log_probs = F.log_softmax(student_logits, dim=-1)

    # KL(target || student) = Σ_v p_target · (log p_target - log p_student)
    # F.kl_div(input=log_q, target=log_p, log_target=True) returns KL(p || q),
    # so input=student_log_probs, target=target_log_probs gives KL(target || student).
    per_token = F.kl_div(
        student_log_probs, target_log_probs, reduction="none", log_target=True
    ).sum(dim=-1)

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
