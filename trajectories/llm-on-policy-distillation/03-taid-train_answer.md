RS-KD did what its construction predicted, and the *size* of the move is the diagnosis. `rs_kd` reads GSM8K 0.4541, MATH-500 0.292 — against `dagger` (0.4511 / 0.290) the head-faithful soft target bought almost nothing, $+0.003$ and $+0.002$, within the single-seed noise floor. Replacing the hard one-hot with the teacher's true relative mass on its top 128 tokens, with the $1/a$ over-confidence cured, was correct and near-free, which confirms the soft head carries real information — but it tells me the bottleneck is not "hard vs soft head." Something deeper is still wrong, and RS-KD touched none of it: the *target is fixed at the teacher* from step zero. Dagger's target, RS-KD's target, the scaffold default — all of them ask the small student to match the full far-away teacher immediately. Under a large capacity gap that target may be out of reach for the entire run, and no amount of cleverness in *summarizing* a fixed, unreachable target fixes that. There is a documented, ugly phenomenon behind this: hold a fixed small student and distill it from a progressively larger teacher and the student gets *worse* as the teacher grows, because the teacher distribution drifts too far from anything the student can represent. The on-policy loop already adapts the *data* to the student; I want the same instinct applied to the *target itself*.

I propose **TAID**: distill toward a *moving target* — an intermediate teacher $p_t$ that is the student at $t=0$ and the teacher at $t=1$, with $t$ ramping over training, so at every moment the target is only a little ahead of where the student currently is, close enough to fit and far enough to pull. The place I want to *start* is the most reachable target there is, the student's own current distribution; the place I want to *end* is the teacher. The decision that makes this work is *how* to blend the two. The lazy option is a convex combination of probabilities $t\,p_T + (1-t)\,p_S$, but a probability mixture of two peaked distributions is bimodal — it literally puts a bump on each — washing out or doubling the structure I am trying to hand the student smoothly. The additive geometry lives in the *logits*: since softmax is $\exp(\text{logit})/Z$, a convex combination *in logit space* is a normalized geometric blend, a smooth tempering that preserves the *relative* confidences within each distribution and degrades the internal ranking gracefully from student-like to teacher-like rather than producing a two-humped average. So I define the intermediate teacher at the logit level,
$$\texttt{target\_logits} = (1-t)\cdot \texttt{student\_logits.detach()} + t\cdot \texttt{teacher\_logits},$$
softmax it to get $p_t$, and minimize the forward KL $\mathrm{KL}(p_t \,\|\, p_S)$.

The `.detach()` is load-bearing and easy to get wrong. $p_t$ contains the student's own logits in its $(1-t)$ leg, and I am using $p_t$ as the *target* in a KL whose other argument is also the student. If gradients flowed through the student copy inside $p_t$, I would be differentiating the target with respect to the very parameters I am updating — a degenerate "move the goalposts toward me" channel where the student chases a target made partly of itself. So I treat the $(1-t)\cdot\texttt{student\_logits}$ slice as a constant for the step and let gradient flow only through the student in the KL's other argument. Without the detach the loss is silently degenerate.

The dynamics are worth checking, because they double as the reason this is the right medicine. At small $t$, $p_t$ is almost the student's own detached distribution, so I am distilling the student into a slightly-perturbed copy of itself — essentially self-distillation, which lets the student consolidate its own modes early rather than drowning under a teacher it cannot represent. This is precisely the regime RS-KD never entered, which is why its gain was flat. As $t$ grows the target tilts toward the teacher and the student takes on richer structure, but always from a target only modestly beyond its reach; late in training it is ordinary forward-KL teacher distillation, but now from a student walked up to the teacher's neighborhood instead of dropped in cold. Self-distillation has a known dark side — repeated self-distillation drives collapse, because the recursion only contracts the signal with nothing replenishing it — and the saving grace is the $t\cdot\texttt{teacher}$ leg: the target carries a constant slice of teacher signal that does not depend on the (shrinking) student prediction, flooring the signal at a fraction proportional to $t$ rather than letting it bleed to zero. The constant teacher injection is what makes the moving target safe, and it requires the schedule to keep $t$ rising rather than stalling near zero.

The schedule is where this task's loss is deliberately simpler than TAID's full form. The full method keys $t$'s speed to a momentum-EMA of the student's smoothed relative loss progress, passed through a sigmoid and a $(1-t)$ diminishing factor, so the target races ahead when learning is easy and slows when it is hard — but that adaptive machinery must carry scalar state (a running momentum, a previous-loss buffer) across steps, and the edit surface is a *stateless* loss body called once per step with `step` and `total_steps` and nothing persistent. So I drop the adaptive momentum and use the plain **linear** schedule — the non-adaptive variant the method compares against, and the one its non-collapse proof is written for:
$$t = t_{\text{start}} + (1 - t_{\text{start}})\cdot \min\!\big(1,\ \texttt{step}/\texttt{total\_steps}\big),$$
ramping linearly from $t_{\text{start}}$ to $1.0$, with a fixed mid-value fallback when `total_steps` is unavailable. I set $t_{\text{start}} = 0.4$, not near zero. At step zero the target is then $\sim$60% student / 40% teacher, which skips the trivially-easy earliest phase where the target is almost exactly the student and there is nothing to learn, while leaving a real but reachable gap; starting much lower (an earlier mistaken value of 0.10 gave a $\sim$90%-student step-zero target) would give near-zero distillation signal — another way to reproduce RS-KD's flat result. So $t_{\text{start}}=0.4$ is the load-bearing constant and $t_{\text{end}}=1.0$ so the final target is the genuine teacher.

One implementation reality folds in cleanly. The forward KL $\mathrm{KL}(p_t \,\|\, p_S) = \sum_v p_t(\log p_t - \log p_S)$ has a first term $\sum_v p_t\log p_t$ that is the negative entropy of the *detached* $p_t$, a constant for the step, so it drops from the student's gradient and the KL is gradient-equivalent to the cross-entropy $-\sum_v p_t\log p_S$. I take the KL form with the direction pinned so the intermediate teacher is the *target* and the student is the input being pushed — $\mathrm{KL}(\text{target}\,\|\,\text{student})$ — which under the framework's log-target convention means the student log-probs are the input and the target log-probs are the target. Temperature divides both logit tensors before the interpolation, mask to completion tokens, reduce per token.

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
