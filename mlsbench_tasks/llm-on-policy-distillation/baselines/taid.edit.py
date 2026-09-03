"""TAID baseline — Temporally Adaptive Interpolated Distillation
(Shing et al., ICLR'25 Spotlight, arXiv 2501.16937).

Reference impl (authoritative):
    https://github.com/SakanaAI/TAID/blob/main/src/distil_losses/taid.py

The earlier version of this baseline diverged from the official impl on three
points which together produced an unrealistically weak TAID result:

  1. `lambda_min` was 0.10 (official `t_start = 0.4`) — at step 0 the target
     was ~90% student and ~10% teacher, giving near-zero distillation signal.
  2. Interpolation was done in **log-prob space** (logsumexp of student/teacher
     log-probs). The official impl interpolates in **logit space**
     (`(1-t)·s_logits.detach() + t·t_logits`), which is a different operation.
  3. The student logits were *not* detached in the target, so the target had
     a spurious gradient through the student leg.

This revision matches the official impl on (1)–(3). The adaptive momentum
schedule (which speeds up `t` when loss drops fast) is intentionally left
out — for our 300-step budget a plain linear schedule from `t_start=0.4` to
`1.0` is the standard non-adaptive variant the paper compares against.
"""

_FILE = "trl/trl/experimental/gkd/custom_distill_loss.py"

_BODY = """\
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
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 42,
        "end_line": 65,
        "content": _BODY,
    },
]
