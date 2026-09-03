"""OPD baseline — On-Policy Distillation (Lu et al., Thinking Machines 2025;
Qwen3 Technical Report, arXiv 2505.09388).

Loss: per-token *reverse* KL on student-sampled tokens, γ = 0 (no discount).
This is the β = 1 special case of GKD applied to on-policy rollouts.

    L = KL(p_S ‖ p_T) = Σ_v p_S(v) [log p_S(v) - log p_T(v)]

The OPD recipe additionally pins lmbda = 1.0 (always on-policy). This
baseline does not change lmbda directly — it relies on the trainer's
default lmbda = 0.5 mixing, applying reverse KL regardless of which
sampling source produced the tokens. (The reverse KL formulation is what
makes this OPD, not just on-policy mixing.)
"""

_FILE = "trl/trl/experimental/gkd/custom_distill_loss.py"

_BODY = """\
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
