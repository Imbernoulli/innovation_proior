"""GKD baseline — Generalized Knowledge Distillation (Agarwal et al., ICLR'24).

Paper: https://huggingface.co/papers/2306.13649
Reference impl: vendor/external_packages/trl/trl/experimental/gkd/gkd_trainer.py
                ::generalized_jsd_loss

Loss (TRL convention, matches reference impl):
    β = 0       ⇒  KL(p_T ‖ p_S)
    β = 1       ⇒  KL(p_S ‖ p_T)
    0 < β < 1   ⇒  β · KL(p_T ‖ M) + (1-β) · KL(p_S ‖ M),  M = (1-β)·p_S + β·p_T

This baseline pins β = 0.5 (canonical GKD setting).
"""

_FILE = "trl/trl/experimental/gkd/custom_distill_loss.py"

_BODY = """\
    # GKD (Agarwal et al., ICLR'24) — Generalized JSD with β = 0.5.
    beta_use = 0.5
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    beta_t = torch.tensor(beta_use, dtype=student_log_probs.dtype, device=student_log_probs.device)
    mixture_log_probs = torch.logsumexp(
        torch.stack([student_log_probs + torch.log1p(-beta_t), teacher_log_probs + torch.log(beta_t)]),
        dim=0,
    )
    # PyTorch's F.kl_div(input=log_q, target=log_p) computes KL(p || q).
    kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True).sum(dim=-1)
    kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True).sum(dim=-1)
    per_token = beta_t * kl_teacher + (1 - beta_t) * kl_student  # [B, T]

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
