"""Editable distillation loss for the llm-on-policy-distillation task.

Replaces the default `generalized_jsd_loss` in TRL's GKDTrainer. Called once
per gradient step by trl/experimental/gkd/gkd_trainer.py::compute_loss
inside the standard (non-Liger) path, with:

    student_logits  [B, T, V]  — student logits over completion tokens
    teacher_logits  [B, T, V]  — teacher logits over the same tokens
    labels          [B, T]     — completion token ids (-100 for padding)
    beta            float      — interpolation knob from GKDConfig (0..1)
    temperature     float      — softmax temperature
    reduction       str        — "batchmean" / "sum" / "mean" / "none"
    step            int        — current global training step
    total_steps     int        — total planned training steps (for curricula)
    lmbda           float      — GKDConfig.lmbda, fraction of on-policy batches
                                  (use with `step` to design DAgger-style
                                  on-policy/off-policy hybrid schedules)

Notes:
    * Padding positions (labels == -100) MUST be excluded from the reduction.
    * Return a scalar tensor (or per-element tensor when reduction == "none").
    * The reference GKD implementation is in
      trl/experimental/gkd/gkd_trainer.py::generalized_jsd_loss.
"""

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
    # ================== EDITABLE START ==================
    # Default = forward KL D(teacher || student). Replace this body with a
    # novel distillation loss. Keep the signature unchanged.
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    # KL(p_T || p_S), summed over vocab → per-token KL of shape [B, T]
    kl = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True).sum(dim=-1)

    if labels is not None:
        mask = labels != -100
        kl = kl[mask]
        denom = mask.sum().clamp_min(1)
    else:
        denom = torch.tensor(max(kl.numel(), 1), device=kl.device, dtype=kl.dtype)

    if reduction == "batchmean":
        return kl.sum() / denom
    elif reduction == "sum":
        return kl.sum()
    elif reduction == "mean":
        return kl.mean()
    return kl
    # ================== EDITABLE END ==================
