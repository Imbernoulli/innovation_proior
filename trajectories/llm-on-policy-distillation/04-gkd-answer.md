**Problem (from step 3).** TAID's moving target lifted GSM8K to 0.4685 (the biggest single-rung gain)
but pushed MATH-500 *down* to 0.280 — below dagger and RS-KD. The reachability fix worked; the loss
it reaches the teacher *through* is still forward KL (mass-covering), which on the hardest set spreads
a 0.5B student's mass thin and parks it in the valley between the teacher's reasoning modes. Three
rungs have all used forward KL in some guise. The untouched axis is the *divergence direction*.

**Key idea (GKD, generalized JSD at β=0.5).** Replace forward KL with the symmetric Jensen-Shannon
divergence: mixture `M = β·p_T + (1−β)·p_S`, loss `β·KL(p_T ‖ M) + (1−β)·KL(p_S ‖ M)`, pinned at
`β = 0.5`. This steps onto the *interior* of the forward↔reverse divergence family for the first time
— half mass-covering, half mode-seeking.

**Why.** Forward KL weights by the teacher's probability (mass-covering — bad under capacity gap);
reverse KL weights by the student's (mode-seeking — commits to a coherent mode, what MATH-500 needs).
The generalized JSD interpolates: `lim_{β→0} D_JSD^β/β = KL(p_T ‖ p_S)` (forward), `β→1` gives reverse
by symmetry, `β=0.5` sits exactly between. Bonus: JSD is *bounded* even on disjoint supports (plain KL
is infinite there), taming the early-training blow-up pure reverse KL would risk. Symmetric JSD pulls
the student off pure mass-covering without slamming it into mode-collapse.

**Subsumes the ladder.** Every prior rung lives on the forward-KL face of the (divergence × data)
family; this is the first interior-divergence point, which is why it is the right generalization, not
a fourth ad-hoc loss.

**Numerics / direction trap.** `log M = logsumexp([log p_S + log(1−β), log p_T + log β])` (no leaving
log space). `F.kl_div(input=log_q, target=log_p, log_target=True) = KL(p ‖ q)`, so `KL(p_T ‖ M)` is
mixture-input, teacher-target; getting it backwards minimizes a silently wrong objective.

**Paper-vs-task note.** The full method exposes `β` as a tunable knob with explicit `β=0`/`β=1`
endpoint branches plus the interior JSD. This baseline does *not* expose the sweep — it hard-pins
`beta_use = 0.5` and always computes the genuine interior JSD, ignoring the `beta` argument. So this
lands specifically the *symmetric Jensen-Shannon* point, not the configurable family.

**Hyperparameters.** `beta_use = 0.5` (symmetric JSD); data axis = trainer's static `lmbda`,
untouched; temperature on both sides; `batchmean` over completion tokens.

**What to watch.** Asymmetric prediction: MATH-500 should *recover* into the low-0.31s (reversing
TAID's 0.280 regression — the divergence fix helps most where mass-covering hurt most), GSM8K a
smaller move into the low-0.47s (short chains forgive mass-covering). If half-JSD recovers MATH-500 but
GSM8K only inches up, the next move is to push all the way to the reverse-KL endpoint. AMC stays noise.

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
```
