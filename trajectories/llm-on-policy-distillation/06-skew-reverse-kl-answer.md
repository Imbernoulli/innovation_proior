**Problem (from step 5).** OPD's pure reverse KL won GSM8K (0.4852, best on the ladder) but its
MATH-500 (0.310) came in a hair under GKD — the mode-collapse tax of *unbounded* reverse KL. Its
gradient coefficient `log(p_S/p_T) + 1` detonates where `p_T ≈ 0`, which on the student's own on-policy
rollouts (sequences the teacher did not write) is the *typical* case. Keep OPD's mode-seeking
commitment; stop the gradient blow-up that overshoots into collapse.

**Key idea (Skewed Reverse KL, SRKL).** Take the reverse KL of the student against a *skewed mixture*
of teacher and student instead of against the raw teacher:
`D_SRKL^α(p_T, p_S) = KL(p_S ‖ (1−α)·p_T + α·p_S)`, with `α = 0.1`. At `α=0` this is exactly OPD's
reverse KL; the `α·p_S` leg floors the denominator.

**Why.** OPD's gradient explodes because the raw teacher `p_T` sits in a denominator and vanishes on
off-distribution (on-policy) tokens. With the mixture `p̃ = (1−α)·p_T + α·p_S`, the skewed gradient
coefficient is `log r + 1 − α·r` with `r = p_S/p̃ ≤ 1/α` (since `p̃ ≥ α·p_S`): the `log r` term is now
*bounded*, and the extra `−α·r` term subtracts and grows with `r`, pulling the coefficient back down
exactly where it would spike. The weighting is still `p_S` — mode-seeking, zero-forcing, the property
that won GSM8K — but the detonation that collapsed MATH-500 is removed.

**Choosing α.** The skewed estimator's L2 error has inverse-`α` terms (larger `α` floors the denominator
and cuts variance), but an Adam-style optimizer divides out a uniformly smaller gradient scale, so the
gradient-scale-normalized L2 error is convex in `α` with an interior optimum. `α = 0.1` sits there — 90%
faithful to the teacher, enough student mixed in to bound the ratio.

**Distinct from GKD's JSD.** JSD is a sum of two skewed KLs with coupled skew (`β`/`1−β`), so it cannot
make both legs mildly skewed; `α=0.1` on the reverse leg would force `0.9` on the forward leg. A single
freely-tuned mild skewed reverse KL reaches the interior optimum JSD cannot. GKD balanced two *unskewed*
directions at `β=0.5`; this skews *one* mode-seeking direction by a small amount — the knob the OPD
failure calls for.

**Paper-vs-task note.** The full streamlined recipe pairs SRKL with two data-side mechanisms — an
adaptive SGO scheduler (ramps self-generation by validation loss) and an off-policy replay buffer
(decaying replay ratio) — both framework-level and out of scope for this single-loss surface. Only the
*loss* is landed here, on the trainer's static `lmbda` mixing, exactly as the baselines consumed it.

**Numerics.** Keep *both* legs of the reverse-KL gradient (`p_S·log p_S` carries the `+1`
normalization); mixture formed in probability space then logged; `±inf` logits guarded to contribute
zero; masked to completion tokens; `batchmean` per-token mean.

**The bar (no feedback — this is the endpoint).** Must clear OPD: GSM8K 0.4852, MATH-500 0.310. The
falsifiable claim: GSM8K holds OPD's lead (≥ 0.4852 — same direction, 90% teacher, bounded gradient
trains no worse) while MATH-500 recovers *above* 0.310 (the floor stops the collapse OPD suffered). If
MATH-500 stays at OPD's level the collapse is the direction, not the gradient; if GSM8K drops, `α` is too
large. The clean win — GSM8K ≥ 0.4852 and MATH-500 > 0.310 — is what the bounded-gradient story predicts.

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
    # Skewed Reverse KL (SRKL, Ko et al., ICML'24) — KL(p_S || (1-a)*p_T + a*p_S), a = 0.1.
    # Mode-seeking like OPD, but the a*p_S leg floors the teacher-side denominator so the
    # gradient coefficient log r + 1 - a*r (r = p_S/p~ <= 1/a) cannot explode where p_T ~= 0
    # (the typical case on student on-policy rollouts), removing OPD's mode-collapse tax.
    alpha = 0.1

    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(student_logits, dim=-1, dtype=torch.float32)
    mixed_probs = (1 - alpha) * teacher_probs + alpha * student_probs        # p~ = (1-a)p_T + a p_S

    student_log_probs = F.log_softmax(student_logits, dim=-1, dtype=torch.float32)
    mixed_log_probs = torch.log(mixed_probs.clamp_min(1e-9))

    # KL(p_S || p~) = Sum_v p_S * (log p_S - log p~). Keep BOTH legs: the p_S*log p_S leg
    # depends on theta and carries the reverse-KL +1 normalization gradient.
    inf_mask = torch.isinf(student_logits) | torch.isinf(teacher_logits)     # guard -inf logits
    prod = torch.masked_fill(student_probs * student_log_probs, inf_mask, 0)
    prod -= torch.masked_fill(student_probs * mixed_log_probs, inf_mask, 0)
    per_token = prod.sum(dim=-1)                                             # [B, T], = +KL(p_S || p~)

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
