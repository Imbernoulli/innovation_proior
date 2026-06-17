# Skew (Reverse) KL Divergence (SKL / SRKL), distilled

Skewed KL is a distillation objective that targets the gradient instability of plain forward/reverse KL
under capacity mismatch by computing the KL against a *skewed mixture* of the teacher and student
instead of against the raw target. It is the loss-function core of a streamlined LLM distillation
recipe (the rest of which is an adaptive student-generated-output scheduler and an off-policy replay
buffer — orthogonal data-pipeline machinery).

## Problem it solves

Distilling a small student `q_θ` from a large teacher `p` works best when (a) the objective is
mode-seeking (reverse KL) so an under-capacity student commits to the teacher's modes instead of
mode-averaging, and (b) training is on student-generated outputs (SGOs) so it matches the inference
distribution. But that exact combination trains badly: the reverse-KL gradient coefficient explodes
where the teacher probability `p ≈ 0`, and on SGOs — sequences not sampled from the teacher — `p ≈ 0`
can be common. Forward KL has the mirror instability (explodes where `q_θ ≈ 0`). Both blow-ups
come from a raw distribution sitting in a denominator and going to zero.

## Key idea

Skew the KL by mixing in a sliver of the other distribution so the denominator is floored away from
zero. With skew parameter `α`:

- **Skewed KL (SKL):** `D_SKL^α(p, q_θ) = D_KL(p, α·p + (1−α)·q_θ)` — floors the student leg
  (mass-covering).
- **Skewed Reverse KL (SRKL):** `D_SRKL^α(p, q_θ) = D_KL(q_θ, (1−α)·p + α·q_θ)` — floors the teacher
  leg (mode-seeking). This is the variant for mode-seeking-on-SGO.

## Why it works

**Stable gradient.** For the minimized scalar KL losses, the plain gradients are
`∇_θ D_KL(p, q_θ) = −r_{p,q_θ}·∇_θ q_θ` (explodes when `q_θ → 0`) and
`∇_θ D_KL(q_θ, p) = (log r_{q_θ,p} + 1)·∇_θ q_θ` (explodes when `p → 0`). Some code accumulates the
negative reverse-KL integrand before a final negation; the minimized loss has the sign shown here.
Skewing gives:

```
∇_θ D_SKL^α(p, q_θ)   = −(1−α)·r_{p, q̃_θ}·∇_θ q_θ,           q̃_θ = α·p + (1−α)·q_θ
∇_θ D_SRKL^α(p, q_θ)  = (log r_{q_θ, p̃} + 1 − α·r_{q_θ, p̃})·∇_θ q_θ,   p̃ = (1−α)·p + α·q_θ
```

For SRKL, the interpolation prevents the denominator of the ratio from reaching zero on
teacher-vanishing samples because `p̃ ≥ α·q_θ`, hence `r_{q_θ,p̃} ≤ 1/α`; the extra `−α·r` term pulls the
coefficient back down where it would run away. For SKL, `q̃_θ ≥ α·p`, hence `r_{p,q̃_θ} ≤ 1/α` where the
student vanishes.

**Small approximation error.** The empirical L2 error of the SKL estimator is bounded by
`c_1(α)/n² + c_2·log²(α n)/n + c_3·log²(c_4 n)/(α² n)` with `c_1(α) = min{1/α², χ²(p,q)²/(1−α)²}`,
for `α < 1/8`. The bound is not a claim that every displayed term is globally monotone in `α`; the
important edge is that moving away from `α=0` reduces the inverse-`α` contribution that makes the raw-KL
limit noisy.

**Choosing α.** A modern (Adam-style) optimizer normalizes out a uniformly smaller gradient scale, so
the relevant quantity rescales the L2 error by the inverse approximate gradient scale, `1/(1−α)`. That
normalized bound appears convex in `α` (the inverse-`α` pieces want `α` large, the
inverse-`(1−α)` pieces want `α` small), and the normalized-coefficient results report that SRKL is best at
`α = 0.1` and worsens for larger values. The selected value is `α = 0.1`: 10% mixing floors the
denominator while leaving the reverse target 90% teacher.

**Distinct from JSD.** Generalized JSD is
`D_JSD^β(p, q_θ) = β·D_SKL^β(p, q_θ) + (1−β)·D_SRKL^{1−β}(p, q_θ)`, with
`M = β·p + (1−β)·q_θ`. The same `β` ties the two skew values (`β` and `1−β`), so it cannot make *both*
legs mildly skewed; choosing `α=0.1` on one term forces `0.9` on the other. In the official DistiLLM
code, `js_distance(lam)` uses the equivalent convention `lam = 1−β`.

## Working code

The canonical DistiLLM implementation computes the loss in probability/log space, keeps both terms of the
reverse-KL objective (dropping the `q_θ·log q_θ` term would remove the `+1` coefficient), guards `±inf`
logits, masks to completion tokens (`label ≠ −100`), and averages over valid tokens. For forward
KL/SKL, the code intentionally uses the gradient-equivalent cross-entropy form and omits the
`Σ p·log p` target-entropy constant; add that term only when the literal scalar divergence value is
needed for reporting.

```python
import torch
import torch.nn.functional as F


def skewed_reverse_kl(logits, teacher_logits, no_model_batch, lam=0.1):
    # SRKL: KL(q_theta, (1-lam)*p + lam*q_theta). lam = alpha = 0.1. Mode-seeking,
    # with q_theta added to the teacher-side denominator for stability on SGOs.
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_probs = (1 - lam) * teacher_probs + lam * student_probs        # p~ = (1-a)p + a q

    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    mixed_logprobs = torch.log(mixed_probs)

    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    prod_probs = torch.masked_fill(student_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss


def skewed_forward_kl(logits, teacher_logits, no_model_batch, lam=0.1):
    # SKL: KL(p, lam*p + (1-lam)*q_theta). Mass-covering; p is added to the
    # student-side denominator for stability where q_theta is near zero. This
    # canonical training loss drops the theta-constant target entropy Sum p log p.
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_probs = lam * teacher_probs + (1 - lam) * student_probs        # q~ = a p + (1-a) q
    mixed_logprobs = torch.log(mixed_probs)

    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    prod_probs = torch.masked_fill(teacher_probs * mixed_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss
```

## Relation to prior methods

- **Forward KL / supervised KD** (Hinton 2015; Kim & Rush 2016) — `α=0` of SKL; mass-covering,
  unstable where `q_θ → 0`.
- **Reverse KL** (Gu et al. 2023; Agarwal et al. 2023) — `α=0` of SRKL; mode-seeking, unstable where
  `p → 0` (i.e. on SGOs).
- **Generalized JSD** (Agarwal et al. 2023) — a sum of SKL and SRKL with coupled skew `β`/`1−β`;
  cannot make both legs mildly skewed.
- **SGO / on-policy training** (Lin et al. 2020; Agarwal et al. 2023) — the data-side fix for
  training-inference mismatch; skewed KL is what makes mode-seeking-on-SGO trainable, and (with an
  adaptive SGO scheduler + off-policy replay buffer) efficient.
