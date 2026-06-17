# SCRUB, distilled

SCRUB (SCalable Remembering and Unlearning unBound) is an approximate machine-unlearning
algorithm that casts forgetting as a teacher-student problem. A frozen copy of the original
model is the *teacher*; a *student*, initialized from it, is trained to **selectively disobey**
the teacher: pulled toward the teacher on the retain set `D_r` and pushed away from it on the
forget set `D_f`, with agreement measured by KL divergence between softened output
distributions. It needs no Hessian, no retraining, and no assumption that the unlearned weights
stay close to the original — so it scales and works for large (e.g. whole-class) forget sets.

## Problem it solves

Given a model `f(·; w^o)` trained on `D`, a forget set `D_f ⊂ D`, and the retain set
`D_r = D \ D_f`, produce `w^u` that loses competence on `D_f` while preserving utility on `D_r`
and on held-out data, in a few passes over the data, without retraining from scratch.

## Key idea

Treat `w^o` as a frozen teacher and the model being updated as a student `w^u` initialized from
it. Because the teacher was trained on all of `D`, the student is already good on `D_r` at
initialization — half the goal for free. Define the per-example distance between teacher and
student as the temperature-softened KL divergence

    d(x; w^u) = KL( p_T(f(x; w^o)) ‖ p_T(f(x; w^u)) ),

where `p_T(z) = softmax(z / T)`. Then:

- **Forget** = make the student *disobey* the teacher on `D_f`: **maximize** `d` there.
  This uses a softened teacher-disagreement signal instead of raw hard-label cross-entropy
  ascent; the objective is still a KL and should not be treated as globally upper-bounded over
  all logits, but it avoids making "drive the true-class probability to zero" the forget target.
- **Retain** = make the student *obey* the teacher on `D_r`: **minimize** `d` there, matching
  the teacher's full softened distribution (its "dark knowledge" / inter-class similarity
  structure), which is a stronger constraint than matching hard labels.
- **Stay correct** on `D_r`: also minimize cross-entropy to the true labels, an independent
  incentive that does not inherit the teacher's small errors.

The combined objective is

    min_{w^u}  (α/N_r) Σ_{x_r ∈ D_r} d(x_r; w^u)
             + (γ/N_r) Σ_{(x_r,y_r) ∈ D_r} ℓ(f(x_r; w^u), y_r)
             − (1/N_f) Σ_{x_f ∈ D_f} d(x_f; w^u),

with `ℓ` cross-entropy and `α, γ` scalars (`γ ≫ α`: the task loss dominates the retain side,
the KL is a light anchor).

## Why it works where the baselines don't

- **vs. finetuning on `D_r`:** finetuning has no force against `D_f`, so it barely forgets. The
  negated-KL max-step is an explicit forgetting pressure.
- **vs. NegGrad / NegGrad+:** cross-entropy ascent on `D_f` is unbounded and explodes. Maximizing
  teacher-student KL gives a soft distributional disagreement target; the teacher anchor
  protects the *function* on `D_r`, not just its labels.
- **vs. Fisher / NTK scrubbing:** no Hessian/Fisher inverse (those scale quadratically in the
  dataset) and no small-edit / closeness assumption that fails for class-sized forget sets.
- **vs. Bad-T:** pushing *away from the one good teacher* on `D_f` is a sharper, cleaner signal
  than pulling *toward a random incompetent teacher* (near-uniform, noisy, utility-degrading).

## Optimization recipe (handling the min-max instability)

Minimizing on `D_r` and maximizing on `D_f` through shared weights interferes and oscillates
(a min-max objective, like GAN training). The fix, by analogy to training a discriminator for
several steps per generator update:

1. **Capture the teacher** once (frozen copy of `w^o`); init student `w^u ← w^o`.
2. For the first `msteps` epochs: do a **max-epoch** on `D_f` (gradient ascent on `d`, i.e.
   descend `−d`), then a **min-epoch** on `D_r` (descend `α·d + γ·CE`).
3. After `msteps`: do **min-epochs only**, restoring retain performance the max-steps disturbed
   without re-teaching `D_f`.
4. Stop when forget error is high and retain error is unharmed — typically a few epochs.

Implementation defaults used here: `msteps = 2`, `T = 4`, `α = 0.01`, `γ = 0.99`, small learning
rate (~`5e-4`) with decay (decay is important to tame the oscillation); forget and retain sets
may use different batch sizes to tune the max:min iteration ratio.

## SCRUB+R (rewinding, for privacy)

The base algorithm drives forget error *maximal*, ideal for removing biases / resolving
confusion but a liability for privacy: an abnormally high forget error is itself detectable by a
membership-inference attack. SCRUB+R calibrates instead. Build a held-out validation set from
the *same distribution* as `D_f`; its error under the trained student approximates the error of
a model that never saw that distribution. Checkpoint every epoch and **rewind** to the
checkpoint whose forget error is closest to that reference — "just high enough," not
conspicuous.

## Working code

Fills the `unlearn_step` slot of the unlearning harness. The teacher is a frozen deep copy of
the model captured on the first step; the max-step runs while `epoch < msteps`, the min-step
every epoch.

```python
import copy
import torch
import torch.nn.functional as F


class UnlearningMethod:
    """SCRUB: min-max KL distillation against a frozen original-model teacher."""

    def __init__(self):
        self.msteps = 2        # epochs of max-step before switching to min-only
        self.kd_T = 4.0        # distillation temperature
        self.alpha = 0.01      # weight on KL(teacher || student) on the retain set
        self.gamma = 0.99      # weight on cross-entropy to true labels on the retain set
        self.teacher = None    # frozen copy of the original model, captured lazily

    def _kd_kl(self, student_logits, teacher_logits):
        # d(x) = T^2 * KL( softmax(teacher/T) || softmax(student/T) ).
        # The T^2 compensates the 1/T^2 scaling of soft-target gradients,
        # keeping the distillation term's weight stable across temperatures.
        T = self.kd_T
        p_s = F.log_softmax(student_logits / T, dim=1)
        p_t = F.softmax(teacher_logits / T, dim=1)
        return F.kl_div(p_s, p_t, reduction="batchmean") * (T * T)

    def _capture_teacher(self, model):
        self.teacher = copy.deepcopy(model)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

    def unlearn_step(self, model, retain_batch, forget_batch, optimizer, step, epoch):
        if self.teacher is None:
            self._capture_teacher(model)            # student starts as the teacher

        retain_x, retain_y = retain_batch
        forget_x, _ = forget_batch

        # ---- Max-step on D_f (only for the first msteps epochs): push AWAY ----
        forget_kl_val = 0.0
        if epoch < self.msteps:
            optimizer.zero_grad()
            s_forget = model(forget_x)
            with torch.no_grad():
                t_forget = self.teacher(forget_x)
            forget_kl = self._kd_kl(s_forget, t_forget)
            (-forget_kl).backward()                 # minimize -KL == maximize KL
            optimizer.step()
            forget_kl_val = forget_kl.item()

        # ---- Min-step on D_r (every epoch): pull TOWARD + stay CORRECT ----
        optimizer.zero_grad()
        s_retain = model(retain_x)
        with torch.no_grad():
            t_retain = self.teacher(retain_x)
        retain_ce = F.cross_entropy(s_retain, retain_y)
        retain_kl = self._kd_kl(s_retain, t_retain)
        loss = self.gamma * retain_ce + self.alpha * retain_kl
        loss.backward()
        optimizer.step()

        return {
            "loss": float(loss.item()),
            "retain_ce": float(retain_ce.item()),
            "retain_kl": float(retain_kl.item()),
            "forget_kl": float(forget_kl_val),
        }
```
