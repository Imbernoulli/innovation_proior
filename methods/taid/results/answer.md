# TAID, distilled

TAID (Temporally Adaptive Interpolated Distillation) is a knowledge-distillation objective for
compressing a large language-model teacher into a small student. Instead of distilling toward the
fixed teacher distribution, it distills toward a **time-dependent intermediate teacher** `p_t` that
starts as the student's own distribution and migrates to the teacher's as a scalar `t` ramps from 0
to 1 over training. This keeps the target just ahead of the current student, addressing the capacity
gap and balancing mode averaging against mode collapse. It is logit-only — no student sampling, no
extra frozen network.

## Problem it solves

Distilling from a teacher much larger than the student. Forward KL `KL(p || q)` is mass-covering, so
a low-capacity student oversmooths to cover all teacher modes (**mode averaging**). Reverse KL
`KL(q || p)` is mode-seeking, so the student concentrates on a few dominant modes and drops the rest
(**mode collapse**). Methods that interpolate the two (generalized JSD, skew-KL) use a *fixed*
coefficient against a target anchored at the *fixed* teacher; under a large **capacity gap** that
target is unreachable from the first step to the last, and student quality can *decrease* as the
teacher grows. TAID makes the target move with the student.

## Key idea

Define an intermediate teacher by interpolating teacher and (detached) student **in logit space**:

```
p_t(y_s | y^{<s}) = softmax( (1 - t) * logit_{q'_theta}(y_s | y^{<s}) + t * logit_p(y_s | y^{<s}) )
```

- `q'_theta` is the student logits **detached** (stop-gradient): `p_t` is a target, not a second
  trainable copy, so gradient flows only through the student denominator of the KL.
- Interpolation is in logit space, not probability space, so `p_t` is a smooth (normalized
  geometric) tempering between the two distributions that preserves each one's *relative* confidence,
  rather than a bimodal probability average.
- `t = 0` gives the student's own distribution (self-distillation-like; consolidates the student's
  modes), `t = 1` gives the teacher; in between, a target only modestly ahead of the student.

The objective is forward KL against this moving target:

```
J_TAID^(t)(p, q_theta) = J_KL(p_t, q_theta) = (1/S) sum_s sum_y p_t(y_s|y^{<s}) log( p_t / q_theta )
```

Because `p_t` is detached, its entropy is constant in `theta`, so this has the same student gradient
as the cross-entropy `- sum_y p_t log q_theta` — the form actually implemented.

## Adaptive schedule for t

A linear ramp `t_linear = t_start + (t_end - t_start) * n/N` is the proof baseline and is
non-collapsing under the signal condition in the theorem. The adaptive version moves `t` faster when
the student is learning fast:

```
delta_n   = ( J^(t_{n-1}) - J^(t_n) ) / ( J^(t_{n-1}) + eps )      # scale-free relative loss drop
m_n       = beta * m_{n-1} + (1 - beta) * delta_n                  # momentum EMA, smooth the signal
Delta t   = alpha * sigmoid(m_n) * (1 - t_n)                       # bounded, diminishing toward t_end
t_{n+1}   = min( t_end, max( t_linear, t_n + Delta t ) )           # monotone, >= linear, <= teacher
```

- `sigmoid(m_n) in (0,1)`: monotone in progress, saturates so no lurching jump.
- `(1 - t_n)`: the remaining distance to the teacher, so the step shrinks as `t -> 1`.
- `max(t_linear, .)`: floors progress at the linear schedule; `min(t_end, .)`: caps at the teacher.

Defaults: `t_start in [0.2, 0.4]` (skip the trivially-easy initial phase; ~0.4 for a large gap),
`t_end = 1.0`, `beta = 0.99` (smooth the noisy relative-progress signal), `alpha = 5e-4` (slow,
stable adaptation).

## Theory (non-collapse)

In Mobahi et al.'s least-square-regression proxy (interpolation regime, kernel regularizer,
nonlinear-ridge solution `f = V^T D (lambda I + D)^{-1} V y`, collapse iff `||y||^2 <= N eps`), the
TAID update with interpolated label `y~_t = (1 - t/T) y_t + (t/T) y_0` becomes, in rotated
coordinates `z = V y` with `A_t = D(lambda_t I + D)^{-1}`:

```
z~_t = (1 - t/T) * A_{t-1} * z~_{t-1} + (t/T) * z_0
```

Unlike self-distillation's homogeneous contraction `z_t = A_{t-1} z_{t-1}` (which only shrinks the
signal and inevitably collapses), TAID has a constant teacher injection `(t/T) z_0`. Unrolling gives
`z~_t = A_bar_t z_0`, and since the last-injected slice contributes a `(t/T) I` diagonal term,
`sigma_min(A_bar_t) >= t/T`, hence `||z~_t|| >= (t/T) ||z_0||`. With `r_0 = ||y_0||/sqrt(N eps)` and
condition number `kappa = d_max/d_min`, non-collapse `||z~_t|| > sqrt(N eps)` holds for some
`gamma in [0,1]` when

```
t < min{ (1/(gamma+kappa)) (r_0 - gamma) + o(1),  (gamma/r_0) T }    or    t > T / r_0,
```

(the late bound `t > T/r_0` from the floor; the early bound from an affine-recursion lower bound on
`r_t` using `sigma_min(A_{t-1}) >= (r_{t-1}-1)/(r_{t-1}-1+kappa)` and its linear under-estimate). At
`gamma = 1` this yields the clean corollary: if

```
||y_0|| = Omega( ( (1 + sqrt(1 + 4 T (1 + kappa))) / 2 ) sqrt(N eps) ),
```

the student never collapses for any `t`. Self-distillation survives only for `t <= (r_0 - 1)/kappa`
and then inevitably collapses; TAID covers the full late phase at the price of a constant-factor
weaker early window (critical step `~ r_0/(gamma+kappa)` vs `~ r_0/kappa`, so `gamma` must stay
bounded away from 0).

## Working code

The TAID loss together with the adaptive schedule:

```python
from typing import Dict, Union
import torch
from torch import nn
from torch.nn import functional as F
from lightning import LightningModule


class DistilLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs) -> Union[Dict, torch.Tensor]:
        raise NotImplementedError


def forward_kl(logits, teacher_logits, mask, teacher_probs=None, student_logprobs=None):
    if teacher_probs is None:
        teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    if student_logprobs is None:
        student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(logits)
    prod_probs = torch.masked_fill(teacher_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    return -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)


class TAID(DistilLoss):
    def __init__(self, t_start=0.4, t_end=1.0, alpha=5e-4, beta=0.99, disable_adaptive=False):
        super().__init__()
        assert 0.0 <= t_start < 1.0
        assert 0.0 < t_end <= 1.0
        assert 0.0 <= alpha <= 1.0
        self.t_start = t_start
        self.t_end = t_end
        self.alpha = alpha
        self.beta = beta
        self.disable_adaptive = disable_adaptive
        self.register_buffer("t", torch.tensor(float(t_start), dtype=torch.float32))
        self.register_buffer("prev_loss", torch.tensor(float("inf"), dtype=torch.float32))
        self.register_buffer("momentum", torch.zeros((), dtype=torch.float32))

    def update_t(self, loss, global_step, num_train_steps):
        if torch.isinf(self.prev_loss):
            self.prev_loss = loss
            return

        relative_change = (self.prev_loss - loss) / (self.prev_loss + 1e-15)
        self.momentum = self.beta * self.momentum + (1 - self.beta) * relative_change

        adaptive_delta = torch.sigmoid(self.momentum)
        t_target = self.t_start + (self.t_end - self.t_start) * (global_step / num_train_steps)
        delta_t = self.alpha * adaptive_delta * (1 - self.t)
        t = (min(self.t_end, max(t_target, self.t + delta_t))
             if not self.disable_adaptive else t_target)
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=self.t.device, dtype=self.t.dtype)
        self.t = t
        self.prev_loss = loss
        return delta_t

    def compute_loss(self, logits, teacher_logits, mask):
        p_t = (1 - self.t) * logits.detach() + self.t * teacher_logits
        p_t = F.softmax(p_t, dim=-1, dtype=torch.float32)
        return forward_kl(
            logits=logits,
            teacher_logits=teacher_logits,
            mask=mask,
            teacher_probs=p_t,
        )

    def forward(self, lightning_module: LightningModule, logits, teacher_logits, mask, **kwargs):
        loss = self.compute_loss(logits, teacher_logits, mask)
        delta_t = self.update_t(
            loss.detach().clone(),
            global_step=lightning_module.trainer.global_step,
            num_train_steps=lightning_module.trainer.estimated_stepping_batches,
        )
        return {
            "distil_loss": loss,
            "tiki_t": self.t,
            "delta_t": delta_t,
        }
```
