# A-GEM (Averaged Gradient Episodic Memory), distilled

A-GEM is a continual-learning method that prevents catastrophic forgetting by *projecting the
current-task gradient* so that, to first order, it does not increase the **average** loss over
a small episodic memory of past tasks. It is the efficient successor to GEM: where GEM imposes
one inequality constraint per past task (a quadratic program per training step), A-GEM imposes
a single averaged constraint against one reference gradient, which has a closed-form projection
— no QP, one extra backward pass, two dot products per step.

## Problem it solves

A learner sees a stream of tasks `t = 1, ..., T`, one pass each, and is evaluated on all tasks
at the end. Plain sequential fine-tuning forgets earlier tasks. A-GEM mitigates forgetting
under a bounded memory budget and per-step compute that stays flat in the number of tasks, and
it permits positive backward transfer (old tasks may improve).

## Key idea

A single SGD step `θ ← θ − α g` changes a past task's loss by `≈ −α ⟨g, g_k⟩` to first order,
so the sign of the gradient inner product is the forgetting diagnostic. Keep a small episodic
memory `M = ∪_{k<t} M_k`. While learning task `t`, require only that the *average* memory loss
not increase:

```
minimize_θ  ℓ(f_θ, D_t)   s.t.  ℓ(f_θ, M) ≤ ℓ(f_θ^{t-1}, M).
```

Linearized, with `g = ∇_θ ℓ(f_θ, D_t)` and `g_ref = ∇_θ ℓ(f_θ, M)` (estimated from a random
memory minibatch), this is the single constraint `g̃ᵀ g_ref ≥ 0`. Projecting `g` to the closest
feasible gradient is a projection onto one halfspace:

```
if  gᵀ g_ref ≥ 0:   g̃ = g                                    # already feasible
else:               g̃ = g − (gᵀ g_ref / g_refᵀ g_ref) g_ref   # remove the offending component
```

In the projected case `g̃ᵀ g_ref = 0`: the corrected gradient is orthogonal to the reference
(neither helping nor hurting the average past task to first order) while staying as close to
`g` as the constraint allows. Then `θ ← θ − α g̃`.

## Why this over GEM

- GEM keeps `t−1` constraints (`⟨g̃, g_k⟩ ≥ 0` for every past task) and solves a dual QP in
  `t−1` variables every step, needing one backward pass per past task to build `G`. Cost grows
  with the stream.
- A-GEM replaces "every task" with "the average task," collapsing to one constraint with a
  closed-form projection: cost is one extra backward (for `g_ref`) and two dot products,
  independent of `t`.
- Trade-off: GEM guarantees no individual task's (memory) loss rises (worst-case forgetting);
  A-GEM constrains the *average* memory loss, aligning the update rule with the average-accuracy
  objective while giving up GEM's per-task worst-case guarantee.

## Update-rule derivation (closed form)

```
minimize_z (1/2)‖g − z‖²  s.t.  z ᵀ g_ref ≥ 0
  drop constant (1/2)gᵀg, write −zᵀg_ref ≤ 0:  minimize (1/2)zᵀz − gᵀz  s.t. −zᵀg_ref ≤ 0
  L(z, α) = (1/2)zᵀz − gᵀz − α zᵀg_ref,   α ≥ 0
  ∂L/∂z = z − g − α g_ref = 0     ⇒  z* = g + α g_ref
  θ_D(α) = −(1/2)gᵀg − α gᵀg_ref − (1/2)α² g_refᵀg_ref   (concave in α)
  ∂θ_D/∂α = 0                     ⇒  α* = − gᵀg_ref / g_refᵀg_ref
  constraint violated (gᵀg_ref < 0) ⇒ α* > 0 active ⇒ z* = g − (gᵀg_ref/g_refᵀg_ref) g_ref
  constraint satisfied (gᵀg_ref ≥ 0) ⇒ clamp α = 0 ⇒ z* = g
```

## Episodic memory

Keep a bounded buffer of examples from completed tasks and draw the reference minibatch from
that buffer. The current task contributes no reference constraint, so the first task trains as
plain SGD.

## Working code

A compact implementation of the update inside a continual-learning model. `store_grad` and
`overwrite_grad` are the existing GEM helpers for moving between per-parameter `.grad` fields
and flat buffers; `grad_xy = g` and `grad_er = g_ref` are the two flat buffers. `project` is the
closed form, and projection is applied exactly when `dot_prod.item() < 0`.

```python
import numpy as np
import torch

from models.gem import overwrite_grad, store_grad
from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer


def project(gxy: torch.Tensor, ger: torch.Tensor) -> torch.Tensor:
    corr = torch.dot(gxy, ger) / torch.dot(ger, ger)
    return gxy - corr * ger


class AGem(ContinualModel):
    NAME = 'agem'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(AGem, self).__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size)

        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())
        self.grad_xy = torch.Tensor(np.sum(self.grad_dims)).to(self.device)
        self.grad_er = torch.Tensor(np.sum(self.grad_dims)).to(self.device)

    def end_task(self, dataset):
        samples_per_task = self.args.buffer_size // dataset.N_TASKS
        loader = dataset.train_loader
        cur_y, cur_x = next(iter(loader))[1:]
        self.buffer.add_data(
            examples=cur_x.to(self.device),
            labels=cur_y.to(self.device)
        )

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.zero_grad()
        p = self.net.forward(inputs)
        loss = self.loss(p, labels)
        loss.backward()

        if not self.buffer.is_empty():
            store_grad(self.parameters, self.grad_xy, self.grad_dims)

            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size,
                transform=self.transform,
                device=self.device,
            )
            self.net.zero_grad()
            buf_outputs = self.net.forward(buf_inputs)
            penalty = self.loss(buf_outputs, buf_labels)
            penalty.backward()
            store_grad(self.parameters, self.grad_er, self.grad_dims)

            dot_prod = torch.dot(self.grad_xy, self.grad_er)
            if dot_prod.item() < 0:
                g_tilde = project(gxy=self.grad_xy, ger=self.grad_er)
                overwrite_grad(self.parameters, g_tilde, self.grad_dims)
            else:
                overwrite_grad(self.parameters, self.grad_xy, self.grad_dims)

        self.opt.step()
        return loss.item()
```
