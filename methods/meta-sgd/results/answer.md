# Meta-SGD, distilled

Meta-SGD is a gradient-based few-shot meta-learner that meta-learns **both** the learner's
initialization `θ` **and** a learnable per-parameter learning-rate **vector** `α` of the same
shape as `θ`. The inner-loop adaptation on a task's support set is a single elementwise-scaled
gradient step,

```
θ' = θ − α ∘ ∇L_T(θ),        (∘ = elementwise / Hadamard product)
```

where `α ∘ ∇L_T(θ)` is a vector whose *orientation* is the per-coordinate-rescaled update
direction and whose norm is the effective step size. For a nonzero gradient with at least two
active coordinates, the rescaled vector stays parallel to the raw gradient only when the active
entries of `α` are all the same scalar. Otherwise the coordinate-wise rescaling tilts the step
off the raw-gradient direction. Both `θ` and `α` are
meta-trained jointly by ordinary backprop, with no recurrent network and no
backpropagation-through-time over an optimizer.

## Problem it solves

Few-shot learning: adapt a high-capacity differentiable learner `f_θ` to a new task from only
`K` examples (often 1 or 5), fast (ideally one inner step), without overfitting, by
meta-training across a distribution of related tasks `p(T)`. The contribution is the *inner
adaptation rule*: which way and how far each parameter moves during adaptation.

## Key idea

Of the three ingredients of an optimizer — initialization, update direction, learning rate —
meta-learn all three, but realize the "learned step" with the cheapest object that still
rescales per coordinate and tilts off the gradient: a **diagonal preconditioner**, i.e. a single
learned vector `α`.

- A *scalar* learning rate (as in plain SGD inner loops) gives one global rate and locks the
  direction to `∇L`. A *full matrix* preconditioner `A ∇L` could rotate arbitrarily but costs
  `|θ|²` memory and, per step, reintroduces the cost that makes recurrent optimizers
  unscalable. The *diagonal* `α ∘ ∇L` is the middle ground: linear memory, a distinct rate per
  coordinate, and an off-gradient direction.
- `α` is *learned across tasks*, not computed from a gradient history (hand-designed adaptive
  optimizers like AdaGrad/RMSProp/Adam set per-coordinate scaling from history, which one-step
  few-shot adaptation does not have). It encodes, for this task family, how far and which
  combined way each coordinate should move when the support gradient points a given way.

## Meta-training objective

Optimize the *query* loss of the *adapted* parameters, across tasks, over both `θ` and `α`:

```
min_{θ, α}  E_{T ~ p(T)} [ L_test(T)( θ − α ∘ ∇L_train(T)(θ) ) ].
```

This is differentiable in both `θ` and `α`. For one task, with `g = ∇_θ L_train(θ)`,
`H = ∇^2_θ L_train(θ)`, `θ' = θ − α ∘ g`, and `v = ∇_{θ'} L_test(θ')`:

```
∂L_test(θ')/∂α = −v ∘ g
∂L_test(θ')/∂θ = (I − diag(α) H)^T v
```

The `θ`-gradient is the second-order gradient-through-a-gradient path (a Hessian-vector product,
as in MAML); the `α`-gradient is the chain rule through the linear `α`-dependence of `θ'`. Solve
by SGD over task batches (Adam in practice). A first-order variant detaches `g`, keeps the
identity path through `θ`, and drops the `diag(α)H` term.

## Algorithm (supervised, one inner step shown; iterate the same rule for T steps)

```
Initialize θ (learner init), α (per-parameter rates, e.g. all entries = inner_lr)
while not done:
    sample a batch of tasks T_i ~ p(T)
    for each T_i:
        L_train ← (1/|train(T_i)|) Σ_{(x,y)∈train(T_i)} ℓ(f_θ(x), y)
        θ'_i   ← θ − α ∘ ∇L_train(θ)                      # differentiable inner step
        L_test_i ← (1/|test(T_i)|) Σ_{(x,y)∈test(T_i)} ℓ(f_{θ'_i}(x), y)
    (θ, α) ← (θ, α) − β ∇_{(θ,α)} Σ_i L_test_i(θ'_i)       # meta-update (β = meta-LR)
```

For multiple inner steps, apply `θ ← θ − α ∘ ∇L_train(θ)` repeatedly, reusing the same learned
`α` at each step and keeping the graph differentiable across steps.

## Relation to prior methods

- **MAML** (Finn, Abbeel & Levine 2017): freeze `α` to a single constant scalar in every
  coordinate and do not learn it ⇒ `θ' = θ − α ∇L`, plain SGD inner loop with only the
  initialization meta-learned. Meta-SGD strictly generalizes it by making `α` a non-uniform,
  learned vector (per-coordinate rate + learned direction).
- **Meta-Learner LSTM** (Ravi & Larochelle 2017): pursue the same ambition (learn init,
  direction, and rate) by using a recurrent network to emit per-coordinate gates each step. BPTT
  over the unrolled LSTM costs `O(T · #states · dim(θ))` memory and does not scale to
  convolutional learners; Meta-SGD keeps the learned-step ambition but removes the recurrence.

## Working code

The inner-loop rule, filling the harness's adaptation slot. `lrs` is the learned per-parameter
`α` (one tensor per parameter, same shape), exposed as meta-parameters; `adapt` runs the
differentiable `θ ← θ − α ∘ ∇L` for the requested number of inner steps.

```python
from typing import List
import torch
import torch.nn.functional as F
from torch import Tensor, nn

INNER_LR = 0.5  # initial value for every entry of alpha


class InnerLoopOptimizer(nn.Module):
    """Meta-SGD inner-loop optimizer: a learned per-parameter learning-rate vector
    alpha (the `lrs`), meta-optimized by the outer loop jointly with the learner
    initialization. Inner step:  theta <- theta - alpha (.) grad   (elementwise)."""

    def __init__(self, model: nn.Module, inner_lr: float = INNER_LR,
                 first_order: bool = False):
        super().__init__()
        self.inner_lr = inner_lr
        self.first_order = first_order
        # one learnable LR tensor per parameter, same shape, initialized uniform.
        self.lrs = nn.ParameterList([
            nn.Parameter(torch.ones_like(p) * inner_lr)
            for p in model.parameters()
        ])

    def adapt(self, model: nn.Module, support_x: Tensor, support_y: Tensor,
              n_steps: int) -> nn.Module:
        model.train()
        second_order = not self.first_order
        for _ in range(n_steps):
            loss = F.cross_entropy(model(support_x), support_y)
            # second_order keeps the Hessian-vector path through the inner gradient.
            grads = torch.autograd.grad(
                loss,
                model.parameters(),
                retain_graph=second_order,
                create_graph=second_order,
            )
            model = self._apply_update(model, grads)
        return model

    def _apply_update(self, model: nn.Module, grads: List[Tensor]) -> nn.Module:
        """Differentiable p <- p - lr * grad update using learnable rates."""
        for p, lr, g in zip(model.parameters(), self.lrs, grads):
            p.grad = g
            p._lr = lr

        def reroute(module: nn.Module) -> nn.Module:
            for name in module._parameters:
                p = module._parameters[name]
                if p is not None and p.grad is not None:
                    module._parameters[name] = p - p._lr * p.grad
                    p.grad = None
                    p._lr = None

            for name in module._buffers:
                buff = module._buffers[name]
                if buff is not None and buff.grad is not None and getattr(buff, "_lr", None) is not None:
                    module._buffers[name] = buff - buff._lr * buff.grad
                    buff.grad = None
                    buff._lr = None

            for name in module._modules:
                module._modules[name] = reroute(module._modules[name])
            return module

        return reroute(model)

    def meta_parameters(self) -> List[Tensor]:
        # alpha is meta-optimized alongside the initialization.
        return list(self.lrs.parameters())
```

This fills the same `InnerLoopOptimizer` slot as the scaffold while preserving the canonical
`learn2learn` `MetaSGD` mechanics: store
`lrs = ParameterList([Parameter(ones_like(p) * lr) for p in model.parameters()])`, compute
gradients with `autograd.grad(..., retain_graph=second_order, create_graph=second_order)`, and
re-route parameter tensors as `p ← p − lr_p · grad_p`. The surrounding harness supplies the cloned
learner to `adapt`, and the outer loop optimizes `θ` (the learner initialization) and `α` (the
`lrs`) together.
