# Context: gradient-based few-shot meta-learning (circa 2019)

## Research question

The goal is to make a high-capacity neural network learn a brand-new task from only a handful of
labeled examples — classify novel image classes after one or five examples per class, fit a new
continuous function from a few points — and do it *fast*, in a few gradient steps, after having
meta-trained on many related tasks drawn from a distribution `p(T)`. Each task `T` brings a small
support set and a held-out query set; the unit of generalization is the task. The object we get to
optimize ahead of time is the *adaptation procedure itself*: the initialization, the update
direction, and the learning rate of the inner loop.

By early 2019 the dominant frame is gradient-based meta-learning (MAML and its variants): meta-learn
a shared initialization and adapt to a new task by a few ordinary gradient steps on its support set,
differentiating the post-adaptation *query* loss back into the initialization. This works and scales,
but the inner update is plain SGD — the gradient direction is fixed by the support loss and the rate is
usually a scalar or, at most, coordinate-wise. The precise open question is whether the inner update can
be made better conditioned without leaving the MAML frame: standard gradient descent performs poorly
on ill-conditioned loss surfaces, and the inner-loop optimizer's choice is known to matter, but a
few-shot learner cannot afford a huge curvature matrix, matrix inversions, or a curvature estimate that
simply overfits the support examples. A solution must provide a richer update direction than scalar SGD,
remain cheap enough for convolutional networks, and be optimized for post-adaptation query
generalization rather than support-set fitting alone.

## Background

The problem is **few-shot meta-learning**. A distribution over tasks `p(T)`; each task `T_i` supplies
a support set `train(T_i)`, a query set `test(T_i)`, and a loss `L_{T_i}`. During meta-training the
system sees tasks sampled from `p(T)`; at meta-test it adapts to held-out tasks from `K` examples. A
learner is a parametric differentiable map `f_θ`; the standard inner adaptation is iterated gradient
descent on the support loss, `θ^t = θ^{t-1} − α ∇L_T(θ^{t-1})`, with `α` a scalar rate and `θ^0` a
learned initialization. Three ingredients fully specify such an optimizer — initialization, update
direction, learning rate — and the few-shot regime makes the defaults (follow the gradient, one
hand-set rate) liabilities.

Two threads inform a curvature-based answer. First, **second-order optimization**: Newton's method
preconditions the gradient with the inverse Hessian, `θ − α H^{-1} ∇L`, taking a local quadratic view
of the loss; natural gradient descent preconditions with the inverse Fisher information matrix, `θ − α
F^{-1} ∇L`, a steepest-descent direction in distribution space. Both accelerate gradient descent on
ill-conditioned surfaces, and both are infeasible at full scale, which is why approximations exist —
online updates, **Kronecker-factored approximations (K-FAC)** that approximate the Fisher as `F ≈ A ⊗
G` (A from input-unit activations, G from output-unit gradients), and diagonal approximations. But all
of these *compute* curvature from the current training loss and data, so with only `K` examples they
estimate curvature from almost nothing and tend to converge fast to a poor local minimum — they do not
consider *generalization*. Second, **tensor algebra**: a layer's weights and gradients are naturally
multidimensional tensors (`C_out × C_in × d` for a conv layer with filter size `d = h·w`, `d = 1` for
a linear layer), and an `n`-mode product `X ×_n M` multiplies a tensor along its `n`-th mode by a
matrix M — which lets a structured transform of the gradient tensor be factored into small per-mode
matrices instead of one giant matrix over all coordinates.

## Baselines

These are the prior gradient-based meta-learners a curvature method is measured against and reacts to.

**MAML — Model-Agnostic Meta-Learning (Finn, Abbeel & Levine, ICML 2017).** Meta-learn only the
*initialization* `θ`; the inner loop is ordinary SGD at a scalar rate, `θ'_i = θ − α ∇L_{T_i}(f_θ)`.
The meta-objective optimizes the query loss of the adapted parameters across tasks, `min_θ Σ_i
L_{T_i}(f_{θ'_i})`, and the meta-update `θ ← θ − β ∇_θ Σ_i L_{T_i}(f_{θ'_i})` differentiates through
the inner step (a Hessian-vector product; FOMAML drops it). Architecture-agnostic, no extra
parameters. **Gap:** the inner update is plain SGD — one global learning rate, the direction locked to
the raw gradient. The only thing meta-learned is *where to start*; *how* to move is a fixed rigid rule.

**Meta-SGD (Li, Zhou, Chen & Li, 2017).** Meta-learn the initialization *and* a per-parameter
learning-rate vector `α` of the same shape as `θ`; the inner step is the elementwise product `θ' = θ −
α ∘ ∇L`, with `α` meta-trained jointly with `θ`. This is a *diagonal* preconditioner of the gradient:
a distinct learnable rate per coordinate, and (when the entries differ) a step that tilts off the raw
gradient. **Gap:** the preconditioner is diagonal — it scales each coordinate independently but cannot
model *dependencies between* gradient coordinates (between input channels, output channels, or filter
positions within a layer). The curvature of a real loss is not diagonal.

**K-FAC (Martens & Grosse 2015) as an inner optimizer.** The natural candidate for a structured
preconditioner: approximate the Fisher by `A ⊗ G` and precondition with its inverse. **Gap:** A ∈
`R^{C_in d × C_in d}` is expensive to maintain computationally and spatially even for small nets;
computed from training-loss statistics it ignores generalization; and applied as a meta-learned
inner optimizer it tends to overfit the meta-training set. It motivates the factorization but not the
recipe.

The state of the field before the target work: MAML scales but its inner step is plain SGD; Meta-SGD
adds a learned coordinate-wise preconditioner but does not model cross-coordinate dependencies; and
second-order methods have the right geometric motivation but compute curvature from data statistics and
do not scale cleanly as an inner-loop few-shot optimizer. The gap is a way to improve the update
geometry while preserving the support/query meta-learning contract.

## Evaluation settings

The standard few-shot yardsticks of the period (datasets/protocols predate this method):

- **Few-shot image classification.** *Omniglot* (Lake et al. 2011), *MiniImagenet* (Ravi &
  Larochelle 2017 split, 64/16/20 classes, 84×84), and *tieredImagenet* (Ren et al. 2018). Episodic
  `N`-way `K`-shot sampling with a held-out query set per episode (Vinyals et al. 2016); metric is
  mean classification accuracy over many test episodes with 95% confidence intervals. Standard
  backbone: four conv modules (3×3 conv → batch-norm → ReLU → 2×2 max-pool), following the MAML setup.
- **Few-shot regression.** Sine curves with varying amplitude/phase, `K` support points, MSE loss, a
  small MLP — tests recovery of an unseen function from few points.
- Protocol: meta-train across sampled task batches, meta-validate hyperparameters, report on held-out
  meta-test tasks; one or a few inner adaptation steps; the comparison is against MAML, Meta-SGD,
  and other MAML variants under matched splits and backbones.

## Code framework

The method plugs into a standard gradient-based meta-learning harness that already exists. The *outer*
loop is fixed: sample a batch of tasks; for each, clone the learner from the shared initialization,
run an inner adaptation on the support set, score the adapted clone on the query set, and meta-update
across tasks by back-propagating the summed query loss *through* the inner adaptation. The data
pipeline (episodic `N`-way `K`-shot tasksets), the backbone, the cross-entropy loss, the
differentiable parameter-update primitive (a functional update that re-routes parameter tensors so
gradients flow back through the inner step to the meta-parameters), and the meta-optimizer (Adam) are
all given. What is *not* settled is the inner adaptation rule — exactly what is to be designed.

The inner rule lives behind a single object with a constructor (which may create learnable optimizer
state), an `adapt` method that takes a *cloned* learner and the support set and returns the adapted
learner using *differentiable* ops (`torch.autograd.grad`, not `torch.optim`, so the graph survives),
and a `meta_parameters` accessor returning any optimizer state to be trained by the outer loop. The
starting default is the plainest inner rule — ordinary SGD with one fixed scalar rate and no trainable
optimizer state — with one neutral empty slot where the designed update rule will go.

```python
from typing import List
import torch
import torch.nn.functional as F
from torch import Tensor, nn

INNER_LR = 0.5  # default scalar inner-loop step size (a hand-set hyperparameter)


class InnerLoopOptimizer(nn.Module):
    """Inner-loop adaptation rule for gradient-based meta-learning.

    Defines HOW the cloned learner's parameters move during fast adaptation on a
    task's support set. The outer (meta) loop is fixed and differentiates THROUGH
    whatever this object does, so adapt() must use differentiable ops and any
    learnable optimizer state must be exposed via meta_parameters().
    """

    def __init__(self, model: nn.Module, inner_lr: float = INNER_LR):
        super().__init__()
        self.inner_lr = inner_lr

    def adapt(self, model: nn.Module, support_x: Tensor, support_y: Tensor,
              n_steps: int) -> nn.Module:
        # model is a CLONE — safe to modify; must stay differentiable so the
        # outer loop can backprop through the inner step.
        model.train()
        for _ in range(n_steps):
            loss = F.cross_entropy(model(support_x), support_y)
            grads = torch.autograd.grad(
                loss, model.parameters(), retain_graph=True, create_graph=True
            )
            model = self._apply_update(model, grads)
        return model

    def _apply_update(self, model: nn.Module, grads: List[Tensor]) -> nn.Module:
        # TODO: define the adaptation rule.
        raise NotImplementedError

    def meta_parameters(self) -> List[Tensor]:
        return list(self.parameters())
```

The outer loop calls `adapt` on each task's clone and meta-optimizes the shared initialization
together with whatever `meta_parameters()` returns; the single empty slot is the inner update rule.
