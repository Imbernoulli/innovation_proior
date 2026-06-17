## Research question

A learner is shown a sequence of tasks `t = 1, ..., T`, one after another. It trains on task 1,
then task 2, and so on, and after the whole sequence is finished it is evaluated on *every*
task it ever saw. The data arrives as a stream — ideally each task's examples are seen once,
in a single pass, because the point of the setting is to learn quickly and cheaply from a
continuum, not to replay a fixed dataset many times. The central failure mode is
**catastrophic forgetting**: when ordinary training (gradient descent on the current task's
loss) moves the parameters to fit task `t`, the accuracy on tasks `1, ..., t-1` collapses,
because nothing in the objective ties the parameters to what made the old tasks work.

The precise goal is an algorithm that, while training on the current task, keeps the loss on
the earlier tasks from going up — but it must do so under hard *deployment* constraints that
rule out the obvious cures. It may **not** simply store and retrain on all past data (that
turns lifelong learning back into ordinary multitask learning and defeats the streaming
premise), and its memory must not grow without bound. Just as important, the **per-step
compute** must stay close to that of plain SGD: a method whose cost per gradient step grows
with the number of tasks already seen does not scale to a long stream. And it should keep
*positive backward transfer* on the table — learning a new task is allowed to *improve* old
ones, so a cure that freezes old behavior outright is giving something up. A solution has to
hit all of these at once: limited samples (single pass), limited memory, limited per-step
compute, forgetting under control, transfer not forbidden.

## Background

By this time the field has settled into a few broad families of cure, each with a
characteristic cost.

**Regularization-based methods** add a penalty that anchors the parameters near the values
that solved past tasks. The prototype is Elastic Weight Consolidation (Kirkpatrick et al.,
2017): after task `t-1` it estimates, per parameter, how *important* that parameter was —
using the diagonal of the Fisher information `F_i`, the expected squared gradient of the
log-likelihood — and while learning task `t` it adds `Σ_i (λ/2) F_i (θ_i − θ*_i)^2`, pulling
important parameters back toward their old values `θ*` and letting unimportant ones move
freely. Synaptic Intelligence (Zenke et al., 2017) and RWalk (Chaudhry et al., 2018) compute
the per-parameter importance differently (path integral of the loss; a KL-based importance)
but share the quadratic-anchor shape. These are extremely cheap — a single extra penalty
term, memory linear in the number of parameters, no stored examples.

**Modular / architectural methods** sidestep interference by giving each task its own
capacity. Progressive Networks (Rusu et al., 2016) add a fresh column of units per task and
freeze the old ones, so old tasks are literally untouched. PathNet and expert-gate variants
route each task through a learned subset of modules.

**Episodic-memory methods** keep a *small* buffer of raw examples from past tasks and use it
to constrain or rehearse. iCaRL (Rebuffi et al., 2017) stores a handful of exemplars per
class and uses them with a distillation loss. The line that matters most here treats the
stored examples not as a rehearsal set to fit, but as a way to *measure* whether a proposed
update would harm past tasks.

The empirical facts that frame the problem are well established. Plain sequential fine-tuning
forgets badly: accuracy on task `j` drops sharply once training moves on to later tasks.
Regularization methods, which were validated mostly in the **multi-epoch** regime where the
network can settle into a good basin, are observed to be much weaker in the **single-pass**
streaming regime, and their behavior is sensitive to the regularization strength `λ` — too
small and they forget, too large and they cannot learn the new task. Modular methods either
grow memory/parameters with the number of tasks (Progressive Networks) or require searching
over architectures, which is not sample-efficient under a single-pass budget. And it is a
basic geometric fact about gradient descent that the harm a single update does to a past task
is, to first order, governed by the **angle** between the current update direction and the
direction that would reduce the past task's loss: if the proposed step has a positive inner
product with the past-task gradient it tends to *decrease* that task's loss, and if the inner
product is negative it tends to *increase* it. This linear-around-a-small-step picture is the
diagnostic everything below is built on.

## Baselines

These are the prior methods a new continual-learning algorithm would be measured against and
would react to.

**Sequential fine-tuning (the naive baseline).** Just minimize the current task's loss with
SGD/AdamW and move on: at each step `θ ← θ − α ∇_θ ℓ(f_θ, D_t)`. Nothing constrains the
parameters relative to past tasks. **Gap:** maximal forgetting — it is the failure mode the
whole field is trying to fix.

**Elastic Weight Consolidation and the regularization family (Kirkpatrick et al., 2017;
Zenke et al., 2017; Chaudhry et al., 2018).** While learning task `t`, optimize
`ℓ(f_θ, D_t) + Σ_i (λ/2) F_i (θ_i − θ*_i)^2`, where `F_i` is the estimated importance of
parameter `i` for past tasks and `θ*` are the post-task-`t-1` values. Importance is computed
from the diagonal Fisher (EWC), a path integral (SI), or a KL/Riemannian measure (RWalk).
This is memory-cheap (per-parameter importances, no stored data) and compute-cheap (one extra
penalty). **Gap:** it bakes the constraint into a soft scalar penalty whose strength `λ` must
be tuned and is fragile; it was designed for and works best with many epochs per task, where
the optimizer has time to find a basin that satisfies both old and new tasks, and is observed
to underperform when the learner gets only a single pass over each task's data. It also tends
to *forbid* movement of important parameters rather than allow movement that happens to help
old tasks, so it does not naturally support positive backward transfer.

**Progressive Networks (Rusu et al., 2016).** Allocate a new network column per task and
freeze previous columns, with lateral connections so a new task can reuse old features.
Forgetting is zero by construction. **Gap:** memory and parameter count grow linearly with
the number of tasks, and at test time every column is kept around — infeasible for a long
stream under a fixed memory budget.

**Gradient Episodic Memory (Lopez-Paz & Ranzato, 2017).** The closest relative and the one to
study in detail. Keep a small episodic memory `M_k` of raw examples for each past task `k`.
The idea is to phrase "don't make past tasks worse" as a *hard inequality constraint* rather
than a soft penalty: while learning task `t`,

```
minimize_θ   ℓ(f_θ, D_t)
subject to   ℓ(f_θ, M_k) ≤ ℓ(f_{θ}^{t-1}, M_k)   for all k < t,
```

i.e. the loss on each past task's memory may *decrease* (positive backward transfer is
allowed) but may not *increase* beyond its value at the end of task `t-1`. Two observations
make this tractable. First, one need not store the old predictor `f_{θ}^{t-1}`: it suffices
to guarantee that each update does not raise the past-task losses, which around a small step
is a condition on the gradient. Second, by local linearity the constraint
`ℓ(f_θ, M_k) ≤ ℓ(f_{θ}^{t-1}, M_k)` becomes, for the proposed update gradient `g = ∇_θ
ℓ(f_θ, D_t)` and the past-task memory gradients `g_k = ∇_θ ℓ(f_θ, M_k)`,

```
⟨g, g_k⟩ ≥ 0   for all k < t.
```

If every inner product is non-negative the proposed step is unlikely to harm any past task
and is taken as-is. If one or more are negative, GEM replaces `g` by the *closest feasible*
gradient `g̃` in Euclidean norm:

```
minimize_{g̃}   (1/2) ‖g − g̃‖_2^2
subject to      ⟨g̃, g_k⟩ ≥ 0   for all k < t.
```

This is a quadratic program in `P` variables — the number of network parameters, in the
millions — which is solved through its dual, a much smaller QP in `t − 1` variables (one per
past task):

```
minimize_v   (1/2) vᵀ G Gᵀ v + gᵀ Gᵀ v   subject to   v ≥ 0,
```

with `G = −(g_1, ..., g_{t-1})` and the recovered update `g̃ = Gᵀ v* + g`. **Gap:** the cost
is structural and grows with the stream. At *every* training step, GEM must compute the full
matrix `G` — one backward pass over each of the `t − 1` past-task memories — and then solve a
QP with a numerical solver (quadprog). As the number of tasks and the memory size grow, both
the `t − 1` extra backward passes per step and the per-step QP become prohibitive, and it must
also keep gradients for all tasks in memory. GEM gives a strong *worst-case* guarantee — no
individual past task's (memory) loss is allowed to rise — but pays for it with a per-step cost
that scales with how much it has already learned.

## Evaluation settings

The natural yardsticks already in use for streaming continual learning:

- **Permuted MNIST** — a sequence of tasks, each a fixed random permutation of the 784 input
  pixels applied to all MNIST digits; classify the digit. Tasks share label space but differ
  in input statistics. A long sequence (e.g. 20 tasks) under a single pass per task.
- **Split CIFAR** — partition the classes into disjoint groups, each group a
  task; the learner sees one group at a time. Tests class-incremental forgetting.
- **Split CUB / AWA** — disjoint subsets of bird or animal classes as a task
  stream, with small per-task budgets to stress few-shot learning.
- Metrics: **Average Accuracy** `A_T = (1/T) Σ_{j=1}^{T} a_{T,j}`, the mean test accuracy over
  all tasks after the last task is learned (the headline number); **Forgetting** `F`, the
  average over tasks of (best accuracy ever reached on a task) minus (its final accuracy); and
  a **learning-curve / speed** metric capturing how fast a new task is picked up.
- Protocol: a strict single pass over each task's training data for the final run; because
  some methods are sensitive to a regularization/strength hyper-parameter, that
  hyper-parameter is selected on a *separate, disjoint* stream of validation tasks rather than
  by replaying the evaluation stream, so the evaluation stream is genuinely seen once.

## Code framework

The method plugs into a generic continual-learning model object. The harness already owns a
network `net`, a loss function, an optimizer `opt`, a bounded replay-style `Buffer`, and the
standard minibatch entry point `observe(inputs, labels, not_aug_inputs, epoch=None)`. Existing
episodic-memory baselines also have helpers for flattening parameter gradients into a single
vector and writing a flat vector back into `.grad` fields. What is not settled is the rule
inside `observe`: after the current minibatch gradient has been computed, how, if at all, should
the stored examples affect the gradient before `opt.step()`?

```python
import torch

from models.gem import overwrite_grad, store_grad


class Buffer:
    """Bounded store of examples from completed tasks."""

    def __init__(self, size):
        self.size = size

    def is_empty(self):
        pass

    def add_data(self, examples, labels):
        pass

    def get_data(self, batch_size, transform=None, device=None):
        pass


class ContinualModel:
    """Generic streaming continual-learning harness."""

    def __init__(self, backbone, loss, args, transform, dataset=None):
        self.net = backbone
        self.loss = loss
        self.args = args
        self.transform = transform
        self.dataset = dataset
        self.buffer = Buffer(args.buffer_size)

    def parameters(self):
        return self.net.parameters()

    def zero_grad(self):
        self.net.zero_grad()

    def end_task(self, dataset):
        # TODO: decide what, if anything, to retain from the finished task.
        pass

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.zero_grad()
        outputs = self.net.forward(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()

        if not self.buffer.is_empty():
            # TODO: decide how memory should influence the current gradient.
            pass

        self.opt.step()
        return loss.item()
```

The outer loop supplies one minibatch to `observe`; everything about how that step interacts
with past examples lives in the body of `observe` and in the post-task bookkeeping of
`end_task`.
