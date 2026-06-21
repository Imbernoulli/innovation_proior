# Context: continual learning from a stream of tasks under a small memory (circa 2018-2019)

## Research question

A learner is shown a sequence of supervised tasks one after another — task 1, then task 2, and
so on — and must keep performing *all* of them. The hard fact about this setting is catastrophic
forgetting (McCloskey & Cohen 1989; Robins 1995): if you simply keep training the same network on
each new task, gradient descent on task `t` overwrites the weights that solved tasks `< t`, and
accuracy on the old tasks drops. The problem is to learn each new task well while losing as little
as possible of what was already learned.

The regime the field treats as realistic: **a single pass over each task's data**, full
supervision, an integer task id available at train and test time to pick the right output head,
and only a **small, fixed-size episodic memory** in which the learner may stash a handful of past
examples. The single-pass constraint says learning is online; the small-memory constraint is what
separates continual learning from ordinary multi-task learning, where the complete dataset of every
task is available at every step. The question is how to update the network from each new minibatch
in a way that keeps old-task accuracy high, given only that tiny buffer of past examples.

## Background

The field state. The dominant framings split into two families. **Regularization-based** methods
protect the weights that mattered for old tasks: after finishing a task they store a per-parameter
importance measure and, on later tasks, add a penalty that pulls each weight back toward its old
value in proportion to that importance. **Memory-based** (rehearsal) methods keep a small set of
past examples and use them to stop the loss on old tasks from climbing. In the single-pass studies
of 2018-2019, memory-based methods were reported to do better than regularization-based ones in this
online regime, and the prominent memory-based methods used the memory as a **constraint on the
optimization** — letting the memory veto or rotate a gradient step that would damage old tasks.

The gradient-constraint line noted that minimizing the current-example loss *together with* the loss
on the episodic memory "results in overfitting to the examples stored in [the memory]" (Lopez-Paz &
Ranzato 2017), since with only a few stored examples per class repeated gradient steps on them risk
memorizing that handful.

The load-bearing concepts. **Catastrophic forgetting** as the core obstacle. The **single-pass /
fixed-memory protocol** (Chaudhry et al. 2019) as the evaluation contract: a small cross-validation
stream of a few tasks for tuning, then one-and-only-one pass over the evaluation stream.
**Average accuracy** and **forgetting** as the two metrics. The idea of an **episodic memory** —
a small buffer holding past `(input, task-id, label)` triplets — and the question of *how to
write into it* under a stream of unknown length, for which the standard tool is **reservoir
sampling** (Vitter 1985): to keep a uniform random subset of an unbounded stream in a buffer of
size `mem_sz`, retain the `n`-th arriving item with probability `mem_sz / n`, overwriting a random
slot. And from reinforcement learning, the long-standing observation (Lin 1992; Mnih et al.
2013, 2015) that replaying stored transitions from a large buffer over many passes stabilizes
learning — though in supervised continual learning the buffer is tiny and the stream is seen once.

## Baselines

**Fine-tune (Vanilla).** Initialize task `t` from the parameters left by task `t-1` and train with
ordinary SGD and cross-entropy, no memory, no penalty. The cheapest possible method and the
reference point.

**EWC** (Kirkpatrick et al. 2017). A regularization method. After learning a task it estimates,
for each weight, how important that weight was, via the diagonal of the Fisher information matrix
`F_i` (the expected squared gradient of the log-likelihood). On the next task it minimizes
`L_t(θ) + Σ_i (λ/2) F_i (θ_i − θ*_i)^2`, anchoring important weights near their old values `θ*`.
Memory cost in the worst case equals the number of parameters (one importance scalar per weight).

**GEM** (Lopez-Paz & Ranzato 2017). A memory method that treats the loss on each past task's
memory as an **inequality constraint**: minimize the current-task loss subject to
`ℓ(θ, M_k) ≤ ℓ(θ^{t-1}, M_k)` for every `k < t` — never let any past-task memory loss increase.
Operationally, at each step it computes a gradient `g_k` on each past task's memory and, if the
proposed update `g` makes a negative-inner-product violation with some `g_k`
(`⟨g, g_k⟩ < 0`), projects `g` to the nearest vector `g̃` (in L2) satisfying
`⟨g̃, g_k⟩ ≥ 0 ∀k < t`. That projection is a quadratic program; solved in the dual it has `t-1`
variables, and it requires assembling the matrix `G = (g_1, …, g_{t-1}) ∈ R^{(t-1)×P}` every step.
GEM also allows *positive backward transfer* (old-task loss may decrease).

**A-GEM** (Chaudhry et al. 2019). GEM with the `t-1` per-task constraints replaced by a single
*averaged* one: sample one random minibatch from the union of all memories, take its gradient
`g_ref`, and require only `⟨g̃, g_ref⟩ ≥ 0`. The projection then has a closed form — if
`g^T g_ref ≥ 0`, keep `g`; otherwise

```
g̃ = g − (g^T g_ref / g_ref^T g_ref) g_ref
```

i.e. subtract off the component of `g` that points against the memory gradient. No QP, no stored
`G` — one extra gradient and one inner product per step, much faster than GEM with comparable
single-pass accuracy. The memory acts as a *veto*: it is used only when there is a violation, and
then it removes a component of the current-task gradient.

**MER** (Riemer et al. 2018). Also keeps an episodic memory, but combines current and memory
examples through a Reptile-style meta-learning objective whose inner updates approximately maximize
the dot products between gradients of current and past tasks (encouraging gradient alignment).
Reported strong in the single-pass regime; the meta-update carries an inner loop per step.

## Evaluation settings

The standard yardsticks for this regime, all pre-existing:

- **Datasets.** Permuted MNIST (each task a fixed random pixel permutation; 23 tasks). Split
  CIFAR-100 (the 100 classes cut into 20 disjoint 5-way tasks). Split miniImageNet (100 classes,
  600 images each, into 20 disjoint 5-way tasks). Split CUB (200 fine-grained bird classes split
  into 20-way subsets). In each, a few tasks (3) form the cross-validation stream and the rest the
  evaluation stream.
- **Protocol.** A single pass over each evaluation-stream task; metrics reported on the held-out
  test sets of the evaluation stream; hyper-parameters tuned only on the cross-validation stream.
- **Architectures.** A 2×256-unit ReLU MLP for MNIST; a reduced ResNet18 for CIFAR and
  miniImageNet; an ImageNet-pretrained ResNet18 for CUB. An integer task id selects a task-specific
  classifier head. Cross-entropy loss; plain SGD; training minibatch of 10; the memory minibatch is
  also 10 regardless of how big the buffer is.
- **Metrics.** Average accuracy `A_T = (1/T) Σ_{j=1}^{T} a_{T,j}`, the mean over tasks of accuracy
  after the final task. Forgetting `F_T = (1/(T-1)) Σ_{j=1}^{T-1} (max_{l∈{1,...,T-1}} a_{l,j} − a_{T,j})`,
  the mean drop from each task's best historical accuracy to its final accuracy.

## Code framework

The substrate is an ordinary continual-learning training harness that already knows how to
fine-tune through a sequence of tasks. There is a base learner that owns a model (here a
behavior-cloning policy producing a task loss via `compute_loss`), an optimizer, and a scheduler, and that
exposes four lifecycle hooks the outer loop calls: `__init__` once, `start_task` before each task,
`observe` on every training minibatch (move to device, zero grads, compute the loss, backprop,
optional grad-clip, step), and `end_task` after each task. The plain fine-tune baseline lives
entirely in these hooks with empty `start_task`/`end_task` bodies and a bare single-batch
`observe`. What a forgetting-mitigation strategy actually *does* inside these hooks is the open slot.

```python
import torch
import torch.nn as nn


class Sequential(nn.Module):
    """Fine-tuning base learner and superclass for continual-learning strategies.
    Owns the policy (-> compute_loss), optimizer, scheduler. The strategy fills the hooks."""

    def __init__(self, n_tasks, cfg, **policy_kwargs):
        super().__init__()
        self.cfg = cfg
        self.n_tasks = n_tasks
        self.loss_scale = cfg.train.loss_scale
        self.policy = build_policy(cfg)        # BC policy: self.policy.compute_loss(data) -> loss
        self.current_task = -1

    def start_task(self, task):
        """Called before training on each new task; sets up optimizer/scheduler.
        # TODO: any per-task state a strategy needs goes here."""
        self.current_task = task
        self.optimizer = build_optimizer(self.policy.parameters(), self.cfg)
        self.scheduler = build_scheduler(self.optimizer, self.cfg)

    def observe(self, data):
        """One training step on a minibatch of the current task.
        # TODO: a strategy may change what `data` (or the gradient) is before/at the step."""
        data = self.map_tensor_to_device(data)
        self.optimizer.zero_grad()
        loss = self.policy.compute_loss(data)
        (self.loss_scale * loss).backward()
        if self.cfg.train.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.train.grad_clip)
        self.optimizer.step()
        return loss.item()

    def end_task(self, dataset, task_id, benchmark, env=None):
        """Called after finishing a task.
        # TODO: post-processing for the strategy (if any) goes here."""
        pass


# Pre-existing data utilities the harness already ships:
#   DataLoader, ConcatDataset, RandomSampler         (PyTorch)
#   TruncatedSequenceDataset(dataset, n)             (truncate a dataset to n sequences)
#   self.map_tensor_to_device(data)                  (move a nested dict of tensors to GPU)
```

The strategy is whatever fills `__init__`, `start_task`, `observe`, and `end_task` — the harness
above is everything that exists before that choice is made.
</content>
</invoke>
