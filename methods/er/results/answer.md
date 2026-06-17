# Experience Replay (ER) for continual learning, distilled

Experience Replay is the simplest strong baseline for supervised continual learning: keep a small
episodic memory of past examples, and at every training step stack a random memory minibatch onto
the current-task minibatch and take one ordinary gradient step on the union. No constraints, no
projection, no quadratic program — two changes to plain fine-tuning (update the memory; double the
batch), at near-zero extra cost.

## Problem it solves

Learn a stream of tasks one at a time without catastrophic forgetting, under the realistic regime:
a single pass over each task's data, an available task id, and only a small fixed episodic memory
of past examples (the small-memory constraint is what distinguishes continual learning from
multi-task learning).

## Key idea

Train *directly* on the memory instead of using it as an optimization constraint. With current-task
gradient `g` (from batch `B_n`) and memory gradient `g_ref` (from a random memory batch `B_M`), one
SGD step on the concatenation `B_n ∪ B_M` descends `(g + g_ref) / 2` when both losses are averaged
over equal-size minibatches, equivalently the same direction as `g + g_ref` up to a constant
learning-rate scale. This contrasts with the constraint-based ancestors:

- **GEM** projects `g` to the nearest `g̃` with `⟨g̃, g_k⟩ ≥ 0` for every past task `k` (a per-step
  QP over the stored gradient matrix `G`).
- **A-GEM** replaces those with one averaged constraint: keep `g` if `g^T g_ref ≥ 0`, else project
  `g̃ = g − (g^T g_ref / g_ref^T g_ref) g_ref`. The memory acts only as a *veto*, only on
  violations, and only caps the old-task loss.
- **ER** averages instead of projecting: it uses the memory *every* step and actively *descends*
  the old-task loss, with no projection solve and no QP.

## Why training on a tiny memory does not overfit

The objection (from the constraint line) was that repeatedly fitting a handful of stored examples
must overfit them. That holds only when the memory is trained on *in isolation*. In ER the memory
is always co-trained with the large current-task dataset `D_t`, which acts as an **implicit,
data-dependent regularizer** on the repeated learning of the memory. Two knobs govern it:

- **Strength** = size of `D_t` (more current-task data pulls harder away from memorizing the few
  stored examples).
- **Effectiveness** = relatedness of the tasks. Closely related tasks → co-training even gives
  positive transfer to the old task without the memory; moderately related → still helps;
  near-adversarial (e.g. a rotated 2 looking like a 5) → the regularization can hurt.

So close fitting of the stored examples does not by itself imply poor old-task generalization:
generalization can come from the companion data, not from the memory size alone. Constraint-based
methods (A-GEM) underfit — their cautious updates never fully fit the memory or the new task — and
so never fully reap this regularization.

## Algorithm

```
M <- empty buffer of size mem_sz ; n <- 0
for task t = 1 ... T:
    for minibatch B_n drawn without replacement from D_t:
        B_M <- random minibatch from M                 # only if M is non-empty
        theta <- SGD_step(theta, B_n  union  B_M, lr)  # one step on the stacked batch
        M <- UpdateMemory(mem_sz, t, n, B_n)           # write strategy (below)
        n <- n + |B_n|
return theta, M
```

## Memory write strategies

- **Reservoir sampling** (Vitter 1985): after `n` stream examples have been seen including the
  candidate, keep that candidate with probability `mem_sz / n` by drawing a uniform index over the
  seen-example count and overwriting a memory slot only if the index lands inside the buffer.
  Uniform random subset; best when the memory is reasonably sized. Risk in the tiny regime: can
  evict an entire early class, spiking its forgetting.
- **Ring buffer**: per-class fixed-size FIFO of size `mem_sz/C`; keeps last few per class.
  Guarantees class balance (best when memory is tiny); stored old samples are static.
- **k-Means / Mean-of-Features**: per class, store examples closest to online k-Means centroids
  (feature coverage) or to a running mean feature vector (mode coverage); both class-balanced.
- **Hybrid**: start reservoir, switch to a balanced scheme once any class drops below a minimum;
  needs no advance knowledge of the task count. Best across memory sizes.

## Metrics

Average accuracy `A_T = (1/T) Σ_{j=1}^{T} a_{T,j}` and forgetting
`F_T = (1/(T-1)) Σ_{j=1}^{T-1} (max_{l∈{1,...,T-1}} a_{l,j} − a_{T,j})`, on the held-out test sets after a single pass
over the evaluation stream.

## Working code

Filling the continual-learning harness's lifecycle hooks. The episodic memory is the list of
finished-task datasets; replay is built by truncating each finished-task dataset to
`cfg.lifelong.n_memories` sequences, concatenating those truncated buffers, and cycling a
random-sampled DataLoader over them; `observe` doubles the batch before one ordinary BC step.

```python
import collections

import numpy as np
import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, RandomSampler

from libero.lifelong.algos.base import Sequential
from libero.lifelong.datasets import TruncatedSequenceDataset
from libero.lifelong.utils import *


def cycle(dl):
    # replay must outlive the (different-length) current-task loader
    while True:
        for data in dl:
            yield data


def merge_datas(x, y):
    # BC data is a nested dict of obs modalities; concatenate matching tensors along batch dim
    if isinstance(x, (dict, collections.OrderedDict)):
        if isinstance(x, dict):
            new_x = dict()
        else:
            new_x = collections.OrderedDict()

        for k in x.keys():
            new_x[k] = merge_datas(x[k], y[k])
        return new_x
    elif isinstance(x, torch.FloatTensor) or isinstance(x, torch.LongTensor):
        return torch.cat([x, y], 0)


class ER(Sequential):
    """Experience replay: train on the current minibatch stacked with a random memory minibatch."""

    def __init__(self, n_tasks, cfg, **policy_kwargs):
        super().__init__(n_tasks=n_tasks, cfg=cfg, **policy_kwargs)
        # Finished-task datasets are truncated into replay buffers when replay is used.
        self.datasets = []
        self.descriptions = []
        self.buffer = None

    def start_task(self, task):
        super().start_task(task)
        if self.current_task > 0:
            buffers = [
                TruncatedSequenceDataset(dataset, self.cfg.lifelong.n_memories)
                for dataset in self.datasets
            ]
            buf = ConcatDataset(buffers)
            self.buffer = cycle(
                DataLoader(
                    buf,
                    batch_size=self.cfg.train.batch_size,
                    num_workers=self.cfg.train.num_workers,
                    sampler=RandomSampler(buf),
                    persistent_workers=True,
                )
            )

    def end_task(self, dataset, task_id, benchmark):
        self.datasets.append(dataset)

    def observe(self, data):
        if self.buffer is not None:
            buf_data = next(self.buffer)
            data = merge_datas(data, buf_data)

        data = self.map_tensor_to_device(data)

        self.optimizer.zero_grad()
        loss = self.policy.compute_loss(data)
        (self.loss_scale * loss).backward()
        if self.cfg.train.grad_clip is not None:
            grad_norm = nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.cfg.train.grad_clip
            )
        self.optimizer.step()
        return loss.item()
```
