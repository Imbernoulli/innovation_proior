I need a learner that can absorb a stream of supervised tasks one after another and still perform all of them at the end. The hard fact is catastrophic forgetting: ordinary fine-tuning on task t rewrites the weights that solved tasks 1..t−1, so old-task accuracy collapses. The regime that matters is unforgiving: a single pass over each task's data, a known task id for choosing the right head, and only a small fixed episodic memory that can hold a handful of past examples. That small memory is the whole point — if I could keep every task's full dataset I would just do multi-task learning. So the real question is how to retain old tasks with almost no extra compute and a tiny buffer.

The existing answers each sacrifice something. Regularization methods like EWC protect weights by pinning them with a per-parameter Fisher importance, but that stores one scalar per weight and the quadratic anchor is blind to which directions in weight space actually preserve the old function; under a single pass it barely beats fine-tuning. Memory-based methods do better, but the strongest ones treat the memory as a constraint rather than data. GEM solves a per-step quadratic program to project the current gradient so no past-task memory loss can rise, which is expensive and scales with the number of tasks. A-GEM trims this to a single averaged half-space constraint with a closed-form projection, making it fast enough for the online regime. Yet the memory still only acts as a veto: most steps it does nothing, and even when it intervenes it only removes the component of the current gradient that would hurt old tasks. It caps the old-task loss but never drives it down, which is why A-GEM underfits both the memory and the new task.

The method I propose is Experience Replay, or ER. The idea is to stop being clever about constraints and simply train on the memory. At each step I take the current task minibatch, stack it with a random minibatch drawn from the episodic memory, and perform one ordinary SGD step on the union. If the two minibatches are the same size, the gradient of the mean loss on the concatenation points along (g + g_ref)/2, where g is the current-task gradient and g_ref is the memory gradient. So ER always descends both the new task and the old tasks, every step, instead of merely guarding a ceiling. It is actually simpler than A-GEM: no extra backward to compute a reference gradient for projection, no inner product, no flat gradient buffers, no quadratic program — just feed a double-sized batch through the same forward/backward and step once.

The obvious objection is that repeatedly fitting a tiny memory should overfit those few stored examples and generalize badly. That objection is correct only when the memory is trained on in isolation. In ER the memory is always co-trained with the full current-task dataset, and that large companion acts as an implicit, data-dependent regularizer: it prevents the network from contorting itself to memorize the few old examples, because doing so would wreck its fit on the abundant new-task data. The strength of this regularizer is set by the amount of current-task data, and its helpfulness is set by how related the tasks are. In the normal case of moderately or closely related tasks — the LIBERO-Spatial setting, where every task is the same pick-and-place skill with only layout changes — co-training helps rather than hurts. Only near-adversarial task pairs can turn the regularizer against the old task, and those are rare. This is why A-GEM's caution backfires: by never descending the memory loss, it never exposes itself to the beneficial regularization that makes direct replay work.

The remaining design choice is what to store. When the memory is reasonably large relative to the number of tasks, reservoir sampling gives a uniform random subset of the stream and keeps the stored set fresh. When the memory is tiny, a balanced writer is safer: a per-class ring buffer, online k-Means in feature space, or a mean-of-features selection all guarantee that no early class is accidentally evicted entirely. A hybrid that starts with reservoir and switches to a balanced scheme once any class falls below a minimum count works across memory sizes without knowing the task count in advance. In the behavior-cloning implementation below the replay buffer is built by truncating each finished task to a fixed number of sequence windows, concatenating them, and cycling a randomly sampled DataLoader. The observe hook pulls one replay batch, recursively concatenates it onto the current nested observation dict, and runs the ordinary BC step.

```python
import collections

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, RandomSampler

from libero.lifelong.algos.base import Sequential
from libero.lifelong.datasets import TruncatedSequenceDataset
from libero.lifelong.utils import *


def cycle(dl):
    while True:
        for data in dl:
            yield data


def merge_datas(x, y):
    if isinstance(x, (dict, collections.OrderedDict)):
        new_x = dict() if isinstance(x, dict) else collections.OrderedDict()
        for k in x.keys():
            new_x[k] = merge_datas(x[k], y[k])
        return new_x
    elif isinstance(x, torch.FloatTensor) or isinstance(x, torch.LongTensor):
        return torch.cat([x, y], 0)


class Custom(Sequential):
    """ER (Experience Replay): train on current batch stacked with a replay batch."""

    def __init__(self, n_tasks, cfg, **policy_kwargs):
        super().__init__(n_tasks=n_tasks, cfg=cfg, **policy_kwargs)
        self.n_memories = 1000
        self.datasets = []
        self.buffer = None

    def start_task(self, task):
        super().start_task(task)
        if self.current_task > 0:
            buffers = [
                TruncatedSequenceDataset(dataset, self.n_memories)
                for dataset in self.datasets
            ]
            buf = ConcatDataset(buffers)
            self.buffer = cycle(
                DataLoader(
                    buf,
                    batch_size=self.cfg.train.batch_size,
                    num_workers=self.cfg.train.num_workers,
                    sampler=RandomSampler(buf),
                    persistent_workers=(self.cfg.train.num_workers > 0),
                )
            )

    def observe(self, data):
        if self.buffer is not None:
            buf_data = next(self.buffer)
            data = merge_datas(data, buf_data)

        data = self.map_tensor_to_device(data)

        self.optimizer.zero_grad()
        loss = self.policy.compute_loss(data)
        (self.loss_scale * loss).backward()
        if self.cfg.train.grad_clip is not None:
            nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.cfg.train.grad_clip
            )
        self.optimizer.step()
        return loss.item()

    def end_task(self, dataset, task_id, benchmark, env=None):
        self.datasets.append(dataset)
```
