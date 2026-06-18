## Research question

Lifelong robot manipulation: the agent learns ten LIBERO-Spatial pick-and-place tasks **one at a
time, in sequence**. Every task shares the same action goal (pick up a black bowl, place it on a
plate) and differs only in spatial layout. It trains on task 0 to convergence, then task 1, and so on
through task 9; only after the last task is it evaluated on **all ten**. The single thing being
designed is the *lifelong learning strategy* layered on a fixed behavior-cloning learner — the
mechanism, if any, that stops task k's training from erasing what tasks 0..k−1 learned. The pain is
**catastrophic forgetting**: plain sequential fine-tuning rewrites the very weights the earlier tasks
relied on, so by the time the final task finishes the early tasks have collapsed. Everything else about
the agent — the transformer policy, the BC loss, AdamW, the cosine schedule, the 50-epochs-per-task
budget — is frozen.

## Prior art before the first rung (continual-learning lineage)

The first rung reacts to the standard continual-learning toolbox; these are the ancestors the ladder
climbs out of, each with the gap that pushes past it.

- **Joint / multi-task training (the unreachable ceiling).** Train on all ten tasks' data
  interleaved. Forgetting vanishes because the weights are jointly fit. But the sequential regime
  forbids it: once a task finishes, its full dataset is gone, and interleaving would require hoarding
  every task's data, with storage growing in the number of tasks. Gap: remembers by hoarding data, not
  by remembering anything about the network — violates the streaming budget.
- **Uniform L2 anchoring (naive regularization).** After a task, snapshot the weights θ\* and add
  `Σ_i (λ/2)(θ_i − θ\*_i)²` while training the next task — constant memory, no old data. But the single
  global stiffness λ cannot tell apart the weights an old task needed from the weights the new task must
  move: large λ freezes the whole net (can't learn the new task), small λ fails to hold the old one. Gap:
  a uniform penalty has no notion of *which* weights mattered.
- **Plain dropout / weight decay.** Resist forgetting indirectly, mostly by pushing toward larger nets
  (spare capacity), and stop working after a couple of sequential tasks. Gap: not protecting any
  *specific* knowledge.
- **Progressive Networks (Rusu et al. 2016).** Give each task its own column and freeze the rest, so
  forgetting is literally zero. But parameters and memory grow with the number of tasks, and every
  column is carried to test time. Gap: unbounded growth on a long stream.

These split the design space into two camps the ladder will test: *protect the weights that mattered*
(regularization) versus *keep a few old examples and use them* (rehearsal). The fixed substrate below is
the BC learner all of them bolt onto.

## The fixed substrate

A behavior-cloning continual learner is frozen and must not be touched. A transformer policy
`self.policy` maps multimodal observations (RGB views, proprioception) to actions; its BC objective is
`self.policy.compute_loss(data)` (a scalar; `reduction="none"` is available for per-sample losses).
The parent `Sequential` class drives the lifecycle — it constructs the AdamW `self.optimizer` and the
cosine-annealing `self.scheduler` in its own `start_task`, exposes `self.loss_scale`,
`self.current_task`, `self.n_tasks`, `self.cfg`, and the helper `self.map_tensor_to_device(data)` that
moves a nested observation dict to the GPU. Each task trains for 50 epochs; after the last task the agent
is rolled out in every task's environment and scored. The loop also provides module-level helpers a
strategy may use: `cycle(dl)` (an infinite iterator over a DataLoader, needed because a replay stream
must outlast the current-task loader), `merge_datas(x, y)` (recursively concatenate two nested data
dicts along the batch dimension), `TruncatedSequenceDataset(dataset, n)` (keep the first `n` sequence
windows of a dataset), `safe_device`, `TensorUtils.map_tensor`, and the PyTorch data utilities
`DataLoader`, `ConcatDataset`, `RandomSampler`.

## The editable interface

Exactly one region is editable — the `Custom(Sequential)` class in
`LIBERO/libero/lifelong/algos/custom.py` (lines 37–65). Every method on the ladder is a fill of this
same contract by overriding any of four lifecycle hooks:

- `__init__(self, n_tasks, cfg, **policy_kwargs)` — allocate the strategy's state (buffers,
  regularizers, flat gradient scratch).
- `start_task(self, task)` — called before each task; set up task-specific state (e.g. build the replay
  iterator from finished tasks). Must call `super().start_task(task)` so the parent rebuilds the
  optimizer/scheduler.
- `observe(self, data)` — called every training step; compute the loss, backprop, step the optimizer.
  Must return the (forward) loss as a float.
- `end_task(self, dataset, task_id, benchmark, env=None)` — called after each task; post-process (store
  replay data, compute Fisher information).

The starting point is the scaffold default: **plain sequential fine-tuning** — no buffer, no penalty,
no projection. `observe` does the ordinary BC step and `end_task` does nothing. Each later method
replaces exactly this class body.

```python
# EDITABLE region of LIBERO/libero/lifelong/algos/custom.py (lines 37-65) — default fill (Sequential)
class Custom(Sequential):
    """Custom lifelong learning algorithm.

    Override __init__, start_task, observe, and/or end_task to implement
    your lifelong learning strategy. The goal is to minimize catastrophic
    forgetting across sequential robot manipulation tasks.
    """

    def __init__(self, n_tasks, cfg, **policy_kwargs):
        super().__init__(n_tasks=n_tasks, cfg=cfg, **policy_kwargs)

    def start_task(self, task):
        super().start_task(task)

    def observe(self, data):
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
        pass
```

## Evaluation settings

LIBERO-Spatial, ten tasks learned sequentially, 50 epochs per task, seed 42. After the final task the
policy is rolled out in all ten task environments. Metrics, all higher-is-better: **avg_final_success**
(mean success rate across the ten tasks, evaluated after the last task), and **auc** — the canonical
LIBERO lifelong score (area under the per-task success curve, which folds forward learning and backward
retention into one number); the leaderboard also records **fwt** (forward transfer) and **nbt**
(negative backward transfer / forgetting). The ranking metric is `auc`.
