## Research question

Lifelong robot manipulation in LIBERO-Spatial: the agent learns ten pick-and-place tasks **one at a time, in sequence**. Every task shares the same action goal (pick up a black bowl, place it on a plate) and differs only in spatial layout. It trains on task 0 to convergence, then task 1, and so on through task 9; only after the last task is it evaluated on **all ten**. The single design freedom is the *lifelong learning strategy* layered on a fixed behavior-cloning learner — the mechanism that governs how task k's training interacts with what tasks 0..k−1 learned. Everything else is frozen: the transformer policy, the BC loss, AdamW, the cosine schedule, and the 50-epochs-per-task budget.

## Prior art / Background / Baselines

Available continual-learning approaches include:

- **Joint / multi-task training.** Train on all ten tasks' data interleaved. Forgetting is avoided because the weights are jointly fit across all tasks.

- **Uniform L2 anchoring.** After a task, snapshot the weights θ* and add `(λ/2)||θ − θ*||²` while training the next task; constant memory, no old data required.

- **Plain dropout / weight decay.** These encourage generic capacity regularization across the network.

- **Progressive Networks.** Give each task its own column and freeze the rest, so earlier representations are not overwritten.

## Fixed substrate / Code framework

The behavior-cloning continual learner is frozen. The transformer policy `self.policy` maps multimodal observations (RGB views, proprioception) to actions; its BC objective is `self.policy.compute_loss(data)` (a scalar; `reduction="none"` is available for per-sample losses). The parent `Sequential` class drives the lifecycle: it constructs the AdamW `self.optimizer` and the cosine-annealing `self.scheduler` in its own `start_task`, exposes `self.loss_scale`, `self.current_task`, `self.n_tasks`, `self.cfg`, and the helper `self.map_tensor_to_device(data)`. Each task trains for 50 epochs; after the last task the agent is rolled out in every task environment and scored. The loop also provides module-level helpers a strategy may use: `cycle(dl)` (an infinite iterator over a DataLoader), `merge_datas(x, y)` (recursively concatenate two nested data dicts along the batch dimension), `TruncatedSequenceDataset(dataset, n)` (keep the first `n` sequence windows), `safe_device`, `TensorUtils.map_tensor`, and PyTorch data utilities `DataLoader`, `ConcatDataset`, `RandomSampler`.

## Editable interface

Exactly one region is editable — the `Custom(Sequential)` class in `LIBERO/libero/lifelong/algos/custom.py` (lines 37–65). A strategy implements any of four lifecycle hooks:

- `__init__(self, n_tasks, cfg, **policy_kwargs)` — allocate the strategy's state (buffers, regularizers, flat gradient scratch).
- `start_task(self, task)` — called before each task; set up task-specific state. Must call `super().start_task(task)` so the parent rebuilds the optimizer/scheduler.
- `observe(self, data)` — called every training step; compute the loss, backprop, step the optimizer. Must return the forward loss as a float.
- `end_task(self, dataset, task_id, benchmark, env=None)` — called after each task; post-process (store replay data, compute statistics).

The starting point is plain sequential fine-tuning: no buffer, no penalty, no projection. `observe` does the ordinary BC step and `end_task` does nothing. A strategy replaces exactly this class body.

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

LIBERO-Spatial, ten tasks learned sequentially, 50 epochs per task, seed 42. After the final task the policy is rolled out in all ten task environments. Metrics, all higher-is-better: **avg_final_success** (mean success rate across the ten tasks after the last task), and **auc** — the canonical LIBERO lifelong score (area under the per-task success curve, which folds forward learning and backward retention into one number); the leaderboard also records **fwt** (forward transfer) and **nbt** (negative backward transfer / forgetting). The ranking metric is `auc`.
