**Problem.** A-GEM lifted avg_final to 0.22 (auc 0.2166) but its nbt stayed high at 0.573: the memory
acts only as a *veto* — it caps the old-task loss rather than driving it down, so old-task rollouts keep
failing. The fix is to descend the old loss directly.

**Key idea (Experience Replay).** Drop the constraint/projection entirely; just put the old examples in
the batch. Each step, stack a random memory minibatch onto the current minibatch and take one ordinary
BC step. The gradient on the concatenation is the *average* direction `(g + g_ref)` (constant absorbed
into the LR), so — unlike A-GEM's veto — ER *always* descends both the current task and the memory, every
step. The standard objection (repeatedly fitting a small buffer overfits it) holds only if the memory is
trained on *alone*; co-trained with the large current-task dataset, that dataset is an implicit
data-dependent regularizer, so memorizing the buffer and generalizing to the old task stop conflicting.
On LIBERO-Spatial the ten tasks share one skill and differ only in layout (related, never adversarial),
the regime where co-training helps most.

**This task's implementation.** No reservoir / balanced writer: `end_task` appends the whole finished
dataset; `start_task` (past task 0) truncates each to its first `n_memories = 1000` windows with
`TruncatedSequenceDataset`, `ConcatDataset`s them, and `cycle`s a `RandomSampler` DataLoader. `observe`
pulls one replay batch and `merge_datas` (recursive nested-dict concat) it onto the current batch, then
one ordinary BC step — the concatenation *is* the average-gradient move. `persistent_workers` is guarded
on `num_workers > 0`. Task 0 trains as plain BC.

**Why over A-GEM.** Descends the old-task loss every step instead of vetoing increases, reaching the
co-training regularization A-GEM's caution never exposed itself to — and it is computationally *simpler*
(no `g_ref` projection, no flat-gradient buffers).

**Expectations.** avg_final well past 0.22 and auc above 0.2166; nbt well below 0.573; fwt may dip
slightly (half of each batch is now old data), the correct retention-for-speed trade on an
after-final-task metric.

**Hyperparameters.** `n_memories = 1000` windows per finished task; replay batch = `train.batch_size`;
one merged step per training step.

```python
# EDITABLE region of LIBERO/libero/lifelong/algos/custom.py (lines 37-65) — step 3: ER
class Custom(Sequential):
    """ER (Experience Replay) lifelong learning algorithm."""

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
