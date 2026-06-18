**Problem.** EWC's decayed weight-spring did not hold the early tasks (avg_final 0.045, auc 0.0487, nbt
0.17): anchoring *coordinates* protects neither the old policy's *function* nor permits backward
transfer. The next step protects the old tasks at the source of forgetting — the gradient inner product
— using a small episodic memory of real past examples.

**Key idea (A-GEM).** A step `θ ← θ − αg` changes a past task's loss by `≈ −α⟨g, g_j⟩`, so the sign of
the inner product is the forgetting diagnostic. Require only that the *average* past loss not rise:
with `g` the current gradient and `g_ref` the gradient on a random batch from the union memory, the
constraint is the single inequality `⟨g̃, g_ref⟩ ≥ 0`, whose closest-feasible projection is closed form:

```
if  gᵀg_ref ≥ 0:  g̃ = g                                  # already feasible — take the step
else:             g̃ = g − (gᵀg_ref / g_refᵀg_ref) g_ref   # remove the offending component
```

In the projected case `g̃ᵀg_ref = 0` (orthogonal to the reference: no average harm, minimal change to
`g`). One extra backward + two dot products per step, flat in the number of tasks; one-sided so
backward transfer is allowed (unlike EWC's symmetric anchor).

**This task's implementation.** Episodic memory = the list of finished-task datasets, each truncated to
`n_memories = 1000` sequence windows; the reference stream is a `cycle`d `RandomSampler` DataLoader over
their `ConcatDataset`. `g` and `g_ref` are flattened into preallocated buffers (`grad_xy`, `grad_er`)
via module-level `_store_grad`/`_overwrite_grad`; `_project` is the closed form. Backward-pass
sequencing is load-bearing: current-task backward → store `g` → zero_grad → reference backward → store
`g_ref` → project or restore → grad-clip → step. Task 0 trains as plain BC (empty buffer).

**Why over EWC.** Constrains the *average memory loss* (aligned with avg-final-success) using actual old
data, instead of a blunt coordinate anchor that decays the early tasks out. Expected to clear EWC on
avg_final and auc with lower nbt.

**Limitation that motivates the next rung.** The memory acts only as a *veto* — on the common
`⟨g, g_ref⟩ ≥ 0` step it does nothing; it caps the old loss but never drives it down, so it underfits
both the new task and the memory.

**Hyperparameters.** `n_memories = 1000` windows per finished task; reference batch = `train.batch_size`;
projection applied exactly when `dot_prod < 0`.

```python
# EDITABLE region of LIBERO/libero/lifelong/algos/custom.py (lines 37-65) — step 2: A-GEM
def _project(gxy, ger):
    """Project gradient gxy so it does not conflict with ger."""
    corr = torch.dot(gxy, ger) / torch.dot(ger, ger)
    return gxy - corr * ger


def _store_grad(params, grads, grad_dims):
    """Store parameter gradients into a flat vector."""
    grads.fill_(0.0)
    count = 0
    for param in params():
        if param.grad is not None:
            begin = 0 if count == 0 else sum(grad_dims[:count])
            end = np.sum(grad_dims[: count + 1])
            grads[begin:end].copy_(param.grad.data.view(-1))
        count += 1


def _overwrite_grad(params, newgrad, grad_dims):
    """Overwrite parameter gradients from a flat vector."""
    count = 0
    for param in params():
        if param.grad is not None:
            begin = 0 if count == 0 else sum(grad_dims[:count])
            end = sum(grad_dims[: count + 1])
            this_grad = newgrad[begin:end].contiguous().view(param.grad.data.size())
            param.grad.data.copy_(this_grad)
        count += 1


class Custom(Sequential):
    """A-GEM (Averaged Gradient Episodic Memory) lifelong learning algorithm."""

    def __init__(self, n_tasks, cfg, **policy_kwargs):
        super().__init__(n_tasks=n_tasks, cfg=cfg, **policy_kwargs)
        self.n_memories = 1000
        self.datasets = []
        self.buffer = None

        self.grad_dims = []
        for pp in self.policy.parameters():
            self.grad_dims.append(pp.data.numel())
        self.grad_xy = torch.Tensor(np.sum(self.grad_dims)).to(self.cfg.device)
        self.grad_er = torch.Tensor(np.sum(self.grad_dims)).to(self.cfg.device)

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
        data = self.map_tensor_to_device(data)
        self.optimizer.zero_grad()
        loss = self.policy.compute_loss(data)
        (loss * self.loss_scale).backward()

        if self.buffer is not None:
            _store_grad(self.policy.parameters, self.grad_xy, self.grad_dims)
            buf_data = next(self.buffer)
            self.policy.zero_grad()

            buf_data = self.map_tensor_to_device(buf_data)
            buf_loss = self.policy.compute_loss(buf_data)
            buf_loss.backward()
            _store_grad(self.policy.parameters, self.grad_er, self.grad_dims)

            dot_prod = torch.dot(self.grad_xy, self.grad_er)
            if dot_prod.item() < 0:
                g_tilde = _project(gxy=self.grad_xy, ger=self.grad_er)
                _overwrite_grad(self.policy.parameters, g_tilde, self.grad_dims)
            else:
                _overwrite_grad(self.policy.parameters, self.grad_xy, self.grad_dims)

        if self.cfg.train.grad_clip is not None:
            nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.cfg.train.grad_clip
            )
        self.optimizer.step()
        return loss.item()

    def end_task(self, dataset, task_id, benchmark, env=None):
        self.datasets.append(dataset)
```
