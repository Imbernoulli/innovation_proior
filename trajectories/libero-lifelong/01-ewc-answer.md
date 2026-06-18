**Problem.** Ten LIBERO-Spatial tasks learned in sequence; plain fine-tuning erases earlier tasks
because the task-k objective has no term protecting the weights tasks 0..k−1 needed. The cheapest fix
stores no old data — it tries to protect forgetting by penalizing changes to the weights an old task
relied on.

**Key idea (online EWC).** Read training as MAP estimation; all of a finished task's knowledge lives in
the posterior `p(θ|D_A)`, which a Laplace approximation turns into a Gaussian centered at the
post-task weights θ\* with precision = the Hessian of the old negative log posterior. Replace the
intractable Hessian by the (diagonal) **Fisher information**, an average of squared first-order
gradients. The task-B objective becomes a per-weight quadratic spring
`L_B(θ) + e_lambda · Σ_i F_i (θ_i − θ\*_i)²`: weights an old task cared about (large `F_i`) are held
nearly rigid, the rest stay free.

**This task's implementation (not the textbook EWC).**
- **Online, decayed Fisher**, not one-per-task: a *single* running `fish` updated `fish ← γ·fish + F`
  with `γ = 0.9`, and a single re-snapped anchor `checkpoint`. Constant memory; early tasks' curvature
  decays away.
- **Empirical Fisher from the demonstrated actions**, not the model-distribution Fisher: `end_task`
  loops the finished task's data, backprops `(−nll).mean()` from `compute_loss(..., reduction="none")`,
  and accumulates `grads²`. This is `E_{(x,y)∼data}[(score)²]`, exact only at a good fit.
- **`e_lambda = 50000`, no 1/2 factor** (absorbed into e_lambda); penalty applied only for
  `current_task > 0`; `get_params`/`get_grads` flatten the transformer's parameters/gradients so the
  spring is one elementwise op.

**Why start here.** No replayed data, so the entire defense is a blunt weight anchor that protects
coordinates, not the old policy's function; the decay drops the earliest tasks and the large e_lambda
fights the new task. Expected to sit at the bottom — that weakness motivates putting old examples back
into the batch (rehearsal) at the next rung.

**Hyperparameters.** `e_lambda = 50000`, `gamma = 0.9`; Fisher over the full finished-task dataloader;
penalty added every step once past task 0.

```python
# EDITABLE region of LIBERO/libero/lifelong/algos/custom.py (lines 37-65) — step 1: EWC (online)
class Custom(Sequential):
    """EWC (Elastic Weight Consolidation) lifelong learning algorithm."""

    def __init__(self, n_tasks, cfg, **policy_kwargs):
        super().__init__(n_tasks=n_tasks, cfg=cfg, **policy_kwargs)
        self.e_lambda = 50000
        self.gamma = 0.9
        self.checkpoint = None
        self.fish = None

    def get_params(self):
        return torch.cat([p.reshape(-1) for p in self.policy.parameters()])

    def get_grads(self):
        return torch.cat(
            [
                p.grad.reshape(-1)
                if p.grad is not None
                else torch.zeros_like(p).reshape(-1)
                for p in self.policy.parameters()
            ]
        )

    def penalty(self):
        if self.checkpoint is None:
            return safe_device(torch.tensor(0.0), self.cfg.device)
        else:
            penalty = (self.fish * ((self.get_params() - self.checkpoint) ** 2)).sum()
            return penalty

    def start_task(self, task):
        super().start_task(task)

    def observe(self, data):
        data = self.map_tensor_to_device(data)
        self.optimizer.zero_grad()
        loss = self.policy.compute_loss(data)
        forward_loss = loss.item()
        if self.current_task > 0:
            loss += self.e_lambda * self.penalty()
        assert not torch.isnan(loss)
        (loss * self.loss_scale).backward()
        if self.cfg.train.grad_clip is not None:
            nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.cfg.train.grad_clip
            )
        self.optimizer.step()
        return forward_loss

    def end_task(self, dataset, task_id, benchmark, env=None):
        self.policy.train()
        fish = torch.zeros_like(self.get_params())

        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=self.cfg.train.num_workers,
        )

        for data in dataloader:
            data = TensorUtils.map_tensor(
                data, lambda x: safe_device(x, device=self.cfg.device)
            )
            self.policy.zero_grad()
            nll = self.policy.compute_loss(data, reduction="none")
            (-nll).mean().backward()
            grads = self.get_grads()
            fish += grads ** 2

        fish /= len(dataloader)

        if self.fish is None:
            self.fish = fish
        else:
            self.fish *= self.gamma
            self.fish += fish

        self.checkpoint = self.get_params().data.clone()
```
