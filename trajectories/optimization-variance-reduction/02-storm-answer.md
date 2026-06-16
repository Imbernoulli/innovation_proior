**Problem.** SVRG's once-per-epoch full-gradient snapshot goes *stale* mid-sweep: on the stiff `κ = 100`
conditioned problem the iterate drifts away from the snapshot within a few inner steps, the control
variate stops reducing variance, the full-magnitude first-step-of-epoch overshoots, and the loss diverges
(measured `nan` / `8.5e10`). The anchor must move with the iterate.

**Key idea (STORM = recursive momentum).** Keep a running gradient estimate and re-anchor it *every*
step at the previous iterate, with a momentum contraction folded in:
`d_t = (1−a) g_t + a (d_{t-1} + g_t − g'_{t-1})`, i.e. `d_t = ∇f_i(x_t) + (1−a)(d_{t-1} − ∇f_i(x_{t-1}))`,
then `x_{t+1} = x_t − η d_t`. The same-sample two-point difference `g_t − g'_{t-1}` is `O(L||x_t−x_{t-1}||)`
— one step fresh, never stale — and the `(1−a)` momentum contracts accumulated error. Tracking
`ε_t = d_t − ∇F(x_t)` gives a contraction `(1−a)ε_{t-1}` plus a small same-step correction, so the error
settles at a small equilibrium continuously, with no per-epoch reset.

**Why it should help.** Continual re-anchoring is the direct cure for SVRG's staleness/divergence: the
correction always uses the tiny consecutive-step gap and the first step of each epoch is never a
full-magnitude snapshot gradient. The conditioned MSE should go from `nan` to finite and small.

**This-harness specifics (NOT the adaptive paper STORM).** The learning rate is the fixed handed-in
`self.lr` — there is no AdaGrad cube-root step here. With `η` fixed, the momentum cannot be tied to `η²`
adaptively; it is set once per epoch from the inner-step count `T = n//b` as `a = 1 − 1/√T`, the same
scalar formula on all three problems. The recursion is seeded by **one** full gradient on the **first
epoch only** (the single allowed full pass), after which the running `d` carries across epochs and no more
full gradients are needed — total gradient cost far below SVRG's per-epoch snapshots. The two-point
gradient uses the same load-previous/restore mechanism: two stochastic-gradient calls per batch.

**Hyperparameters.** `a = 1 − 1/√(n//b)` (fixed per epoch); `η = self.lr`; `full_grad_count = 1` on the
first epoch, 0 thereafter.

```python
# EDITABLE region of custom_vr.py (lines 286-370) -- step 2: STORM
class VarianceReductionOptimizer:
    """STORM: STochastic Recursive Momentum.

    Maintains a momentum-based gradient estimator that achieves variance
    reduction without requiring periodic full gradient computations (unlike
    SVRG/SARAH).  The key idea is to use an exponential moving average of
    recursively corrected stochastic gradients:

        d_t = (1-a) * g_t + a * (d_{t-1} + g_t - g_{t-1}')

    where g_t = grad_i(x_t), g_{t-1}' = grad_i(x_{t-1}), and a is a
    momentum coefficient.  The first epoch uses a full gradient to warm-start.
    """

    def __init__(self, model: nn.Module, lr: float, l2_reg: float,
                 loss_type: str, n_train: int, batch_size: int,
                 device: torch.device):
        self.model = model
        self.lr = lr
        self.l2_reg = l2_reg
        self.loss_type = loss_type
        self.n_train = n_train
        self.batch_size = batch_size
        self.device = device
        self.params = list(model.parameters())
        # Momentum coefficient (STORM paper recommends a = 1 - 1/sqrt(T))
        n_steps_per_epoch = max(1, n_train // batch_size)
        self.momentum = 1.0 - 1.0 / math.sqrt(n_steps_per_epoch)
        # Running gradient estimator
        self.d = None
        # Previous parameters for correction term
        self.prev_params = None
        self.initialized = False

    def _save_params(self):
        return [p.data.clone() for p in self.params]

    def _load_params(self, saved):
        for p, s in zip(self.params, saved):
            p.data.copy_(s)

    def train_one_epoch(self, X_train: torch.Tensor,
                        y_train: torch.Tensor) -> dict:
        self.model.train()
        n = X_train.size(0)
        a = self.momentum
        full_grad_count = 0

        # Initialize with full gradient on first epoch
        if not self.initialized:
            self.d = compute_full_gradient(
                self.model, X_train, y_train, self.loss_type,
                self.l2_reg, self.device
            )
            self.prev_params = self._save_params()
            # First step using full gradient
            with torch.no_grad():
                for p, di in zip(self.params, self.d):
                    p.data.add_(di, alpha=-self.lr)
            self.initialized = True
            full_grad_count = 1

        indices = torch.randperm(n)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            idx = indices[start:end]
            Xb = X_train[idx].to(self.device)
            yb = y_train[idx].to(self.device)

            # Current stochastic gradient g_t = grad_i(x_t)
            current_params = self._save_params()
            g_current = compute_stochastic_gradient(
                self.model, Xb, yb, self.loss_type, self.l2_reg
            )

            # Previous stochastic gradient g_{t-1}' = grad_i(x_{t-1})
            self._load_params(self.prev_params)
            g_prev = compute_stochastic_gradient(
                self.model, Xb, yb, self.loss_type, self.l2_reg
            )
            self._load_params(current_params)

            # STORM update: d_t = (1-a)*g_t + a*(d_{t-1} + g_t - g_{t-1}')
            with torch.no_grad():
                for i, (p, gc, gp, di) in enumerate(zip(
                        self.params, g_current, g_prev, self.d)):
                    self.d[i] = (1 - a) * gc + a * (di + gc - gp)
                    p.data.add_(self.d[i], alpha=-self.lr)

            self.prev_params = self._save_params()

            # Track loss
            with torch.no_grad():
                loss = compute_loss_on_batch(
                    self.model, Xb, yb, self.loss_type, self.l2_reg
                )
                total_loss += loss.item()
            n_batches += 1

        return {"avg_loss": total_loss / max(n_batches, 1),
                "full_grad_count": full_grad_count}
```
