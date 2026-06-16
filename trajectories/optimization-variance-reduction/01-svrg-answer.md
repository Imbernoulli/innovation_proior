**Problem.** Vanilla mini-batch SGD on a finite sum has a gradient-variance floor that survives at the
optimum (`∇f_i(x*) ≠ 0` while their average is zero), so a constant step leaves a noise ball and the rate
collapses to `O(1/t)` unless the step is annealed. The variance is the villain; kill it and a big
constant step survives. The conditioned (`κ = 100`) MSE problem is where this floor is the binding
constraint.

**Key idea (SVRG control variate).** Keep a snapshot `x̃` of the parameters and its exact full gradient
`μ̃ = ∇F(x̃)` (one full pass per epoch). Replace the mini-batch gradient with the control-variate
direction `v = ∇f_i(x) − ∇f_i(x̃) + μ̃`. It is unbiased for `∇F(x)` for any `x̃`, and as both `x` and `x̃`
approach `x*` the matched-index difference cancels the per-example offset that floored SGD, so `v → 0`
and a constant step needs no annealing. State is `x̃` and `μ̃` only — `O(d)` memory, no per-example table.

**Why it should help.** The variance-reduced direction drives the residual noise ball toward zero, which
is exactly what the strongly convex, MSE-scored conditioned problem rewards; on the accuracy-scored
logistic/MLP problems the floor was never binding, so SVRG should roughly match SGD there.

**This-harness specifics.** The budget caps `compute_full_gradient` at once per epoch, so the snapshot is
taken at the *start of every epoch* and the inner loop is one pass of `n/b` batches — the epoch *is* the
inner loop (`m = n/b`), not a freely chosen `2n`/`5n`. There is no cached-scalar path for `∇f_i(x̃)`, so
each inner step physically loads the snapshot parameters, evaluates the stochastic gradient there, and
restores — three gradient touches per batch (current point, snapshot point, plus the snapshot pass),
roughly twice an SGD epoch's cost.

**Hyperparameters.** Step size `self.lr` used directly for cross-entropy. For the `mse` loss only, the
step is capped by the inverse full-gradient norm, `effective_lr = min(self.lr, 1.0/||μ̃||)`, to bound the
full-magnitude first step of each epoch on the ill-conditioned quadratic (the first batch's direction is
exactly `μ̃`, which can overshoot and diverge). `full_grad_count = 1` per epoch.

```python
# EDITABLE region of custom_vr.py (lines 286-370) -- step 1: SVRG
class VarianceReductionOptimizer:
    """SVRG with adaptive step sizing and geometric growth cap.

    At the start of each epoch, computes a full gradient at the current
    snapshot point.  Each inner iteration uses the control-variate estimator:
        v_t = grad_i(x_t) - grad_i(x_snap) + mu   (where mu = full_grad(x_snap))

    Step size: eta = min(lr, 0.01 * ||w||/||g||, eta_max).
    eta_max grows geometrically at 1.5x per epoch, allowing the step to
    increase as training progresses (gnorm decreases) while preventing the
    runaway growth that caused divergence in v2.
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
        self.snapshot_params = None
        self.full_grad = None
        self.eta_max = None

    def _save_snapshot(self):
        self.snapshot_params = [p.data.clone() for p in self.params]

    def _load_snapshot(self):
        saved = [p.data.clone() for p in self.params]
        for p, sp in zip(self.params, self.snapshot_params):
            p.data.copy_(sp)
        return saved

    def _restore_params(self, saved):
        for p, s in zip(self.params, saved):
            p.data.copy_(s)

    def train_one_epoch(self, X_train: torch.Tensor,
                        y_train: torch.Tensor) -> dict:
        self.model.train()
        n = X_train.size(0)

        # --- Snapshot ---
        self._save_snapshot()
        self.full_grad = compute_full_gradient(
            self.model, X_train, y_train, self.loss_type,
            self.l2_reg, self.device
        )

        # Standard SVRG: use the provided lr directly. For ill-conditioned
        # MSE problems cap the first-step magnitude by 1/||∇F|| to prevent
        # divergence (previous adaptive 1.5x-geometric schedule blew up to
        # eta≈1e5 and gave final MSE≈1e34).
        if self.loss_type == 'mse':
            gnorm = math.sqrt(sum(
                g.pow(2).sum().item() for g in self.full_grad)) + 1e-8
            effective_lr = min(self.lr, 1.0 / gnorm)
        else:
            effective_lr = self.lr

        indices = torch.randperm(n)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            idx = indices[start:end]
            Xb = X_train[idx].to(self.device)
            yb = y_train[idx].to(self.device)

            grad_at_x = compute_stochastic_gradient(
                self.model, Xb, yb, self.loss_type, self.l2_reg
            )

            saved = self._load_snapshot()
            grad_at_snap = compute_stochastic_gradient(
                self.model, Xb, yb, self.loss_type, self.l2_reg
            )
            self._restore_params(saved)

            # SVRG update: v = grad_i(x_t) - grad_i(x_snap) + mu
            with torch.no_grad():
                for p, gx, gs, mu in zip(self.params, grad_at_x,
                                         grad_at_snap, self.full_grad):
                    vr_grad = gx - gs + mu
                    p.data.add_(vr_grad, alpha=-effective_lr)

            with torch.no_grad():
                loss = compute_loss_on_batch(
                    self.model, Xb, yb, self.loss_type, self.l2_reg
                )
                total_loss += loss.item()
            n_batches += 1

        return {"avg_loss": total_loss / max(n_batches, 1),
                "full_grad_count": 1}
```
