**Problem.** STORM killed SVRG's divergence (conditioned MSE finite) but left the exact residual it
warned of: on the stiff `κ = 100` problem the final MSE is consistently *above* the best (3.91 vs 3.51,
4.06 vs 0.85) with a 4× seed spread — overshoot/oscillation near the optimum from a fixed step and fixed
momentum that cannot respond to local curvature.

**Key idea (adaptive-enhanced STORM).** Keep the STORM recursive-momentum estimate
`d_t = (1−a_t) g_t + a_t (d_{t-1} + g_t − g'_{t-1})`, but make all three knobs per-step adaptive:
a *decaying* momentum from a global counter `a_t = min(1 − 1/√(t+1), 0.999)` (more fresh gradient early,
harder averaging late); a *trust-region* step `η_t = min(self.lr, 0.01·||w||/||d||)` (full step on flat
directions, clamped to ~1% of the parameter norm when the estimate spikes — the direct overshoot guard);
and a *clip* `||d|| → 3·||g||` when `||d|| > 3·||g||` (forbids the rare runaway where the recursive
estimate is several times the raw gradient).

**Why it should help.** Each addition answers a measured STORM failure: the decaying momentum addresses
the late-epoch staleness (MLP best-to-final gap), the adaptive step removes the conditioned overshoot
(final-above-best), and the clip removes the per-seed spike. On `κ = 100` the conditioned MSE should drop
one-to-two orders of magnitude with a tight seed spread and final ≈ best.

**This-harness specifics (NOT the published parameter-free STORM+).** The published STORM+ sets
`η = (Σ||d||²/a)^{-1/3}` with no tuning constants; this version instead uses the handed-in `self.lr` as a
ceiling and the `0.01·||w||/||d||` trust-region as the adaptive arm, the global-counter `1 − 1/√(t+1)`
momentum, and an explicit `3×` clip — a heuristically-enhanced STORM, not the cube-root adaptive method.
Seed/gradient mechanics are STORM's: one full gradient on the first epoch (warm-started with the same
adaptive step), two stochastic-gradient calls per inner step.

**Hyperparameters.** `a_t = min(1 − 1/√(global_step+1), 0.999)`; `η_t = min(self.lr, 0.01·||w||/(||d||+1e-8))`;
clip threshold `3.0·||g||`; `full_grad_count = 1` on the first epoch.

```python
# EDITABLE region of custom_vr.py (lines 286-370) -- step 3: STORM+ (adaptive-enhanced)
class VarianceReductionOptimizer:
    """STORM+ with adaptive momentum and per-step adaptive lr.

    d_t = (1-a_t)*g_t + a_t*(d_{t-1} + g_t - g_{t-1}')
    a_t = min(1 - 1/sqrt(t+1), 0.999)

    Full gradient warmstart on first epoch.
    Per-step lr: min(lr, 0.01 * ||w|| / ||d||).
    Gradient clipping: scale d if ||d|| > 3*||g||.
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
        self.d = None
        self.prev_params = None
        self.initialized = False
        self.global_step = 0

    def _save_params(self):
        return [p.data.clone() for p in self.params]

    def _load_params(self, saved):
        for p, s in zip(self.params, saved):
            p.data.copy_(s)

    def _gnorm(self, grads):
        return math.sqrt(sum(g.pow(2).sum().item() for g in grads))

    def _step_lr(self, direction):
        dnorm = self._gnorm(direction)
        pnorm = math.sqrt(sum(
            p.data.pow(2).sum().item() for p in self.params)) + 1e-8
        return min(self.lr, 0.01 * pnorm / (dnorm + 1e-8))

    def train_one_epoch(self, X_train, y_train):
        self.model.train()
        n = X_train.size(0)
        full_grad_count = 0

        if not self.initialized:
            self.d = compute_full_gradient(
                self.model, X_train, y_train, self.loss_type,
                self.l2_reg, self.device)
            self.prev_params = self._save_params()
            eta = self._step_lr(self.d)
            with torch.no_grad():
                for p, di in zip(self.params, self.d):
                    p.data.add_(di, alpha=-eta)
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

            self.global_step += 1
            a = min(1.0 - 1.0 / math.sqrt(self.global_step + 1), 0.999)

            current_params = self._save_params()
            g_current = compute_stochastic_gradient(
                self.model, Xb, yb, self.loss_type, self.l2_reg)

            self._load_params(self.prev_params)
            g_prev = compute_stochastic_gradient(
                self.model, Xb, yb, self.loss_type, self.l2_reg)
            self._load_params(current_params)

            with torch.no_grad():
                for i, (gc, gp, di) in enumerate(zip(
                        g_current, g_prev, self.d)):
                    self.d[i] = (1 - a) * gc + a * (di + gc - gp)

                # Clip
                d_norm = self._gnorm(self.d)
                g_norm = self._gnorm(g_current)
                if d_norm > 3.0 * g_norm and g_norm > 1e-8:
                    scale = 3.0 * g_norm / d_norm
                    for di in self.d:
                        di.mul_(scale)

                eta = self._step_lr(self.d)
                for p, di in zip(self.params, self.d):
                    p.data.add_(di, alpha=-eta)

            self.prev_params = self._save_params()

            with torch.no_grad():
                loss = compute_loss_on_batch(
                    self.model, Xb, yb, self.loss_type, self.l2_reg)
                total_loss += loss.item()
            n_batches += 1

        return {"avg_loss": total_loss / max(n_batches, 1),
                "full_grad_count": full_grad_count}
```
