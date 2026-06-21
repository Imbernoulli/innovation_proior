STORM did the one thing I most needed: it killed SVRG's divergence. Where SVRG returned `nan` and $8.5\times 10^{10}$ on the conditioned MSE, STORM came back finite on every seed — best mean 1.76, final mean 2.96 — confirming that a continually re-anchored running estimate is the right structural fix; it matched SVRG on logistic (92.50/92.36) and tightened the MLP best-to-final gap (53.91/52.49 versus SVRG's 52.67/50.09). But the conditioned numbers also carry the exact residual I had flagged. The per-seed best MSE is 3.51 on seed 42 but 0.85 and 0.91 on seeds 123 and 456 — a 4× spread — and the final MSE is *worse than the best* on two of three seeds (3.91 vs 3.51, 4.06 vs 0.85), equal only on the seed that happened to settle. That signature — final consistently above best, with wide seed-to-seed variance — is not divergence and not a noise floor; it is the iterate *overshooting and oscillating* near the optimum on the stiff $\kappa = 100$ quadratic, because the fixed $\eta = \texttt{self.lr}$ and fixed $a = 1 - 1/\sqrt{T}$ are set once and never respond to what the gradients are doing. The same step that is fine along low-curvature directions is too aggressive along the high-curvature one. The fix is to make the step and the momentum *adapt, per step*, to the observed iterate and gradient magnitudes, and to add a brake on the rare step where the recursive estimate runs away from the raw gradient.

I propose **STORM+**, an adaptive-enhanced recursive momentum. I keep the STORM recursion verbatim — $d_t = (1-a_t)\,g_t + a_t\,(d_{t-1} + g_t - g'_{t-1})$, the re-anchored estimate that fixed the divergence — and attack the three knobs the conditioned spread exposed, each as a direct response to a measured failure.

First the momentum. A fixed $a$ sets the contraction $(1-a)$ to one value for the whole run, but the error recursion says $a$ should be *larger early* (the iterate moves fast and the estimate is freshly seeded, so fold in more fresh gradient and trust the carry less) and *smaller late* (consecutive iterates are close, the correction is tiny, so let the contraction $(1-a)$ be small and average accumulated error away hard). A *decaying* momentum on a **global** step counter $t$ does this:

$$a_t = \min\!\Big(1 - \frac{1}{\sqrt{t+1}},\ 0.999\Big).$$

At $t=0$, $a_0 = 0$, so the first inner step is pure fresh gradient — exactly right when there is no reliable running estimate yet; as $t$ grows, $a_t \to 1$, the contraction $(1-a_t) = 1/\sqrt{t+1} \to 0$, and the estimate leans on the smoothly averaged history while the correction $(1-a_t)(g_t - g'_{t-1})$ shrinks on both factors. The cap $0.999$ stops $(1-a_t)$ collapsing to exactly zero, which would freeze the estimate and stop folding in new information on a non-stationary problem. The counter is *global*, not per-epoch, because the right decay is "how long have I been optimizing," not "where am I in this epoch" — resetting it every epoch would re-inject the early high-$(1-a)$ regime over and over and reproduce STORM's late oscillation. This is the harness-native analogue of decaying the momentum like $t^{-2/3}$: I cannot use the cube-root accumulator, but $1 - 1/\sqrt{t+1}$ is the same shape from a single counter.

Second, the step size — the lever that directly addresses the overshoot. STORM's fixed `self.lr` was simultaneously too large along the stiff direction (overshoot, the final-above-best signature) and too small along the flat directions (slow progress, seed-42's high best-MSE); a fixed scalar cannot win that trade. What I can compute every step, with no knowledge of the smoothness constant, is the ratio of the parameter scale to the update scale. The classic trust-region-in-normalized-terms step is $\eta \propto \|w\|/\|d\|$ — a step whose length is a fixed *fraction* of the current parameter norm regardless of the estimate magnitude:

$$\eta_t = \min\!\Big(\texttt{self.lr},\ 0.01 \cdot \frac{\|w_t\|}{\|d_t\| + \varepsilon}\Big).$$

When $\|d_t\|$ is small (near convergence, or a flat direction), the second arm is large, the $\min$ falls back to `self.lr`, and I take the full handed-in step — no artificial slowdown. When $\|d_t\|$ is large (a stiff direction, or a momentary spike in the estimate), the $\min$ selects $0.01\|w\|/\|d\|$ and clamps the step to 1% of the parameter norm — the overshoot guard the final-above-best numbers were screaming for. The $0.01$ is the trust radius: moving the parameters by at most $\sim$1% per update is stable on $\kappa = 100$ yet not so small that 30 epochs cannot make progress. This is per-step and per-problem automatically — on logistic and the MLP the $\min$ almost always picks `self.lr` so the method runs as plain STORM with decaying momentum, while on the conditioned problem the clamp engages. I apply the same adaptive step to the first-epoch full-gradient warm-start too, so even the seed step — the place SVRG's full-magnitude step first went wrong — is overshoot-guarded.

Third, a brake on the estimate itself. The recursion can, on a rare step, produce a $d_t$ whose magnitude far exceeds the raw gradient $g_t$ if the carried estimate and the correction add constructively — the kind of transient that, even with the adaptive step, can knock the iterate off the stiff quadratic and show up as seed-42's high best-MSE. So I cap the *direction*, not just the length: if $\|d_t\| > 3\|g_t\|$, rescale $d_t$ down to $3\|g_t\|$. The factor 3 is deliberately loose — the estimate *should* differ from the raw gradient (that difference is the variance reduction; clamping to $\sim\|g\|$ would throw the method back to SGD) — so a 3× cap forbids only the pathological runaway. The guard $\|g_t\| > \varepsilon$ avoids rescaling when the raw gradient is near zero and the ratio is meaningless.

Together: the same recursive-momentum estimate, but a per-step decaying momentum $a_t = \min(1 - 1/\sqrt{t+1},\ 0.999)$ from a global counter, a per-step trust-region step $\eta_t = \min(\texttt{self.lr},\ 0.01\|w\|/\|d\|)$, and a $\|d\| > 3\|g\|$ clip. The seed and gradient mechanics are STORM's unchanged — one full gradient on the first epoch only, two stochastic-gradient calls per inner step via load-previous/restore — so the gradient cost matches STORM and stays well below SVRG. Each addition answers a specific measured failure: the decaying momentum to the late-epoch staleness behind the MLP gap, the adaptive step to the conditioned overshoot, and the clip to the per-seed spike. I expect the conditioned MSE to fall one to two orders of magnitude with a tight spread and final ≈ best, while logistic stays a wash and the MLP edges up slightly on both metrics.

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
