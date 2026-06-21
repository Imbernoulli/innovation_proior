The baseline I am replacing is vanilla mini-batch SGD on a finite sum $F(x) = \frac{1}{n}\sum_i f_i(x)$, and its one binding flaw is sharp enough to see at the optimum itself. SGD draws a batch and follows $g = \frac{1}{b}\sum_{i\in B}\nabla f_i(x)$, which is an unbiased estimate of $\nabla F(x)$, so it descends honestly; the trouble is the variance. At a minimizer $x^*$ we have $\nabla F(x^*) = 0$ by definition, yet the individual $\nabla f_i(x^*)$ are in general nonzero — only their average vanishes. So standing exactly at $x^*$ and drawing a batch yields a nonzero gradient and a step *away* from the solution. Writing $\sigma^2 = \frac{1}{n}\sum_i \|\nabla f_i(x^*)\|^2 > 0$, the squared-distance recursion $\mathbb{E}\|x_t - x^*\|^2 = \|x_{t-1}-x^*\|^2 - 2\eta(x_{t-1}-x^*)\cdot\nabla F(x_{t-1}) + \eta^2\,\mathbb{E}\|g\|^2$ keeps a last term that floors at $\approx \eta^2\sigma^2/b$ and does not vanish as $x_{t-1}\to x^*$. The fixed point is therefore not the optimum but a noise ball of radius $O(\eta\sigma^2/(b\gamma))$, and the only way to shrink it is to anneal $\eta$ to $O(1/t)$, which collapses the rate to $O(1/t)$. The villain is the variance $\sigma^2$, not any bias, and the whole game is to manufacture, out of mini-batch gradients, a direction whose variance *shrinks* as the iterate settles, so a constant step survives.

I propose **SVRG**, the snapshot control-variate method. The mechanism that SAG and SDCA exploited, stated plainly, is the Monte-Carlo control variate: to estimate $\mathbb{E}[X]$ when I have a correlated $Y$ whose mean $\mathbb{E}[Y]$ I know exactly, I form $X - Y + \mathbb{E}[Y]$, which is unbiased for any such $Y$ and has variance $\mathrm{Var}(X) + \mathrm{Var}(Y) - 2\mathrm{Cov}(X,Y)$, much smaller than $\mathrm{Var}(X)$ when $X$ and $Y$ are strongly correlated. I map this onto gradients. Let $X = \nabla f_i(x)$ be the noisy thing I sample at the current point. The natural companion, indexed by the *same* $i$ so the per-example idiosyncrasy is shared, is the gradient of the same example at a fixed reference $\tilde{x}$: $Y = \nabla f_i(\tilde{x})$. They are correlated because, by smoothness, $\nabla f_i(x) - \nabla f_i(\tilde{x}) \approx \nabla^2 f_i\cdot(x-\tilde{x})$, so the "example $i$ happens to have a big gradient" part cancels in the difference. And I know its mean exactly: $\mathbb{E}_i[Y] = \frac{1}{n}\sum_i \nabla f_i(\tilde{x}) = \nabla F(\tilde{x})$, computable with one full pass — call it $\tilde\mu$. The variance-reduced direction is

$$v = \nabla f_i(x) - \nabla f_i(\tilde{x}) + \tilde\mu.$$

Two properties make it the right object. It is unbiased: $\mathbb{E}_i[v] = \nabla F(x) - \nabla F(\tilde{x}) + \tilde\mu = \nabla F(x)$ for *any* reference $\tilde{x}$, so I can drop $v$ straight into the SGD update and in expectation I am still doing gradient descent. And its variance vanishes at the solution: as both $x\to x^*$ and $\tilde{x}\to x^*$, the matched-index difference $\nabla f_i(x) - \nabla f_i(\tilde{x}) \to 0$ example by example while $\tilde\mu = \nabla F(\tilde{x})\to 0$, so $v\to 0$. The very offset $\nabla f_i(x^*)$ that floored SGD now appears in both terms at the same $i$ and cancels against itself; as $v\to 0$ the step $\eta v\to 0$ on its own, with no annealing of $\eta$. The stored state is just $\tilde{x}$ and $\tilde\mu$, both the size of the parameters — $O(d)$ memory, no $n$-sized per-example table — which is what SAG/SDCA refused to keep small and what frees this method from any linear-model special case.

The engineering question is what $\tilde{x}$ is and how often $\tilde\mu$ is refreshed, and the harness answers it for me. The exact mean is one full pass, which I cannot afford every step (that would rebuild full GD); but the reference cannot be frozen forever either, because the variance reduction holds only while $x$ stays near $\tilde{x}$. The resolution is to amortize in epochs, and the harness *forces* the cadence: `compute_full_gradient` may be called at most once per epoch. So I snapshot the current parameters as $\tilde{x}$ at the start of each epoch, pay once for $\tilde\mu = \nabla F(\tilde{x})$, then run one full pass of $n/b$ cheap inner steps $x \leftarrow x - \eta\,(\nabla f_i(x) - \nabla f_i(\tilde{x}) + \tilde\mu)$ against that held $(\tilde{x}, \tilde\mu)$. The epoch *is* the inner loop: there is no freely chosen $m = 2n$ or $5n$ here, only $m = n/b$. A second harness wrinkle: there is no cached-scalar path for $\nabla f_i(\tilde{x})$, because `compute_stochastic_gradient` always evaluates at the model's *current* parameters. To get the snapshot gradient for the inner batch I must physically load the snapshot parameters into the model, evaluate, and restore — three gradient touches per batch (current point, snapshot point, plus the once-per-epoch snapshot pass), roughly twice an SGD epoch's cost. That is the honest price of having no per-example cache.

The remaining design choice is the step size, and it is the load-bearing risk on the conditioned problem. The appeal of the control variate is a constant $\eta$, and for the well-scaled cross-entropy problems (logistic, MLP) I use the handed-in `self.lr` directly — the SVRG direction is unbiased and a constant step is the point. But the conditioned problem is ill-conditioned linear regression with $\kappa = 100$, and the *first* inner step of every epoch is dangerous: at the snapshot the SVRG direction for the first batch is $\nabla f_i(\tilde{x}) - \nabla f_i(\tilde{x}) + \tilde\mu = \tilde\mu = \nabla F(\tilde{x})$, the *full* gradient at full magnitude. On a stiff quadratic with large $\|\nabla F(\tilde{x})\|$, a fixed-$\eta$ step along $\tilde\mu$ can overshoot the high-curvature direction; iterate that across epochs and the strongly convex MSE grows geometrically and diverges. So for the `mse` loss only I cap the step by the inverse full-gradient norm, $\eta_{\text{eff}} = \min(\texttt{self.lr},\ 1/\|\tilde\mu\|)$, the standard "do not step longer than $1/\|g\|$ in normalized terms" guard. I gate it on the loss type precisely so the convex/non-convex classification problems run clean constant-step SVRG and only the ill-conditioned regression gets the cap. I am clear-eyed that this bounds *one* step per epoch, not the compounding across epochs — a single full-gradient anchor riding stale across a $\sim$600-step inner sweep on a stiff quadratic is the most fragile thing on this ladder, and if it fails, the diagnosis already points at the next move: replace the stale once-per-epoch snapshot with a reference that re-anchors continually as the iterate moves.

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
