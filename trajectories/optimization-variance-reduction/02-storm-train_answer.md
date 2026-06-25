SVRG vindicated the variance-reduction idea on the benign problems — logistic landed 92.63/92.62 best/final, indistinguishable from a competent SGD, and the MLP came in mid-50s (52.67 best, 50.09 final) — but it failed exactly where the floor analysis said it would matter, and it failed catastrophically. On the stiff $\kappa = 100$ conditioned problem the best MSE averaged 1617 and the final MSE was $8.5\times 10^{10}$ on one seed with literal `nan` on the other two. That is not a stalled noise floor, it is divergence. The cause is precise: SVRG anchors a single full-gradient snapshot $\tilde\mu = \nabla F(\tilde{x})$ at the start of each epoch and rides it for the whole $\sim 78$-step inner sweep, but on a stiff quadratic the iterate drifts away from $\tilde{x}$ within a few steps, the correlation $\nabla f_i(x) \approx \nabla f_i(\tilde{x})$ that the control variate relies on collapses, and the "variance-reduced" direction becomes $\nabla f_i(x) - \nabla f_i(\tilde{x}) + \tilde\mu$ with a *stale* $\tilde\mu$ pointing where the snapshot used to be good. The $\min(\texttt{lr}, 1/\|\tilde\mu\|)$ cap bounded one step but not the compounding across epochs. The lesson is unambiguous: the anchor must not be allowed to go stale; I need a reference that *moves with the iterate*, re-anchored continually.

I propose **STORM**, stochastic recursive momentum. The variance-reduction lever is always the same — by smoothness, $\|\nabla f_i(x) - \nabla f_i(x')\| \le L\|x - x'\|$, and that difference is small *in variance* (the per-sample idiosyncrasy of example $i$ is shared and cancels) precisely when the two points are close. SVRG's two points were "current iterate" and "epoch-start snapshot," and the snapshot drifted far. The obvious repair is to move the anchor to the *previous iterate*, so the gap is exactly one step. That is the SARAH recursion $v_t = \nabla f_i(x_t) - \nabla f_i(x_{t-1}) + v_{t-1}$, which never goes stale — but SARAH still reseeds each inner loop with a fresh full gradient, which on a once-per-epoch budget *is* the stale-anchor structure I am escaping. So I want SARAH's continual re-anchoring without leaning on a per-epoch seed. The key observation is that SARAH's recursion, rewritten as $v_t = \nabla f_i(x_t) + (v_{t-1} - \nabla f_i(x_{t-1}))$, has the same shape as momentum $d_t = (1-a)d_{t-1} + a\nabla f_i(x_t)$ — both carry a running estimate forward and fold in the new gradient. I run both at once: graft a $(1-a)$-weighted copy of the same-sample two-point difference onto the momentum recursion. With $g_t = \nabla f_i(x_t)$ and $g'_{t-1} = \nabla f_i(x_{t-1})$,

$$d_t = (1-a)\,g_t + a\,(d_{t-1} + g_t - g'_{t-1}), \qquad x_{t+1} = x_t - \eta\,d_t,$$

which collects to $d_t = \nabla f_i(x_t) + (1-a)(d_{t-1} - \nabla f_i(x_{t-1}))$ — exactly SARAH's recursion with a $(1-a)$ weight on the carry-forward instead of a hard 1. SARAH is the $a=0$ corner; plain momentum is the corner where the correction is dropped; the interior, $a$ small but nonzero, is the new object. It is safe by construction: as the run converges, $x_t \approx x_{t-1}$, so $g_t - g'_{t-1} \to 0$ and $d_t$ collapses to plain momentum — the correction matters most early, when steps are large, which is exactly where SVRG's stale anchor killed the conditioned problem.

That it genuinely reduces variance falls out of tracking the direction error $\varepsilon_t := d_t - \nabla F(x_t)$. Subtracting $\nabla F(x_t)$ and adding and subtracting $(1-a)\nabla F(x_{t-1})$,

$$\varepsilon_t = (1-a)\,\varepsilon_{t-1} + a\,(\nabla f_i(x_t) - \nabla F(x_t)) + (1-a)\big[(\nabla f_i(x_t) - \nabla f_i(x_{t-1})) - (\nabla F(x_t) - \nabla F(x_{t-1}))\big].$$

The first term *contracts* the previous error by $(1-a)$ — the averaging SVRG's per-epoch reset could never do mid-sweep. The second is fresh single-sample noise of size $\sim\sigma$, but premultiplied by the small $a$, so I squash it by shrinking $a$. The third is the same-sample two-point difference minus its mean, which by smoothness is $O(\|x_t - x_{t-1}\|) = O(\eta\|d_{t-1}\|)$ — small whenever the step is small, and crucially using the *one-step* gap, never a stale epoch-wide gap. So heuristically $\|\varepsilon_t\| \sim (1-a)\|\varepsilon_{t-1}\| + Z$ with $Z$ small, a contraction-plus-small-input recursion that settles at $\|\varepsilon\| \sim Z/a$. The error stabilizes at a small equilibrium *continuously*, never resetting to a full-gradient seed and never riding a stale anchor.

Landing this in the harness is what makes my STORM differ from the fully adaptive version, and the differences are forced. The fully adaptive STORM sets $\eta_t = k/(w + \sum\|g_i\|^2)^{1/3}$ and ties $a_{t+1} = c\,\eta_t^2$ with the cube-root exponent from the optimal nonconvex rate; I cannot, because the learning rate here is handed to me and fixed, and inventing a per-coordinate AdaGrad accumulator that must behave across MNIST logistic, a CIFAR MLP, and a $\kappa = 100$ quadratic with one code path is the wrong altitude. So $\eta = \texttt{self.lr}$, full stop. With $\eta$ fixed there is nothing adaptive to tie $a$ to, but the equilibrium $\|\varepsilon\| \sim Z/a$ still demands a value balancing contraction against fresh noise. I set it once from the only structural quantity I have, the inner-step count $T = n // b$:

$$a = 1 - \frac{1}{\sqrt{T}},$$

so the contraction $(1-a) = 1/\sqrt{T}$ is strong enough to average accumulated error down hard over the epoch, while the fresh-gradient weight $a \approx 1$ keeps the estimate tracking $\nabla F$ as the iterate moves and the variance reduction is carried by the $(1-a)$-weighted correction term. One scalar, same formula on all three problems, no per-problem-tuned accumulator. For the seed, I am *allowed* one full gradient per epoch, and the first epoch is the one place a low-noise seed costs nothing in steady state: I warm-start by setting $d = \nabla F(x_1)$, take the first step along it, and only then begin the recursion; every later epoch carries the running $d$ across and needs no full gradient. This inverts SVRG's structure — SVRG paid a full gradient *every* epoch and rode it stale; STORM pays one full gradient *once*, on epoch one, to seed a running estimate that re-anchors itself every step thereafter, so the total gradient count falls well below SVRG's. The two-point mechanics reuse SVRG's load/restore: save the current parameters and evaluate $g_t$, load the previous iterate's parameters and evaluate $g'_{t-1}$ on the same batch, restore, apply the update, then save the now-current parameters as the next step's "previous" — two stochastic-gradient calls and two swaps per batch, spent on a one-step-fresh correction. I expect this to turn SVRG's `nan`/`8.5e10` into a finite, far smaller conditioned MSE, while leaving logistic and the MLP essentially unchanged; the residual worry I carry forward is seed-to-seed spread, since a fixed $a$ and fixed $\eta$ cannot adapt to the local curvature the iterate currently sits in.

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
        # Momentum coefficient (a = 1 - 1/sqrt(T))
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
