A great many learning problems reduce to minimizing a finite average of per-example losses, $P(w) = \frac{1}{n}\sum_{i=1}^{n}\psi_i(w)$, where each $\psi_i$ is convex and $L$-smooth and $P$ is $\gamma$-strongly convex with condition number $\kappa = L/\gamma$. The two textbook ways to descend this objective sit at opposite ends of a brutal trade-off. Full gradient descent steps along $\nabla P(w) = \frac{1}{n}\sum_i \nabla\psi_i(w)$ and, with a constant step $\eta < 1/L$, contracts the suboptimality by a fixed factor each iteration — a linear rate $O((1-\gamma/L)^t)$, reaching accuracy $\varepsilon$ in about $\kappa\ln(1/\varepsilon)$ iterations — but every iteration touches all $n$ examples, so the real bill is $n\kappa\ln(1/\varepsilon)$ gradient evaluations, painful for large $n$. Stochastic gradient descent draws one example $i_t$ uniformly and steps $w \leftarrow w - \eta_t\nabla\psi_{i_t}(w)$ at a cost of one gradient per step, independent of $n$, and is honestly unbiased since $\mathbb{E}_i[\nabla\psi_i(w)] = \nabla P(w)$; but its rate is only sublinear $O(1/t)$, and the step must be driven to zero or it does not converge at all.

I want the mechanism behind that wall, not the folklore. The cleanest place to look is the optimum itself. At $w^*$ we have $\nabla P(w^*) = 0$, but the individual $\nabla\psi_i(w^*)$ are in general *not* zero — only their average is. Standing exactly at $w^*$ and running an SGD step, I draw some $i$, $\nabla\psi_i(w^*) \neq 0$, and walk *away* from the answer. Quantify the leftover by $\sigma^2 = \frac{1}{n}\sum_i \|\nabla\psi_i(w^*)\|^2 > 0$. Expanding the squared distance, $\mathbb{E}\|w_t - w^*\|^2 \lesssim (1 - 2\eta\gamma + \dots)\|w_{t-1} - w^*\|^2 + \eta^2\sigma^2$, whose fixed point is not zero but a floor of order $\eta\sigma^2/\gamma$: the iterate contracts into a ball of that radius and then rattles around inside it. Shrinking the ball means shrinking $\eta$, but a shrinking $\eta$ also weakens the contraction, and the balance lands at $\eta_t = O(1/t)$ and the rate at $O(1/t)$. The villain is named — it is the variance $\sigma^2$, not any bias. Kill the variance and a constant, large step survives. The $O(1/t)$ rate is provably optimal, but only for an oracle restricted to *unbiased noisy gradient measurements*, a fresh anonymous sample each time; my functions are a fixed finite list that recur, so I can occasionally look at all of them, and that loophole is exactly what SAG and SDCA already exploit to cross into a linear rate. The trouble is *how* they pay for it: SAG keeps a table of one stored gradient $y_i$ per example and steps along the running average $w \leftarrow w - \frac{\eta}{n}\sum_i y_i$, refreshing only $y_{i_t}$; SDCA optimizes the regularized dual one coordinate at a time and stores the $n$ dual variables. Both reach a linear rate, and both carry $O(n)$-sized per-example state — a gradient table (compressible to one scalar each only for linear models $\psi_i(w) = \phi_i(w^\top x_i)$) or $n$ duals. On a structured-prediction loss or a neural network there is no such table I can afford, and SDCA's dual machinery does not even make sense there. SAG's convergence proof is also a tangle — a joint Lyapunov function over gradients and iterates — which leaves no clean picture of *why* the rate is linear.

I propose SVRG — Stochastic Variance Reduced Gradient. Reading SAG's averaged-table step in one sentence, it replaces the high-variance single-sample direction with something whose variance shrinks as the iterate settles. That is variance reduction, and there is a completely standard tool for it from Monte Carlo: to estimate $\mathbb{E}[X]$ when I have a correlated $Y$ whose mean $\mathbb{E}[Y]$ I know exactly, form $X - Y + \mathbb{E}[Y]$. Its mean is $\mathbb{E}[X]$ for *any* such $Y$, and its variance is $\mathrm{Var}(X) + \mathrm{Var}(Y) - 2\,\mathrm{Cov}(X,Y)$, which is much smaller than $\mathrm{Var}(X)$ when $X$ and $Y$ are strongly correlated. Take $X = \nabla\psi_i(w)$, the SGD direction I sample, and for $Y$ the most natural correlated companion: the gradient of the *same* example $i$ at a fixed reference point $\tilde w$, $Y = \nabla\psi_i(\tilde w)$. By smoothness $\nabla\psi_i(w) - \nabla\psi_i(\tilde w) \approx \nabla^2\psi_i\cdot(w - \tilde w)$, so when $w$ is near $\tilde w$ the two move together example-by-example and the idiosyncrasy of "example $i$ has a big gradient" cancels in $X - Y$. And its mean is computable: $\mathbb{E}_i[\nabla\psi_i(\tilde w)] = \frac{1}{n}\sum_i \nabla\psi_i(\tilde w) = \nabla P(\tilde w) =: \tilde\mu$, exact, at the cost of one full pass at $\tilde w$. The update direction is

$$v = \nabla\psi_i(w) - \nabla\psi_i(\tilde w) + \tilde\mu.$$

Two checks settle everything. First, it stays an honest descent direction: $\mathbb{E}_i[v] = \nabla P(w) - \nabla P(\tilde w) + \tilde\mu = \nabla P(w)$, unbiased for *any* reference $\tilde w$, so I can drop $v$ straight into $w \leftarrow w - \eta v$ and in expectation I am still doing gradient descent on $P$ — whatever reference I pick, I never optimize the wrong thing. Second, and this is the whole point, the variance vanishes where SGD's did not. Suppose $w \to w^*$ and the reference $\tilde w \to w^*$ as well; then $\nabla\psi_i(w) - \nabla\psi_i(\tilde w) \to \nabla\psi_i(w^*) - \nabla\psi_i(w^*) = 0$ example-by-example, and $\tilde\mu = \nabla P(\tilde w) \to \nabla P(w^*) = 0$, so $v \to 0$. The very offset $\nabla\psi_i(w^*)$ that floored SGD — the part specific to example $i$ that never averages away — now appears in *both* $\nabla\psi_i(w)$ and $\nabla\psi_i(\tilde w)$ at the same $i$ and cancels against itself, leaving only the part that genuinely depends on how far $w$ is from the optimum. As $v \to 0$ the step $\eta v \to 0$ on its own, with no need to shrink $\eta$. There is a second reading that confirms nothing ad hoc was smuggled in: with $\tilde\psi_i(w) = \psi_i(w) - (\nabla\psi_i(\tilde w) - \tilde\mu)^\top w$ one has $\nabla\tilde\psi_i(w) = v$ exactly, and since the slopes average to zero, $\frac{1}{n}\sum_i(\nabla\psi_i(\tilde w) - \tilde\mu) = \tilde\mu - \tilde\mu = 0$, the objective is unchanged, $\frac{1}{n}\sum_i\tilde\psi_i = P$. So $v$ is plain SGD on a re-centered representation of the very same $P$, re-centered so each example's reference gradient sits at the common mean.

The one expensive piece is the exact mean $\tilde\mu = \nabla P(\tilde w)$, a full pass; recomputing it every step would just rebuild full gradient descent. But $\tilde w$ cannot be frozen forever either, because the variance reduction is only good while $w$ stays near $\tilde w$. So I amortize, in epochs: at the start of an epoch snapshot the current parameters as $\tilde w$, pay once for $\tilde\mu$ in one full pass, set $w_0 = \tilde w$, then run $m$ cheap inner steps against the held $(\tilde w, \tilde\mu)$,

$$w_t = w_{t-1} - \eta\big(\nabla\psi_{i_t}(w_{t-1}) - \nabla\psi_{i_t}(\tilde w) + \tilde\mu\big),$$

and refresh $\tilde w$ at the end. As the whole thing converges each epoch's $\tilde w$ creeps toward $w^*$, $\tilde\mu$ shrinks, the inner variance shrinks with it, and the step is constant throughout. The memory is the punchline: only $\tilde w$ and $\tilde\mu$ are stored, both $O(d)$ — no per-example table, no duals — which is precisely the bill SAG and SDCA refused to keep small, and it is small because the reference is a single shared snapshot rather than $n$ separately stored gradients. That immediately frees SVRG from the linear-model special case and ports it to structured losses and neural nets. Counting on the fair axis of gradient-evaluations-per-$n$, an epoch costs $n$ for $\tilde\mu$ plus $2m$ for the two example-gradients per inner step (one at $w_{t-1}$, one at $\tilde w$), so I want $m = O(n)$ — large enough to amortize the snapshot pass, not so large that $w$ drifts hopelessly far from $\tilde w$ — say $m = 2n$ for convex problems and $m = 5n$ for the rougher nonconvex landscape. For refreshing $\tilde w$, the last inner iterate $w_m$ is the natural practical choice (option I).

The linear rate is a short, mechanical theorem, and its spine is turning "variance" into "function-value gap." The lemma: fix $i$, set $g_i(w) = \psi_i(w) - \psi_i(w^*) - \nabla\psi_i(w^*)^\top(w - w^*)$, so $\nabla g_i(w^*) = 0$ and $w^*$ minimizes the convex $L$-smooth $g_i$ with $g_i(w^*) = 0$. Minimizing the smoothness upper bound of one gradient step, $0 = g_i(w^*) \le \min_{\eta'}[g_i(w) - \eta'\|\nabla g_i(w)\|^2 + \frac{1}{2}L\eta'^2\|\nabla g_i(w)\|^2] = g_i(w) - \frac{1}{2L}\|\nabla g_i(w)\|^2$, giving $\|\nabla\psi_i(w) - \nabla\psi_i(w^*)\|^2 \le 2L[\psi_i(w) - \psi_i(w^*) - \nabla\psi_i(w^*)^\top(w - w^*)]$. Averaging over $i$, the linear terms vanish because $\nabla P(w^*) = 0$, leaving

$$\frac{1}{n}\sum_i \|\nabla\psi_i(w) - \nabla\psi_i(w^*)\|^2 \le 2L\,[P(w) - P(w^*)]. \tag{8}$$

Now bound the second moment of $v_t = \nabla\psi_{i_t}(w_{t-1}) - \nabla\psi_{i_t}(\tilde w) + \tilde\mu$. Insert $\nabla\psi_{i_t}(w^*)$ and split $v_t = a + b$ with $a = \nabla\psi_{i_t}(w_{t-1}) - \nabla\psi_{i_t}(w^*)$ and $b = \nabla\psi_{i_t}(w^*) - \nabla\psi_{i_t}(\tilde w) + \tilde\mu$. Using $\|a+b\|^2 \le 2\|a\|^2 + 2\|b\|^2$, then recognizing $b = -(\xi - \mathbb{E}\xi)$ for $\xi = \nabla\psi_{i_t}(\tilde w) - \nabla\psi_{i_t}(w^*)$ whose mean is $\nabla P(\tilde w) = \tilde\mu$, so $\mathbb{E}\|b\|^2 = \mathbb{E}\|\xi\|^2 - \|\mathbb{E}\xi\|^2 \le \mathbb{E}\|\xi\|^2$ — this is exactly where subtracting the known mean $\tilde\mu$ pays off in the algebra — and applying (8) at $w_{t-1}$ and at $\tilde w$,

$$\mathbb{E}\|v_t\|^2 \le 4L\,[P(w_{t-1}) - P(w^*)] + 4L\,[P(\tilde w) - P(w^*)]. \tag{$\star$}$$

This is the variance reduction made rigorous: as $w_{t-1}, \tilde w \to w^*$ the right side $\to 0$, with no separate variance bound needed. The per-step contraction follows from $\mathbb{E}_{i_t}[v_t] = \nabla P(w_{t-1})$ and convexity, $-(w_{t-1} - w^*)^\top\nabla P(w_{t-1}) \le P(w^*) - P(w_{t-1})$, plus $(\star)$:

$$\mathbb{E}\|w_t - w^*\|^2 \le \|w_{t-1} - w^*\|^2 - 2\eta(1 - 2L\eta)[P(w_{t-1}) - P(w^*)] + 4L\eta^2[P(\tilde w) - P(w^*)],$$

whose middle coefficient is a genuine decrease iff $\eta < 1/(2L)$ — a constant ceiling that does not shrink with $t$. Fix $\tilde w = \tilde w_{s-1}$, $w_0 = \tilde w$, sum over $t = 1,\dots,m$ so the $\|\cdot - w^*\|^2$ terms telescope, take full expectation, drop $\mathbb{E}\|w_m - w^*\|^2 \ge 0$, and use option II — $\tilde w_s$ a uniformly random inner iterate, which makes $\frac{1}{m}\sum_t \mathbb{E}[P(w_{t-1}) - P(w^*)] = \mathbb{E}[P(\tilde w_s) - P(w^*)]$ with no Jensen gap, the reason the analysis prefers it even though I run last-iterate — together with $\|\tilde w - w^*\|^2 \le \frac{2}{\gamma}[P(\tilde w) - P(w^*)]$ from strong convexity. This gives $2\eta(1 - 2L\eta)m\,\mathbb{E}[P(\tilde w_s) - P(w^*)] \le (\frac{2}{\gamma} + 4Lm\eta^2)\,\mathbb{E}[P(\tilde w) - P(w^*)]$, and dividing through,

$$\mathbb{E}[P(\tilde w_s) - P(w^*)] \le \alpha\,\mathbb{E}[P(\tilde w_{s-1}) - P(w^*)], \qquad \alpha = \frac{1}{\gamma\eta(1 - 2L\eta)m} + \frac{2L\eta}{1 - 2L\eta},$$

so $\mathbb{E}[P(\tilde w_s) - P(w^*)] \le \alpha^s[P(\tilde w_0) - P(w^*)]$. The first term of $\alpha$ is $O(1/m)$ and the second is small for $\eta$ a constant fraction of $1/L$, so $\alpha < 1$ is attainable by picking $\eta$ a constant fraction of $1/L$ and $m = O(\kappa)$. At the telling case $\kappa = L/\gamma = n$, taking $\eta = 0.1/L$ makes $2L\eta = 0.2$ and the second term $0.25$, and $m = 50n$ drives the first term to $0.25$, so $\alpha = 0.5$: reaching $\varepsilon$ takes $O(n\ln(1/\varepsilon))$ gradient evaluations, a factor of $n$ past batch GD's $n^2\ln(1/\varepsilon)$, matching SAG and SDCA but with $O(d)$ memory. For smooth convex $P$ without strong convexity the same machinery yields $O(1/T)$; for a nonconvex model the estimator stays unbiased and the matched-index cancellation still drives $v$ toward zero, so warm-starting with a few SGD steps and then running SVRG near a locally strongly convex basin gives local geometric convergence with a constant step. And the vanishing-variance condition unifies the family: SAG's stored per-example gradients and SDCA's stored duals — where $\nabla\phi_i(w) + \lambda n\,\alpha_i \to 0$ as $(w,\alpha) \to (w^*,\alpha^*)$ — are both per-example references that cancel the offending offset as things converge; the three methods are one mechanism, with SVRG the memory-light, model-agnostic member that keeps the reference as a single shared snapshot plus its mean.

```python
import torch


class FiniteSumOptimizer:
    """Stochastic Variance Reduced Gradient for P(w) = (1/n) Σ_i ψ_i(w).

    Each epoch: snapshot w̃, pay one full pass for μ̃ = ∇P(w̃), then run m cheap inner steps
    along the control-variate direction
        v = ∇ψ_i(w) − ∇ψ_i(w̃) + μ̃     (E_i[v] = ∇P(w); variance → 0 as w, w̃ → w*),
    so the step size stays CONSTANT and the rate is linear. Only w̃ and μ̃ are stored (O(d)).
    """

    def __init__(self, problem, lr, inner_steps):
        self.problem = problem            # grad_batch(w_state, idx), full_grad(), loss_batch(idx)
        self.params = problem.params      # the live parameters w
        self.lr = lr                      # constant η < 1/(2L)
        self.inner_steps = inner_steps    # m random inner updates per epoch
        self.snapshot = None              # w̃
        self.mu = None                    # μ̃ = ∇P(w̃)

    def _clone_params(self):
        return [p.data.clone() for p in self.params]

    def train_one_epoch(self):
        # snapshot: hold w̃ and compute the exact mean μ̃ = ∇P(w̃) (one full pass)
        self.snapshot = self._clone_params()
        self.mu = self.problem.full_grad()

        n, b = self.problem.n, self.problem.batch_size
        total_loss, n_batches = 0.0, 0

        for _ in range(self.inner_steps):
            idx = torch.randint(n, (b,))

            grad_cur = self.problem.grad_batch(self.params, idx)     # ∇ψ_i(w_{t-1})
            grad_snap = self.problem.grad_batch(self.snapshot, idx)  # ∇ψ_i(w̃), same indices

            # v = ∇ψ_i(w) − ∇ψ_i(w̃) + μ̃   (unbiased for ∇P(w))
            with torch.no_grad():
                for p, gc, gs, mu in zip(self.params, grad_cur, grad_snap, self.mu):
                    p.data.add_(gc - gs + mu, alpha=-self.lr)        # constant-step in-place update

            total_loss += float(self.problem.loss_batch(idx))
            n_batches += 1

        # refresh reference from the last inner iterate (option I)
        return {"avg_loss": total_loss / max(n_batches, 1), "full_grad_count": 1}
```
