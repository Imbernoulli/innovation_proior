We are handed a tiny number of linear measurements $A(X) = y$ of an unknown matrix $X \in \mathbb{R}^{n\times n}$, with $A(X)_i = \langle A_i, X\rangle$ and $m \ll n^2$, and asked to fill in the rest; matrix completion is the special case where each $A_i$ reads off one entry. The honest starting point is that fitting the data is trivial and useless. The feasible set $\{X : A(X) = y\}$ is a huge affine family — $n^2$ unknowns against $m \ll n^2$ constraints — and minimizing $F(X) = \|A(X) - y\|_2^2$ reaches zero on every member of it. For completion I can set all unobserved entries to zero and already have a perfect zero-loss solution that recovers nothing. So the question is never whether we can fit, but *which* fit the algorithm lands on, and that is a property of the optimizer, not the objective.

The classical answer is to add structure: assume the truth is low rank and ask for the lowest-rank $X$ that fits. Rank minimization is NP-hard, so the field relaxes rank to its convex envelope, the nuclear norm $\|X\|_* = \sum_k \sigma_k(X)$, and solves $\min \|X\|_*$ subject to $A(X) = y$ as an SDP — Recht–Fazel–Parrilo and Candès–Recht give exact recovery from $O(nr)$ measurements, and Srebro–Shraibman identify low nuclear norm as exactly the quantity that controls generalization. So we know *what* we want: a low-nuclear-norm fit. The remaining options each fall short of getting it cheaply. Explicit nuclear-norm minimization works but requires choosing the regularizer by hand and running an SDP rather than simply training a model. Gradient descent directly on $X$ is the cleanest baseline to analyze but gives the wrong bias: $\nabla F(X) = A^*(A(X)-y)$ always lies in the $m$-dimensional span $L = \{A^*(s)\}$, so from $X = 0$ every iterate stays in $L$, and a zero-error limit satisfies the KKT conditions of $\min \|X\|_F^2$ subject to $A(X)=y$ — it is the minimum-Frobenius-norm fit, which for completion is the useless impute-zeros matrix. Low-rank factorized descent with a small inner dimension $d$ works when the rank is known, but bakes the rank preference in as a hard architectural cap and so says nothing about what an expressive model does. What we want instead is for the low-rank preference to emerge from the dynamics, without any explicit penalty or cap.

I propose Shallow Matrix Factorization: parameterize the recovered matrix as the end-to-end product of two full-dimensional, bias-free linear layers, $W = W_2 W_1^\top$, initialize both factors with tiny Gaussian weights so the product starts a hair above zero, and descend the masked squared error on the observed entries with a small step size. There is no nuclear-norm penalty, no rank cap (the hidden dimension is $n$), no explicit regularizer; the entire bias toward the minimum-nuclear-norm fit is supplied by gradient descent on the factorization from near zero. The reason this works is geometric. Writing the symmetric PSD form $X = U U^\top$ with $U \in \mathbb{R}^{n\times n}$, gradient flow on $U$ is $\dot U_t = -A^*(r_t)\,U_t$ with $r_t = A(X_t) - y$, and the chain rule turns this into a flow on $X$ itself,
$$\dot X_t = -A^*(r_t)\,X_t - X_t\,A^*(r_t).$$
The factor $U$ has dropped out — the $X$-dynamics depend only on $X$ and the residual, not on the arbitrary square root — and that is the crux. Descent on $X$ had velocity $-A^*(r_t)$, an element of the *flat* span $L$, so $X$ slid around inside an affine subspace and stopped at the min-Frobenius point. Here the same residual direction $A^*(r_t)$ is multiplied by the current $X$ on both sides, so the velocity is suppressed wherever $X$ is small and the available directions depend on where the flow currently is. That is multiplicative, self-reinforcing growth, and it is what swaps the implicit norm from Frobenius to nuclear.

What makes the claim provable is that this flow stays on a curved manifold. When the measurement matrices commute, $A_i A_j = A_j A_i$, they are simultaneously diagonalizable, $A^*(s)$ commutes with its own time derivative, and the exponential ansatz solves the flow exactly,
$$X_t = \exp\!\big(A^*(s_t)\big)\,X_0\,\exp\!\big(A^*(s_t)\big), \qquad s_t = -\!\int_0^t r_\tau\,d\tau.$$
Initialize at $X_0 = \gamma I$ and set $\beta = -\tfrac12\log\gamma$ (equivalently $\gamma = \alpha^2$, $\beta = -\log\alpha$ for two factors started at $\alpha I$). In the shared eigenbasis $v_1,\dots,v_n$, the eigenvalue of the limit along $v_k$ obeys
$$\lambda_k\big(X_\infty(\gamma I)\big) = \gamma\,\exp\!\big(2\,\lambda_k(A^*(s_\infty))\big) = \exp\!\big(2\,\lambda_k(A^*(s_\infty)) - 2\beta\big).$$
As $\gamma \to 0$, $\beta \to \infty$. For a direction with positive limiting eigenvalue $\lambda_k(\hat X) > 0$, take logs and divide by $2\beta$: with the rescaled dual $\nu(\beta) = s_\infty(\beta)/\beta$, the $\log\lambda_k(\hat X)/(2\beta)$ term vanishes and $\lambda_k(A^*(\nu(\beta))) \to 1$. For a direction with $\lambda_k(\hat X) = 0$, the same expression $\big(\exp(\lambda_k(A^*(\nu)) - 1)\big)^{2\beta} \to 0$ forces the base below $1$, so $\lambda_k(A^*(\nu)) \le 1$ in the limit. Together these give $A^*(\nu) \preceq I$ (dual feasibility) and $(I - A^*(\nu))\hat X = 0$ (complementary slackness) — exactly the KKT conditions of $\min_{X \succeq 0}\|X\|_*$ subject to $A(X) = y$, the other two conditions ($A(X)=y$ and $X \succeq 0$) being given by reaching a zero-error optimum and by the factorization. So the limit is the minimum-nuclear-norm fit. The mechanism is power-iteration-like: $\exp(A^*(s))$ amplifies $X_0$ along the dominant *signed* eigendirections, and starting from a vanishing product forces $|s_\infty| \to \infty$, which is precisely what collapses the amplifier onto the eigenspace where the rescaled dual saturates at eigenvalue $1$.

Each ingredient is forced. Factorizing at all, rather than descending on $X$, is what swaps min-Frobenius (impute-zeros) for min-nuclear (rank-promoting). The full dimension $d = n$ is what removes any explicit rank cap, so the low-rank preference comes entirely from the dynamics — desirable because the true rank is usually unknown and because the point is that over-parameterized models generalize without the constraint. Near-zero initialization is what the $\alpha \to 0$ limit literally requires: reaching a finite-scale $\hat X$ from a vanishing start forces $|s_\infty| \to \infty$ and selects the low-nuclear-norm point, whereas a large start behaves like descent on $X$ and lands at a generic high-nuclear-norm optimum. The small step size is not a tuning nicety but a necessity: the selecting manifold $M = \{\exp(A^*(s))\,X_{\text{init}}\,\exp(A^*(s))\}$ is curved — its tangent space differs at every point — and only infinitesimal steps stay on it, while finite steps or momentum shoot off it and lose the guarantee (the min-Frobenius story tolerated momentum precisely because its manifold was flat). The same robustness argument shows that any control $r_t$ that drives the residual to zero while following $\dot X = -A^*(r)X - X A^*(r)$ lands on a nuclear-norm minimizer, so a different reasonable loss or stochastic subsets of measurements per step inherit the bias. The fully general, non-commuting case loses the single exponential — the true solution becomes a time-ordered exponential and the Lie closure of generic non-commuting $A_i$ fills all of $\mathbb{R}^{n\times n}$, so no confining manifold survives — and there the nuclear-norm selection remains a well-motivated conjecture, supported by the heuristic that a smooth slowly-varying residual control cannot accumulate the wild $1/\epsilon^2$ wiggle needed to exploit commutator directions before the shrinking initialization pins it to the commutative manifold. The low-rank bias is also visible directly in the spectrum: for a depth-$N$ factorization the singular values evolve as $\dot\sigma_r = -N(\sigma_r^2)^{1-1/N}\langle\nabla\ell(W), u_r v_r^\top\rangle$, so at $N=1$ (plain convex descent) the rate is size-independent and flat, while at $N=2$ the prefactor is $|\sigma_r|$ — the growth rate is proportional to the singular value itself, large values race ahead and small ones are throttled, opening a power-law gap that self-organizes the spectrum into a few large values and many near-zero ones, i.e. an effectively low-rank, low-nuclear-norm matrix.

```python
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer


class MatrixRecoveryStrategy:
    def recover(self, observed_values, observed_mask, n, rank_hint,
                device, max_iters, log_iters):
        raise NotImplementedError


class GroupRMSprop(Optimizer):
    """Reference optimizer: one RMS scale shared across all parameter gradients."""

    def __init__(self, params, lr=1e-3, alpha=0.99, eps=1e-4):
        defaults = dict(lr=lr, alpha=alpha, eps=eps, adjusted_lr=lr)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            state = self.state
            if len(state) == 0:
                state["step"] = 0
                state["square_avg"] = torch.tensor(0.0)

            square_avg = state["square_avg"]
            alpha = group["alpha"]
            square_avg.mul_(alpha)
            state["step"] += 1

            for param in group["params"]:
                if param.grad is not None:
                    square_avg.add_((1.0 - alpha) * param.grad.detach().pow(2).sum().cpu())

            avg = square_avg.div(1.0 - alpha ** state["step"]).sqrt_().add_(group["eps"])
            adjusted_lr = group["lr"] / avg
            group["adjusted_lr"] = adjusted_lr

            for param in group["params"]:
                if param.grad is not None:
                    param.data.add_(-adjusted_lr.to(param.device) * param.grad.data)
        return loss


class ShallowMatrixFactorization(MatrixRecoveryStrategy):
    def __init__(self, init_scale=1e-3, lr=1e-3, train_thres=1e-6):
        self.init_scale = float(init_scale)
        self.lr = float(lr)
        self.train_thres = float(train_thres)

    def _build(self, n, device):
        layers = [nn.Linear(n, n, bias=False) for _ in range(2)]
        scale = (self.init_scale ** 0.5) * (n ** -0.5)
        for layer in layers:
            nn.init.normal_(layer.weight, mean=0.0, std=scale)
        return nn.Sequential(*layers).to(device)

    @staticmethod
    def _e2e(model):
        weight = None
        for layer in model:
            weight = layer.weight.t() if weight is None else layer(weight)
        return weight

    def recover(self, observed_values, observed_mask, n, rank_hint,
                device, max_iters, log_iters):
        model = self._build(n, device)
        optimizer = GroupRMSprop(model.parameters(), lr=self.lr, eps=1e-4)
        target = observed_values.to(device)
        mask = observed_mask.to(device)
        denom = max(int(mask.sum().item()), 1)
        log_every = max(int(log_iters), 1)

        for it in range(max_iters):
            estimate = self._e2e(model)
            residual = (estimate - target) * mask
            loss = residual.pow(2).sum() / denom

            optimizer.zero_grad()
            loss.backward()

            if it % log_every == 0 or it == max_iters - 1 or loss.item() <= self.train_thres:
                print(f"TRAIN_METRICS iter={it} train_mse={loss.item():.6e}", flush=True)
                if loss.item() <= self.train_thres:
                    break

            optimizer.step()

        with torch.no_grad():
            return self._e2e(model).detach().cpu()


def build_strategy() -> MatrixRecoveryStrategy:
    return ShallowMatrixFactorization(init_scale=1e-3, lr=1e-3, train_thres=1e-6)
```
