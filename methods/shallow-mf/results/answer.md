# Shallow Matrix Factorization (depth-2 factorized gradient descent)

## Method

Recover the matrix as the end-to-end product of two full-dimensional, bias-free linear layers,

```text
W = W2 W1^T,
```

and minimize masked squared error on the observed entries by gradient descent on the factors.
The hidden dimension is `n`, so there is no explicit rank cap. The factors are initialized near
zero with the same Gaussian scaling as the reference deep-matrix-factorization implementation:
per-layer standard deviation

```text
init_scale^(1/depth) * n^(-1/2),   depth = 2.
```

Thus the end-to-end Frobenius norm starts on the order of `init_scale * sqrt(n)`. The rank
preference is supplied by the factorized descent dynamics, not by a nuclear-norm penalty, early
stopping rule, or explicit low-rank architecture.

## Result

For the symmetric PSD form `X = U U^T`, consider the gradient-flow normalization

```text
Udot = -A*(A(UU^T)-y) U.
```

Different squared-loss constants only rescale time. For `X_t = U_t U_t^T` and
`r_t = A(X_t)-y`, the induced dynamics are

```text
Xdot = -A*(r_t) X_t - X_t A*(r_t).
```

Direct gradient descent on `X` instead follows `Xdot = -A*(r_t)` and, from `X=0`, stays in
`{A*(s)}`; a zero-error limit is therefore the minimum-Frobenius-norm fit. Factorized descent
changes the geometry to a multiplicative flow.

The main claim is: with a full-rank initialization whose scale goes to zero, if the factorized
flow converges to a zero-error global optimum, the selected optimum is the minimum-nuclear-norm
solution. This is a conjecture in general and is proved when the measurement matrices commute.

In the commuting case,

```text
X_t = exp(A*(s_t)) X_0 exp(A*(s_t)),   s_t = -∫_0^t r_tau d tau.
```

Use product-level initialization `X_0 = gamma I` and set `beta = -1/2 log gamma`; equivalently,
if two factors are initialized as `alpha I`, then `gamma = alpha^2` and `beta = -log alpha`.
For a shared eigenvector `v_k`,

```text
lambda_k(X_infty(gamma I))
  = gamma exp(2 lambda_k(A*(s_infty)))
  = exp(2 lambda_k(A*(s_infty)) - 2 beta).
```

With `nu(beta) = s_infty(beta) / beta`, positive limiting eigenvalues force
`lambda_k(A*(nu(beta))) -> 1`; zero limiting eigenvalues force the corresponding dual
eigenvalues to have limit at most `1`. Hence `A*(nu) <= I` and `(I-A*(nu)) Xhat = 0`, the dual
feasibility and complementary-slackness KKT conditions for

```text
min_{X >= 0} ||X||_*  subject to A(X)=y.
```

For the asymmetric depth-2 product used in code (`W2 @ W1.T`, equivalently `W2 W1` under the
factor convention), the balanced flow with factors initialized at `alpha I` reduces to the same
PSD product dynamics with `W(0)=alpha^2 I`. The induced singular-value dynamics give the same
low-rank intuition: at depth `N=2`,

```text
sigma_dot_r = -2 |sigma_r| <grad ell(W), u_r v_r^T>,
```

so small singular values move slowly while larger ones move faster.

## Code

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
