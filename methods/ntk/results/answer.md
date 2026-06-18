# The Neural Tangent Kernel

## Problem

Gradient descent on a neural network's parameters minimizes a highly non-convex
parameter-space loss, yet wide networks often reach zero training loss and
generalize. In the infinite-width NTK parameterization, this behavior can be
described by following the *function* computed by the network instead of the
weights.

## Key idea

Track the realization map `F(theta)=f_theta`. Under gradient flow
`theta_dot = -grad_theta (C o F)`, with `partial_f C` represented by
`<d, .>_{p_in}`, the chain rule gives

    partial_t f_theta(x) = - E_{x'~p_in}[Theta(x, x') d(x')],

where the tangent kernel is

    Theta(x, x') = sum_p partial_{theta_p} f_theta(x) partial_{theta_p} f_theta(x')^T.

For squared loss on `N` samples this is

    f_dot_i = -(1/N) sum_j Theta(x_i, x_j) (f_j - y_j),

or `f_dot = -Pi(f-y)` if `Pi` denotes the empirical kernel operator including
the `1/N` average.

With preactivations

    alpha_tilde^(l+1) = (1/sqrt(n_l)) W^(l) alpha^(l) + beta b^(l),

iid standard Gaussian parameters, and hidden widths tending to infinity, the
kernel has a deterministic limit

    Sigma^(1)(x,x')       = x^T x' / n0 + beta^2
    Sigma^(L+1)(x,x')     = E[sigma(f(x)) sigma(f(x'))] + beta^2
    dotSigma^(L+1)(x,x')  = E[sigma'(f(x)) sigma'(f(x'))]
    Theta_inf^(1)         = Sigma^(1)
    Theta_inf^(L+1)       = Theta_inf^(L) dotSigma^(L+1) + Sigma^(L+1).

The training-time constancy theorem assumes a Lipschitz twice-differentiable
activation with bounded second derivative and a stochastically bounded
`int_0^T ||d_t||_{p_in} dt`; then `Theta(t) -> Theta_inf^(L) tensor Id`
uniformly on finite time intervals. Each individual hidden preactivation and
scaled weight displacement is small, while the aggregate of many small tangent
features remains order one.

## Consequences

- Least-squares training becomes the fixed-kernel linear ODE
  `f_t = f* + exp(-t Pi)(f_0 - f*)`.
- For `beta > 0`, depth `L >= 2`, and non-polynomial Lipschitz `sigma`, the
  limiting kernel is positive definite on the unit sphere; the proof uses
  Daniely's Hermite dual and the Schoenberg/Gneiting sphere criterion.
- Components along kernel principal directions decay as `exp(-lambda_i t)`, so
  large-eigenvalue directions are fit first.
- At convergence, the mean predictor is ridgeless kernel regression with
  `Theta_inf^(L)`; the residual Gaussian fluctuation is pinned to zero on the
  training points.

## Code

This is the scalar fully connected ReLU specialization of the Neural Tangents
`stax.Dense(..., parameterization="ntk", W_std=1, b_std=beta)` plus `Relu`
recursion, and the empirical kernel mirrors Neural Tangents'
Jacobian-contraction NTK.

```python
import numpy as np
import torch


def relu_dual(cov_xx, cov_xpxp, cov_xxp):
    """Return E[ReLU(X)ReLU(X')] and E[ReLU'(X)ReLU'(X')]."""
    denom = np.sqrt(cov_xx * cov_xpxp)
    rho = np.clip(cov_xxp / np.maximum(denom, 1e-12), -1.0, 1.0)
    angle = np.arccos(rho)
    nngp = denom / (2.0 * np.pi) * (
        np.sin(angle) + (np.pi - angle) * np.cos(angle)
    )
    nngp_dot = (np.pi - angle) / (2.0 * np.pi)
    return nngp, nngp_dot


def infinite_ntk(X, Xp, depth, beta=0.1):
    """Compute (Theta_inf^(L), Sigma^(L)) for a depth-L ReLU MLP."""
    if depth < 1:
        raise ValueError("depth counts affine layers and must be at least 1")
    n0 = X.shape[1]
    beta2 = beta ** 2
    sig = X @ Xp.T / n0 + beta2
    sig_xx = (X * X).sum(axis=1) / n0 + beta2
    sig_pp = (Xp * Xp).sum(axis=1) / n0 + beta2
    theta = sig.copy()

    for _ in range(depth - 1):
        nngp, nngp_dot = relu_dual(sig_xx[:, None], sig_pp[None, :], sig)
        sig_xx = relu_dual(sig_xx, sig_xx, sig_xx)[0] + beta2
        sig_pp = relu_dual(sig_pp, sig_pp, sig_pp)[0] + beta2
        sig = nngp + beta2
        theta = theta * nngp_dot + sig
    return theta, sig


class WideMLP(torch.nn.Module):
    """Finite-width MLP in the NTK parameterization."""
    def __init__(self, n0, width, depth, beta=0.1):
        super().__init__()
        if depth < 1:
            raise ValueError("depth counts affine layers and must be at least 1")
        self.beta = beta
        sizes = [n0] + [width] * (depth - 1) + [1]
        self.Ws = torch.nn.ParameterList(
            torch.nn.Parameter(torch.randn(out_dim, in_dim))
            for in_dim, out_dim in zip(sizes[:-1], sizes[1:])
        )
        self.bs = torch.nn.ParameterList(
            torch.nn.Parameter(torch.randn(out_dim)) for out_dim in sizes[1:]
        )
        self.scales = [in_dim ** -0.5 for in_dim in sizes[:-1]]

    def forward(self, x):
        a = x
        last = len(self.Ws) - 1
        for i, (W, b) in enumerate(zip(self.Ws, self.bs)):
            a = self.scales[i] * (a @ W.T) + self.beta * b
            if i != last:
                a = torch.relu(a)
        return a.squeeze(-1)


def _jacobian_rows(net, X):
    params = list(net.parameters())
    rows = []
    for xi in X:
        out = net(xi.unsqueeze(0)).squeeze()
        grads = torch.autograd.grad(out, params)
        rows.append(torch.cat([g.reshape(-1) for g in grads]))
    return torch.stack(rows)


def empirical_ntk(net, X, Xp=None):
    """Finite NTK: J_theta f(X) J_theta f(Xp)^T."""
    J = _jacobian_rows(net, X)
    Jp = J if Xp is None else _jacobian_rows(net, Xp)
    return (J @ Jp.T).detach().cpu().numpy()


def kernel_regression(K_train, K_test, y, ridge=0.0):
    """Ridgeless kernel-regression mean; use ridge > 0 for conditioning."""
    A = K_train + ridge * np.eye(K_train.shape[0])
    return K_test @ np.linalg.solve(A, y)
```
