# Mean-field Langevin training for multi-index models

## Core object

The target model is

```text
y = g(U x) + xi,
U = (u_1/sqrt(k), ..., u_k/sqrt(k))^T,
```

with orthonormal `u_i`, zero-mean subGaussian noise `xi`, and fixed `k << d`. The source
normalization matters: for isotropic `Sigma = I_d`,

```text
c_x = sqrt(tr(Sigma)),       r_x = || Sigma^{1/2} U^T ||_F,
d_eff = c_x^2 / r_x^2 = d.
```

For general covariance, `d_eff` is small only when input variance is concentrated in directions
that overlap the target subspace.

## Measure-space lift

Write the width-`m` first layer as an empirical measure

```text
mu_W = (1/m) sum_j delta_{w_j},
yhat(x; mu) = integral Psi(x; w) dmu(w).
```

The prediction is linear in `mu`, so a convex pointwise loss makes the risk convex as a functional
of `mu`. The Euclidean free energy is

```text
F_{beta,lambda}(mu)
  = empirical_risk(mu) + (lambda/2) integral ||w||^2 dmu + beta^{-1} H(mu).
```

Since

```text
(lambda/2) integral ||w||^2 dmu + beta^{-1} H(mu | Leb)
  = beta^{-1} H(mu | gamma) + const,
gamma = N(0, (lambda beta)^{-1} I),
```

weight decay is exactly the Gaussian-base KL regularizer in the lifted problem.

## Dynamics and constants

The mean-field Langevin SDE is

```text
d w_t = - grad_w J'[mu_t](w_t) dt + sqrt(2 / beta) dB_t,
```

and the Eq. 10 Euler-Maruyama MFLA update is

```text
w_j^{l+1}
  = w_j^l - m eta grad_{w_j} Jhat_lambda(W^l)
    + sqrt(2 eta / beta) xi_j^l,
xi_j^l ~ N(0, I).
```

The `m` factor is required because the per-neuron gradient of the averaged network objective is
`O(1/m)`. Expanding `Jhat_lambda` gives the concrete drift

```text
- lr * m * data_gradient - lr * lambda * W.
```

The official notebook uses the same gradient and weight decay scaling, but its empirical noise line
is

```python
2 * np.sqrt(lr * inv_temp) * torch.randn_like(W)
```

with `inv_temp = beta^{-1}`. That is `sqrt(4 lr / beta)`, whereas Eq. 10 is
`sqrt(2 lr / beta)`.

## Guarantees

The proximal Gibbs law is

```text
nu_mu(dw) proportional exp(-beta J'[mu](w)) tau(dw).
```

If these laws satisfy an LSI with constant `C_LSI`, then

```text
F_beta(mu_t) - F_beta(mu_beta*)
  <= exp(-2t / (beta C_LSI))
     (F_beta(mu_0) - F_beta(mu_beta*)).
```

In Euclidean space, bounded smooth activations and a Lipschitz loss give the Holley-Stroock bound

```text
C_LSI <= (1 / (beta lambda)) exp(4 C_rho iota beta).
```

With `lambda = lambda_tilde r_x^2` and `beta = Otilde(d_eff / lambda_tilde)`, the main theorem
gives sample complexity

```text
n = Otilde(d_eff),
```

independent of the information/leap exponent of `g`, while width and iteration bounds are
exponential in `d_eff` in the worst case.

The compact Riemannian result is conditional. On a manifold with
`Ric >= rho d * metric`, if there exists a good reference measure with empirical risk
`bar_epsilon` and entropy `bar_Delta` against the uniform measure, then for
`beta < rho d/(C_rho K)`,

```text
C_LSI <= (rho d - beta C_rho K)^{-1}.
```

This gives a polynomial-time route under the entropy-spread assumption. It does not prove that
this assumption yields `bar_Delta ~ d_eff` for fixed-`k` multi-index models; that case remains
open.

## Faithful reference implementation

```python
import math
import torch


def predict(X, W, a, phi):
    return phi(X @ W.T) @ a


def first_layer_gradient(X, y, W, a, phi, phiprime):
    """Matches the notebook gradient: two 1/sqrt(n) factors give the mean loss gradient."""
    n = X.shape[0]
    v1 = phiprime(W @ X.T) * a.reshape(-1, 1) / math.sqrt(n)
    v2 = X * (predict(X, W, a, phi) - y).reshape(-1, 1) / math.sqrt(n)
    return v1 @ v2


def relu(z):
    return torch.maximum(z, torch.zeros_like(z))


def reluprime(z):
    return (z >= 0).to(z.dtype)


def make_signed_second_layer(m, *, device=None, dtype=None):
    """Notebook convention: scalar neurons with fixed +1/m and -1/m output weights."""
    a = torch.cat([torch.ones(m // 2), -torch.ones(m - m // 2)])
    return (a / m).to(device=device, dtype=dtype)


def init_first_layer_on_sphere(m, d, *, device=None, dtype=None):
    """Notebook initialization; the notebook does not project after later updates."""
    W = torch.randn(m, d, device=device, dtype=dtype)
    return W / W.norm(dim=1, keepdim=True).clamp(min=1e-8)


def mfla_step(
    W,
    X,
    y,
    a,
    phi,
    phiprime,
    lr,
    weight_decay,
    inv_temp,
    *,
    match_notebook_noise=True,
):
    """One first-layer update.

    match_notebook_noise=True reproduces the companion notebook:
        noise = 2 * sqrt(lr * inv_temp) * N(0, I).
    match_notebook_noise=False uses Eq. 10:
        noise = sqrt(2 * lr * inv_temp) * N(0, I).
    """
    m = W.shape[0]
    grad = first_layer_gradient(X, y, W, a, phi, phiprime)
    coeff = 2.0 if match_notebook_noise else math.sqrt(2.0)
    noise = coeff * math.sqrt(lr * inv_temp) * torch.randn_like(W)
    return W - lr * m * grad - lr * weight_decay * W + noise


def train_mfla(
    X,
    y,
    *,
    n_iters=3000,
    width=50,
    lr=0.1,
    inv_temp=0.001,
    weight_decay=0.01,
    append_bias=True,
    match_notebook_noise=True,
):
    if append_bias:
        X = torch.cat([X, torch.ones(X.shape[0], 1, device=X.device, dtype=X.dtype)], dim=1)
    W = init_first_layer_on_sphere(width, X.shape[1], device=X.device, dtype=X.dtype)
    a = make_signed_second_layer(width, device=X.device, dtype=X.dtype)
    losses = []
    for _ in range(n_iters):
        W = mfla_step(
            W,
            X,
            y,
            a,
            relu,
            reluprime,
            lr,
            weight_decay,
            inv_temp,
            match_notebook_noise=match_notebook_noise,
        )
        losses.append(torch.mean((predict(X, W, a, relu) - y) ** 2).item())
    return W, a, losses
```

The theoretical Euclidean architecture fixes all outer weights at `+1` and represents signed
outputs by paired neurons `Psi(x; w) = phi(<x_tilde, omega_1>) - phi(<x_tilde, omega_2>)` with
`w in R^{2d+2}`. The notebook's signed scalar-neuron implementation is the corresponding practical
encoding used for experiments.
