# Mean-Field Langevin Algorithm (MFLA) for multi-index models, distilled

The mean-field Langevin algorithm trains a two-layer network by **noisy, weight-decayed
gradient descent** and reads that training as a convex optimization over the *distribution* of
neurons. Lifting the width-`m` network to a measure `mu` over first-layer weights makes the
risk a convex functional of `mu`; adding entropy regularization makes the Wasserstein gradient
flow of that functional exactly a Langevin diffusion, whose injected Gaussian noise performs
the saddle-escape that single-trajectory SGD must pay for with extra samples. The payoff:
sample complexity scaling almost linearly with an *effective dimension* `d_eff`, independent of
the information/leap exponent of the link.

## Problem it solves

Learning a multi-index target `y = g(Ux) + xi` (`U in R^{k x d}` orthonormal, `k = O(1)`,
`g` unknown) with a *standard* first-order training procedure at sample complexity near the
information floor `n ~ d`, for arbitrary `g` — escaping the `n ~ d^{s-1}` wall that gradient
descent pays for links of information/leap exponent `s >= 3`.

## Key idea

Represent the network by the empirical measure `mu = (1/m) sum_j delta_{w_j}` over its
first-layer neuron weights:

```
yhat(x; mu) = integral Psi(x; w) dmu(w),    R(mu) = integral ||w||^2 dmu(w).
```

The prediction is linear in `mu` and the loss is convex, so the risk `J(mu)` is **convex in
`mu`** even though it is non-convex in `W`. Regularize the *measure* with entropy and minimize
the free energy

```
F_beta(mu) = J(mu) + (1/beta) H(mu | tau),    H(mu | tau) = integral ln(dmu/dtau) dmu,
```

with inverse temperature `beta`. Its Wasserstein gradient flow is the **mean-field Langevin
dynamics**

```
dw_t = - grad J'[mu_t](w_t) dt + sqrt(2 beta^{-1}) dB_t,
```

where `J'[mu]` is the first variation of `J`. The entropy term *is* the diffusion: the noise
escapes the near-flat region around initialization that traps SGD.

**Weight decay = KL regularizer.** An `L2` penalty plus entropy is a relative entropy to a
Gaussian base measure:

```
(lambda/2) integral ||w||^2 dmu + (1/beta) H(mu | Leb) = (1/beta) H(mu | gamma) + const,
    gamma = N(0, (lambda beta)^{-1} I).
```

So the drift splits into the data gradient plus a `-lambda w` weight-decay term, and the
algorithm is just noisy weight-decayed gradient descent.

## Convergence

The controlling object is the **proximal Gibbs distribution** of the current measure,
`p_mu  ∝  exp(-beta J'[mu])`, which fills the duality gap. If `p_mu` satisfies a
**log-Sobolev inequality** with constant `C_LSI` uniformly along the trajectory
(`H(mu | p_mu) <= (C_LSI/2) I(mu | p_mu)`, the measure-space analogue of strong convexity),
the free energy contracts geometrically:

```
F_beta(mu_t) - F_beta(mu*_beta) <= exp(-2t / (beta C_LSI)) (F_beta(mu_0) - F_beta(mu*_beta)).
```

The finite-width, discrete-time iteration contracts one step at a time,

```
F^m(mu_{l+1}) - F(mu*) <= exp(-eta / (2 beta C_LSI)) (F^m(mu_l) - F(mu*)) + eta * A,
    A = O(eta + 1/m + ...),
```

so accuracy `epsilon` needs `eta, 1/m` below `epsilon` and `l ~ (beta C_LSI / eta) log(1/epsilon)`.

## Sample vs. compute: the effective dimension

Define the **effective dimension**

```
d_eff = tr(Sigma) / || Sigma^{1/2} U^T ||_F^2 = c_x^2 / r_x^2,
    c_x = tr(Sigma)^{1/2},    r_x = || Sigma^{1/2} U^T ||_F.
```

With `lambda = lambdatilde r_x^2` and `beta = Theta(d_eff / lambdatilde)`, the **sample
complexity is `n = Otilde(d_eff)`**, almost linear and *independent of the link's
information/leap exponent*. Isotropic data gives `d_eff ~ d`; covariances that concentrate
variance in the relevant directions give `d_eff = polylog(d)`.

The exponent that once governed the *statistics* is paid instead in *compute*, through
`C_LSI`:

- **Euclidean (free weights).** Holley–Stroock gives `C_LSI <= (1/(beta lambda)) exp(4 C_rho
  iota beta)`. With `beta ~ d_eff`, iterations/width scale as `exp(d_eff)` — quasipolynomial in
  `d` only when `d_eff = polylog(d)`. (An `exp(beta)` LSI is unavoidable in the Euclidean
  worst case.)
- **Riemannian (weights on the sphere).** Constrain neurons to `S^{d-1}`; drop the `L2`
  penalty (the manifold is compact) and project after each step. Positive Ricci curvature
  (`Ric >= (d-2) g`) via Bakry–Émery gives `C_LSI <= (rho d - beta C_rho K)^{-1} ~ 1/d` for
  `beta` up to order `d` — **polynomial in `d`**, no `exp(beta)`.

## Architecture choices

- **Second layer frozen at signed `+-1/m`** (half `+`, half `-`): keeps the predictor a plain
  average over a measure of first-layer neurons (the convex lift is over the first-layer
  distribution), while signed halves let non-negative activations represent signed targets.
- **Bounded smooth activation** `phi_{kappa,iota}` (`C^2`, `|phi| <= iota`, bounded `phi'`,
  `phi''`): boundedness lets Holley–Stroock certify the LSI; smoothness controls the
  Euler–Maruyama error. Recovers ReLU as `kappa, iota -> infinity`; plain ReLU is used in code.
- **Spherical initialization / projection:** realizes the polynomial-LSI Riemannian setting and
  keeps neuron norms `O(1)`.
- **Bias coordinate** (append `1` to `x`): a learnable threshold per neuron.
- **Large fresh-sample pool:** the drift `-grad J'[mu]` is a population gradient; drawing from
  a large pool approximates it and reflects the `n ~ d_eff` statistical statement and
  propagation of chaos.

## Final update

Euler–Maruyama of the Langevin SDE, with the data gradient scaled by width `m` (per-neuron
gradient is `O(1/m)`; the measure-space drift is `O(1)`):

```
w_j^{l+1} = w_j^l - m eta grad_{w_j} Jhat_lambda(W) + sqrt(2 eta beta^{-1}) xi_j^l,
            xi_j^l ~ N(0, I) iid,
```

i.e. `subtract eta * data_grad`, `subtract eta * lambda * w_j` (decay), `add sqrt(2 eta / beta)`
Gaussian noise; in the Riemannian variant, renormalize `w_j` to the sphere afterward. The noise
constant is the Euler–Maruyama factor `sqrt(2 eta beta^{-1})` (equivalently `sqrt(2 eta) *
noise_std` with `noise_std = 1/sqrt(beta)`), not `sqrt(eta/beta)`.

## Working code

```python
import math
import torch


def predict(X, W, a, phi):
    """Two-layer net with fixed second layer a: yhat = phi(X W^T) @ a."""
    return phi(X @ W.T) @ a


def first_layer_gradient(X, y, W, a, phi, phiprime):
    """Gradient of the per-batch squared loss w.r.t. first-layer weights W -> [m, d]."""
    n = X.shape[0]
    residual = (predict(X, W, a, phi) - y).reshape(1, -1)          # [1, n]
    backprop = phiprime(W @ X.T) * a.reshape(-1, 1)                # [m, n]
    return (backprop * residual) @ X / n                          # [m, d]


def relu(z):
    return torch.clamp(z, min=0.0)


def reluprime(z):
    return (z >= 0).to(z.dtype)


def make_fixed_second_layer(m):
    """Signed halves +1/m and -1/m so non-negative activations represent signed targets."""
    return torch.cat([torch.ones(m // 2), -torch.ones(m - m // 2)]) / m


def init_first_layer_on_sphere(m, d):
    """Neurons (= particles of the measure mu) initialized uniformly on the unit sphere."""
    W = torch.randn(m, d)
    return W / W.norm(dim=1, keepdim=True).clamp(min=1e-8)


def mfla_step(W, X, y, a, phi, phiprime, lr, weight_decay, beta, project_to_sphere=True):
    """One mean-field Langevin update = noisy, weight-decayed gradient descent.

      drift  = -(data gradient) - lambda * w           (data fit + weight decay = KL drift)
      noise  = sqrt(2 * lr / beta) * standard Gaussian (Euler-Maruyama of the diffusion)
    The m prefactor rescales the O(1/m) per-neuron gradient to the O(1) measure-space drift.
    """
    m = W.shape[0]
    g = first_layer_gradient(X, y, W, a, phi, phiprime)           # [m, d]
    noise = math.sqrt(2.0 * lr / beta) * torch.randn_like(W)
    W = W - lr * m * g - lr * weight_decay * W + noise
    if project_to_sphere:                                         # Riemannian retraction
        W = W / W.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return W


def train_mfla(X, y, n_iters=3000, m=50, lr=0.1, weight_decay=0.01,
               inv_temp=0.001, project_to_sphere=True):
    """inv_temp = 1/beta. Append a bias coordinate to X before calling for a learnable threshold."""
    d = X.shape[1]
    beta = 1.0 / inv_temp
    a = make_fixed_second_layer(m).to(X)
    W = init_first_layer_on_sphere(m, d).to(X)
    losses = []
    for _ in range(n_iters):
        W = mfla_step(W, X, y, a, relu, reluprime, lr, weight_decay, beta, project_to_sphere)
        losses.append(torch.mean((predict(X, W, a, relu) - y) ** 2).item())
    return W, losses
```
