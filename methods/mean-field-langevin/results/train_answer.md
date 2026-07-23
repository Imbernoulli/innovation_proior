We are trying to learn a regression target $y = g(Ux) + \xi$ that depends on a high-dimensional input only through a few hidden directions: $U$ has rows $u_i^T/\sqrt{k}$ with the $u_i$ orthonormal, the noise $\xi$ is mean-zero subGaussian, and $k \ll d$. Counting degrees of freedom, the relevant subspace carries only order $d$ information, so on isotropic data the statistical floor for locating it should be $n$ proportional to $d$. The problem is that ordinary training does not get near that floor. At a random initialization a neuron's overlap with a true direction is only order $d^{-1/2}$, and if the first nonvanishing correlation term of the link has degree $s$, the population pull near the uninformative equator scales like overlap raised to the $s-1$ power. For $s \ge 3$ that signal is so weak that plain two-layer SGD spends a long search phase before it moves; for multi-index targets the same obstruction appears as leap complexity, where some coordinates only become visible after earlier ones are learned. So existing baselines all pay a price tied to the algebraic structure of the link rather than to the dimension of the hidden subspace. Plain SGD follows the information or leap exponent. Layer-wise and two-stage methods isolate the feature step but still need the first-layer search signal to be strong. Random-feature methods freeze the first layer and dodge the non-convex search, but then they cannot adapt to the hidden low-dimensional structure. The convex infinite-width formulations show that lifting a wide network to a distribution over neurons makes the loss convex, but they leave open a general finite-sample, finite-training guarantee for arbitrary multi-index links under general covariance. What we need, then, is not a better way to represent $g$ — the network already represents it — but a training dynamics whose sample complexity is not throttled by that weak local gradient.

I propose mean-field Langevin training for the first layer, which I will call MFLA. The starting point is the permutation symmetry of a wide two-layer network: its output is an average over neurons, so I represent the first layer by the empirical measure $\mu_W = \frac{1}{m}\sum_j \delta_{w_j}$ and write the predictor $\hat y(x;\mu) = \int \Psi(x;w)\,d\mu(w)$, which is linear in $\mu$. With a convex pointwise loss the risk becomes convex as a functional of $\mu$, even though it is non-convex in the finite weight matrix $W$. This reframing is what changes the game: instead of one trajectory hunting a hidden direction by a tiny signal, I run a particle system that performs descent in a convex space of measures. A bare Wasserstein gradient flow of the risk is still not enough — I need a unique regularized minimizer and a convergence handle — so I add entropy. The Euclidean free energy is
$$F_{\beta,\lambda}(\mu) = \widehat{\mathcal R}(\mu) + \frac{\lambda}{2}\int \|w\|^2\,d\mu + \frac{1}{\beta}H(\mu).$$
The quadratic penalty and the entropy are not two separate heuristics: completing the square shows
$$\frac{\lambda}{2}\int\|w\|^2\,d\mu + \frac{1}{\beta}H(\mu\mid \mathrm{Leb}) = \frac{1}{\beta}H(\mu\mid\gamma) + \text{const},\qquad \gamma = \mathcal N\!\big(0,(\lambda\beta)^{-1}I\big),$$
so the ordinary $L^2$ weight decay is exactly the Gaussian reference measure for the entropic problem. The gradient flow of this free energy is a nonlinear Langevin diffusion: a drift from the first variation plus a $\beta^{-1}$ Laplacian, with particle SDE
$$dw_t = -\nabla_w J'[\mu_t](w_t)\,dt + \sqrt{2/\beta}\,dB_t.$$
The signs and constants are load-bearing. The drift is the negative gradient of the first variation, and the diffusion coefficient is $\sqrt{2\beta^{-1}}$, so an Euler–Maruyama step of length $\eta$ injects a standard Gaussian scaled by $\sqrt{2\eta/\beta}$. To match the finite-width scaling I have to account for the averaging: the empirical predictor and the regularizer are both averages over $m$ neurons, so the gradient of the network objective with respect to one neuron's weight is of order $1/m$, while the measure-space drift is of order one per particle. The finite-particle update therefore multiplies the ordinary gradient by $m$:
$$w_j^{l+1} = w_j^l - m\,\eta\,\nabla_{w_j}\widehat J_\lambda(W^l) + \sqrt{2\eta/\beta}\,\xi_j^l,\qquad \xi_j^l \sim \mathcal N(0,I).$$
Expanding $\widehat J_\lambda$, whose regularizer is $\frac{\lambda}{2m}\sum_j\|w_j\|^2$, the scaled penalty gradient contributes exactly $-\eta\lambda w_j$, so the drift is the concrete combination $-\,\mathrm{lr}\cdot m\cdot(\text{data gradient}) - \mathrm{lr}\cdot\lambda\cdot W$. In words, MFLA is data-gradient descent, plus weight decay, plus Gaussian parameter noise — and each piece is forced by the measure-space derivation rather than chosen by hand.

What makes this provably reach the floor is the convergence analysis through the proximal Gibbs law. Freezing the current measure in the first variation, the relevant density is $\nu_\mu(dw) \propto \exp\!\big(-\beta J'[\mu](w)\big)\,\tau(dw)$; the flow dissipates KL toward this Gibbs law, and a log-Sobolev inequality with constant $C_{\mathrm{LSI}}$ gives the free-energy contraction
$$F_\beta(\mu_t) - F_\beta(\mu_\beta^\star) \le \exp\!\big(-2t/(\beta C_{\mathrm{LSI}})\big)\,\big(F_\beta(\mu_0) - F_\beta(\mu_\beta^\star)\big),$$
where the factor $2$ in the exponent is the standard LSI normalization. The Euclidean LSI constant is where the cost hides. The first variation contains the quadratic $\frac{\lambda}{2}\|w\|^2$, so the Gaussian reference has LSI constant $1/(\beta\lambda)$; the data-fit term is a bounded perturbation because the activation is bounded by $\iota$ and the loss derivative by $C_\rho$, giving an oscillation of at most $4C_\rho\iota$, and Holley–Stroock yields
$$C_{\mathrm{LSI}} \le \frac{1}{\beta\lambda}\exp(4C_\rho\iota\beta).$$
The statistical side uses the effective dimension. With the scaled convention $r_x = \|\Sigma^{1/2}U^T\|_F$ and $d_{\mathrm{eff}} = \operatorname{tr}(\Sigma)/r_x^2$ — so isotropic data has $r_x^2 = 1$ and $d_{\mathrm{eff}} = d$ — setting $\lambda = \tilde\lambda\, r_x^2$ and $\beta = \tilde O(d_{\mathrm{eff}}/\tilde\lambda)$ makes the empirical regularized minimizer generalize with $n = \tilde O(d_{\mathrm{eff}})$. The crucial outcome is that the information or leap exponent of $g$ has moved entirely out of the sample complexity; what remains is the exponential-in-$\beta$ price living in the LSI constant, so worst-case width and iteration counts are exponential in $d_{\mathrm{eff}}$ while the sample complexity is not. I keep the Riemannian variant in its proper lane: constraining the weights to a positively curved compact manifold removes the Euclidean quadratic confinement and gives a Bakry–Émery bound $C_{\mathrm{LSI}} \le (\rho d - \beta C_\rho K)^{-1}$ when $\beta < \rho d/(C_\rho K)$, which looks polynomial, but it is conditional on a compact-manifold entropy-spread assumption — a good reference measure with empirical risk at most $\bar\epsilon$ and entropy at most $\bar\Delta$ against the uniform measure — and matching $\bar\Delta$ to the Euclidean $d_{\mathrm{eff}}$ for fixed-$k$ multi-index targets remains open. So spherical projection is a route under extra assumptions, not the proven finite-time algorithm; the Euclidean update above is the main guarantee.

Two implementation subtleties are worth pinning down before the code. In the main Euclidean theory the second layer is fixed at $+1$, and signed functions are represented by pairing each neuron's parameter into two halves, $\Psi(x;w) = \phi(\langle\tilde x,\omega_1\rangle) - \phi(\langle\tilde x,\omega_2\rangle)$; the practical encoding uses scalar neurons with half the output weights $+1/m$ and half $-1/m$. And there is a deliberate constant mismatch to preserve: the Eq. 10 Euler–Maruyama noise is $\sqrt{2\,\mathrm{lr}/\beta}$, while the companion notebook writes $2\sqrt{\mathrm{lr}\cdot\mathrm{inv\_temp}}$ with $\mathrm{inv\_temp} = \beta^{-1}$, i.e. $\sqrt{4\,\mathrm{lr}/\beta}$, a factor $\sqrt 2$ larger; the implementation keeps both, defaulting to the notebook coefficient for code-faithful reproduction and exposing the Eq. 10 coefficient as an option.

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
