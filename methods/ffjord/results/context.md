# Context

## Research question

We want a generative model that simultaneously (1) computes **exact log-likelihoods** (so it can be trained by maximum likelihood and evaluated by held-out likelihood), (2) **samples efficiently in a single pass**, and (3) places **no restriction on the neural-network architecture** of its transformation. Change-of-variables (flow) models give (1) and (2) but only by paying for a tractable Jacobian determinant — and the standard way to make the determinant cheap is to *constrain the architecture* (partition dimensions, force a triangular Jacobian, use rank-one layers). The open problem: can we keep exact likelihood and one-pass sampling while letting the transformation be a **free-form** neural net, and at a cost that scales gracefully with dimension `D`?

## Background

**Change of variables.** A complex normalized density can be specified implicitly by warping a simple base `p_z` through an invertible `f: R^D → R^D`. If `z ∼ p_z` and `x = f(z)`,
`log p_x(x) = log p_z(z) − log |det (∂f(z)/∂z)|`.
The log-determinant of the `D×D` Jacobian costs `O(D³)` in general — the central bottleneck. "Reversible generative models" are those that use this formula while keeping *both* density evaluation and sampling efficient in a single pass.

**The three ways prior flows make the determinant cheap (all constrain `f`).**
- *Restricted functional forms / normalizing flows* (Rezende & Mohamed 2015 planar flows; Berg et al. 2018 Sylvester): pick `f` so a determinant identity applies. Planar flow is literally a one-layer net with a *single* hidden unit per layer — very low capacity per step. They typically lack a tractable analytic inverse, so they cannot both train on data and sample; they are used for variational posteriors.
- *Autoregressive transforms* (IAF, Kingma et al. 2016; MAF, Papamakarios et al. 2017; TAN, Oliva et al. 2018): impose an ordering on dimensions so the Jacobian is triangular and `det = ∏ diagonal`. Excellent density estimation, but inverting (sampling) needs `D` *sequential* evaluations — prohibitive for large `D`.
- *Partitioned / coupling layers* (NICE, Dinh et al. 2014; Real NVP, Dinh et al. 2016; Glow, Kingma & Dhariwal 2018): split the dimensions and affinely transform one block conditioned on the other, giving a cheap triangular determinant and an inverse that costs the same as the forward map. Convolution-friendly, strong on images — but each layer is constrained, so many must be stacked.

The common cost: tractable determinants are bought with hand-engineered, low-capacity-per-layer architectures.

**Continuous normalizing flows (Chen et al. 2018, Neural ODEs).** Replace the discrete stack of layers with continuous-time dynamics: sample `z(t_0) ∼ p_{z0}`, define an ODE `∂z(t)/∂t = f(z(t), t; θ)` with `f` a neural net, and integrate the initial-value problem to `z(t_1) = x`. The change in log-density along the trajectory obeys the **instantaneous change of variables**:
`∂ log p(z(t))/∂t = − Tr(∂f/∂z(t))`,
so `log p(z(t_1)) = log p(z(t_0)) − ∫_{t_0}^{t_1} Tr(∂f/∂z(t)) dt`. The crucial change: the *determinant* of a discrete flow becomes a *trace*. Trace is linear and, unlike a determinant, does not require any structure in the Jacobian — so `f` can be much freer (planar CNF uses a one-layer net with *many* hidden units). Existence and uniqueness of the ODE solution require `f` and its first derivatives to be Lipschitz, satisfied by smooth Lipschitz activations. Computing the trace exactly still costs `O(D²)` (each diagonal entry is a separate derivative of `f`), so cost goes from `O(D³)` per layer to `O(D²)` per solver step — better, but still effectively limiting.

**Backprop through the ODE: the adjoint method** (Pontryagin; Chen et al. 2018). For a scalar loss `L(z(t_1))`, the adjoint `a(t) = −∂L/∂z(t)` and the parameter gradient
`dL/dθ = − ∫_{t_1}^{t_0} (∂L/∂z(t))ᵀ (∂f/∂θ) dt`
are obtained by solving a second ODE backward in time. This is the continuous-time analog of backpropagation and uses `O(1)` memory (no stored activations), so very large batch sizes become feasible.

**Stochastic trace estimation (Hutchinson 1989).** For any `D×D` matrix `A` and any distribution `p(ε)` with `E[ε]=0`, `Cov(ε)=I`,
`Tr(A) = E_{p(ε)}[ εᵀ A ε ]`,
an unbiased Monte-Carlo estimator of the trace. Typical `ε`: standard Gaussian or Rademacher (`±1` entries). Combined with reverse-mode autodiff — which computes a vector-Jacobian product `vᵀ(∂f/∂z)` for about the cost of one evaluation of `f` — this is the standard cheap way to probe a trace without forming the matrix.

**Other generative families (for contrast, not change-of-variables).** GANs (Goodfellow et al. 2014) use unrestricted nets but have no closed-form likelihood (need a discriminator). Autoregressive models (MADE, PixelRNN) give exact likelihood but need `O(D)` passes to sample. VAEs (Kingma & Welling 2013) use unrestricted nets but only give a lower bound on the marginal likelihood.

## Baselines

- **Real NVP / Glow** (coupling-layer flows): train on data, one-pass sample, exact likelihood — but the Jacobian is constrained to a partitioned/triangular form, requiring many stacked coupling layers and special invertible `1×1` convolutions; cannot model some densities well (e.g. cleanly separated multimodal / near-discontinuous 2D densities — coupling flows struggle in the low-probability regions between disconnected modes). Gap: constrained architecture.
- **Autoregressive flows (MAF, IAF, TAN, MAF-DDSF)**: exact likelihood, strong on tabular density estimation, but sampling/inversion is `O(D)` sequential (and MAF-DDSF has no analytic inverse). Gap: no efficient one-pass sampling.
- **Planar / Sylvester normalizing flows**: cheap determinants via functional-form restriction, used as variational posteriors; very low capacity per transform and no usable inverse for sampling from data. Gap: restricted form, used only for inference.
- **Planar CNF** (Chen et al. 2018): continuous flow with the trace formula, free-er `f`, one-pass sampling and exact likelihood — but exact trace costs `O(D²)`, so `f` is still kept relatively restricted/low-width.

The shared gap: no existing change-of-variables model has *all* of {train on data, one-pass sampling, exact likelihood, **free-form Jacobian**} at a per-step cost better than `O(D²)`.

## Evaluation settings

- **Toy 2D densities** (eight-Gaussians, checkerboard, two-spirals): warp an isotropic Gaussian to multimodal and near-discontinuous targets; qualitative density/sample visualization, compared against a 100-layer Glow.
- **Tabular density estimation** (POWER, GAS, HEPMASS, MINIBOONE, BSDS300, preprocessed as in Papamakarios et al. 2017): negative log-likelihood in nats.
- **Image density estimation** (MNIST, CIFAR-10): bits/dim, with single-flow encoder-decoder and multiscale (squeeze-based) architectures à la Real NVP/Glow.
- **Variational inference** (VAE posteriors on MNIST, Omniglot, Frey Faces, Caltech Silhouettes): negative ELBO, encoder/decoder mirroring Berg et al. 2018, comparing flow families for the approximate posterior.
- **Protocol / solver:** Runge–Kutta 4(5) (Dormand–Prince) adaptive solver with tolerances around `1e-5` (tighter for tabular); Adam, lr `1e-3` decayed to `1e-4`; batch sizes up to `10,000` (tabular) and `900` (images), enabled by the `O(1)`-memory adjoint. Train with the stochastic trace estimator; evaluate test likelihood with the exact trace where feasible. Ablations measure number-of-function-evaluations (NFE) vs data dimension and the effect of a hidden bottleneck on estimator variance.

## Code framework

The pieces that already exist: a black-box adaptive ODE solver with an adjoint backward pass (`torchdiffeq.odeint_adjoint`), reverse-mode autodiff for vector-Jacobian products (`torch.autograd.grad`), PyTorch layers, and Adam. The base distribution is a fixed Gaussian. What is *not* yet decided: how to define the time-dependent dynamics network, and — the crux — how to compute the log-density change along the trajectory cheaply enough to allow an unrestricted `f`. The slots below are empty.

```python
import torch, torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

class DiffEqNet(nn.Module):
    """The dynamics f(z(t), t; θ): R^D × R → R^D, a neural net. How time t enters
    and which layers are used is to be chosen; activations must be smooth/Lipschitz."""
    def __init__(self, ...):
        # stack of layers conditioned on t ; smooth nonlinearity
        pass
    def forward(self, t, z):
        # TODO: return dz/dt
        pass

def divergence(dz_dt, z):
    """Tr(∂f/∂z): the rate of change of log-density. How to compute it — and how cheaply —
    is exactly the open question."""
    # TODO
    pass

class ODEfunc(nn.Module):
    """Augmented dynamics: integrates the state z together with its log-density change."""
    def __init__(self, diffeq):
        super().__init__()
        self.diffeq = diffeq
    def forward(self, t, states):
        z = states[0]                       # log-density is the second state
        with torch.set_grad_enabled(True):
            z.requires_grad_(True)
            dz = self.diffeq(t, z)
            dlogp = ???                      # TODO: -Tr(∂f/∂z)
        return (dz, dlogp)

class CNF(nn.Module):
    """Solve the augmented IVP from t0 to t1 (or reversed), returning z(t1) and Δlogp."""
    def __init__(self, odefunc, T=1.0, solver='dopri5', atol=1e-5, rtol=1e-5):
        super().__init__()
        self.odefunc, self.T = odefunc, T
        self.solver, self.atol, self.rtol = solver, atol, rtol
    def forward(self, z, logpz=None, reverse=False):
        logpz = torch.zeros(z.shape[0], 1).to(z) if logpz is None else logpz
        times = torch.tensor([0.0, self.T]).to(z)
        if reverse:
            times = times.flip(0)
        z_t, logpz_t = odeint(self.odefunc, (z, logpz), times,
                              atol=self.atol, rtol=self.rtol, method=self.solver)
        return z_t[-1], logpz_t[-1]
```
