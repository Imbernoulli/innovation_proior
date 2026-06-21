# Context

## Research question

We want a generative model that simultaneously (1) computes **exact log-likelihoods** (so it can be trained by maximum likelihood and evaluated by held-out likelihood), (2) **samples efficiently in a single pass**, and (3) places few restrictions on the neural-network architecture of its transformation. Change-of-variables (flow) models give (1) and (2) by warping a base density through an invertible map, at the cost of a Jacobian determinant. The question: how to specify such a flow so that the transformation `f` can be a free-form neural net while exact likelihood and one-pass sampling are retained, at a cost that scales with dimension `D`.

## Background

**Change of variables.** A complex normalized density can be specified implicitly by warping a simple base `p_z` through an invertible `f: R^D → R^D`. If `z ∼ p_z` and `x = f(z)`,
`log p_x(x) = log p_z(z) − log |det (∂f(z)/∂z)|`.
The log-determinant of the `D×D` Jacobian costs `O(D³)` in general. "Reversible generative models" are those that use this formula while keeping *both* density evaluation and sampling efficient in a single pass.

**The ways prior flows make the determinant cheap.**
- *Restricted functional forms / normalizing flows* (Rezende & Mohamed 2015 planar flows; Berg et al. 2018 Sylvester): pick `f` so a determinant identity applies. A planar flow is a one-layer net with a single hidden unit per layer. They are typically used as variational posteriors.
- *Autoregressive transforms* (IAF, Kingma et al. 2016; MAF, Papamakarios et al. 2017; TAN, Oliva et al. 2018): impose an ordering on dimensions so the Jacobian is triangular and `det = ∏ diagonal`. Inverting (sampling) proceeds through `D` sequential evaluations.
- *Partitioned / coupling layers* (NICE, Dinh et al. 2014; Real NVP, Dinh et al. 2016; Glow, Kingma & Dhariwal 2018): split the dimensions and affinely transform one block conditioned on the other, giving a triangular determinant and an inverse that costs the same as the forward map. Convolution-friendly; many layers are stacked.

**Continuous normalizing flows (Chen et al. 2018, Neural ODEs).** Replace the discrete stack of layers with continuous-time dynamics: sample `z(t_0) ∼ p_{z0}`, define an ODE `∂z(t)/∂t = f(z(t), t; θ)` with `f` a neural net, and integrate the initial-value problem to `z(t_1) = x`. The change in log-density along the trajectory obeys the **instantaneous change of variables**:
`∂ log p(z(t))/∂t = − Tr(∂f/∂z(t))`,
so `log p(z(t_1)) = log p(z(t_0)) − ∫_{t_0}^{t_1} Tr(∂f/∂z(t)) dt`. In continuous time the per-step object is `Tr(∂f/∂z)` rather than a log-determinant. Existence and uniqueness of the ODE solution require `f` and its first derivatives to be Lipschitz, satisfied by smooth Lipschitz activations. Computing the trace exactly costs `O(D²)` (each diagonal entry is a separate derivative of `f`), so per-step cost is `O(D²)` rather than the `O(D³)` per-layer log-determinant.

**Backprop through the ODE: the adjoint method** (Pontryagin; Chen et al. 2018). For a scalar loss `L(z(t_1))`, the adjoint `a(t) = −∂L/∂z(t)` and the parameter gradient
`dL/dθ = − ∫_{t_1}^{t_0} (∂L/∂z(t))ᵀ (∂f/∂θ) dt`
are obtained by solving a second ODE backward in time. This is the continuous-time analog of backpropagation and uses `O(1)` memory (no stored activations), so very large batch sizes become feasible.

**Stochastic trace estimation (Hutchinson 1989).** For any `D×D` matrix `A` and any distribution `p(ε)` with `E[ε]=0`, `Cov(ε)=I`,
`Tr(A) = E_{p(ε)}[ εᵀ A ε ]`.
Typical `ε`: standard Gaussian or Rademacher (`±1` entries). Reverse-mode autodiff computes a vector-Jacobian product `vᵀ(∂f/∂z)` for about the cost of one evaluation of `f`.

**Other generative families (for contrast, not change-of-variables).** GANs (Goodfellow et al. 2014) use unrestricted nets but have no closed-form likelihood (need a discriminator). Autoregressive models (MADE, PixelRNN) give exact likelihood but need `O(D)` passes to sample. VAEs (Kingma & Welling 2013) use unrestricted nets and give a lower bound on the marginal likelihood.

## Baselines

- **Real NVP / Glow** (coupling-layer flows): train on data, one-pass sample, exact likelihood; the Jacobian is partitioned/triangular, with many stacked coupling layers and invertible `1×1` convolutions.
- **Autoregressive flows (MAF, IAF, TAN, MAF-DDSF)**: exact likelihood, strong on tabular density estimation; sampling/inversion is `O(D)` sequential.
- **Planar / Sylvester normalizing flows**: cheap determinants via functional-form restriction, used as variational posteriors.
- **Planar CNF** (Chen et al. 2018): continuous flow with the trace formula, one-pass sampling and exact likelihood, exact trace at `O(D²)`.

## Evaluation settings

- **Toy 2D densities** (eight-Gaussians, checkerboard, two-spirals): warp an isotropic Gaussian to multimodal and near-discontinuous targets; qualitative density/sample visualization, compared against a 100-layer Glow.
- **Tabular density estimation** (POWER, GAS, HEPMASS, MINIBOONE, BSDS300, preprocessed as in Papamakarios et al. 2017): negative log-likelihood in nats.
- **Image density estimation** (MNIST, CIFAR-10): bits/dim, with single-flow encoder-decoder and multiscale (squeeze-based) architectures à la Real NVP/Glow.
- **Variational inference** (VAE posteriors on MNIST, Omniglot, Frey Faces, Caltech Silhouettes): negative ELBO, encoder/decoder mirroring Berg et al. 2018, comparing flow families for the approximate posterior.
- **Protocol / solver:** adaptive Runge-Kutta 4(5) (Dormand-Prince) ODE solves for continuous flows; maximum-likelihood training with Adam; likelihood evaluation by integrating a state and its log-density change between data time and base time.

## Code framework

The available pieces are a black-box adaptive ODE solver with an adjoint backward pass (`torchdiffeq.odeint_adjoint`), reverse-mode autodiff for vector-Jacobian products (`torch.autograd.grad`), PyTorch modules, smooth activations, and a fixed Gaussian base distribution. A continuous-flow harness already has places for a time-dependent dynamics network, a trace computation, an augmented ODE function, and an ODE-solve wrapper.

```python
import numpy as np
import torch, torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

from . import diffeq_layers
from .squeeze import squeeze, unsqueeze

class Swish(nn.Module):
    # TODO: optional smooth activation.
    pass

class Lambda(nn.Module):
    # TODO: wrapper for simple activation functions.
    pass

NONLINEARITIES = {}

class ODEnet(nn.Module):
    """Dynamics dz/dt = f(z(t), t; theta)."""
    def __init__(self, hidden_dims, input_shape, strides=None, conv=False,
                 layer_type="concat", nonlinearity="softplus", num_squeeze=0):
        # TODO: choose time-conditioned linear/convolutional layers and activations.
        pass

    def forward(self, t, y):
        # TODO: return the state derivative.
        pass

def divergence_bf(dx, y, **unused_kwargs):
    # TODO: divergence computation.
    pass

def divergence_approx(f, y, e=None):
    # TODO: divergence computation.
    pass

def sample_rademacher_like(y):
    # TODO.
    pass

def sample_gaussian_like(y):
    # TODO.
    pass

class ODEfunc(nn.Module):
    """Augmented dynamics for (state, accumulated log-density change)."""
    def __init__(self, diffeq, divergence_fn="approximate",
                 residual=False, rademacher=False):
        super().__init__()
        # TODO: store the dynamics and select the trace computation.
        pass

    def before_odeint(self, e=None):
        # TODO: reset per-solve state.
        pass

    def forward(self, t, states):
        # TODO: return (dy/dt, dlogp/dt).
        pass

class CNF(nn.Module):
    """Solve the augmented IVP forward or backward in time."""
    def __init__(self, odefunc, T=1.0, train_T=False,
                 solver="dopri5", atol=1e-5, rtol=1e-5):
        super().__init__()
        # TODO: configure end time, solver, and tolerances.
        pass

    def forward(self, z, logpz=None, integration_times=None, reverse=False):
        # TODO: integrate (z, logpz) and return the final state.
        pass

def _flip(x, dim):
    # TODO: reverse an integration-time tensor along one dimension.
    pass
```
