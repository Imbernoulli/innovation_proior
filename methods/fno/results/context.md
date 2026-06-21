## Research question

A great many problems in science and engineering reduce to solving the *same* partial differential equation over and over, each time for a different input function — a different coefficient field, forcing, or initial condition. In airfoil design, subsurface flow, micro-mechanics, and turbulence one must evaluate a forward model thousands of times while sweeping parameters or sampling a posterior. Each of those evaluations, with classical machinery, is an independent solve on a fine mesh.

Write the input function as `a` and the resulting solution as `u`, both functions on a bounded domain `D ⊂ R^d`. The map we actually care about is the **solution operator**

```
G† : a ↦ u,
```

a map between two infinite-dimensional spaces of functions. The question is whether we can *learn* this operator once, from a finite set of observed pairs `{a_j, u_j}`, so that a new instance costs a single forward evaluation rather than a fresh PDE solve, and how to do so in a way that is tied to function space rather than to one fixed mesh.

## Background

**The continuum object.** Let `A = A(D; R^{d_a})` and `U = U(D; R^{d_u})` be (separable Banach) spaces of functions on `D`. The target `G† : A → U` is typically nonlinear, even when the underlying PDE is linear: for steady Darcy flow `−∇·(a∇u) = f`, the equation is linear in `u` but the coefficient-to-solution map `a ↦ u` is not. We have access to data only through point samples: a discretization `D_j = {x_1,…,x_n} ⊂ D` and the evaluations `a_j|_{D_j}`, `u_j|_{D_j}`.

**Green's functions: the solution operator of a linear PDE is an integral operator.** For a linear differential operator `L` with `Lu = f`, the solution is

```
u(x) = ∫_D G(x, y) f(y) dy,
```

where `G` is the Green's function — the response at `x` to a unit point source at `y`. When `L` has constant coefficients the Green's function is translation invariant, `G(x, y) = G(x − y)`, and the solve becomes a **convolution** `u = G * f`. This is the classical fact that an entire family of solutions (over all forcings `f`) is encoded by a single kernel, and that for translation-invariant problems that kernel is a convolution.

**The convolution theorem and spectral methods.** Spectral PDE solvers exploit that differentiation becomes multiplication under the Fourier transform, and more generally that **convolution becomes pointwise multiplication**: with `F` the Fourier transform,

```
F(κ * v) = F(κ) · F(v),     κ * v = F⁻¹( F(κ) · F(v) ).
```

A convolution, which costs `O(N²)` if done directly, can therefore be computed by two Fourier transforms and a pointwise product; on a uniform grid the Fast Fourier Transform does each transform in `O(N log N)`.

**Spectral content of the data.** The solution fields encountered here have Fourier spectra that decay with wavenumber; for turbulent Navier–Stokes the energy spectrum follows roughly a `k^{−5/3}` slope.

## Baselines

**Classical numerical solvers (FEM / FDM / pseudospectral).** Discretize `D` and solve the resulting algebraic system for one instance. They are accurate and need no data; coarse meshes are fast, fine meshes are accurate, and each run solves a single instance.

**Finite-dimensional neural surrogates.** Parameterize `G†` as a deep convolutional network between Euclidean spaces `R^n → R^n` (image-to-image regression: fully-convolutional nets, U-Net, ResNet, and turbulence-specific spatio-temporal convolutional nets). They amortize across instances and are fast at inference; the parameters are tied to the training discretization and geometry.

**Neural-FEM / physics-informed networks.** Represent the *solution function* `u` itself as a neural network and fit it by minimizing the PDE residual (Deep Ritz; physics-informed neural networks). They are mesh-free and replace a finite basis with the space of networks; each one models a single instance, fitting a fresh optimization problem per coefficient/initial condition given the governing equation.

**Finite-dimensional operator surrogates via reduced bases.** Encode `a` and `u` into finite latent spaces — PCA / POD reduced-basis methods, or a learned PCA autoencoder with a network mapping between latent codes — using a linear basis whose dimension is chosen up front.

**Low-rank / branch–trunk operator networks.** Approximate the kernel by a finite sum of separable terms `κ(x, y) = Σ_{j=1}^r φ_j(x) ψ_j(y)`, equivalently a branch network reading samples of `a` at fixed sensor locations times a trunk network of the query point `x`. The output is mesh-free.

**Graph / nonlocal neural operators.** Cast the operator as an iterative architecture that lifts the input to a higher channel dimension, applies several layers, and projects back. Each layer composes a pointwise linear map with a **kernel integral operator**

```
(K(a) v)(x) = ∫_D κ(x, y, a(x), a(y)) v(y) dy,
```

where `κ` is itself a learned neural network, so that stacking the linear integral operator with pointwise nonlinearities yields a nonlinear operator — the function-space analogue of "linear layer + activation." It is mesh-free and transfers between discretizations, evaluating the integral by message passing / Nyström quadrature on a graph over the sample points (a multi-scale/multipole variant). The integral couples every evaluation point to every other, so a layer is `O(N²)` in the number of points.

## Evaluation settings

Three parametric PDE families serve as the testbed, each posed as an operator-learning problem with data given as point samples on a grid (lower resolutions sub-sampled from a fine reference solve).

- **1-D viscous Burgers' equation**, `∂_t u + ∂_x(u²/2) = ν ∂_xx u` on the unit interval with periodic boundaries; learn the map from the initial condition `u_0` to the solution at time one. Reference solve at `s = 8192` (split-step in Fourier space), sub-sampled to `s ∈ {256, …, 8192}`. Metric: relative `L²` error.
- **2-D steady Darcy flow**, `−∇·(a∇u) = f` on the unit square with Dirichlet boundary; learn the diffusion-coefficient-to-solution map `a ↦ u`. Coefficients drawn from a thresholded Gaussian-random-field measure; reference solve on a `421 × 421` grid by second-order finite differences, sub-sampled to `s ∈ {85, 141, 211, 421}`. Metric: relative `L²` error.
- **2-D Navier–Stokes in vorticity form** on the unit torus, `∂_t w + u·∇w = ν Δw + f`, `∇·u = 0`; learn the map from the vorticity over an initial time window to the vorticity over a later window. Initial vorticity from a Gaussian-random-field measure, fixed forcing, viscosities `ν ∈ {1e−3, 1e−4, 1e−5}` with the final time shortened as the flow becomes chaotic; pseudospectral data on `256 × 256`, sub-sampled to `64 × 64`. Metric: relative `L²` error; train and test fixed at `64 × 64` for the grid-bound baselines.

The natural yardsticks, by family: for the time-independent problems, a pointwise feedforward net, a reduced-basis method, a fully-convolutional net, a PCA-autoencoder operator, the graph and multipole graph operators, and a low-rank operator; for the time-dependent problem, an 18-layer residual convolutional net, a U-Net, and a turbulence-specific spatio-temporal convolutional net. A downstream test is a function-space MCMC for a Bayesian inverse problem (recovering the initial vorticity from sparse noisy late-time observations), where the learned operator stands in as the forward surrogate. Reported metric throughout is relative `L²` error; wall-clock per evaluation and parameter count are also recorded. Hardware: a single 16 GB GPU.

## Code framework

The primitives that already exist: a tensor/autodiff library with an FFT (`torch.fft`), standard layers (`nn.Linear`, the 1×1 convolutions `nn.Conv{1,2,3}d`), pointwise nonlinearities, the Adam optimizer with a step learning-rate schedule, and a relative-`L²` loss for function-valued regression. Data arrives as point samples on a uniform grid, with the grid coordinates available to concatenate as extra input channels. The scaffold below leaves one layer unspecified.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalOperator2d(nn.Module):
    """A global layer mapping one sampled 2-D function representation to another."""
    def __init__(self, in_channels, out_channels, *operator_hparams):
        super().__init__()
        # TODO: parameters for the global linear operator on the spatial grid.
        pass

    def forward(self, v):
        # v: (batch, channels, x, y) samples of a vector-valued function.
        # TODO: apply the global linear operator.
        pass


class OperatorNet2d(nn.Module):
    """Lift to channels, alternate global and pointwise linear maps, project back."""
    def __init__(self, in_channels=1, out_channels=1, width=32, *operator_hparams, padding=0):
        super().__init__()
        self.width = width
        self.padding = padding
        self.fc0 = nn.Linear(in_channels + 2, width)  # lift P
        self.global0 = GlobalOperator2d(width, width, *operator_hparams)
        self.global1 = GlobalOperator2d(width, width, *operator_hparams)
        self.global2 = GlobalOperator2d(width, width, *operator_hparams)
        self.global3 = GlobalOperator2d(width, width, *operator_hparams)
        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)
        self.fc1 = nn.Linear(width, 128)   # projection Q
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, values):
        # values: (batch, x, y, in_channels)
        grid = self.get_grid(values.shape, values.device)
        x = torch.cat((values, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)  # channels first for spatial operators
        if self.padding:
            x = F.pad(x, [0, self.padding, 0, self.padding])
        for global_layer, local_layer in [(self.global0, self.w0), (self.global1, self.w1),
                                          (self.global2, self.w2), (self.global3, self.w3)]:
            x = global_layer(x) + local_layer(x)
            if global_layer is not self.global3:
                x = F.gelu(x)
        if self.padding:
            x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)  # channels last for pointwise projection
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batch, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.linspace(0, 1, size_x, device=device).reshape(1, size_x, 1, 1)
        gridx = gridx.repeat(batch, 1, size_y, 1)
        gridy = torch.linspace(0, 1, size_y, device=device).reshape(1, 1, size_y, 1)
        gridy = gridy.repeat(batch, size_x, 1, 1)
        return torch.cat((gridx, gridy), dim=-1)


def relative_l2(pred, target):
    # function-space error: per-sample L2 norm of the difference over L2 norm of target
    diff = (pred - target).reshape(pred.size(0), -1).norm(dim=1)
    base = target.reshape(target.size(0), -1).norm(dim=1)
    return (diff / base).mean()


def train(model, train_loader, y_normalizer=None, epochs=500, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.5)
    for _ in range(epochs):
        for a, u in train_loader:
            opt.zero_grad()
            pred = model(a).reshape_as(u)
            target = u
            if y_normalizer is not None:
                pred = y_normalizer.decode(pred)
                target = y_normalizer.decode(target)
            loss = relative_l2(pred, target)
            loss.backward()
            opt.step()
        sched.step()
```
