## Research question

We want to understand which functions can be learned by *standard* stochastic gradient descent
on *standard* (regular) neural networks — not networks hand-built to emulate a specific algorithm,
and not networks pushed into a regime where training collapses to a linear model. The clean
testbed is *sparse* functions on the Boolean hypercube. The covariate is `x ~ Unif({+1,-1}^d)`
with `d` arbitrarily large, but the label depends on only a hidden handful of `P` coordinates:
`f*(x) = h*(z)`, where `z = x_I = (x_{i_1},...,x_{i_P})` for an unknown latent subset `I ⊆ [d]`,
`|I| = P`, and `h* : {+1,-1}^P -> R`. The learner does not know `I`.

The precise goal is a *necessary and sufficient* characterization: for which latent functions `h*`
can a two-layer network trained by one-pass batch-SGD reach arbitrarily small test error using only
`O(d)` samples, *uniformly over every choice of the hidden subset `I`*, as `d -> infinity`? Two
3-sparse targets make the question sharp:

```
h*(z) = z1 z2 z3            (a single degree-3 monomial)
h*(z) = z1 + z1 z2 + z1 z2 z3   (a "staircase" of growing degree)
```

Both depend on only three coordinates, and both contain a degree-3 term, so a fixed-feature
(kernel / random-feature) method needs `Omega(d^3)` samples for either. The open question is whether
SGD on a network *separates* them — whether one is `O(d)`-learnable and the other is not — and what
structural property of `h*` decides it. A solution has to (1) go beyond the linear/lazy regime so the
network genuinely builds features adapted to the unknown `I`, (2) come with a *necessity* direction
(some sparse functions must be hard), and (3) give complexity finer than "polynomial in `d`".

## Background

By this time the picture of SGD on wide networks is understood at two opposite scalings of the
output, and the interesting regime sits between them.

**The lazy / linear scaling.** With one parametrization of a wide network, training barely moves the
weights: the network stays close to its first-order Taylor expansion around initialization and behaves
like a *fixed* kernel method (Jacot, Gabriel & Hongler 2018). Here the feature map is frozen at
initialization, so the network cannot discover the latent subset `I`; learning a degree-`k` component
of a sparse function then costs `Omega(d^k)` samples, with no adaptation to sparsity.

**The mean-field scaling.** Write a two-layer network with the `1/N` normalization
`fhat(x;Theta) = (1/N) sum_{j in [N]} a_j sigma(<w_j,x>)`. Under this scaling the weights move an
`O(1)` amount and the dynamics is genuinely nonlinear (Mei, Montanari & Nguyen 2018; Chizat & Bach
2018; Rotskoff & Vanden-Eijnden 2018; Sirignano & Spiliopoulos 2020). The key object is the empirical
distribution of the `N` weight vectors, `rhohat^{(N)} = (1/N) sum_j delta_{theta_j}`. Mei-Montanari-
Nguyen show that the population risk is a function of this distribution alone,
`R(rho) = R_# + 2 int V(theta) rho(dtheta) + int U(theta,theta') rho(dtheta) rho(dtheta')`, with
`V(theta) = -E[y sigma(x;theta)]` and a symmetric PSD interaction `U(theta1,theta2) =
E[sigma(x;theta1) sigma(x;theta2)]`. As `N -> infinity` and step size `eps -> 0`, one-pass SGD with
step `s_k = eps xi(k eps)` drives `rhohat^{(N)}` to a measure `rho_t` that solves a continuity equation
— a Wasserstein gradient flow on `R(rho)` in the space of probability measures:

```
partial_t rho_t = 2 xi(t) nabla_theta . ( rho_t nabla_theta Psi(theta; rho_t) ),
Psi(theta; rho) = V(theta) + int U(theta,theta') rho(dtheta').
```

Intuitively the `N` neurons are particles of a gas whose density `rho_t` descends the risk while
conserving mass locally. This "distributional dynamics" factors out the permutation symmetry of the
neurons and lets one exploit symmetries of the data distribution. Non-asymptotic versions (Mei,
Misiakiewicz & Montanari 2019) make the SGD-to-PDE approximation quantitative with dimension-free
rates. A standard tool throughout is the **Fourier-Walsh** decomposition on the hypercube: any
`h* : {+1,-1}^P -> R` is `h*(z) = sum_{S ⊆ [P]} hhat(S) chi_S(z)` with monomials
`chi_S(z) = prod_{i in S} z_i`, orthonormal under `z ~ Unif`, `<f,g> = E_z[f(z) g(z)]`. The "degree"
of a component is `|S|`, and a degree-`k` monomial is the canonical hard case.

**The diagnostic phenomenon: hierarchical pickup.** It has been observed that regular networks learn
*hierarchically* — a high-level concept is assembled from simpler ones learned first. Abbe, Boix-Adsera,
Brennan, Bresler & Nagaraj (2021) made this concrete on the hypercube with the **staircase function**
`S_k(x) = x1 + x1 x2 + x1 x2 x3 + ... + x1 ... xk`, a sum of monomials of increasing degree where each
support extends the previous one by a single coordinate. Training a fixed regular network with SGD on
the square loss, they observe a dramatic split: the network learns `S_10` to vanishing error while it
*cannot* learn the isolated degree-10 monomial `chi_{1..10}` at all — and when one plots the network's
Fourier coefficients against training iteration, on the staircase target they come up *in order of
increasing degree* (degree 1 first, then degree 2, ...), whereas on the bare monomial no coefficient
ever moves. The reading of this is that the degree-1 part is learned first, which makes the degree-2
part easier to pick up, and so on — the network "climbs the staircase." This is a pre-method fact about
how SGD behaves on structured Boolean targets.

**Why fixed features can't compete.** Lower bounds for linear methods (Ghorbani, Mei, Misiakiewicz &
Montanari 2021; Hsu, Sanford, Servedio & Vlatakis-Gkaragkounis 2021; Kamath, Montasser & Srebro 2020)
establish that any kernel or random-feature method with `poly(d)` features or samples cannot learn
degree-`P` monomials when `P` grows with `d`. For staircases, the usual statistical-query route to such
bounds is not tight — a staircase is efficiently SQ-learnable by querying its monomials in increasing
degree — so separating staircases from linear methods requires a different lower-bound technique.

## Baselines

**Lazy / NTK random-feature training (Jacot, Gabriel & Hongler 2018; Ghorbani et al. 2021).** Take the
same two-layer architecture but freeze the first layer at a random initialization (`w ~ N(0, I_d/d)`,
random biases) and train only the linear readout. The network is then exactly a random-feature model:
`fhat(x) = sum_j a_j phi_j(x)` with `phi_j(x) = sigma(<w_j,x>)` fixed, and the readout is fit by linear
least squares. Core math: with frozen features the optimization is convex and the model lives in the span
of a *fixed* feature map. **Limitation:** the feature map is chosen before seeing which coordinates are
relevant, so it spends representational budget uniformly over all `d` inputs; to express a degree-`k`
component over an unknown subset it needs `min(n, q) = Omega(d^k)` samples-or-features. It does not move
its weights toward `I`, so it cannot exploit the sparsity, and it learns the staircase and the bare
monomial at the same prohibitive cost.

**Layerwise coordinate-descent training of deep sparse nets on staircases (Abbe et al. 2021).** This
prior work proves that staircase functions are learnable by *regular* networks, establishing that
hierarchical structure is exploitable in principle. Core idea: a deep (e.g. `P`-layer) network with
sparse random connectivity, trained by a layerwise *stochastic coordinate descent* variant of SGD,
builds the degree-`k` feature at depth `k` from the degree-`(k-1)` feature below it. **Limitations,
stated as where it stops short:** the construction needs depth that grows with the degree and a training
algorithm that is not plain SGD (sparse layers plus coordinate descent rather than a standard optimizer);
its guarantee is only "polynomial in `d`" with no finer rate; it covers only single nested chains where
each support grows by exactly one (no merging of separately-grown chains); and it gives a sufficiency
result with no matching necessity — it does not say which sparse functions are *out of reach*.

**Mean-field analyses keyed to rotational symmetry (Mei, Montanari & Nguyen 2018; Mei et al. 2019).**
For special data distributions (e.g. classifying isotropic or anisotropic Gaussians) the mean-field PDE
admits a low-dimensional reduction and one can prove global convergence. Core math: the Wasserstein
gradient flow above, with the low-dimensional reduction coming from the *rotational invariance* of the
Gaussian problem. **Limitation:** the reduction is driven by symmetry of the input distribution, not by
sparsity of the target; the dimension `d` does not really shrink for a sparse Boolean target, and these
results give no sample-complexity statement tied to the structure of `h*`. More broadly, generic global-
convergence results for the mean-field flow (Chizat & Bach 2018; Nguyen 2020; Wojtowytsch 2020) need a
"spread-out" initialization with density on an open set around the origin and an assumption that the flow
converges to a limit — assumptions that are exactly what is in question here.

## Evaluation settings

The natural yardstick is fixed before any method is proposed. Data are `x ~ Unif({+1,-1}^d)` with
labels `y = f*(x) + noise`, `f*(x) = h*(x_I)` for a latent `P`-subset `I`. One trains a two-layer
network of width `N` by one-pass batch-SGD on the square loss: at each step a fresh batch of size `b`
is drawn (so `n = b * (#steps)` total samples and no example is reused), the readout and first layer
are updated with step sizes `(eta^a_k, eta^w_k)`, optionally with `ell_2` regularization. Representative
configuration used to compare the SGD trajectory against its limiting dynamics: `d = N = 100`, batch
`b = 150`, step size `eta = 1/2`, no regularization, readout weights from `Unif([-1,1])`, scaled
first-layer coordinates satisfying `sqrt(d) w_{j,l} ~ N(0,1)`, and shifted-sigmoid activation.
Target families span the hierarchy: a vanilla staircase
`z1 + z1 z2 + z1 z2 z3 + ...`, chains that merge two staircases, and isolated monomials such as
`z1 z2 z3`. Metrics are the population square error (test MSE) on held-out samples and the recovery of
the target's individual Fourier coefficients over the latent monomials, tracked as a function of the
number of SGD steps / samples. The yardstick is the `O(d)`-sample budget: does the test error fall to
`eps` uniformly over the choice of `I` within `O(d)` samples?

## Code framework

The procedure plugs into a standard one-pass SGD training harness that already exists. The data sampler
(`x ~ Unif({+1,-1}^d)`, label `h*(x_I)`), the square loss, and the evaluation (test MSE, Fourier
recovery) are fixed by the harness. What is *not* settled is the network to instantiate, the optimizer
to wrap around it, and the per-step update — those are exactly the slots a learning procedure for sparse
functions has to fill. The scaffold is therefore a bare two-layer-network harness with three empty
stubs.

```python
import torch
import torch.nn as nn


def build_model(config) -> nn.Module:
    """Return the trainable two-layer network. Input is [batch, d] with entries in
    {+1,-1}; output is [batch] (or [batch, 1]). Width is config.width, input dim config.d.
    Architecture, parametrization, and initialization are to be designed."""

    class TwoLayerNet(nn.Module):
        def __init__(self, d: int, M: int) -> None:
            super().__init__()
            # A two-layer network: first layer d -> M, readout M -> 1.
            # TODO: the architecture / parametrization / initialization we will design
            #       (output scaling, activation, how each layer is initialized).
            self.M = M

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # TODO: the forward map we will design.
            raise NotImplementedError

    return TwoLayerNet(d=config.d, M=config.width)


def get_optimizer(model: nn.Module, config) -> torch.optim.Optimizer:
    """Return the optimizer used to update model.parameters()."""
    # TODO: the optimizer and step size we will choose.
    raise NotImplementedError


def train_step(model, optimizer, x, y) -> float:
    """One gradient update on a fresh batch (x, y). Return the scalar training loss."""
    optimizer.zero_grad(set_to_none=True)
    pred = model(x).view(-1)
    loss = ((pred - y) ** 2).mean()          # fixed square loss vs the target value
    loss.backward()
    optimizer.step()
    return float(loss.item())


# existing one-pass training loop the procedure plugs into
def train(model, optimizer, sampler, T):
    for _ in range(T):                        # one pass: a fresh batch every step
        x, y = sampler.sample_batch()         # x ~ Unif({+1,-1}^d), y = h*(x_I) + noise
        train_step(model, optimizer, x, y)
```

The three stubs — the network (its parametrization/initialization and output scaling), the optimizer,
and the update — are what the procedure fills in.
