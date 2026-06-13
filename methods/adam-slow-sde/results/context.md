# Context

## Research question

Over-parameterized neural networks have many global minimizers that all achieve
(near-)zero training loss yet generalize very differently. Which minimizer an
optimizer settles into — its *implicit bias* — is therefore a central determinant of
test performance. For full-batch gradient descent and gradient flow the implicit bias
is fairly well understood (max-margin, min-norm, kernel limits). The harder and more
practically relevant question concerns *stochastic* and *adaptive* training: once the
loss is essentially zero and the iterate has reached a connected set of minimizers,
what slow force continues to act, and toward what kind of solution does it push?

For plain stochastic gradient descent this force is known: gradient noise induces a
drift that reduces a sharpness measure of the loss, and sharpness has long correlated
with generalization. But the optimizer that actually trains modern models is almost
never plain SGD — it is an adaptive method that rescales each coordinate's step by a
running estimate of the gradient's second moment. The precise problem is: **give a
rigorous, long-horizon (not just a short transient) characterization of the implicit
bias of adaptive gradient methods near a manifold of minimizers, and pin down exactly
how it differs from SGD's** — including whether the per-coordinate rescaling changes
*which* sharpness measure is reduced, and whether that helps or hurts generalization.
A solution must be valid for the full timescale over which the implicit
regularization actually plays out (empirically `O(η⁻²)` steps), must not rely on
artificial assumptions like vanishingly small gradients, and ideally should cover a
whole family of adaptive methods at once rather than a single hand-tuned case.

## Background

**Sharpness and flatness.** Flat minimizers tend to generalize better than sharp
ones; trace of the Hessian `tr(∇²L)` and related quantities are standard sharpness
measures, and explicitly penalizing sharpness (sharpness-aware minimization) improves
test accuracy. This motivates asking what sharpness, if any, an optimizer reduces on
its own.

**The minimizer manifold.** Empirically, low-loss solutions are connected (mode
connectivity), and language-model training traverses a river-valley landscape whose
floor is a continuum of near-minimizers. This motivates modeling the set of global
minimizers as a smooth, compact, `(d−m)`-dimensional submanifold `Γ` on which the
loss is locally minimized and the Hessian has constant rank `m` (full rank only in
the `m` normal directions). With a small enough learning rate, a converged optimizer
is trapped near such a `Γ`.

**Two phases.** Stochastic optimizers spend `Õ(η⁻¹)` steps converging onto `Γ`, then
linger for `O(η⁻²)` steps during which a much slower, implicit-regularization motion
dominates. A first-order ("conventional") SDE approximation
`dθ = -∇L dt + √η Σ^{1/2} dW` faithfully tracks the *convergence* phase, but its
approximation error cannot be controlled once the iterate is on the manifold — exactly
the regime where the implicit bias lives. Capturing the manifold phase requires a
different device that peels the fast convergence off and follows only the slow drift,
by projecting the trajectory onto `Γ` via the gradient-flow map `Φ` (whose Jacobian on
`Γ` is the orthogonal projector onto the tangent space `T_ζΓ`). This projected
description remains accurate for the full `O(η⁻²)` horizon.

**The drift for SGD.** For SGD this slow, projected dynamics is a stochastic process
on `Γ` whose drift is the negative *semi-gradient* (differentiate only the first
argument, holding the noise covariance fixed) of a sharpness measure built from the
third derivative of the loss and the gradient-noise covariance split into tangent and
normal parts. In words: SGD wanders on `Γ` while slowly reducing
`⟨∇²L(ζ), Σ̂_◇(ζ)⟩`, where `Σ̂_◇` is the normal-space noise weighted by the inverse
of pairwise Hessian-eigenvalue sums.

**Label noise.** A widely used analytically clean regime adds fresh `±δ` noise to each
training label at every step of an over-parameterized `ℓ₂`-regression problem. On the
zero-loss manifold this makes the gradient-noise covariance proportional to the
Hessian, `Σ(ζ) = α ∇²L(ζ)`. Under this condition SGD's slow dynamics degenerates from
a stochastic process to a deterministic flow whose fixed points are stationary points
of `tr(∇²L)` along `Γ` (shown by fixed-point analysis, by the slow-SDE reduction, and
by implicit gradient regularization — three independent routes agreeing).

**Diagnostic settings where the sharpness target matters.** Two over-parameterized
problems make the *form* of the reduced sharpness measure observable. (i) *Sparse
linear regression with a diagonal linear network*: parameters `θ=(u,v)`, estimate
`ŵ = u⊙² − v⊙²`, `ℓ₂` loss against a `κ`-sparse ground truth with `d ≫ n`. On `Γ` the
diagonal of the Hessian equals `4θ⊙²`, and the optimum forces `u_i=0` or `v_i=0`, so
the Hessian's diagonal on the manifold is read off directly from the recovered vector
`ŵ`. For SGD, minimizing `tr(H)` here corresponds to the `ℓ₁` norm of `ŵ` — the
lasso-type penalty, which recovers a sparse signal from fewer samples than a ridge-type
`ℓ₂` penalty.
(ii) *Deep matrix factorization with label noise*: minimizing `tr(H)` is known to be
roughly equivalent to minimizing the nuclear norm of the product matrix, which favors
low rank and thus generalizes well when the ground truth is low-rank. These two
landscapes make the choice of sharpness target directly visible.

**Adaptive methods.** Adaptive optimizers maintain an exponential moving average of a
per-coordinate (or per-block, per-layer, or Kronecker-factored) second moment of the
gradient and divide the momentum step by (a power of) it: a parameter-dependent
preconditioner `S`. Whether convergence onto a manifold even holds is delicate: with the
second-moment decay rate too far from 1 these methods can fail to converge, a
long-standing concern. Prior attempts to characterize their implicit bias were either
restricted to a 2-D loss with a non-rigorous quasistatic approximation, valid only for
the short `Õ(η⁻¹)` transient, dependent on the unrealistic assumption that every
gradient coordinate is below the stabilizing constant, or limited to linearly
separable data — leaving the long-horizon manifold behavior of adaptive methods open.

## Baselines

**Plain SGD (`θ_{k+1} = θ_k − η ∇ℓ_k(θ_k)`).** The reference point. Near `Γ`, its
slow projected dynamics reduces a Hessian-and-noise sharpness measure; under label
noise this is exactly `tr(∇²L)`. Gap: SGD is rarely used to train modern models, and
its analysis leans on the noise entering the update *unmodified* (and on rotational
equivariance), so it says nothing directly about preconditioned updates.

**Blanc et al. (2020) — Ornstein–Uhlenbeck analysis of label-noise SGD.** Showed
that near a zero-loss point, label-noise SGD behaves like an OU process and locally
decreases `tr(∇²L)`, moving away from a minimizer iff the trace is not locally
minimal. Gap: a *local* statement valid only for `Õ(η⁻¹·⁶)` steps and specific to
label noise — too short to capture the full regularization and not a global picture.

**Damian, Ma, Lee (2021) — label-noise SGD prefers flat minimizers.** Extended the
trace-reduction bias to constant learning rate, to any sufficiently smooth loss
satisfying a Kurdyka–Łojasiewicz condition, and to SGD with momentum. Gap: still tied
to SGD's additive-noise structure; no preconditioning, and the analysis does not
reach the `O(η⁻²)` manifold timescale in full generality.

**Li, Wang, Arora (2021) — mathematical framework after SGD reaches zero loss.**
Introduced the long-horizon projected-SDE device (via a Katzenberger-style giant-step
limit): track `Φ(θ)` on `Γ` so the fast convergence cancels, leaving a slow drift
valid for `O(η⁻²)` steps with arbitrary noise covariance; under label noise it reduces
to gradient flow on `tr(∇²L)`. Gap: the derivation is specific to SGD — it uses the
noise entering the gradient directly and the rotational structure that lets one
diagonalize the Hessian and treat `Γ` as a coordinate subspace; it does not extend to
a parameter-dependent preconditioner.

**Gu, Lyu, Huang, Arora (2023) — long-horizon SDE for local SGD.** Cast the same
device in a form designed to generalize across optimizers and supplied reusable
giant-step moment lemmas (first- and second-moment change of the projection over one
giant step). Gap: instantiated for local SGD, a communication variant; the
preconditioned case is not treated.

**Wang et al. (2023) — marginal value of momentum.** Proved SGD and SGD-with-momentum
share the *same* slow SDE at small learning rate. Useful prior: momentum (the first
moment) does not, by itself, change the implicit bias — a fact any adaptive-method
analysis can lean on. Gap: only the first moment; says nothing about the
second-moment preconditioner that distinguishes adaptive methods.

**Conventional SDEs for SGD and adaptive methods (Li et al.; Malladi et al.).**
First-order weak approximations of the *iteration itself*. Gap: accurate only on the
`Õ(η⁻¹)` convergence phase; error is uncontrolled on the manifold, so they cannot
expose the implicit bias.

**Implicit-gradient-regularization for adaptive methods (Cattaneo et al. 2024).**
Derived higher-order ODE corrections suggesting full-batch adaptive updates with
constant learning rate can *anti*-regularize sharpness when `β1 < β2`. Gap: valid only
for an `Õ(η⁻¹)` horizon and the full-batch / deterministic regime — it does not see
the long-horizon, noise-driven behavior.

**Existing convergence guarantees for adaptive methods.** Prior bounds variously
assume convexity, use `1/√t` decaying step sizes, do not vanish as `η→0`, or bound
only the *average* gradient norm or the loss *in expectation*. Gap: the manifold
analysis needs a *high-probability* bound on the *last-iterate* optimality gap that
goes to zero with `η`, holds for a whole family of adaptive methods, and is compatible
with a second-moment decay rate scaling as `Θ(η²)` — none of the prior bounds deliver
this.

## Evaluation settings

**Toy manifolds.** A 2-D elliptical `ℓ₂` loss with `±0.5` label
noise, used to visualize that a stochastic optimizer first converges to the minimizer
curve and then drifts along it toward flatter regions; a 1-D manifold cartoon
contrasting "track the whole iteration" (conventional SDE) with "track the projection
onto the manifold" (the long-horizon device).

**Sparse linear regression with a diagonal linear network.** `d = 10000`,
`κ = 50`-sparse ground truth `w*`, inputs `z_i ∈ {±1}^d`, clean labels
`y_i = ⟨z_i, w*⟩`, fresh label noise added during training, estimate
`ŵ = u⊙² − v⊙²`. Sweep the number of training points; train to convergence; a
configuration "recovers the ground truth" when the (clean) test loss falls below 1.
Metric: final test loss versus training-set size, across optimizers and
hyperparameters.

**Deep matrix factorization with label noise.** `L`-layer linear network
(`L = 2, 5`), product `W_L⋯W_1` fit by MSE to linear measurements of a low-rank
ground-truth matrix `M*`, Gaussian label noise per step. Tracked quantities over
training: `tr(H)`, `tr(Diag(H)^{1/2})`, and the train/test MSE. Standard adaptive
hyperparameters `β1=0.9`, `β2=0.999`, lr `1e-3`; SGD configured to a standard
matrix-factorization protocol.

## Code framework

The runnable scaffold is a small stochastic-optimizer harness for the diagonal-network
diagnostic. The optimizer update rule and the optimizer factory are left as open slots.

```python
import torch

class CoordinateRescaledOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999,
                 eps=1e-8, exponent=0.5):
        pass  # TODO: first moment, second statistic, and coordinate rescaling

    @torch.no_grad()
    def step(self, closure=None):
        pass  # TODO: update state and apply the rescaled stochastic step

def make_optimizer(name, params, lr, exponent=0.5):
    pass  # TODO: choose SGD or the coordinate-rescaled optimizer

def make_diagonal_net(d, kappa, seed=0):
    g = torch.Generator().manual_seed(seed)
    w_star = torch.zeros(d)
    idx = torch.randperm(d, generator=g)[:kappa]
    w_star[idx] = torch.randn(kappa, generator=g)
    u = torch.full((d,), 0.1, requires_grad=True)
    v = torch.full((d,), 0.1, requires_grad=True)

    def predict(z):
        return (z * (u.square() - v.square())).sum()

    return [u, v], predict, w_star

def label_noise_step(predict, z, y_clean, delta, opt, gen):
    noisy = y_clean + delta * (2 * torch.randint(0, 2, (1,), generator=gen).item() - 1)
    loss = 0.5 * (predict(z) - noisy) ** 2
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()

def run_diagnet(opt_name, n_train, d=10000, kappa=50, delta=0.1,
                steps=20000, lr=1e-2, exponent=0.5, seed=0):
    gen = torch.Generator().manual_seed(seed + 1)
    params, predict, w_star = make_diagonal_net(d, kappa, seed)
    Z = (torch.randint(0, 2, (n_train, d), generator=gen) * 2 - 1).float()
    y = Z @ w_star
    Ztest = (torch.randint(0, 2, (2000, d), generator=gen) * 2 - 1).float()
    ytest = Ztest @ w_star
    opt = make_optimizer(opt_name, params, lr, exponent)

    for _ in range(steps):
        i = torch.randint(0, n_train, (1,), generator=gen).item()
        label_noise_step(predict, Z[i], y[i], delta, opt, gen)

    with torch.no_grad():
        u, v = params
        w_hat = u.square() - v.square()
        test_loss = 0.5 * ((Ztest @ w_hat - ytest) ** 2).mean()
    return test_loss.item()
```
