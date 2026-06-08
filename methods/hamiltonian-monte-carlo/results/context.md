# Context

## Research question

I have a continuous target distribution π(q) over q ∈ R^d — most often a Bayesian
posterior, π(q) ∝ prior(q) · likelihood(q | data) — and I want expectations
E_π[f] = ∫ f(q) π(q) dq. I can evaluate π only up to its unknown normalizing
constant, and I can evaluate its gradient ∇ log π(q). The integral is analytically
intractable, so I must estimate it by drawing (correlated) samples q₁, …, q_N and
averaging f over them.

The difficulty is dimension. The probability **mass** of a high-dimensional
distribution does not sit near the mode; it concentrates in a thin shell — the
"typical set" — at an intermediate distance from the mode. This is a volume effect:
the density is largest at the mode, but the volume of a neighborhood at radius δ
grows fast enough with d that the product density × volume, which is what
expectations integrate, peaks away from the mode and is negligible both at the mode
and far out in the tails. Any estimator must put essentially all of its evaluations
inside this thin set.

A sampler that moves through state space by small, undirected, isotropic steps
explores such a thin set as a random walk: net displacement grows only like the
square root of the number of steps, so the number of steps to reach a nearly
independent state grows with dimension. The goal is a transition that instead moves
a long, *coherent* distance along the typical set per iteration — covering it
systematically rather than diffusing — while still leaving π invariant and keeping
the per-step acceptance probability high.

## Background

**The canonical / Boltzmann distribution.** From statistical mechanics, a system
with energy function E(x) at temperature T has equilibrium density
P(x) = (1/Z) exp(−E(x)/T). Read backwards: any density P(x) can be written as a
canonical distribution at T = 1 by setting the energy E(x) = −log P(x) − log Z, for
any convenient constant Z. So "sample from π" and "sample the Boltzmann distribution
of energy U(q) = −log π(q)" are the same problem. Sampling probability and
minimizing/exploring an energy landscape are two views of one object.

**Hamiltonian mechanics.** A physical system with position q ∈ R^d and momentum
p ∈ R^d is governed by a Hamiltonian H(q, p) (its total energy). Its time evolution
obeys Hamilton's equations
  dqᵢ/dt = ∂H/∂pᵢ,   dpᵢ/dt = −∂H/∂qᵢ.
Three properties of this flow are classical and load-bearing here:
- **Conservation of energy:** dH/dt = Σᵢ (∂H/∂qᵢ · dqᵢ/dt + ∂H/∂pᵢ · dpᵢ/dt)
  = Σᵢ (∂H/∂qᵢ · ∂H/∂pᵢ − ∂H/∂pᵢ · ∂H/∂qᵢ) = 0. The trajectory stays on a level
  set of H.
- **Volume preservation (Liouville's theorem):** the flow's vector field has zero
  divergence, Σᵢ (∂/∂qᵢ)(dqᵢ/dt) + (∂/∂pᵢ)(dpᵢ/dt)
  = Σᵢ (∂²H/∂qᵢ∂pᵢ − ∂²H/∂pᵢ∂qᵢ) = 0, so the map from the state at time t to the
  state at time t+s has Jacobian determinant of absolute value 1; it neither
  compresses nor expands phase-space volume.
- **Reversibility / symplecticness:** the flow is a one-to-one map with an inverse
  obtained by negating the time derivatives; equivalently, for a Hamiltonian of the
  form H(q,p) = U(q) + K(p) with K(p) = K(−p), the inverse is "negate p, run the
  flow, negate p again." The Jacobian B_s satisfies Bₛᵀ J⁻¹ Bₛ = J⁻¹ with
  J = [[0, I], [−I, 0]], which implies det(Bₛ)² = 1.

**Molecular dynamics.** Deterministic Newtonian simulation of a many-body system
(Alder & Wainwright 1959): integrate the equations of motion forward in time. This
gives long, directed motion through configuration space, but on its own it is an
approximation — a finite-stepsize integrator does not conserve energy exactly, so it
does not sample any distribution exactly.

**Numerical integration of Hamiltonian flow.** Hamilton's equations must be
discretized with some stepsize ε. The naive choice, Euler's method
  pᵢ(t+ε) = pᵢ(t) − ε ∂U/∂qᵢ(q(t)),  qᵢ(t+ε) = qᵢ(t) + ε pᵢ(t)/mᵢ,
is unstable: applied to a simple harmonic oscillator it spirals outward to infinity,
because it does not preserve phase-space volume. A "leapfrog" / Störmer–Verlet
integrator (Verlet 1967, with antecedents going back to Störmer 1907), studied in
the framework of symplectic integrators (De Vogelaere; Ruth 1983; Leimkuhler & Reich
2004, *Simulating Hamiltonian Dynamics*), interleaves position and momentum updates
in a way that preserves volume and is time-reversible even at finite ε, and whose
energy error stays bounded rather than accumulating. Such integrators exactly solve
a nearby "shadow" Hamiltonian.

## Baselines

**Random-walk Metropolis (Metropolis, Rosenbluth, Rosenbluth, Teller & Teller
1953).** To sample P(x) ∝ exp(−E(x)), build a Markov chain: from x, propose
x* = x + N(0, ς²I) (a symmetric, isotropic perturbation), and accept x* with
probability min[1, exp(−(E(x*) − E(x)))] = min[1, P(x*)/P(x)]; otherwise stay at x.
The symmetric proposal makes the proposal density cancel, and the accept rule
enforces detailed balance P(x) T(x*|x) = P(x*) T(x|x*), so the chain leaves P
invariant. Limitation: the proposal is undirected. In high d almost every proposal
points off the typical set toward the low-density tails, where the density is so
small the acceptance probability is negligible; shrinking ς to stay inside the
typical set keeps acceptance up but makes each move tiny. Either way exploration is a
diffusion — displacement ∝ √(steps) — so the cost to reach a nearly independent
state grows with dimension. (Quantitatively, the mean energy gap of a proposal grows
like d ς², forcing ς ∝ d^{−1/2} and cost ∝ d², with an optimal acceptance rate near
0.23.)

**Metropolis–Hastings (Hastings 1970).** Generalizes the above to an arbitrary,
possibly asymmetric, proposal density Q(x*|x): accept with
min[1, Q(x|x*) P(x*) / (Q(x*|x) P(x))]. This is the general template for "propose,
then correct." It tells me that if I want to use some *other* proposal mechanism — in
particular a deterministic one built from dynamics — I will need to know its proposal
density (its Jacobian), and I will need it to be reversible enough that the ratio is
well-defined.

**Gibbs sampling / coordinate updates.** Update one coordinate (or block) at a time
from its exact conditional. Each scan still moves the state by an amount comparable to
the most-constrained width, so it too explores the least-constrained directions by a
random walk and inherits the same √(steps) diffusion in d.

## Evaluation settings

The natural targets are continuous distributions on R^d whose density and log-density
gradient can be evaluated up to a constant: as controlled cases, multivariate
Gaussians where the answer is known — e.g. a strongly correlated bivariate Gaussian
(correlation 0.95–0.98) that is tightly constrained in one direction, and a
d = 100 Gaussian with a wide spread of marginal standard deviations
(0.01, 0.02, …, 1.00) so the most- and least-constrained directions differ by two
orders of magnitude; and, as the intended application, Bayesian posteriors
(including hierarchical models and neural-network models) known only up to their
normalizing constant. The yardsticks are sampling efficiency at fixed computational
cost — autocorrelation time / effective sample size per gradient evaluation, the
acceptance rate of proposals, and how these scale with dimension d — plus correctness
checks that estimated means and variances match the known target when such checks are
available. The comparison method is random-walk Metropolis (and Gibbs) run for matched
computation.

## Code framework

What exists before the method: a target specified by its energy and gradient, a
Gaussian random-number generator, and a Metropolis–Hastings accept/reject. The chain
is a loop that calls one transition kernel per iteration. The proposal mechanism is
the empty slot.

```r
# Target distribution, supplied by the user:
U = function (q)        # potential energy = -log pi(q), up to an additive constant
  stop("supply U")
grad_U = function (q)   # gradient of U
  stop("supply grad_U")

metropolis_accept = function (current_U, proposed_U)
{
  runif(1) < exp(current_U - proposed_U)
}

transition = function (U, grad_U, epsilon, L, current_q)
{
  # proposal mechanism goes here
  current_q
}

sample_chain = function (U, grad_U, epsilon, L, q0, n_iter)
{
  q = q0
  chain = matrix(NA, nrow=n_iter+1, ncol=length(q0))
  chain[1,] = q
  for (i in 1:n_iter)
  {
    q = transition(U, grad_U, epsilon, L, q)
    chain[i+1,] = q
  }
  chain
}
```
