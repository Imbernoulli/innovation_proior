# Hamiltonian (Hybrid) Monte Carlo

## Problem

Sample from a continuous target π(q) on R^d (typically a Bayesian posterior), known
only up to its normalizing constant but with a computable gradient ∇ log π(q), in
order to estimate expectations E_π[f]. In high dimension the probability mass lives
on a thin "typical set" away from the mode; random-walk Metropolis explores this set
only diffusively (net displacement ∝ √steps), so its cost to reach a near-independent
state scales like d² and its proposals must be tiny to keep acceptance up.

## Key idea

Lift the target to a joint distribution over (q, p) by introducing an auxiliary
**momentum** p, with

  π(q, p) ∝ exp(−H(q, p)),   H(q, p) = U(q) + K(p),
  U(q) = −log π(q),   K(p) = ½ pᵀ M⁻¹ p   (so p ~ N(0, M)).

Because H is additive, q ⟂ p and the q-marginal is exactly π(q). Evolving (q, p) by
**Hamilton's equations** routes the gradient through the momentum, producing
trajectories that glide *along* a constant-probability surface rather than crashing
into the mode. The exact flow (i) **conserves H** (acceptance → 1 for arbitrarily
distant proposals), (ii) **preserves phase-space volume** (Jacobian determinant 1 ⇒
no Jacobian in the accept ratio), and (iii) is **reversible** (⇒ symmetric proposal).
The **leapfrog** integrator preserves (ii) and (iii) *exactly* at finite stepsize,
sacrificing only energy conservation to O(ε²); a single Metropolis accept/reject pays
that residue back *exactly*. Resampling p each iteration moves the chain between
energy level sets.

## Algorithm (one iteration)

Parameters: stepsize ε (below the stability limit ≈ 2·σ_min), number of leapfrog
steps L (trajectory length ε·L). State: current position q.

1. **Resample momentum.** Draw p ~ N(0, M). (Gibbs step; leaves exp(−H) invariant,
   changes H.)
2. **Leapfrog L steps** from (q, p), with K(p)=½pᵀM⁻¹p so ∂K/∂p = M⁻¹p:
   - half-kick: p ← p − (ε/2) ∇U(q)
   - repeat L times: drift q ← q + ε M⁻¹ p; and (except after the last) kick
     p ← p − ε ∇U(q)
   - half-kick: p ← p − (ε/2) ∇U(q)
   Call the result (q*, p*); negate p* (makes the proposal symmetric/reversible).
3. **Metropolis correction.** Accept (q*, p*) with probability
     min[1, exp(−H(q*, p*) + H(q, p))]
        = min[1, exp(U(q) − U(q*) + K(p) − K(p*))];
   otherwise keep q (counted again).

The q-coordinates of the chain are samples from π.

## Why it is exact

- **Volume preservation.** Hamilton's flow has zero-divergence vector field
  (Σ ∂²H/∂qᵢ∂pᵢ − ∂²H/∂pᵢ∂qᵢ = 0), so |det J| = 1. Each leapfrog substep is a *shear*
  (only p, or only q, changes, by an amount depending on the other), whose Jacobian is
  triangular with unit diagonal — determinant exactly 1 at finite ε. So no Jacobian
  term enters the acceptance ratio.
- **Reversibility.** Leapfrog is symmetric in time; "L steps then negate p" is its own
  inverse, so the proposal is symmetric and the proposal densities cancel in the
  Metropolis–Hastings ratio, leaving only the density ratio inside min[1, exp(−ΔH)].
- **Detailed balance.** Partition phase space into equal-volume cells A_k with images
  B_k (also equal-volume, by volume preservation; also a tiling, by reversibility). For
  matched cells, (V/Z)e^{−H_{A}} min[1, e^{−H_B+H_A}] = (V/Z)e^{−H_B} min[1, e^{−H_A+H_B}]
  = (V/Z) min[e^{−H_A}, e^{−H_B}] — identical on both sides. So the leapfrog+MH update
  leaves exp(−H) invariant; combined with the momentum resampling (which also leaves it
  invariant), the full chain has π(q,p) as its stationary distribution and π(q) as its
  marginal.

## Tuning and scaling

- **Stepsize.** For a quadratic mode of width σ, a leapfrog step is a linear map with
  eigenvalues of modulus 1 iff ε/σ < 2 and unstable otherwise; ε is capped by the
  most-constrained direction, and high-dimensional acceptance may require a smaller
  ε. Randomize ε once per trajectory, and vary ε or L, to reduce failures from a
  fixed oversized step and to avoid exact periodicities (ergodicity).
- **Trajectory length** ε·L: long enough to traverse the least-constrained direction
  (distance ∝ L, not √L), short enough to avoid U-turns.
- **Scaling.** Leapfrog energy error over a fixed trajectory is O(ε²); with
  E[Δ₁] ≈ E[Δ₁²]/2, E[Δ_d] ∝ d·ε⁴, so ε ∝ d^{−1/4},
  leapfrog steps to near-independence ∝ d^{1/4}, and wall-clock cost ∝ d^{5/4}
  after the O(d) gradient cost — versus ς ∝ d^{−1/2} and cost ∝ d² for
  random-walk Metropolis. Optimal acceptance ≈ 0.65 for HMC vs ≈ 0.23 for random walk.

## R implementation

```r
# One iteration of Hamiltonian Monte Carlo.
# U(q): potential energy = -log target density (up to constant)
# grad_U(q): gradient of U
# epsilon: leapfrog stepsize; L: number of leapfrog steps; current_q: start position.
# Kinetic energy assumed K(p) = sum(p^2)/2  (mass matrix = identity).
HMC = function (U, grad_U, epsilon, L, current_q)
{
  q = current_q
  p = rnorm(length(q),0,1)  # independent standard normal variates
  current_p = p

  # Make a half step for momentum at the beginning

  p = p - epsilon * grad_U(q) / 2

  # Alternate full steps for position and momentum

  for (i in 1:L)
  {
    # Make a full step for the position

    q = q + epsilon * p

    # Make a full step for the momentum, except at end of trajectory

    if (i!=L) p = p - epsilon * grad_U(q)
  }

  # Make a half step for momentum at the end.

  p = p - epsilon * grad_U(q) / 2

  # Negate momentum at end of trajectory to make the proposal symmetric

  p = -p

  # Evaluate potential and kinetic energies at start and end of trajectory

  current_U = U(current_q)
  current_K = sum(current_p^2) / 2
  proposed_U = U(q)
  proposed_K = sum(p^2) / 2

  # Accept or reject the state at end of trajectory, returning either
  # the position at the end of the trajectory or the initial position

  if (runif(1) < exp(current_U-proposed_U+current_K-proposed_K))
  {
    return (q)  # accept
  }
  else
  {
    return (current_q)  # reject
  }
}
```
