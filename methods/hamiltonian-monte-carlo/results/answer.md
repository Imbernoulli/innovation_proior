# Hamiltonian Monte Carlo

## Method

Given a target density `pi(q)` on `R^d`, known up to a constant, define

```text
U(q) = -log pi(q)
K(p) = 1/2 p^T M^{-1} p
H(q, p) = U(q) + K(p)
```

with auxiliary momentum `p ~ Normal(0, M)`. The joint density is proportional to `exp(-H(q,p))`, and its `q` marginal is the target.

One transition from current position `q`:

1. Draw fresh momentum `p ~ Normal(0, M)`.
2. Starting from `(q, p)`, run `L` reversible volume-preserving leapfrog steps with step size `epsilon` to obtain `(q_L, p_L)`.
3. Set `(q*, p*) = (q_L, -p_L)` to make the deterministic proposal reversible.
4. Accept `(q*, p*)` with probability

```text
alpha = min(1, exp(-H(q*, p*) + H(q, p)))
      = min(1, exp(U(q) - U(q*) + K(p) - K(p*)))
```

and otherwise keep `q`.

## Leapfrog

For `K(p)=1/2 p^T M^{-1}p`, one leapfrog step is

```text
p <- p - (epsilon / 2) grad U(q)
q <- q + epsilon M^{-1} p
p <- p - (epsilon / 2) grad U(q)
```

The half-step/full-step/half-step symmetry gives reversibility. Each substep is a shear map, so the composition preserves phase-space volume exactly.

## Why It Is Exact

Exact Hamiltonian dynamics conserves `H`, preserves phase-space volume, and is reversible. Leapfrog preserves the last two properties exactly and keeps the energy error small when the step size is stable. Because the numerical proposal is reversible and volume preserving, the Metropolis-Hastings ratio has no proposal-density or Jacobian term; it is only the canonical density ratio `exp(-Delta H)`.

The momentum refresh is a Gibbs step for the joint density. The leapfrog-plus-Metropolis step leaves the same joint density invariant by detailed balance. Their composition leaves `exp(-H(q,p))` invariant, so the retained positions have stationary density `pi(q)`.

## Tuning

- `epsilon`: must be below the stability scale of the tightest direction. For `U(q)=q^2/(2 sigma^2)` and unit kinetic energy, leapfrog is stable when `epsilon/sigma < 2`.
- `L`: should make `epsilon L` long enough to traverse broad directions, but not so long that trajectories double back.
- `M`: acts as a mass matrix or preconditioner. Matching target scales and correlations improves trajectory geometry.
- Randomizing `epsilon` or `L` helps avoid periodic trajectories.

For replicated independent dimensions, random-walk Metropolis needs proposal scale proportional to `d^{-1/2}` and costs about `d^2`; HMC uses `epsilon` proportional to `d^{-1/4}` and costs about `d^{5/4}` in the standard replicated-coordinate asymptotic model. The corresponding optimal acceptance is about `0.65` in that model.

## R Artifact

```r
HMC = function (U, grad_U, epsilon, L, current_q)
{
  q = current_q
  p = rnorm(length(q), 0, 1)
  current_p = p

  p = p - epsilon * grad_U(q) / 2

  for (i in 1:L)
  {
    q = q + epsilon * p
    if (i != L) p = p - epsilon * grad_U(q)
  }

  p = p - epsilon * grad_U(q) / 2
  p = -p

  current_U = U(current_q)
  current_K = sum(current_p^2) / 2
  proposed_U = U(q)
  proposed_K = sum(p^2) / 2

  if (runif(1) < exp(current_U - proposed_U + current_K - proposed_K))
  {
    return(q)
  }
  else
  {
    return(current_q)
  }
}
```
