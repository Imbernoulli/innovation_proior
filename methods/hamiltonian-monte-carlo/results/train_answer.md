The problem is to draw samples from a continuous target density pi(q) on R^d when only the unnormalized log density, and ideally its gradient, are available. In high dimensions, almost all of the probability mass lies in a thin typical region where density and volume balance, not near the mode and not in the tails. A Markov chain must spend nearly all of its time in that region if expectations are to be estimated efficiently. The classic random-walk Metropolis approach proposes small symmetric displacements around the current point. Because most isotropic directions point out of the typical region, either the step size is shrunk so far that the chain diffuses slowly, or proposals land in the tails and are rejected. Coordinate-wise updates avoid rejection but still explore broad directions by slow accumulated drift. The gradient of the log density contains useful geometric information, but it points toward regions of higher density, which is toward the mode, not along the typical set. What is needed is a way to convert gradient information into long, coherent moves that stay inside the typical region.

The method that does this is Hamiltonian Monte Carlo. It introduces an auxiliary momentum variable p for every position coordinate q and defines a potential energy U(q) equal to minus the log target density plus a constant, together with a kinetic energy K(p) equal to one half p transpose M inverse p. The joint energy is H(q,p) = U(q) + K(p), and the corresponding joint density is proportional to exp(-H(q,p)). Because the energy is additive, the joint density factorizes into a q factor exp(-U(q)), which is the target, and a Gaussian p factor. Adding momentum therefore gives the chain extra directions to move without changing the marginal distribution of interest.

The dynamics come from Hamilton's equations: the rate of change of q is M inverse p, and the rate of change of p is minus the gradient of U at q. The gradient no longer pulls q toward the mode; instead it curves the momentum, and the momentum carries q across the target landscape. Exact Hamiltonian dynamics conserve H, preserve phase-space volume, and are reversible. If we could integrate exactly, every proposal would be accepted because H would be unchanged. Exact integration is usually impossible, so we use the leapfrog integrator. One leapfrog step updates momentum by half a step, then updates position by a full step using the new momentum, then updates momentum by another half step. Each substep is a shear map, so the whole step preserves volume exactly. The symmetric half-step/full-step/half-step structure makes it reversible. Leapfrog does not conserve energy exactly, but its energy error is small for stable step sizes, and any residual error is corrected by a final Metropolis accept-reject step.

One transition works as follows. First, draw fresh momentum p from Normal(0,M); this is a Gibbs update for the joint density. Then run L leapfrog steps of size epsilon to produce a proposed state (q*,p*). Flip the final momentum to -p* so that the deterministic proposal is reversible. Accept the proposed position q* with probability min(1, exp(-H(q*,p*) + H(q,p))), which is the same as min(1, exp(U(q) - U(q*) + K(p) - K(p*))). Because the proposal is volume preserving and reversible, the Metropolis-Hastings ratio has no proposal density or Jacobian term. The only correction is for the numerical energy error. If the proposal is rejected, the chain keeps the old q. Repeating this process leaves the target density invariant and produces long-range moves with high acceptance.

The main tuning knobs are the step size epsilon, the number of leapfrog steps L, and the mass matrix M. The step size must be small enough that leapfrog remains stable in the tightest direction; for a one-dimensional quadratic with curvature one over sigma squared, stability requires epsilon over sigma to be below two. The trajectory length epsilon times L should be long enough that momentum carries the state across the broad directions of the typical region, but not so long that trajectories double back on themselves. The mass matrix acts as a preconditioner: choosing M close to the target covariance makes the energy levels more regular and exploration more isotropic. Randomizing epsilon or L from iteration to iteration helps avoid periodic behavior.

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
