# The Natural Policy Gradient

## Problem

Maximize the average reward η(θ) = Σ_{s,a} ρ^π(s) π(a;s,θ) R(s,a) of a stochastic policy
π(a;s,θ) by adjusting θ. The vanilla policy-gradient update Δθ = α ∂η/∂θ is **non-covariant**:
it implicitly measures step length by the Euclidean norm Σ(dθᵢ)² in arbitrarily-chosen
coordinates, so the "steepest ascent" direction changes when the policy is reparameterized, and
the rule is even dimensionally inconsistent. The fix must keep the ascent direction a property of
the *policy distribution*, not of the parameter labels — without losing sample-estimability.

## Key idea

Steepest ascent under a metric G is **G⁻¹∇η** (maximize ∇η·dθ subject to dθᵀG dθ = ε²). The
objective lives on the manifold of policy distributions, whose unique reparameterization-invariant
metric — equal to the local Hessian of the KL divergence — is the **Fisher information**. Average
the per-state Fisher matrices by the stationary distribution and step along the resulting
**natural gradient**. Two facts make it land:

1. With **compatible** function approximation (the critic's features are the policy's score), the
   natural gradient equals the critic's least-squares weight vector w̄ — nothing extra to compute.
2. A natural-gradient step scales each action's probability by its compatible-critic (advantage)
   value, moving the policy toward the **greedy** action; in the limit it performs one step of
   approximate policy iteration.

## The construction

**Exact gradient (policy gradient theorem).** With no state-distribution term, hence estimable
from rollouts:

    ∇η(θ) = Σ_{s,a} ρ^π(s) ∇π(a;s,θ) Q^π(s,a),   ψ(s,a) := ∇log π(a;s,θ) = ∇π/π.

**Steepest ascent under a metric.** Maximize η(θ+dθ) s.t. dθᵀG(θ)dθ = ε². The Lagrangian
stationarity condition is ∇η − λGdθ = 0, hence dθ = (1/λ)G⁻¹∇η.

**Fisher metric on the policy manifold.** Per-state Fisher and its ρ^π-average:

    F_s(θ) = E_{a∼π}[ ψ(s,a) ψ(s,a)ᵀ ],     F(θ) = E_{s∼ρ^π}[ F_s(θ) ].

F is positive semidefinite, positive definite when the score features span the parameter space,
invariant to reparameterization, and locally equals the Hessian of
D_KL(π_θ ‖ π_{θ+dθ}) = ½ dθᵀ F dθ + O(dθ³).

**Natural gradient.**   ∇̃η(θ) = F(θ)⁻¹ ∇η(θ), using a pseudoinverse or small ridge if F is
singular.  (Reduces to ∇η exactly when F = I.)

**Compatible critic.** f(s,a;w) = wᵀψ(s,a). Compatibility (∂f/∂w = ∇π/π) keeps the gradient exact
when f replaces Q^π. For Gibbs π ∝ exp(θᵀφ_sa), ψ_sa = φ_sa − Σ_b π(s,b)φ_sb (mean-zero ⇒ f
approximates the advantage A^π = Q^π − V^π).

### Theorem 1 (natural gradient = compatible critic weights)

Let w̄ minimize ε(w) = Σ_{s,a} ρ^π(s) π(a;s,θ) (wᵀψ(s,a) − Q^π(s,a))². For nonsingular F,
**w̄ = ∇̃η(θ)**.

*Proof.* ∂ε/∂w = 0 gives Σ ρ^π π ψ(ψᵀw̄ − Q^π) = 0, i.e.
[Σ ρ^π π ψψᵀ] w̄ = Σ ρ^π π ψ Q^π. The right side is Σ ρ^π ∇π Q^π = ∇η (since πψ = ∇π); the left
bracket is E_{ρ^π}[E_π[ψψᵀ]] = F. Hence F w̄ = ∇η, so w̄ = F⁻¹∇η = ∇̃η when the metric solve is
well-defined. ∎

### Theorem 2 (greedy improvement, exponential family)

For π ∝ exp(θᵀφ_sa) with w̄ minimizing the approximation error, let
π_∞(a;s) = lim_{α→∞} π(a;s, θ + α∇̃η(θ)). Then π_∞(a;s) ≠ 0 **iff** a ∈ argmax_{a'} f(s,a';w̄).

*Proof.* f(s,a;w̄) = ∇̃η(θ)ᵀψ_sa and ψ_sa = φ_sa − E_π[φ] with E_π[φ] independent of a, so
argmax_a f = argmax_a ∇̃η(θ)ᵀφ_sa. After the step π ∝ exp(θᵀφ_sa + α∇̃η(θ)ᵀφ_sa); as α→∞ the
second term dominates, concentrating mass on argmax_a ∇̃η(θ)ᵀφ_sa = argmax_a f(s,a;w̄). ∎

(The vanilla gradient under α→∞ selects only a *better* action, f(a) > E_{a'}f(a'), not the best.)

### Theorem 3 (local move toward the best action, general policy)

For θ' = θ + α∇̃η(θ):   **π(a;s,θ') = π(a;s,θ)(1 + α f(s,a;w̄)) + O(α²).**

*Proof.* Δθ = αw̄ (Thm 1). π(a;s,θ') = π + (∂π/∂θ)ᵀΔθ + O(Δθ²) = π(1 + ψᵀΔθ) + O(Δθ²)
= π(1 + αψᵀw̄) + O(α²) = π(1 + α f(s,a;w̄)) + O(α²). ∎

Initial improvement is guaranteed whenever ∇η ≠ 0 and the metric used for the solve is positive
definite: dη/dα|₀ = ∇η·∇̃η = ∇ηᵀF⁻¹∇η > 0.

### Why F is not the Hessian (and why that is acceptable)

    ∇²η(θ) = Σ_{sa} ρ^π(∇²π Q^π + ∇π∇Q^πᵀ + ∇Q^π∇πᵀ).

F = Σ_{sa} ρ^π ∇π∇πᵀ/π contains none of the value-coupled terms, so F ≠ ∇²η: the natural gradient
is **not** Newton's method and is not asymptotically second-order efficient (unlike natural
gradient in parameter estimation, where Fisher → Hessian and the Cramér–Rao bound is attained).
But ∇²η is objective curvature, not a metric; for maximization it can be indefinite far from a
maximum, and its useful local picture near a maximum is negative curvature. The Fisher is instead a
positive metric after restricting to the nondegenerate score subspace or adding a ridge, so F⁻¹∇η is
a robust, covariant ascent direction with guaranteed local improvement under that metric; conjugate
methods are preferable only near the optimum.

## Algorithm

```python
import numpy as np

def natural_policy_gradient_step(policy, mdp, eps=1e-2, ridge=1e-3):
    # 1. Differential value and exact gradient (policy gradient theorem).
    Q = estimate_value(policy, mdp)                 # Q^π under average reward
    m = policy.theta.shape[0]
    grad = np.zeros(m); F = np.zeros((m, m))
    for s in mdp.states:
        rho = mdp.stationary_prob(s, policy)
        for a in mdp.actions:
            psi = policy.score(s, a)                # ψ = ∇log π = ∇π/π (compatible features)
            p   = policy.prob(s, a)
            grad += rho * p * psi * Q(s, a)         # Σ ρ^π ∇π Q
            F    += rho * p * np.outer(psi, psi)    # Σ ρ^π π ψψ^T  = Fisher metric
    F += ridge * np.eye(m)                          # regularized positive metric if F is singular

    # 2. Natural gradient = compatible-critic weights w̄ (Theorem 1): solve F w̄ = ∇η.
    nat = np.linalg.solve(F, grad)                  # ∇̃η = F^{-1} ∇η = w̄

    # 3. Equal-KL line search: ½ α^2 ∇η^T F^{-1} ∇η = ε.
    denom = grad @ nat
    alpha = 0.0 if denom <= 0 else np.sqrt(2.0 * eps / denom)

    # 4. Covariant step; positive metric ⇒ guaranteed initial improvement; moves toward greedy action.
    policy.theta = policy.theta + alpha * nat
    return policy
```

The natural policy gradient is the policy-search direction that is covariant (independent of the
policy's parameterization), reduces to the vanilla gradient when the Fisher metric is the
identity, coincides with the compatible critic's weight vector, and — taken to the limit — moves
the policy toward the greedy action, i.e. performs one step of approximate policy iteration.
