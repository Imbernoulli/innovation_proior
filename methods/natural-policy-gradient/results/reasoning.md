Let me start from the thing that actually bothers me about following the policy gradient, because it's not a subtlety — it's sitting right in the update rule and I keep stepping over it.

I have a stochastic policy π(a;s,θ), and I want to maximize the average reward η(θ) = Σ_{s,a} ρ^π(s) π(a;s,θ) R(s,a) over the parameters θ. The rule everyone writes down is Δθᵢ = α ∂η/∂θᵢ. Stare at the units for a second. The left side is a change in θᵢ, so it has units of θᵢ. The right side is a derivative with respect to θᵢ, so it has units of 1/θᵢ (times units of η, but that's shared). Those don't match. And the m components of θ don't even have to share units with each other — one might be a gain, another a log-variance, another a feature weight. So I'm summing and stepping along a vector whose components live in incompatible spaces, as if they were all plain numbers in the same Euclidean ℝᵐ. That's not a bug I can patch with a learning rate; it means the *direction* I'm calling "uphill" is an artifact of how I happened to label the parameters.

Let me make that concrete, because "non-covariant" is easy to say and easy to ignore. Suppose I reparameterize: let φ = φ(θ) be some smooth invertible change of coordinates for the same policy class. The same policies, just relabeled. If gradient ascent were really picking out a direction intrinsic to the problem, then the step I take should be the *same physical move on the policy* whichever coordinates I use — the φ-gradient should be the Jacobian image of the θ-gradient. It isn't. The gradient is a covector; under θ→φ with Jacobian J it transforms as ∇_φ η = J⁻ᵀ ∇_θ η, by the inverse-transpose Jacobian — whereas a genuine direction (a tangent vector) would transform as J⁻¹·(direction). So the ascent direction "∇η read as a vector," i.e. G⁻¹∇ with G=I in θ-coordinates, maps to G'⁻¹∇ with G'=I in φ-coordinates, and those two are different physical directions unless J is orthogonal (JᵀJ = I). Two people optimizing the identical policy with different parameter conventions take different steps and chase different "steepest" directions. The steepest-ascent direction is supposed to be a fact about the landscape; here it's a fact about my coordinate chart. That's the disease.

So where did "steepest ascent = the gradient" even come from? Let me re-derive it honestly instead of taking it as a definition. The steepest-ascent direction is the dθ that maximizes η(θ+dθ) for an infinitesimal step of fixed *length*. Fixed length under what notion of length? Implicitly we've been measuring |dθ|² = Σᵢ (dθᵢ)² — the Euclidean norm in the raw coordinates. Write that as a quadratic form: |dθ|² = dθᵀ G dθ with G = I. So "the gradient" is the steepest-ascent direction *only relative to the choice G = I*. And G = I in θ-coordinates is exactly the arbitrary, coordinate-dependent assumption that's poisoning everything. The whole non-covariance is the silent assumption G=I.

Good — that reframes the problem productively. I don't have to abandon "steepest ascent." I have to choose the metric G honestly. Let me redo the steepest-ascent calculation for a general positive-definite G(θ) and see what direction drops out. Maximize η(θ+dθ) ≈ η(θ) + ∇η·dθ subject to dθᵀ G dθ = ε². Lagrangian: ∇η·dθ − (λ/2)(dθᵀ G dθ − ε²). Set the gradient w.r.t. dθ to zero: ∇η − λ G dθ = 0, so G dθ = ∇η/λ, so dθ = (1/λ) G⁻¹ ∇η. The direction is **G⁻¹∇η**. When G=I it collapses back to the plain gradient, as it must. So the object I actually want to step along is G⁻¹∇η for the *right* G — and the entire question becomes: what is the right G?

Now, what makes a metric "right" here? The thing I genuinely care about is not how far θ moves in coordinate space — it's how much the *policy* moves. η is a functional of the distributions π(·;s,θ), not of the bare numbers θ. Two parameter settings that encode nearly the same policy should be "close"; two that encode very different policies should be "far" — and that judgment can't depend on my labeling. Let me sanity-check that the Euclidean metric fails this even on a trivial example. Take a Gaussian policy parameterized by mean and standard deviation, θ = (μ, σ). Move from (μ=0, σ=0.3) to (μ=1, σ=0.3): Euclidean step length 1. Now move from (μ=0, σ=3) to (μ=1, σ=3): also Euclidean length 1. But a unit shift in the mean of a tight σ=0.3 Gaussian almost completely separates the two distributions, whereas the same shift on a broad σ=3 Gaussian barely changes it. Same parameter distance, wildly different distributional change. So the Euclidean metric in (μ,σ) is measuring the wrong thing — it doesn't know that the *effect* of moving μ depends on σ. Capping |dθ| doesn't cap how much the policy actually changes. That's exactly the failure that lets a gradient step overshoot into a region the policy can never recover from.

So I want a metric on the *space of distributions*. This is where I should lean on what Amari worked out for learning in general. His picture: when your parameters index a family of probability distributions, the parameter space isn't flat — it's a curved (Riemannian) manifold, and the natural way to measure an infinitesimal displacement is by how much the distribution it indexes changes. The quantity that does this is the Fisher information matrix, F_ij = E[ ∂ᵢ log p · ∂ⱼ log p ]. There are two reasons this is *the* metric and not just *a* metric. First, invariance: the Fisher information is, up to overall scale, the unique metric on a family of probability distributions that's invariant to reparameterization — it assigns the same distance between two distributions no matter what coordinates I use. That's precisely the coordinate-independence I was missing. Second, it's the local curvature of a distributional distance: expand the KL divergence between p_θ and p_{θ+dθ}. The zeroth order is 0 (same distribution), the first order is 0 (KL is minimized at dθ=0, so its gradient there vanishes), and the leading term is second order: D_KL(p_θ ‖ p_{θ+dθ}) = ½ dθᵀ F dθ + O(dθ³). So dθᵀ F dθ is twice the local KL change, to leading order. That's the notion of "how far did the policy actually change" I was reaching for.

Amari also packaged the consequence: steepest descent on the distribution manifold uses F⁻¹∇ in place of ∇, and he called G⁻¹∇ — with G the Fisher metric — the **natural gradient**. Which is exactly the G⁻¹∇η my Lagrangian just produced, with the metric finally pinned down. The constrained-step view and the curved-manifold view are the same thing: capping dθᵀ F dθ caps twice the local KL change of the policy, and the optimal step under that cap is the natural gradient.

But I can't just lift Amari's setup wholesale, and this is where I have to be careful. His natural gradient is for *one* probability model — one Fisher manifold, one likelihood. My policy isn't a single distribution. For every state s there's a *separate* conditional distribution π(·;s,θ), each its own little probability manifold sharing the parameters θ. And the objective η isn't a log-likelihood; it's an average over the stationary distribution ρ^π of the per-state rewards. So "the Fisher matrix of the policy" isn't immediately defined. Let me build it from the per-state pieces. For a single state s, the Fisher information of the conditional is

    F_s(θ) = E_{a∼π(·;s,θ)} [ ∇log π(a;s,θ) ∇log π(a;s,θ)ᵀ ],

which is positive semidefinite and is the right metric on *that* state's policy manifold. Now I need one metric on θ for the whole problem. The objective weights each state by how often I'm in it, ρ^π(s); a metric that's faithful to the objective should weight each state's distributional distance the same way. So average:

    F(θ) = E_{s∼ρ^π} [ F_s(θ) ].

That's my candidate metric, and, when the averaged Fisher is nonsingular, the natural gradient is ∇̃η(θ) = F(θ)⁻¹ ∇η(θ). If the score features are rank-deficient or the policy is becoming effectively deterministic, I will need the same practical escape every Fisher method needs — a pseudoinverse or a small positive ridge — but the direction I am deriving is still the inverse-metric direction. Let me flag, honestly, that this isn't quite as clean as Amari's single manifold: F here is an *average* of per-state Fisher matrices, a collection rather than the Fisher of one grand distribution, so I shouldn't expect every property of the single-model natural gradient to carry over verbatim. But each F_s is a bona fide invariant metric on its state's policy, and the ρ^π-weighting matches the object I'm optimizing, so this is the principled choice. Notice also that although F_s depends only on the policy parameters and not on the transition model P, the averaging weight ρ^π *does* depend on P through the stationary distribution — the metric isn't oblivious to the dynamics, it just enters only through where I spend my time.

Before I go further I have to make sure I haven't broken the one thing that made policy gradients usable: that ∇η is estimable from samples. The exact gradient — and this is the load-bearing fact from the policy gradient theorem — is

    ∇η(θ) = Σ_{s,a} ρ^π(s) ∇π(a;s,θ) Q^π(s,a),

with **no** term involving ∂ρ^π/∂θ. The way the policy perturbs the state distribution drops out entirely (it cancels through stationarity), which is exactly why I can estimate this by rolling out the current policy and never have to differentiate the dynamics. Multiplying by F⁻¹ doesn't touch that — F is also an expectation under ρ^π of the score outer product, equally estimable from rollouts. So covariance comes for free; I haven't sacrificed sample-ability. Good.

Now, solving Fv = ∇η is correct but it looks like extra work: estimate the gradient, separately estimate an m×m Fisher matrix, and solve a linear system. Both sides are built from the same score ∇log π = ∇π/π, and that can't be a coincidence — so let me see how they interact, which means I first have to pin down what the critic should be.

If I'm going to estimate ∇η I need Q^π, and Q^π is unknown, so I'll approximate it. Sutton's result tells me I can't use just *any* approximator without biasing the gradient — the approximator f(s,a;w) has to be *compatible*: its features have to be the policy's own score. That is, I should take

    f(s,a;w) = wᵀ ψ(s,a),   where ψ(s,a) = ∇log π(a;s,θ) = ∇π/π.

The reason is that compatibility makes the approximation error orthogonal to the gradient direction, so substituting f for Q^π in the gradient formula leaves the gradient exact. (For a Gibbs policy π ∝ exp(θᵀφ_sa) this forces ψ_sa = φ_sa − Σ_b π(s,b)φ_sb, the features centered to mean zero under π — which means the compatible critic is really approximating the *advantage* Q^π − V^π, the relative value of each action, not its absolute value. That's fine; only relative values matter for choosing actions.) So I'm going to fit w by least squares to Q^π, and then this w is the natural object floating around in the actor-critic loop.

Let me fit w explicitly and watch what comes out. Let w̄ minimize the ρ^π-π-weighted squared error

    ε(w) = Σ_{s,a} ρ^π(s) π(a;s,θ) ( f(s,a;w) − Q^π(s,a) )² = Σ_{s,a} ρ^π(s) π(a;s,θ) ( wᵀψ(s,a) − Q^π(s,a) )².

Stationarity at the minimum: ∂ε/∂w = 0 gives

    Σ_{s,a} ρ^π(s) π(a;s,θ) ψ(s,a) ( ψ(s,a)ᵀ w̄ − Q^π(s,a) ) = 0,

i.e.

    [ Σ_{s,a} ρ^π(s) π(a;s,θ) ψ(s,a) ψ(s,a)ᵀ ] w̄ = Σ_{s,a} ρ^π(s) π(a;s,θ) ψ(s,a) Q^π(s,a).

Now look at the two sides separately. The right-hand side: ψ(s,a) = ∇π/π, so π(a;s,θ) ψ(s,a) = ∇π(a;s,θ), and the sum is Σ_{s,a} ρ^π(s) ∇π(a;s,θ) Q^π(s,a) — that is *exactly* ∇η(θ), the policy gradient. The left-hand side bracket: Σ_{s,a} ρ^π(s) π(a;s,θ) ψ ψᵀ = E_{s∼ρ^π} E_{a∼π} [ ∇log π ∇log πᵀ ] = E_{s∼ρ^π}[ F_s(θ) ] = F(θ), my Fisher metric, verbatim. So the normal equations read

    F(θ) w̄ = ∇η(θ).

Solve, when F is nonsingular: w̄ = F(θ)⁻¹ ∇η(θ). That's the natural gradient. The weight vector of the compatible critic, the thing I was going to fit anyway to estimate the gradient, *is* the natural gradient — there's nothing extra to compute, no separate Fisher inversion, no second estimation problem beyond the least-squares solve itself. ∇̃η(θ) = w̄. I sat down to make the gradient covariant and the machinery I already had for variance-reduced gradient estimation turns out to be solving for the covariant direction. The least-squares critic and the natural gradient are the same object because both are built from the score ∇log π — its expectation against Q gives the gradient, its outer-product expectation gives the metric, and least squares is exactly the operation that hits one with the inverse of the other.

What does w̄ = F⁻¹∇η actually *do* to the policy? The vanilla gradient, pushed hard, moves probability toward actions that are merely *better* than average — it increases π wherever Q exceeds the state's mean value. Does the natural gradient do something sharper? Let me test it on the cleanest case, the exponential family, where the geometry is flat enough to push the step to its limit. Take π(a;s,θ) ∝ exp(θᵀφ_sa). The compatible critic is f(s,a;w̄) = w̄ᵀψ_sa = ∇̃η(θ)ᵀ ψ_sa with ψ_sa = φ_sa − E_{a'∼π}[φ_{sa'}]. The subtracted term E_π[φ] doesn't depend on a, so it can't change which action maximizes f:

    argmax_a f(s,a;w̄) = argmax_a ∇̃η(θ)ᵀ φ_sa.

Now take a natural-gradient step θ' = θ + α ∇̃η(θ). The new policy is

    π(a;s,θ') ∝ exp( θᵀφ_sa + α ∇̃η(θ)ᵀ φ_sa ).

As α → ∞, the second term α ∇̃η(θ)ᵀ φ_sa dominates the first (provided ∇̃η ≠ 0), so all the probability mass concentrates on the actions that maximize ∇̃η(θ)ᵀ φ_sa — which is exactly argmax_a f(s,a;w̄). In the limit, π_∞(a;s) ≠ 0 if and only if a ∈ argmax_a f(s,a;w̄). An infinite natural-gradient step lands on the policy that is **greedy with respect to the compatible critic** — i.e. it performs exactly one step of approximate policy iteration, choosing the best action under the fitted value, not merely a better one. Contrast the vanilla gradient pushed to α→∞: θᵀφ_sa + α∇η(θ)ᵀφ_sa concentrates on argmax_a ∇η(θ)ᵀφ_sa, which only guarantees an action with f above the mean, not the maximizer. The covariant direction is the one that actually points at the greedy improvement. So the natural gradient bridges the smooth, safe world of policy gradients and the aggressive, fast world of policy iteration: it's the policy-gradient direction that, taken to its extreme, *is* a policy-iteration step.

I leaned on the exponential family to take α→∞ cleanly, but I want the statement for a general policy and a finite, line-searched step. Let me expand the updated policy to first order. The step is Δθ = α∇̃η(θ) = α w̄. Then

    π(a;s,θ') = π(a;s,θ) + (∂π/∂θ)ᵀ Δθ + O(Δθ²)
              = π(a;s,θ)( 1 + ψ(s,a)ᵀ Δθ ) + O(Δθ²)          [since ∂π/∂θ = π · ∇log π = π · ψ]
              = π(a;s,θ)( 1 + α ψ(s,a)ᵀ w̄ ) + O(α²)
              = π(a;s,θ)( 1 + α f(s,a;w̄) ) + O(α²).

So to first order the natural-gradient update *scales up* the probability of each action a in proportion to its compatible-critic value f(s,a;w̄) — the local linear approximation to Q^π (really to the advantage). Locally, the natural gradient moves toward the best action, weighted by the line-search step α. And the first move is an improvement whenever ∇η is nonzero and the metric solve is well-defined: along the natural gradient the rate of change of η is ∇η·∇̃η = ∇ηᵀ F⁻¹ ∇η > 0 for a positive-definite Fisher, with a ridge playing that role in the singular case. So I get guaranteed initial improvement under the metric I actually step with, and the direction of that improvement is "increase the better actions, in proportion to how much better the critic says they are." This is the general statement the exponential-family argument was the extreme case of.

Now I owe myself an honest accounting of what this metric does and doesn't buy, because there's a temptation to call F⁻¹∇η a Newton step and claim second-order convergence, and I don't think that's right here. In plain parameter estimation, the Fisher information converges to the Hessian of the loss, so the natural gradient becomes Newton's method and attains the Cramér–Rao bound — Amari's efficiency result. Does that hold for my η? Let me write the actual Hessian of the average reward. Differentiating ∇η = Σ_{sa} ρ^π ∇π Q^π again (the policy-gradient form),

    ∇²η(θ) = Σ_{sa} ρ^π(s) ( ∇²π(a;s) Q^π(s,a) + ∇π(a;s) ∇Q^π(s,a)ᵀ + ∇Q^π(s,a) ∇π(a;s)ᵀ ).

Look at the three terms. My metric F = Σ_{sa} ρ^π π ψψᵀ = Σ_{sa} ρ^π (∇π ∇πᵀ / π) is built purely from first derivatives of the *policy* — it has no Q in it at all. The last two Hessian terms involve ∇Q^π and are completely invisible to F; they're the coupling between how the policy changes and how the values change, and F throws all of that away. The first term, ∇²π Q^π, has a ∇²π that F's ∇π∇πᵀ might be loosely related to, but it's still weighted by Q, and F neglects that weighting. So F is **not** the Hessian of η. The Fisher-equals-Hessian story that gave Amari second-order efficiency simply doesn't transfer, because here the "loss" is coupled to the state-action values, not a self-contained log-likelihood. So I won't oversell this: the natural gradient here is not a Newton step and isn't asymptotically second-order efficient.

But — and this is why I still want it — that's arguably a feature far from the optimum. The true Hessian of η is not a metric; for a maximization problem it can be indefinite far from a maximum, and the useful local second-order picture near a maximum comes from its negative curvature. Until θ is close to such a maximum, that curvature information can be useless or actively misleading. F is different: after restricting to the nondegenerate score subspace or adding a ridge, it is a positive metric, so F⁻¹∇η is a genuine ascent direction with guaranteed initial improvement under that metric, throughout the parameter space, not just near a maximum. Near the optimum a conjugate-gradient or true-Hessian method would presumably do better on the final convergence rate; the natural gradient's strength is robust, covariant, improvement-guaranteed progress that doesn't get fooled by coordinates or by the plateaus that flatten the vanilla gradient. The right reading isn't "Newton" — it's "the steepest-ascent direction on the policy manifold, which happens to coincide with the critic I already fit, and which points at the greedy improvement."

One practical loose end. I need F to run this online, and I don't want to form Q-free objects I can't estimate. But F is just the running average of the score outer product along a trajectory: maintain F ← F + ∇log π(aₜ;sₜ,θ) ∇log π(aₜ;sₜ,θ)ᵀ over a T-length rollout, and F/T is a consistent estimate. I'm already computing ∇log π for the gradient, so this is nearly free. One numerical caveat: as the policy sharpens toward determinism, ∇log π → 0 and F becomes singular, so I'll add a small ridge — on the order of 10⁻³ I — to keep F⁻¹ well-defined. And I never even have to invert F explicitly if I instead *solve* the compatible-critic least-squares problem, since its solution w̄ is the natural gradient directly.

Roll out the current policy. Estimate the gradient ∇η = Σ ρ^π ∇π Q^π and the Fisher F = E_{ρ^π,π}[∇log π ∇log πᵀ] from the same samples (or equivalently, fit the compatible critic f = wᵀψ by least squares against the returns). Form the natural gradient by solving F∇̃η = ∇η, which gives the critic weights w̄. Step θ ← θ + α ∇̃η, with α a line search. If I want every small step to spend the same local KL budget ε, then ½(α∇̃η)ᵀF(α∇̃η) = ½α²∇ηᵀF⁻¹∇η = ε, so α = √(2ε / ∇ηᵀF⁻¹∇η). Repeat. The update is covariant — independent of how I parameterized π — it reduces to the vanilla gradient exactly when F = I, it gives local improvement under the positive metric used for the solve, and pushed hard it moves the policy toward the action the critic calls best, which is one step of approximate policy iteration.

The causal chain, start to end: the vanilla gradient is non-covariant because Δθ = α∂η/∂θ silently assumes the metric G=I in coordinates I chose arbitrarily; steepest ascent under any positive-definite metric G is G⁻¹∇η; the objective lives on the manifold of policy distributions, whose unique invariant metric — the local Hessian of KL — is the Fisher information F = E_{ρ^π}[E_π[∇log π ∇log πᵀ]]; so the right direction is the natural gradient F⁻¹∇η, still estimable from rollouts because the policy gradient has no state-distribution term; the compatible critic (features = the score, as forced by the requirement that approximating Q doesn't bias the gradient) has least-squares weights w̄ satisfying F w̄ = ∇η, so w̄ = F⁻¹∇η — the natural gradient *is* the critic; and a natural-gradient step scales each action's probability by its compatible-critic (advantage) value, moving the policy toward the greedy action and, in the limit, performing one step of approximate policy iteration.

```python
import numpy as np

class Policy:
    """Differentiable stochastic policy π(a; s, θ). Gibbs example: π ∝ exp(θ·φ(s,a))."""
    def __init__(self, theta):
        self.theta = theta

    def prob(self, s, a):                         # π(a; s, θ)
        ...

    def sample(self, s):                          # a ~ π(·; s, θ)
        ...

    def score(self, s, a):
        # ψ(s,a) = ∇_θ log π(a;s,θ) = ∇π/π  — the compatible features.
        # For Gibbs π ∝ exp(θ·φ): ψ(s,a) = φ(s,a) − Σ_b π(s,b) φ(s,b)   (mean-zero ⇒ advantage features)
        ...


def policy_gradient(policy, mdp, Q):
    # ∇η(θ) = Σ_{s,a} ρ^π(s) ∇π(a;s,θ) Q^π(s,a) = Σ ρ^π π ψ Q   (no ∂ρ/∂θ term — sample-estimable)
    g = np.zeros_like(policy.theta)
    for s in mdp.states:
        rho = mdp.stationary_prob(s, policy)
        for a in mdp.actions:
            g += rho * policy.prob(s, a) * policy.score(s, a) * Q(s, a)
    return g


def fisher_metric(policy, mdp, ridge=1e-3):
    # F(θ) = E_{s~ρ^π} E_{a~π} [ ψ ψ^T ]  — the average per-state Fisher = local KL Hessian.
    # Online form along a trajectory: F ← F + ψ_t ψ_t^T ; F/T is consistent.
    m = policy.theta.shape[0]
    F = np.zeros((m, m))
    for s in mdp.states:
        rho = mdp.stationary_prob(s, policy)
        for a in mdp.actions:
            psi = policy.score(s, a)
            F += rho * policy.prob(s, a) * np.outer(psi, psi)
    return F + ridge * np.eye(m)        # regularized positive metric when F is singular/ill-conditioned


def ascent_direction(policy, mdp, grad):
    # The natural gradient: steepest ascent under the Fisher metric on the policy manifold.
    #   ∇̃η = F^{-1} ∇η  — covariant, reduces to `grad` when F = I.
    # Equivalently it is the compatible-critic weight vector w̄ solving F w̄ = ∇η
    # (least-squares fit of w^T ψ to Q^π): the critic we already fit *is* the natural gradient.
    F = fisher_metric(policy, mdp)
    return np.linalg.solve(F, grad)     # = F^{-1} grad = w̄


def kl_step_size(grad, nat_grad, eps):
    # Fix the local KL budget: ½ α^2 ∇η^T F^{-1} ∇η = ε  ⇒  α = sqrt(2ε / (∇η·∇̃η)).
    denom = grad @ nat_grad
    if denom <= 0:
        return 0.0
    return np.sqrt(2.0 * eps / denom)


def train(policy, mdp, n_iters, eps=1e-2):
    for _ in range(n_iters):
        Q = estimate_value(policy, mdp)             # differential Q^π under average reward
        grad = policy_gradient(policy, mdp, Q)
        nat = ascent_direction(policy, mdp, grad)   # ∇̃η = F^{-1} ∇η = w̄  (greedy-pointing direction)
        alpha = kl_step_size(grad, nat, eps)        # equal local-KL step under the regularized metric
        policy.theta = policy.theta + alpha * nat
    return policy
```
