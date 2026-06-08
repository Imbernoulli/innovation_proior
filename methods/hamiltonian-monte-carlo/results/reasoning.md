Let me start from what actually goes wrong, because the symptom is specific and I think it points somewhere. I have a target π(q) on R^d that I know only up to a constant — a posterior, say — and I want ∫ f(q) π(q) dq. The honest way to get it is to draw samples and average. The classic machine for that is the chain of Metropolis, Rosenbluth, Rosenbluth, Teller and Teller from 1953: to sample P(x) ∝ exp(−E(x)), I sit at x, propose x* = x + N(0, ς²I), and accept it with probability min[1, exp(−(E(x*) − E(x)))]; if I reject I stay put and count x again. The proposal is symmetric, so its density cancels, and the accept rule is exactly what makes detailed balance hold — P(x) T(x*|x) = P(x*) T(x|x*) — so the chain leaves P invariant. That part is airtight and I won't touch it.

What's wrong is the *proposal*. It's a blind, isotropic kick. And in high dimension that's a disaster, and I want to be precise about *why*, not just wave at "the curse of dimensionality."

I keep coming back to the same question. Where does the probability of a high-dimensional distribution actually live? Not at the mode. The density is biggest at the mode, yes, but expectations integrate density against *volume*, and volume in high d is overwhelmingly *away* from any given point. Picture boxing up parameter space into cells around the mode: in 1D the mode's cell has two neighbors, in 2D it has eight, in 3D twenty-six, in general 3^d − 1 — the volume immediately around the mode is dwarfed by the volume just outside it, and this only worsens as I go further out. So the product π(q) dq — density times volume, the thing that actually contributes to the integral — doesn't peak at the mode and doesn't peak in the far tails where the density has died; it concentrates in a thin shell at intermediate radius. Call it the typical set. Any sampler that's going to estimate ∫ f π well has to spend essentially all its evaluations inside this shell.

Now look at what the random walk does to a thin shell. The proposal is biased toward volume — toward the *outside* of the shell, the tails — because that's where the volume is. So a large ς proposes points off the shell into the low-density tail, where exp(−ΔE) is tiny and I reject almost everything; the chain just sits. I can fix the acceptance by shrinking ς until proposals stay inside the shell — but then each accepted move is tiny, comparable to the shell's *thinnest* width. And critically, consecutive steps are uncorrelated in direction: it's a diffusion. The net displacement after n steps grows like √n, not n. So to traverse the long, least-constrained direction of the distribution I need a number of steps that goes like the square of the ratio of the widths.

Let me actually pin the exponent so I believe it. Take U(q) = Σᵢ uᵢ(qᵢ) and an independent proposal per coordinate. Expand U(x*) ≈ U(x) + c U'(x) + ½ c² U''(x) for a step of size c; average Δ₁ = U(x*) − U(x) over c = +a and c = −a and the first-order term cancels, leaving ½a² U''(x); average over the proposal scale and over x and E[Δ₁] ∝ ς². Over d coordinates E[Δ_d] ∝ d ς². There's also a beautiful identity I shouldn't forget: at equilibrium, for a reversible proposal that pairs every move with its reverse, 1 = E[P(x*)/P(x)] = E[exp(−Δ)], so by Jensen E[Δ] ≥ 0, and expanding exp(−Δ₁) ≈ 1 − Δ₁ + Δ₁²/2 gives E[Δ₁] ≈ E[Δ₁²]/2 — the variance of the energy gap is twice its mean. So to keep acceptance from collapsing I have to keep E[Δ_d] of order one, which forces ς ∝ d^{−1/2}; and since exploration is diffusive the number of iterations to a near-independent state goes like ς^{−2} ∝ d, so wall-clock cost ∝ d². Gibbs, updating a coordinate at a time, has exactly the same √n problem — each scan moves about a constrained width and the directions don't add up coherently. Two orders of magnitude in dimension is four orders in cost. That's the wall.

So I don't want a blinder, undirected kick. I want a proposal that *knows which way the typical set runs* and slides a long way along it before I stop and check. The only directional information I have for free is the gradient ∇ log π(q) = −∇U(q). So my first instinct is: follow the gradient. Step in the direction of steepest ascent of log π.

That fails, and it fails for an instructive reason. The gradient points *toward the mode*. If I follow it I'm pulled straight off the shell and into the mode — exactly the low-volume neighborhood I established carries almost no probability mass. The gradient is sensitive to the shape of π but it points *across* the typical set, inward, not *along* it. Think of the mode as a planet and −∇U as its gravity: a particle that just falls along gravity crashes into the surface. I need something that orbits — that uses the gravitational information to stay *on* a shell and travel around it, not plunge through it.

What keeps a satellite in orbit instead of crashing? Momentum. If I give the particle sideways momentum, gravity bends its path but doesn't capture it into the center; as it falls inward the momentum builds and flings it back out, as it drifts outward the momentum bleeds off and gravity reins it back — the exchange between the two balances along an orbit that stays at roughly constant radius. That's exactly the curve I want: motion that hugs a level set rather than diving for the minimum. So the move is to stop trying to push q directly with the gradient, and instead *route* the gradient through an auxiliary momentum.

Concretely: invent a momentum variable p, one component per component of q, doubling the state to (q, p), and channel the dynamics so the gradient acts on p, and p in turn carries q. The mechanism for "energy that trades between a configuration term and a momentum term and is conserved" is Hamiltonian mechanics. Let me set up a Hamiltonian H(q, p) and let the system flow by Hamilton's equations.

But I have to choose p's distribution so that this whole detour actually samples π and isn't just a physics fantasy. Here's the lever. The canonical/Boltzmann correspondence says any density is exp(−energy) up to a constant. So lift the target to a *joint* density on (q, p) that is itself canonical in a Hamiltonian:

  π(q, p) = (1/Z) exp(−H(q, p)),   with   H(q, p) = U(q) + K(p),   U(q) = −log π(q).

Because H splits additively, the joint factorizes: π(q, p) ∝ exp(−U(q)) exp(−K(p)) — q and p are *independent*, and the q-marginal is exactly exp(−U(q)) ∝ π(q). That's the contract: whatever I do in the joint space, as long as I leave exp(−H) invariant, I can throw p away at the end and what's left in q is distributed as π. The momentum is pure scaffolding; I get to pick K(p) however is convenient. I'll take K quadratic, K(p) = pᵀM⁻¹p / 2, which makes p a zero-mean Gaussian with covariance M (the "mass matrix"; default M = I). Quadratic K has two virtues I'll cash in shortly: ∂K/∂p = M⁻¹p is linear, and resampling p from its Gaussian is trivial.

Now the dynamics. Hamilton's equations:

  dqᵢ/dt = ∂H/∂pᵢ = [M⁻¹p]ᵢ,
  dpᵢ/dt = −∂H/∂qᵢ = −∂U/∂qᵢ.

Stare at this. The gradient of U — the thing that on its own pulled me into the mode — now drives the *momentum*, not the position directly; and the momentum drives the position. That's the twist I was after: the gradient is being filtered through p, so instead of pointing me at the mode it bends a moving trajectory so it curves along the level set. The puck slides over the surface −log π; on a slope its momentum is converted to height and back, and it skirts around rather than rolling to the bottom.

Three facts about this flow turn it from a nice picture into a legal proposal, and I want to derive them because the whole method's correctness rests on them.

First, energy is conserved. dH/dt = Σᵢ (∂H/∂qᵢ · dqᵢ/dt + ∂H/∂pᵢ · dpᵢ/dt) = Σᵢ (∂H/∂qᵢ · ∂H/∂pᵢ − ∂H/∂pᵢ · ∂H/∂qᵢ) = 0. So a trajectory of the exact flow stays on a surface of constant H — which is a surface of constant joint probability density. This is the payoff: if I run the flow for some time and propose the endpoint as my Metropolis move, the energy difference is *zero*, so the acceptance probability min[1, exp(−ΔH)] is *one*. I can propose a point arbitrarily far from where I started and still accept it with certainty. That is precisely the long, coherent, high-acceptance move the random walk could never make.

Second — and this is the subtle one that makes the Metropolis correction even *applicable* — the flow preserves phase-space volume. If I'm going to use a deterministic map as a proposal inside Metropolis–Hastings, the general accept ratio (Hastings 1970) is min[1, Q(z|z*) π(z*) / (Q(z*|z) π(z))], and for a deterministic, invertible map the proposal "density" carries the Jacobian of the map. If the map stretched and squashed volume I'd have to compute |det J| of the L-step flow — generally hopeless. Hamiltonian flow saves me: its Jacobian determinant is exactly 1. The cleanest way to see it is Liouville — the divergence of the flow's vector field vanishes:

  Σᵢ [ (∂/∂qᵢ)(dqᵢ/dt) + (∂/∂pᵢ)(dpᵢ/dt) ]
   = Σᵢ [ (∂/∂qᵢ)(∂H/∂pᵢ) − (∂/∂pᵢ)(∂H/∂qᵢ) ]
   = Σᵢ [ ∂²H/∂qᵢ∂pᵢ − ∂²H/∂pᵢ∂qᵢ ] = 0,

the mixed partials cancel, zero divergence means an incompressible flow, |det J| = 1. So no Jacobian appears in my accept ratio at all. (More than volume — the flow is symplectic, Bₛᵀ J⁻¹ Bₛ = J⁻¹ with J = [[0, I],[−I, 0]], from which det(Bₛ)² = 1; volume preservation is the piece I need.)

And third, reversibility. Hamilton's flow is invertible, and for H = U + K with K(p) = K(−p) the inverse map is concretely "negate p, run the flow the same duration, negate p again." I'll use this to make the proposal symmetric so Q cancels.

So in the *exact*-dynamics world I'd have a perfect sampler: resample p from its Gaussian, run the flow a while, propose the endpoint, accept with probability one. But I can't integrate Hamilton's equations exactly for a general U. I have to discretize with some stepsize ε. The moment I discretize, two things are at risk: energy is no longer exactly conserved (so acceptance is no longer exactly one), and — far more dangerous — volume preservation and reversibility could break, which would *invalidate the Metropolis correction itself*. Conservation I can afford to lose a little of, because I have an accept/reject step to clean up the residue. Volume preservation and reversibility I cannot lose at all, because they're what make that accept/reject step *correct*.

Try the obvious discretization first, Euler:
  pᵢ(t+ε) = pᵢ(t) − ε ∂U/∂qᵢ(q(t)),
  qᵢ(t+ε) = qᵢ(t) + ε pᵢ(t)/mᵢ.
On the simplest test, H = q²/2 + p²/2 (whose exact flow is just clockwise rotation on a circle, q = r cos(a+t), p = −r sin(a+t)), Euler spirals *outward* to infinity. It doesn't even stay bounded, let alone conserve energy. The reason is exactly the property I can't lose: Euler isn't volume-preserving, and a map that systematically inflates volume drifts off to infinity. Dead end, but a diagnostic one — it tells me the discretization must be volume-preserving by construction.

Small change: when I update q, use the *new* momentum instead of the old:
  pᵢ(t+ε) = pᵢ(t) − ε ∂U/∂qᵢ(q(t)),
  qᵢ(t+ε) = qᵢ(t) + ε pᵢ(t+ε)/mᵢ.
Now each of the two substeps is a *shear*: in the first only p changes, by an amount depending only on q; in the second only q changes, by an amount depending only on the (already-updated) p. A shear's Jacobian is triangular with ones on the diagonal — determinant exactly 1, *at finite ε*, no approximation. So this modified scheme preserves volume exactly, and on the rotation test it no longer blows up; it tracks the true circle, wobbling but stable. That's the structural fix: build the step out of shears.

One thing still bugs me: that modified-Euler step isn't *reversible* — it treats p then q asymmetrically, so "negate p, redo, negate p" doesn't land me back exactly. I want symmetry, both because it'll give a symmetric proposal and because reversibility forces the error to behave. So symmetrize it — split the momentum kick into two half-kicks bracketing a full drift:

  pᵢ(t + ε/2) = pᵢ(t) − (ε/2) ∂U/∂qᵢ(q(t)),
  qᵢ(t + ε)   = qᵢ(t) + ε pᵢ(t + ε/2)/mᵢ,
  pᵢ(t + ε)   = pᵢ(t + ε/2) − (ε/2) ∂U/∂qᵢ(q(t + ε)).

Half-kick the momentum, drift the position a full step using that momentum, half-kick the momentum again using the new gradient. This is the leapfrog (Störmer–Verlet) integrator. Each of the three substeps is a shear, so the whole step has unit-Jacobian — volume preserved exactly at finite ε. And it's manifestly symmetric in time: negate p, apply the same L steps, negate p, and I'm exactly back at the start — reversible exactly. So both of the properties the Metropolis correction depends on survive discretization *exactly*; only energy conservation is approximate. That's precisely the trade I wanted. And there's a small efficiency bonus: when I chain steps, the trailing half-kick of one step and the leading half-kick of the next combine into a single full kick, so a trajectory of L leapfrog steps costs L+1 gradient evaluations rather than 2L.

How big is the energy error, and does it pile up? Leapfrog has local error O(ε³) per step and, because it's reversible, global error of *even* order — O(ε²) over a fixed trajectory length (any reversible method has even-order global error; that's why this symmetric scheme beats first-order Euler). More importantly, the error doesn't accumulate the way a generic integrator's would: because leapfrog preserves volume exactly, it can be shown to *exactly* conserve a nearby "shadow" Hamiltonian, so the numerical trajectory stays pinned near a level set of H and the energy error oscillates within a bound rather than drifting away — even over a long trajectory. On the rotation test at ε = 0.3 the leapfrog orbit is visually indistinguishable from the true circle; push to ε = 1.2 and the error is visible but the orbit is still stable and stays stable indefinitely; only as ε → 2 does it finally go unstable.

Let me nail the stability threshold, because it'll tell me how to choose ε. For H = q²/2σ² + p²/2 a leapfrog step is a linear map; writing it out from the three substeps gives the matrix
  [[ 1 − ε²/2σ²,            ε        ],
   [ −ε/σ² + ε³/4σ⁴,   1 − ε²/2σ² ]],
with eigenvalues (1 − ε²/2σ²) ± (ε/σ)√(ε²/4σ² − 1). When ε/σ > 2 the eigenvalues are real and one exceeds 1 in magnitude — the iteration diverges. When ε/σ < 2 they're complex with squared magnitude (1 − ε²/2σ²)² + (ε²/σ²)(1 − ε²/4σ²) = 1 exactly — neutrally stable, the trajectory neither grows nor decays. So ε must stay below 2σ, i.e. below twice the *most constrained* width of the distribution. That's the leapfrog stability budget, not the whole tuning rule; in high dimensions I may have to go lower to keep the summed energy error acceptably small. (In many directions at once the binding stability constraint is the smallest σ — the narrowest direction sets the ceiling.)

So I've lost a little energy conservation and I need to pay it back exactly. This is where the accept/reject earns its place. I propose by running L leapfrog steps from (q, p), then negating the momentum, giving (q*, p*). Why negate? Because the deterministic map "L leapfrog steps" by itself is invertible, but applying that same forward map from the endpoint usually does not return to the start; the reverse transition would have zero proposal probability unless the trajectory happened to be periodic. Augmenting with a momentum flip fixes this: "L leapfrog steps then negate p" is its own inverse (leapfrog's symmetry), so from (q*, p*) the same operator proposes (q, p) with equal density. The proposal is now symmetric, the Q's cancel, and — because the map is volume-preserving — there's no Jacobian either. What's left in the Metropolis–Hastings ratio is just the density ratio:

  accept with probability min[1, exp(−H(q*, p*) + H(q, p))]
              = min[1, exp(U(q) − U(q*) + K(p) − K(p*))].

This is the exact correction for the leapfrog's energy drift. If the integrator had been perfect, ΔH = 0 and I'd accept with probability one; in practice ΔH is the O(ε²) leapfrog error, which stays small and bounded, so I accept distant proposals with high probability anyway. (In code I can skip the explicit p-negation: K(p) = K(−p) makes it invisible to the accept ratio, and p gets overwritten next iteration.)

Let me check that this accept/reject really leaves the joint distribution invariant, by detailed balance, because I want to be sure the volume-preservation and reversibility are doing exactly the work I claimed. Partition (q, p) space into tiny cells A_k of equal small volume V. Let B_k be the image of A_k under "L leapfrog steps + negate p." Reversibility means this operator is a bijection, so the B_k also tile the space; volume preservation (and negation, which also preserves volume) means each B_k also has volume V. For i ≠ j the operator maps A_i nowhere near A_j, so the proposal probabilities T(A_i | B_j) = T(B_j | A_i) = 0 and detailed balance is trivial. For the matched cell (i = j = k), as cells shrink H is effectively constant on each cell with values H_{A_k}, H_{B_k}, so detailed balance reads
  (V/Z) exp(−H_{A_k}) · min[1, exp(−H_{B_k} + H_{A_k})]
   = (V/Z) exp(−H_{B_k}) · min[1, exp(−H_{A_k} + H_{B_k})].
Multiply out: the left side is (V/Z) min[exp(−H_{A_k}), exp(−H_{B_k})], and so is the right — they're identical. Detailed balance holds, so this Metropolis update leaves exp(−H) invariant. (And invariance follows: if the current state is canonical, the probability the next state lands in B_k is P(B_k)R(B_k) + Σᵢ P(A_i) T(B_k | A_i), where R is the rejection probability; using detailed balance the sum becomes P(B_k) Σᵢ T(A_i | B_k) = P(B_k)(1 − R(B_k)), and the two terms add back to P(B_k).) The exact volume preservation and reversibility of leapfrog are *exactly* what let one Metropolis step fix the discretization error without any Jacobian and without bias.

There's still a gap: the dynamics conserves H, so a single trajectory is stuck on one level set of the joint — it never changes the *value* of H, hence never really changes U(q) on its own. If I only ever ran trajectories I'd be confined to one energy shell. The fix is built into the lift: resample the momentum. So each full iteration is two moves. First, draw a fresh p ~ N(0, M), ignoring the old p. Since q and p are independent in the joint, p's conditional given q equals its marginal, so this Gibbs step leaves exp(−H) invariant trivially — and it *changes* H, because it changes K(p). Since K(p) is nonnegative for the quadratic kinetic energy, fixed-H dynamics alone would keep U(q) ≤ H; resampling p is what lets the chain move between energy shells. Second, the MH-corrected leapfrog move, which I just showed leaves exp(−H) invariant. Each move preserves the joint, so their composition does, and the q-marginal is π. That's the whole algorithm.

Does it actually scale better than the random walk, or did I just trade one cost for another? Run the same energy-gap accounting on HMC. With K = Σ pᵢ²/2 and U = Σ uᵢ(qᵢ) the (qᵢ, pᵢ) pairs don't interact during a trajectory, so trajectory length needn't grow with d. The acceptance depends on the total leapfrog energy error, a sum over the d pairs. The per-pair H-error is O(ε²); by the same E[Δ₁] ≈ E[Δ₁²]/2 relation, squaring the O(ε²) error gives E[Δ₁] ∝ ε⁴, so the total E[Δ_d] ∝ d ε⁴. To keep that of order one I need ε ∝ d^{−1/4}. The number of leapfrog steps to a near-independent point goes like ε^{−1} ∝ d^{1/4}; each gradient evaluation costs O(d), so wall-clock cost ∝ d^{5/4} — against d² for the random walk. The momentum buys me coherent, ballistic motion: distance grows like the number of steps, not its square root, so the cost advantage is roughly the ratio of the least- to most-constrained widths. And working the acceptance algebra (Δ_d is a sum of d independent gaps, so by the central limit theorem it's Gaussian with variance twice its mean; accept rate = 2Φ(−√(μ/2))), the optimal acceptance for HMC comes out near 0.65, versus 0.23 for random walk — HMC *wants* to accept more, because each move is doing real, directed work.

A couple of tuning realities fall out. ε is capped by stability at the most-constrained width, so a high-curvature region forces a small ε — and if I pick ε from preliminary runs in a wide region it may be too large elsewhere, producing proposals with huge H-error and tiny acceptance. Randomizing ε once per trajectory can occasionally supply a safer smaller step, and randomizing ε or L also helps avoid exact periodicities that would break ergodicity (e.g. when Lε hits a period of the dynamics and the trajectory loops back to where it started). And the trajectory length εL should be long enough to cross the least-constrained direction but not so long it U-turns and wastes gradient evaluations.

Let me write the single iteration concretely, with K(p) = Σ pᵢ²/2 (M = I), the half-step bracketing, and the merged interior full steps:

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

Tracing the causal chain once more, in the order it actually forced itself on me: the probability mass of a high-d target lives on a thin typical set, and an undirected random-walk proposal explores it only diffusively, so cost scales like d². The gradient is the only directional information I have, but followed directly it points off the set toward the mode and crashes. Routing the gradient through an auxiliary momentum — lifting π to the canonical joint exp(−H), H = U + K, so the q-marginal is exactly π — turns gravity into orbit: Hamilton's flow conserves H, so it glides along a constant-probability surface, making distant proposals that would be accepted with probability one. Discretizing with leapfrog keeps the two properties the correction depends on — volume preservation and reversibility — *exactly*, sacrificing only energy conservation to O(ε²); and a single Metropolis accept, min[1, exp(−ΔH)], pays that residue back exactly, with no Jacobian because volume is preserved and a symmetric proposal because the flip makes it reversible. Resampling the momentum each iteration moves the system between energy shells. The result moves a long, coherent distance per iteration at high acceptance, and scales like d^{5/4} instead of d².
