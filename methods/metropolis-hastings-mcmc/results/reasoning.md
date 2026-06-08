I have an observable F and I want its thermal average over a system of a few hundred interacting particles. The thing I actually have to compute is a ratio of integrals over configuration space,

  F̄ = ∫ F(x) e^{−E(x)/kT} dx / ∫ e^{−E(x)/kT} dx,

x being the positions of all the particles, a point in a space of several hundred dimensions. The momenta have already integrated out because the forces don't depend on velocity, so the whole difficulty sits in these two configuration integrals. The numerator and denominator are both hopeless one at a time — the denominator is the partition function, itself a several-hundred-dimensional integral I can't do — but their ratio is a finite, physical number, and that ratio is all I want. So I should stop thinking about the integrals separately and think about the probability density they define: the Boltzmann weight p(x) ∝ e^{−E(x)/kT}. F̄ is just the expectation of F under p. If I could draw configurations distributed according to p, I could replace the integral by an average over draws.

Quadrature is out. A grid with even a few nodes per axis has an absurd number of nodes once I have hundreds of axes; the curse of dimensionality kills it outright. The thing that does survive in high dimension is random sampling — that's the lesson of the Monte Carlo integration von Neumann and Ulam set going: throw down random points, average the integrand, and the error falls like 1/√N regardless of how many dimensions there are. So my instinct is to scatter configurations at random and weight them.

Let me try the most naive version and see where it breaks. Draw configurations x_i uniformly over the box and estimate

  F̄ ≈ Σ_i F(x_i) e^{−E(x_i)/kT} / Σ_i e^{−E(x_i)/kT}.

This is correct in the limit and it never needs the partition function — the normalizer cancels between the two sums. But now picture the particles at liquid or solid density. If I drop particle centers down uniformly at random, the chance that no two of them overlap is microscopic. Almost every configuration I generate has some pair of particles essentially on top of each other, the potential energy is huge, and e^{−E/kT} is effectively zero. So nearly every draw contributes nothing to either sum, and the whole estimate is held up by the occasional fluke configuration that happens to have no bad overlaps. I'm searching a haystack for the few needles that carry the probability. The sample size is enormous but the *effective* sample size — the number of draws that actually matter — is tiny. This is the wall: uniform sampling spends all its effort exactly where the distribution doesn't live.

The textbook fix for "the integrand lives in a small region" is importance sampling: don't draw from the uniform measure, draw from some density q that's concentrated where the integrand is big, and correct each draw by the weight p(x)/q(x). For J = ∫ f(x) p(x) dx that gives Ĵ = (1/N) Σ_i f(x_i) p(x_i)/q(x_i). Fine in low dimension. But in a few hundred dimensions the weights p/q misbehave badly — for any q I can actually write down, a handful of draws get astronomically large weights and the rest get negligible ones, so the estimator's variance is dominated by a few points and I can't even trust the error bars. And there's a second problem that's specific to my situation: p's normalizer is the partition function, which I don't know, so the raw unbiased weight p(x)/q(x) is not available as an absolute number. A self-normalized ratio could cancel that constant, but it would still be built out of the same unstable weights. So importance sampling, at least in this direct form, is no good here.

Step back. What's the common thread in everything that just failed? I keep trying to *generate independent draws* from p — uniformly, or from some clever q — and the trouble is always that I'm trying to land, in one independent shot, inside a tiny high-probability region of a huge space, without knowing where it is and without knowing the normalizer. Independent draws are the problem. What if I give up on independence? Suppose I already have one configuration that's a "good" one — no bad overlaps, the kind that carries real weight — and instead of drawing the next configuration from scratch, I make a *small local change* to the one I have. A small move from a good configuration is likely to land on another good configuration. If I can arrange the rule for these local moves so that, run for a long time, the sequence of configurations is distributed according to p, then I never need to find the good region from nothing — I just walk around inside it. And crucially, a local move only ever asks me about an *energy difference*, E(new) − E(old), which I can compute, never the absolute partition function.

So I'm no longer drawing iid samples; I'm building a stochastic process — a chain — where each configuration depends only on the previous one. That's a Markov chain. And I know the relevant fact about finite Markov chains: an irreducible chain has a unique stationary distribution π, the one fixed by π = πP, meaning Σ_i π_i p_ij = π_j for every j, where p_ij is the probability of stepping from state i to state j. With aperiodicity, the chain forgets its start and settles into π, and a time average along one long run, (1/N) Σ_t F(X(t)), converges to the expectation of F under π. The classical use of such chains went one way: the physics hands you the transition law p_ij a priori — how radiation scatters, say — and you compute the stationary distribution that results. I need to run that backwards. I am handed the distribution I want, π = p = Boltzmann, and I have to *construct* a transition law p_ij that has it as its stationary distribution. The needle-in-a-haystack search becomes: find a P whose fixed point is the distribution I already know how to write down up to a constant.

So the unknown is now a matrix P, or rather a rule, and the constraint is Σ_i π_i P_ij = π_j for all j, where P_ij is the probability of going from i to j in one step. That's a global condition — it couples a whole column of P together — and there are a vast number of P's that satisfy it. I need a constructive handle: some *local*, per-pair condition I can impose on the transition between two configurations that *guarantees* the global stationarity falls out, and that I can actually enforce one move at a time without ever summing over the whole space.

The local condition I can try is to demand that, in equilibrium, the probability flow between any two states balances pairwise:

  π_i P_ij = π_j P_ji for every pair i, j.

Read it as a statement about flux: at equilibrium the number of systems making the transition i → j equals the number making the transition j → i. Each individual pair of states is in balance, not just the totals — that's why it deserves to be called *detailed* balance. Does it give me stationarity? Sum over the source state i while holding the destination j fixed:

  Σ_i π_i P_ij = Σ_i π_j P_ji = π_j Σ_i P_ji = π_j · 1 = π_j.

The last sum is a row sum: with j fixed, P_ji ranges over all destinations i reachable from source j, including the stay-at-j term P_jj, so it is exactly one. That's π = πP. So detailed balance is a *sufficient* condition for π to be stationary, and it's the local handle I wanted: I never have to look at a whole column at once. If every single transition I ever make respects π_i P_ij = π_j P_ji, the chain's stationary distribution is automatically π. (It's only sufficient, not necessary — there are stationary chains that don't balance pairwise — but sufficiency is all I need, and the locality is the prize.)

Now build a transition rule that satisfies it. I'll split each step into two pieces: first *propose* a move, then *decide* whether to take it. Let me propose moves with some rule q — pick a particle, displace it to a random new position inside a small square centered on its current position. By the symmetry of "a small square centered on where it is," the a-priori probability of proposing i → j equals the a-priori probability of proposing j → i: the proposal is symmetric, q_ij = q_ji. If I just *accepted* every proposal, the chain's transition probability would be q itself, and q has no idea that I want π = Boltzmann; its stationary distribution would be uniform, not Boltzmann. So I need an accept/reject filter on top of the proposal — borrowing the spirit of acceptance–rejection sampling, propose then keep-or-throw — and the filter's job is to bend the symmetric proposal into something that balances against π.

For i ≠ j, let the moving part of the actual transition be P_ij = q_ij α_ij, where α_ij ∈ [0,1] is the probability I accept a proposed move from i to j. The diagonal is whatever probability mass remains after all possible rejections, P_ii = 1 − Σ_{j≠i} P_ij. Plug the off-diagonal part into detailed balance:

  π_i q_ij α_ij = π_j q_ji α_ji.

With the symmetric proposal q_ij = q_ji, the q's cancel and I'm left with

  π_i α_ij = π_j α_ji, i.e. α_ij / α_ji = π_j / π_i.

So the *ratio* of the two acceptance probabilities is pinned to π_j/π_i, but their overall scale is free. To mix as fast as possible I want acceptances as large as possible — rejecting wastes a step — so I should push α up against its ceiling of 1. Suppose π_j ≥ π_i, i.e. the move goes "uphill" in probability, toward a more probable configuration. Then make that direction certain: α_ij = 1. The constraint then forces the reverse, downhill, direction to be α_ji = π_i/π_j ≤ 1, which is a legal probability. If instead π_j < π_i, the move is downhill in probability; set α_ji = 1 for the reverse uphill move and α_ij = π_j/π_i ≤ 1 for this one. Both cases collapse into one formula:

  α_ij = min(1, π_j / π_i).

Always accept a move to a more probable state; accept a move to a less probable state with probability equal to the ratio of the two probabilities. Look at what's inside the min: it's π_j / π_i, a *ratio* of the target at two configurations. The partition function is in π_j and in π_i identically, so it cancels:

  π_j / π_i = e^{−E(x_j)/kT} / e^{−E(x_i)/kT} = e^{−(E_j − E_i)/kT} = e^{−ΔE/kT}.

The acceptance depends only on the energy *difference* between the proposed and current configurations. The unknown normalizer — the very thing that made the integral and importance sampling impossible — never appears. That's not a convenience; that's the property that makes the whole thing usable. I can know my target only up to its normalizing constant and still build a chain whose fixed point is exactly that target, because every decision the algorithm makes is a ratio in which the constant divides out.

So the move rule is concrete: pick a particle, propose displacing it into a small square about its position; compute ΔE = E(proposed) − E(current). If ΔE ≤ 0 the proposed configuration is at least as probable — accept, move the particle. If ΔE > 0, draw a uniform random number ξ in (0,1) and accept if ξ < e^{−ΔE/kT}, otherwise leave the particle where it is. Then average F over the configurations visited.

Now a subtle point that I'd get wrong if I weren't careful, and it's exactly where the detailed-balance bookkeeping bites. When I reject a move, what do I record? The temptation is to discard the rejected step and only tally configurations I actually moved to. That's wrong. The transition probability of *staying* at i is P_ii = 1 − Σ_{j≠i} P_ij — the leftover probability mass from all the proposals I declined. That self-loop is a real part of the Markov chain, and the stationarity proof above used the full row sum of P with the diagonal included. Think of it as conservation of probability: each step the chain has to go *somewhere*, and "stay put" is a legitimate destination carrying its share of the mass. If I refuse to count the repeats, I'm silently deleting that diagonal mass, and the configurations I declined to leave get under-represented — probability piles up in the wrong places and the distribution I sample is no longer π. So every step, accepted or not, counts as a member of the ensemble; a rejected move means the current configuration is recorded *again*. The reject isn't a wasted draw to be thrown away — it's the chain's way of giving extra weight to a configuration whose neighbors are less probable than it is.

Let me check this didn't just secretly assume things I have to earn. First, ergodicity. Detailed balance makes π stationary, but convergence asks for more: the chain has to be able to reach every part of the support and avoid a fixed cycle. If the physically allowed configuration space is connected, and if the small moves can be chained while staying inside that allowed space, then the local proposal gives positive probability to paths between allowed configurations. On that connected support the chain is irreducible; with the self-loop from rejection, or the usual continuous-state aperiodicity condition, the stationary distribution is the one it converges to. Good — "any state reachable from any other" is what I have to keep checking.

Second, the proposal-step size. There's a free knob I glossed over: how big to make the little square the particle can jump into. This matters and the failure is two-sided. If the square is huge, a proposed move almost always shoves the particle into some other particle — ΔE is large and positive, e^{−ΔE/kT} is tiny, and almost every move is rejected. The chain just sits there; nothing changes; mixing is glacial. If the square is tiny, every move is accepted because ΔE ≈ 0, but each move barely budges the configuration, so it takes forever to diffuse across the space — again slow to equilibrate. The right step size is in between, where a healthy fraction of moves — not all, not almost none — are accepted; a rule of thumb in the neighborhood of half the moves accepted keeps both pathologies away. The acceptance ratio is itself the diagnostic: watch it, tune the step until it's sensible.

Third — and this is the warning I should keep in view — detailed balance, reachability, and aperiodicity can give convergence *eventually*, but they say nothing about *how fast*. If the distribution has two well-separated high-probability regions divided by a deep low-probability valley, a small local move almost never crosses the valley, so in any feasible run the chain stays stuck near whichever region it starts in and reports averages as if the other doesn't exist. The math is right; the run can still be too short. So in practice I have to throw away an initial stretch while the chain forgets its starting configuration, and I should worry about whether one realization has actually explored everything — comparing different segments of the run, or runs from different starts, for signs of non-convergence.

I need to sample from a hundred-dimensional density whose normalizer I can't compute, and the inner loop only evaluates e^{−ΔE/kT} for a single displaced particle — a local energy change — and compares it to a random number. The hundred dimensions, the partition function, the global structure of the distribution: none of them appear in the inner loop. They're all absorbed into the fixed point of the chain.

Now let me push on the one structural assumption I leaned on: the symmetric proposal q_ij = q_ji. That's what made the q's cancel and gave the clean min(1, π_j/π_i). It came for free from "a small square centered on the current position." But it's a real restriction. What if I want to propose moves with a *biased* rule — drawing candidates from some distribution that's easier to sample, or that pushes preferentially in a useful direction, or that proposes a brand-new configuration independent of the current one? Then q_ij ≠ q_ji in general, and my derivation breaks: re-run it without the symmetry assumption and detailed balance reads

  π_i q_ij α_ij = π_j q_ji α_ji ⟹ α_ij / α_ji = (π_j q_ji) / (π_i q_ij).

The q's no longer cancel. But the structure is identical — it's still a ratio pinning the two acceptances — so the same maximize-acceptance argument applies. Define the test ratio

  R = (π_j q_ji) / (π_i q_ij),

and whichever direction has R ≥ 1 gets acceptance 1, forcing the other to R^{−1}. Both cases collapse to

  α_ij = min(1, (π_j q_ji) / (π_i q_ij)).

The symmetric case is just q_ij = q_ji, where the proposal ratio is 1 and I'm back to min(1, π_j/π_i). The new factor q_ji/q_ij is a correction that exactly undoes the asymmetry of the proposal: if my proposal mechanism makes i → j easier than j → i, the correction shaves the acceptance of i → j by the proposal's own bias, restoring the balance that the proposal alone would have broken. And the normalizer of π still cancels — R only ever uses π_j/π_i — so I'm still free of the partition function. Now q is no longer limited to symmetric local moves. I can use an independence proposal, a biased random walk, a group-valued move, whatever is convenient, provided I know the forward and reverse proposal densities on the moves I permit and I don't choose a support that cuts the state space into unreachable pieces. This is a strict widening of what I can sample and how cheaply.

Worth asking whether min(1, R) is the only sensible filter or just one choice. On a pair with positive R, the constraint was only α_ij / α_ji = R; the min rule is the particular solution that pushes both acceptances as high as they'll go. I could instead write the general form

  α_ij = s_ij / (1 + (π_i q_ij)/(π_j q_ji)) = s_ij / (1 + 1/R),

with s_ij a symmetric function of i, j. The symmetry is what makes the ratio work: swapping i and j replaces R by 1/R, so α_ji = s_ij/(1 + R), and therefore α_ij/α_ji = (1 + R)/(1 + 1/R) = R. The probability bounds for both directions require 0 ≤ s_ij ≤ 1 + min(R, 1/R).

The largest admissible symmetric choice is s_ij = 1 + min(R, 1/R). Check both branches. If R ≥ 1, then s_ij = 1 + 1/R, so α_ij = (1 + 1/R)/(1 + 1/R) = 1, while α_ji = (1 + 1/R)/(1 + R) = 1/R. If R < 1, then s_ij = 1 + R, so α_ij = (1 + R)/(1 + 1/R) = R, while the reverse direction has α_ji = 1. That is exactly α_ij = min(1, R), the choice that drives the larger of the two acceptances to 1. The other natural choice is s_ij = 1, giving Barker's filter α_ij = 1/(1+1/R) = R/(1 + R) = (π_j q_ji)/(π_i q_ij + π_j q_ji); for a symmetric proposal that's π_j/(π_i + π_j). Compare the two when π_i = π_j: my min rule accepts the move with probability 1, while Barker's accepts with probability only 1/2. So Barker's rule sits still half the time even when the two configurations are equally probable and there's nothing to gain by staying — it wastes moves. The min rule accepts as often as detailed balance permits, which means faster decorrelation per step.

Let me make sure the whole thing carries over from a discrete state space to the continuous configuration space I actually have, because the particles live in ℝ^d, not on a finite list of states. Replace the masses π_i by a target density π(x) (known up to a constant — its normalizer K, the partition function, which I'll never need), and the proposal q_ij by a candidate-generating density q(x, y): from the current point x it proposes y with density q(x, y), ∫ q(x, y) dy = 1. The transition kernel has a density part plus an atom for staying put:

  P(x, dy) = q(x, y) α(x, y) dy + r(x) δ_x(dy), r(x) = 1 − ∫ q(x, y) α(x, y) dy,

where r(x) is the probability the chain remains at x — the continuous version of the diagonal self-loop, again carrying the rejected mass. Detailed balance now reads π(x) q(x, y) α(x, y) = π(y) q(y, x) α(y, x), and the same derivation gives

  α(x, y) = min(1, [π(y) q(y, x)] / [π(x) q(x, y)]).

Let me verify, the careful way, that detailed balance on the density part really does make π invariant in continuous space — I want to see the staying-put atom pull its weight, because that's the piece I keep being tempted to drop. Write p(x, y) = q(x, y) α(x, y) for the off-diagonal density, so it satisfies π(x) p(x, y) = π(y) p(y, x). The stationarity I need is ∫ P(x, A) π(x) dx = ∫_A π(y) dy for every set A: starting from π and taking one step leaves me in π. Expand the left side using the kernel's two pieces:

  ∫ P(x, A) π(x) dx = ∫ [ ∫_A p(x, y) dy ] π(x) dx + ∫ r(x) δ_x(A) π(x) dx.

In the first term swap the order of integration and use detailed balance, p(x, y) π(x) = p(y, x) π(y):

  ∫_A [ ∫ p(x, y) π(x) dx ] dy = ∫_A [ ∫ p(y, x) π(y) dx ] dy = ∫_A π(y) [ ∫ p(y, x) dx ] dy = ∫_A (1 − r(y)) π(y) dy,

since ∫ p(y, x) dx is the total probability of *moving* away from y, which is 1 − r(y). The second term is just the staying-put contribution: δ_x(A) is 1 exactly when x ∈ A, so ∫ r(x) δ_x(A) π(x) dx = ∫_A r(x) π(x) dx. Add them:

  ∫_A (1 − r(y)) π(y) dy + ∫_A r(y) π(y) dy = ∫_A π(y) dy.

The r terms cancel exactly — the move-away deficit (1 − r) in the first piece is filled in precisely by the stay-put mass r in the second — and I land on ∫_A π(y) dy. So π is invariant. That cancellation is the formal statement of the conservation-of-probability bookkeeping I worried about earlier: the rejected mass r(y) is not optional decoration, it is exactly what makes the books balance. With the usual irreducibility and aperiodicity conditions — proposal support that connects the support of π and positive acceptance along those paths — the chain converges to π and time averages converge to E_π(F).

I need an expectation against a hundred-dimensional distribution I know only up to its normalizer. Independent samples fail — uniform draws miss the tiny region that carries the mass, and importance weights blow up in high dimension and need the normalizer anyway. The escape is to stop drawing independently and instead walk a Markov chain whose stationary distribution *is* the target; then time-averaging along one run gives the expectation. To construct such a chain locally I impose detailed balance, π_i P_ij = π_j P_ji, which sums up to stationarity π = πP when the full transition matrix, including the stay-put mass, has rows summing to one. Writing each move as propose-then-accept, detailed balance pins the ratio of forward and reverse acceptances, and pushing acceptances to their ceiling gives α = min(1, [π(y) q(y, x)] / [π(x) q(x, y)]) — accept always if the test ratio exceeds 1, otherwise accept with that probability, and on a rejection record the current configuration again so the probability books conserve. Because the rule depends on the target only through the ratio π(y)/π(x), the unknown normalizer cancels, which is the one property that makes a distribution-known-up-to-a-constant samplable at all. The symmetric proposal recovers the bare min(1, π(y)/π(x)); the general q(y, x)/q(x, y) correction lets me use asymmetric proposals without breaking balance.

```python
import numpy as np

def transition(x, log_weight, candidate_rule, rng):
    """One transition.

    log_weight(x)          : unnormalized log target; any additive normalizing
                             constant cancels because only differences are used.
    candidate_rule(x, rng) : returns (y, log_q_forward, log_q_reverse), the
                             candidate y and the log proposal densities
                             log q(x->y), log q(y->x).
    returns the next state: y on accept, x itself on reject. The rejected state
    is counted again as the diagonal mass that conserves probability.
    """
    y, log_q_forward, log_q_reverse = candidate_rule(x, rng)

    # log[ pi(y) q(y->x) / ( pi(x) q(x->y) ) ]; the normalizer cancels.
    log_R = (log_weight(y) - log_weight(x)) + (log_q_reverse - log_q_forward)

    # alpha = min(1, R): accept immediately when R >= 1,
    # otherwise compare log(u) with log_R to avoid underflow.
    if log_R >= 0.0:
        return y
    if np.log(rng.uniform()) < log_R:
        return y          # accept: move
    else:
        return x          # reject: stay -- record x again

def sample(x0, log_weight, candidate_rule, n_steps, burn_in, rng):
    """Run the chain and return post-burn-in states."""
    x = x0
    samples = []
    for t in range(n_steps):
        x = transition(x, log_weight, candidate_rule, rng)
        if t >= burn_in:                # discard the transient while the chain
            samples.append(np.copy(x))  # forgets x0 and reaches stationarity
    return samples

def symmetric_random_walk_candidate(step):
    """Symmetric local candidate; the proposal-density correction vanishes."""
    def candidate_rule(x, rng):
        y = x + step * rng.uniform(-1.0, 1.0, size=np.shape(x))  # small box about x
        return y, 0.0, 0.0   # log q(x->y) == log q(y->x)
    return candidate_rule

def estimate(samples, observable):
    """Average the observable over retained states, including repeats."""
    return np.mean([observable(c) for c in samples], axis=0)
# step is tuned so that a healthy fraction of moves (~half) are accepted:
# too large -> almost all proposals rejected, chain frozen;
# too small -> moves accepted but configuration barely diffuses.
```
