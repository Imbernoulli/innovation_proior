# The Metropolis–Hastings algorithm

## Problem

Compute expectations E_π(F) = ∫ F(x) π(x) dx against a target distribution π on a high-dimensional space when π is known only up to a normalizing constant: π(x) = f(x)/K with f computable but K = ∫ f(x) dx intractable. The motivating case is the canonical-ensemble average in statistical mechanics, π(x) ∝ e^{−E(x)/kT}, where K is the partition function. Direct quadrature fails (curse of dimensionality), uniform Monte Carlo wastes nearly all draws on near-zero-probability configurations, importance-sampling weights p/q degenerate in high dimension (and the raw unbiased form asks for K), and acceptance–rejection needs a global dominating envelope.

## Key idea

Do not try to draw independent samples from π. Instead build a Markov chain whose **stationary distribution is π**, run it, and average F along the trajectory; by ergodicity (1/N) Σ_t F(X(t)) → E_π(F). The chain is constructed by enforcing **detailed balance** through a propose-then-accept/reject step. Crucially, the accept/reject decision depends on π only through the **ratio** π(y)/π(x), so the unknown normalizer K cancels — that is what makes a distribution-known-up-to-a-constant samplable.

## The algorithm

Given the current state x:
1. Propose a candidate y from a candidate-generating density q(x, ·) (∫ q(x,y) dy = 1).
2. Compute the acceptance probability

   α(x, y) = min( 1, [ π(y) q(y, x) ] / [ π(x) q(x, y) ] ).

   Only ratios π(y)/π(x) enter, so K cancels: with π = f/K, the ratio is f(y)q(y,x)/[f(x)q(x,y)].
3. Draw u ∼ Uniform(0,1). If u ≤ α(x, y), set the next state to y (accept); otherwise set it to x (reject — **re-record the current state**).

The transition kernel is

  P(x, dy) = q(x, y) α(x, y) dy + r(x) δ_x(dy),  r(x) = 1 − ∫ q(x, y) α(x, y) dy,

the δ_x atom holding the rejected mass. Run for n steps, discard an initial burn-in while the chain forgets x₀, and average the observable over every retained step (rejected steps counted as repeats).

**Special cases.**
- **Metropolis (symmetric proposal)** q(x,y) = q(y,x) (e.g. a random walk y = x + symmetric noise): the q factors cancel and α(x,y) = min(1, π(y)/π(x)). Uphill-in-probability moves are always accepted; downhill moves accepted with probability π(y)/π(x). For the Boltzmann target this is min(1, e^{−ΔE/kT}).
- **Hastings (asymmetric proposal)** the factor q(y,x)/q(x,y) corrects for the proposal's bias, freeing q to be asymmetric as long as the relevant forward/reverse proposal densities are known and the proposal support preserves reachability.
- **General Hastings family**: for positive R = π(y)q(y,x)/[π(x)q(x,y)], any symmetric s(x,y) with 0 ≤ s ≤ 1 + min(R,1/R) gives α(x,y) = s(x,y)/(1 + 1/R). The largest choice, s = 1 + min(R,1/R), gives the min rule: if R ≥ 1 then α = 1; if R < 1 then α = R. Barker's filter is s = 1, hence α = R/(1+R) = π(y)q(y,x)/[π(x)q(x,y) + π(y)q(y,x)]; for equal target-proposal fluxes it accepts with probability 1/2 while the min rule accepts with probability 1.

## Why it works (detailed balance ⇒ stationarity)

**Detailed balance / reversibility.** The construction enforces, for the moving part of the transition,

  π(x) q(x, y) α(x, y) = π(y) q(y, x) α(y, x).  (∗)

*Derivation of α from (∗).* Generically a proposal violates reversibility; suppose π(x)q(x,y) > π(y)q(y,x), so the chain would move x→y too often and y→x too rarely. Throttle the over-frequent direction with α(x,y) < 1 and let the rare direction run at its ceiling α(y,x) = 1. Then (∗) requires

  π(x) q(x,y) α(x,y) = π(y) q(y,x) · 1 ⟹ α(x,y) = [π(y) q(y,x)] / [π(x) q(x,y)],

and the reverse case gives α(y,x) when the inequality flips. Both branches combine into α(x,y) = min(1, [π(y)q(y,x)]/[π(x)q(x,y)]).

**Detailed balance ⇒ π is stationary.** *Discrete:* use P_ij for the full transition matrix, including P_ii = 1 − Σ_{j≠i}P_ij. If π_i P_ij = π_j P_ji for all i,j, then for each destination j,

  Σ_i π_i P_ij = Σ_i π_j P_ji = π_j Σ_i P_ji = π_j,

because with j fixed, Σ_i P_ji is the row sum out of state j over every destination i, including the self-loop. That is πP = π.

*Continuous (with the staying-put atom):* let p(x,y) = q(x,y)α(x,y) be the moving density and P(x,dy) = p(x,y)dy + r(x)δ_x(dy), with r(x) = 1 − ∫p(x,y)dy. For any set A,

  ∫ P(x,A) π(x) dx = ∫_A [∫ p(x,y) π(x) dx] dy + ∫ r(x) δ_x(A) π(x) dx
       = ∫_A [∫ p(y,x) π(y) dx] dy + ∫_A r(y) π(y) dy    (use (∗) in the first term)
       = ∫_A (1 − r(y)) π(y) dy + ∫_A r(y) π(y) dy
       = ∫_A π(y) dy,

using ∫ p(y,x) dx = 1 − r(y), the total probability of moving away from y. The r terms cancel exactly — the rejected mass r is precisely what conserves probability and makes π invariant; dropping the re-counted rejects breaks this. ∎

**Convergence.** Detailed balance gives invariance; to converge to π the chain must also be irreducible and aperiodic. A globally positive proposal, or a local proposal whose repeated accepted moves connect the support of π, gives reachability; together with the invariant probability π and aperiodicity, this makes π the limiting distribution and time averages converge to E_π(F) (and are asymptotically normal under the usual regularity conditions). Detailed balance, irreducibility, and aperiodicity give convergence but not its *rate*: well-separated modes divided by low-probability regions can trap a local-move chain for any feasible run, so one discards a burn-in, tunes the proposal scale to a moderate acceptance rate (too large ⇒ almost all moves rejected and the chain freezes; too small ⇒ moves accepted but the state barely diffuses), and checks different segments/starts for non-convergence.

## Implementation

```python
import numpy as np

def transition(x, log_weight, candidate_rule, rng):
    """One transition.

    log_weight(x)          : unnormalized log target; any additive normalizing
                             constant cancels because only differences are used.
    candidate_rule(x, rng) : returns (y, log_q_forward, log_q_reverse), the
                             candidate y and the log proposal densities
                             log q(x->y), log q(y->x).
    returns the next state: y on accept, x (re-counted) on reject.
    """
    y, log_q_forward, log_q_reverse = candidate_rule(x, rng)
    # log[ pi(y) q(y->x) / ( pi(x) q(x->y) ) ]; normalizer of pi cancels.
    log_R = (log_weight(y) - log_weight(x)) + (log_q_reverse - log_q_forward)
    # alpha = min(1, R): accept immediately when R >= 1,
    # otherwise compare log(u) with log_R to avoid underflow.
    if log_R >= 0.0:
        return y
    return y if np.log(rng.uniform()) < log_R else x

def sample(x0, log_weight, candidate_rule, n_steps, burn_in, rng):
    x, samples = x0, []
    for t in range(n_steps):
        x = transition(x, log_weight, candidate_rule, rng)
        if t >= burn_in:
            samples.append(np.copy(x))
    return samples

def symmetric_random_walk_candidate(step):
    """Symmetric local candidate; the proposal-density correction vanishes."""
    def candidate_rule(x, rng):
        y = x + step * rng.uniform(-1.0, 1.0, size=np.shape(x))
        return y, 0.0, 0.0    # log q(x->y) == log q(y->x)
    return candidate_rule

def estimate(samples, observable):
    """Average the observable over retained states, including repeats."""
    return np.mean([observable(c) for c in samples], axis=0)
```

With a symmetric random-walk candidate, the proposal correction is zero and the rule reduces to accepting with probability min(1, e^{−ΔE/kT}) for a Boltzmann target. With an asymmetric candidate, the log_q_reverse − log_q_forward term supplies the q(y,x)/q(x,y) correction. Working in log space makes the normalizer cancellation explicit and avoids underflow.
