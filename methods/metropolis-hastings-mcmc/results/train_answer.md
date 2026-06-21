The method I am presenting is the Metropolis-Hastings Markov chain Monte Carlo algorithm. It is designed for the situation where we need expectations under a probability distribution whose density we can evaluate only up to an unknown normalizing constant. This is the canonical case in statistical mechanics, where the target is proportional to an unnormalized Boltzmann weight such as exp(-E(x)/kT), and the normalizer is the partition function, itself a high-dimensional integral we cannot compute. It is also the standard situation in Bayesian inference, where the posterior is proportional to likelihood times prior, and the marginal likelihood is intractable. The algorithm turns the problem of sampling from such a target into the problem of simulating a Markov chain whose long-run distribution is exactly the target, using only ratios of unnormalized densities.

I start from the observation that independent Monte Carlo methods fail here. Plain uniform Monte Carlo over configuration space would waste almost all of its samples on regions where the target weight is essentially zero. Importance sampling can help if we already have a good proposal density, but in high dimensions the importance weights often collapse, with nearly all mass concentrated on one or a few samples. Acceptance-rejection sampling requires a global envelope above the unnormalized target, which is itself a difficult high-dimensional object. These methods all try to draw fresh independent samples from the whole space, and that is the wrong computational object when the target is concentrated in a tiny, complicated region.

The key insight is to abandon independence and instead construct a local random walk that stays inside the important region. If we already have a reasonable configuration, a small perturbation of it is much more likely to remain probable under the target than a fresh draw from the ambient space. The question becomes how to choose the transition rule so that the resulting Markov chain has the target as its stationary distribution. We can think of this as an inverse problem: instead of being given a Markov chain and asked for its stationary distribution, we are given a desired stationary distribution and asked to design a transition kernel that preserves it.

A sufficient condition is detailed balance. For a discrete target distribution pi and transition matrix P, detailed balance requires pi_i P_ij = pi_j P_ji for every pair of states i and j. If this holds, summing over i gives sum_i pi_i P_ij = pi_j sum_i P_ji = pi_j, so pi is stationary. Detailed balance is stronger than stationarity, but it is local and constructive, which makes it the right tool for building a transition rule from pairwise considerations.

I split each transition into a proposal and an acceptance decision. Let q_ij be the probability of proposing state j from state i, and let alpha_ij be the probability of accepting that proposal. For i not equal to j, the transition probability is P_ij = q_ij alpha_ij, while the diagonal term P_ii captures all rejected proposals. Detailed balance then becomes pi_i q_ij alpha_ij = pi_j q_ji alpha_ji. In the special case of a symmetric proposal, where q_ij = q_ji, the proposal terms cancel and the acceptance probabilities must satisfy alpha_ij / alpha_ji = pi_j / pi_i. To maximize the probability of accepting moves while respecting this ratio, we set the larger acceptance probability to one. This gives the Metropolis acceptance probability alpha_ij = min(1, pi_j / pi_i). For the Boltzmann target, this becomes min(1, exp(-(E_j - E_i)/kT)), which accepts all energy-decreasing moves and accepts energy-increasing moves with probability exp(-Delta E / kT).

The unknown normalizing constant cancels in the ratio pi_j / pi_i, which is why the algorithm works with unnormalized densities. The acceptance rule is not a heuristic; it is precisely the correction that makes the proposal flow satisfy detailed balance with respect to the target. A rejected proposal is also part of the chain: if we propose a worse state and reject it, the next state is the current state again. This repetition must be counted in the sample path, because it corresponds to the diagonal mass in the transition matrix and is required for the stationarity proof.

For an asymmetric proposal, where q_ij is not equal to q_ji, we cannot cancel the proposal terms. Returning to detailed balance, we need alpha_ij / alpha_ji = (pi_j q_ji) / (pi_i q_ij). Setting the larger direction to one gives the Hastings acceptance probability alpha_ij = min(1, (pi_j q_ji) / (pi_i q_ij)). This general form corrects for proposal bias: if the proposal makes a move too easy in one direction compared with the reverse, the acceptance probability is reduced until the target-weighted flows match. The target's normalizing constant still cancels, so the algorithm remains applicable whenever the target density is known up to a constant and the forward and reverse proposal densities can be evaluated.

This maximal acceptance rule is the efficient choice among a family of reversible acceptance filters. For example, Barker's filter R / (1 + R) also satisfies detailed balance but accepts less often. Peskun's ordering formalizes the intuition that, for a fixed proposal, moving probability mass off the diagonal and onto accepted transitions reduces asymptotic variance. The Metropolis-Hastings choice is therefore not arbitrary; it is variance-optimal within this reversible family.

Detailed balance gives invariance, but convergence from an arbitrary initial state also requires irreducibility and aperiodicity on the support of the target. In continuous state spaces, the transition kernel can be written as P(x, dy) = q(x, y) alpha(x, y) dy + r(x) delta_x(dy), where r(x) is the total rejection mass. The detailed balance condition on the moving part, together with the stay-put atom, ensures that one step of the kernel preserves pi. The practical rule is therefore to propose y from q(x, .), compute log R = log f(y) - log f(x) + log q(y, x) - log q(x, y), accept y if log R is nonnegative or if a uniform logarithmic random variable is less than log R, and otherwise return x again.

The proposal still matters greatly for mixing. If the proposal step size is too large, most proposals land in low-probability regions and are rejected, so the chain barely moves. If the step size is too small, the chain diffuses slowly and takes a long time to explore the target support. If the target has well-separated modes, a local proposal may remain trapped in one mode for an extremely long time. These are important practical issues, but they concern the speed of convergence, not the correctness of the limiting distribution. The invariant distribution is correct by construction thanks to the detailed balance condition.

The Metropolis-Hastings algorithm has become the standard foundation of MCMC methods because it separates the design of a local proposal, which is problem-specific, from the correction that enforces the correct limiting distribution, which is universal. Any proposal with the right support and evaluable forward and reverse densities can be used, and the acceptance rule automatically compensates for the proposal's bias. The algorithm applies whenever we can evaluate an unnormalized log density and simulate a local perturbation, which covers an enormous range of problems in physics, statistics, and machine learning.

```python
import numpy as np

def mh_step(x, log_f, proposal, rng):
    """One Metropolis-Hastings transition.

    proposal(x, rng) returns (y, log_q_xy, log_q_yx).
    log_f is the unnormalized log target; additive constants cancel.
    """
    y, log_q_xy, log_q_yx = proposal(x, rng)
    log_R = (log_f(y) - log_f(x)) + (log_q_yx - log_q_xy)
    if log_R >= 0.0 or np.log(rng.uniform()) < log_R:
        return y
    return x

def run_chain(x0, log_f, proposal, n_steps, burn_in, rng):
    x = x0
    kept = []
    for t in range(n_steps):
        x = mh_step(x, log_f, proposal, rng)
        if t >= burn_in:
            kept.append(np.array(x, copy=True))
    return kept

def symmetric_gaussian_proposal(scale):
    def propose(x, rng):
        y = x + rng.normal(scale=scale, size=np.shape(x))
        return y, 0.0, 0.0
    return propose

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # Target: standard normal, up to a constant.
    log_f = lambda x: -0.5 * float(x) ** 2
    proposal = symmetric_gaussian_proposal(scale=1.0)
    samples = run_chain(x0=0.0, log_f=log_f, proposal=proposal,
                        n_steps=200_000, burn_in=10_000, rng=rng)
    estimate = np.mean(samples)
    second_moment = np.mean([s ** 2 for s in samples])
    print(f"mean estimate: {estimate:.4f}")
    print(f"second moment estimate: {second_moment:.4f}")
```
