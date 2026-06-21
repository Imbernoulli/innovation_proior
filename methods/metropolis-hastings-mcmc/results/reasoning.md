I start with a ratio of integrals over a configuration space so large that a grid is not even a candidate. The density I care about is proportional to a computable weight, in the physical case `exp(-E(x)/kT)`, but the proportionality constant is itself the integral I cannot compute. I can evaluate a configuration, and I can compare nearby configurations, but I cannot normalize the whole space.

The first idea is plain Monte Carlo: draw configurations uniformly and form the weighted ratio `sum F(x_i) w(x_i) / sum w(x_i)`. The constant cancels there, which looks promising. Then I picture a dense particle system. A uniformly drawn configuration almost surely overlaps particles or puts them into extremely high energy positions, so its Boltzmann weight is essentially zero. I am spending random draws on the part of the space the target almost never visits. The estimator is formally correct, but in finite time it waits for rare useful configurations and has a tiny effective sample size.

Importance sampling tries to fix the location of the draws by sampling from a density `q` and correcting by `p/q`. That helps only if `q` shadows the target well. In high dimension the correction weights can become nearly all negligible with a few huge outliers, so the estimate and its error assessment are controlled by accidents. Acceptance-rejection has a different problem: it gives exact independent samples only if I have a global envelope above the target, and in this setting a practical envelope is just another hard high-dimensional object. I keep running into the same wall: independent candidates from a global distribution are too blunt.

So I stop asking for independent draws. If I already have a reasonable configuration, a small local move from it is much more likely to stay in the important region than a fresh draw from the whole box. A long dependent sequence might be enough if I can choose its transition rule so that its long-run distribution is exactly the target. This turns the problem around. Instead of being given a Markov chain and asking for its stationary distribution, I am given the desired stationary distribution `pi` and I have to build a transition matrix `P` with `pi = pi P`.

The stationarity equation `sum_i pi_i P_ij = pi_j` is global. It asks a whole column of `P` to add up correctly for every destination `j`. I want a local rule, something checked on a pair of states at a time. The natural local conservation law is to make the equilibrium flow from `i` to `j` equal the flow from `j` to `i`:

`pi_i P_ij = pi_j P_ji`.

If this holds for every pair, then summing over `i` gives `sum_i pi_i P_ij = sum_i pi_j P_ji = pi_j sum_i P_ji = pi_j`, because `sum_i P_ji` is the full row sum out of state `j`, including the possibility of staying at `j`. Pairwise balance is stronger than stationarity, but it is constructive. It gives me a local constraint that implies the global fixed point.

Now I split a move into a proposal and a decision. Let `q_ij` be the probability of proposing `j` from `i`, and let `alpha_ij` be the probability that I accept that proposal. For `i != j`, the moving part is `P_ij = q_ij alpha_ij`; the leftover probability goes on the diagonal, `P_ii = 1 - sum_{j!=i} P_ij`. The balance condition becomes

`pi_i q_ij alpha_ij = pi_j q_ji alpha_ji`.

For the original local particle displacement, the proposal is symmetric. If I can move from one configuration to the other by displacing a particle inside a centered box, then the reverse displacement has the same a priori probability, so `q_ij = q_ji`. The proposal terms cancel and I get `alpha_ij / alpha_ji = pi_j / pi_i`. This determines only a ratio, not both probabilities. To avoid wasting moves, I push the larger allowable probability to 1. If `pi_j >= pi_i`, I accept `i -> j` with probability 1 and accept the reverse with probability `pi_i/pi_j`; if `pi_j < pi_i`, I accept `i -> j` with probability `pi_j/pi_i`. The single rule is

`alpha_ij = min(1, pi_j/pi_i)`.

For the Boltzmann target, the ratio is `exp(-E_j/kT)/exp(-E_i/kT) = exp(-(E_j-E_i)/kT)`. The partition function cancels before I ever need to know it. I accept energy-lowering moves; I accept energy-raising moves with probability `exp(-Delta E/kT)`. The accept/reject decision is therefore not an arbitrary heuristic. It is the correction that makes the proposal flow satisfy detailed balance with respect to the target, and it works with an unnormalized density because detailed balance only asks for ratios.

I have to be careful about what a rejection means. If I propose a worse configuration and reject it, I do not erase the step from the sample history. The chain has gone somewhere: it has stayed at the current state. That staying probability is the diagonal term in `P`, and the proof that pairwise balance implies stationarity used the full row sum including that term. If I keep only accepted moves, I remove the diagonal mass and change the transition law. The rejected configuration must be counted as the current configuration appearing again. That is the bookkeeping version of conserving probability.

The symmetric proposal is useful, but it is too narrow. If I use a biased proposal, or an independence proposal, the forward and reverse proposal probabilities are not the same. I return to the balance equation without canceling `q`:

`alpha_ij / alpha_ji = (pi_j q_ji) / (pi_i q_ij)`.

Let `R = (pi_j q_ji)/(pi_i q_ij)`. Again I set the more frequent direction's counterpart to probability 1 and throttle the over-frequent direction. The maximal rule is

`alpha_ij = min(1, R) = min(1, (pi_j q_ji)/(pi_i q_ij))`.

This extra factor `q_ji/q_ij` is exactly the correction for proposal bias. If the proposal makes `i -> j` too easy compared with `j -> i`, acceptance of `i -> j` is reduced until the target-weighted flows match. The unknown normalizing constant in `pi` still cancels. The algorithm can now use any proposal whose forward and reverse densities are known on the moves it makes and whose support can reach the target support.

This maximal rule is also the efficient member of a larger reversible family. Hastings' general form uses a symmetric factor `s_ij`, and Barker's choice gives `R/(1+R)`. That also balances the flows, but it accepts less often. When the two directions are equally weighted, Barker accepts with probability `1/2`, while the maximal rule accepts with probability `1`. Peskun's ordering explains why pushing probability off the diagonal matters: for a fixed proposal, larger off-diagonal transition probabilities reduce asymptotic variance among reversible chains.

I still separate invariance from convergence. Detailed balance gives the target as a stationary distribution. To reach it from an arbitrary start, the chain must be irreducible and aperiodic on the target support. In continuous space, I can express the transition kernel as a moving density plus a staying atom:

`P(x,dy) = q(x,y) alpha(x,y) dy + r(x) delta_x(dy)`,

where `r(x) = 1 - int q(x,y) alpha(x,y) dy`. If `p(x,y)=q(x,y)alpha(x,y)` satisfies `pi(x)p(x,y)=pi(y)p(y,x)`, then for any set `A`, the moving part contributes `int_A (1-r(y)) pi(y) dy`, and the staying atom contributes `int_A r(y) pi(y) dy`; together they give `int_A pi(y) dy`. The rejected mass is exactly what fills the gap.

The practical rule is now clear. I propose a state `y` from `q(x,.)`, compute `log_R = log f(y) - log f(x) + log q(y,x) - log q(x,y)`, accept immediately if `log_R >= 0`, otherwise accept if `log u < log_R`, and on rejection I return `x` again. Running this chain and averaging retained states estimates the expectation under the target. The move scale still matters: too large produces too many rejections, too small produces slow diffusion, and separated modes can trap a local chain for a long run. Those are mixing problems, not failures of the invariant distribution. The core insight is that the accept/reject correction enforces detailed balance for the target, while the target's normalizing constant never appears because every correction is a ratio.
