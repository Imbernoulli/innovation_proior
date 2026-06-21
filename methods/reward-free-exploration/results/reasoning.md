I start from the fact that the reward is the wrong thing to trust during exploration. If I optimize one reward while collecting data, I learn exactly enough about the transition system to support policies that matter for that reward. That is fine until the reward changes. Then the places I ignored can suddenly become the only places that matter, and the old interaction budget no longer buys me correctness.

So I do not ask first what reward to optimize. I ask what a dataset must certify before any reward appears. The value-difference lemma gives me the shape of the certificate. If I plan in an empirical model `Phat` but evaluate in the true model `P`, then for a fixed policy the error is a sum of terms like `(Phat_h - P_h) Vhat_{h+1}` weighted by where that policy goes. A bad transition estimate is harmless if no later policy uses it, but a reward chosen after seeing my data can make a policy route through precisely the weak spot. Therefore the dataset must dominate the occupancy of every policy that might later be optimal for some reward.

This is already different from setting the reward to zero. Zero reward gives me no adversary and no value function that exposes what the planner will later need. Optimism with an artificial unknown-state payoff can force visits to unknown regions, but it is still a coarse known-set objective. The later reward does not care whether I have crossed a knownness threshold; it cares whether the transition error, multiplied by the future value function and the later policy occupancy, stays small. The object I need is coverage, not reward-free optimism for its own sake.

Now I hit the obvious impossibility. I cannot cover every state uniformly in an MDP. Some states are behind branches with probability `10^-6`; some are unreachable. If I demand a fixed number of samples for each state-action pair, the goal becomes impossible for reasons unrelated to planning. But if a state is barely reachable by any policy, then even an adversarial reward cannot route much probability through it. A state whose maximum probability of being reached at time `h` is below `delta` contributes at most about `H delta` to any value at that step. Across `S` states and `H` steps, the total tail is at most `H^2 S delta`.

That resolves the wall. I only need strong coverage for states whose maximum reachability is at least `delta`, and I choose `delta` later so the ignored tail fits inside the error budget. Coverage has to be relative to reachability: if the best policy reaches `(s,h)` with probability `M`, the data should visit it proportional to `M`, not with a fixed absolute probability. A state with `M=1` deserves many samples; a state with `M=delta` deserves just enough samples to make its possible value contribution safe.

I still need a way to produce such a distribution without knowing the reward. For one target `(s,h)`, the problem "reach this target" is an ordinary RL problem if I invent the right measuring reward. I set a synthetic reward equal to one when the trajectory is at state `s` at step `h`, and zero everywhere else. Under this probe, the value of a policy is exactly `P_h^pi(s)`, and the optimal value is `max_pi P_h^pi(s)`. This reward is not a downstream task. It is an instrument for measuring and reaching a cell of the transition geometry.

If I do this for every `(s,h)`, I can learn policies that are good at reaching each state at each time. Then I can mix all those policy sets and sample trajectories from the mixture. The action at the target state needs one more adjustment: reaching `(s,h)` is determined before the action at time `h`, so I can replace the target action distribution by uniform over `A` without reducing the reaching probability. That converts state coverage into state-action coverage.

A naive regret oracle is not enough, because some important targets are only barely above the threshold. If I pay a full worst-case exploration cost for every target, the construction becomes too expensive. This is where the value-dependent property of EULER matters. For the target indicator reward, the total reward along a trajectory is at most one, and the optimal value is the maximum reaching probability `M`. I can plug this directly into EULER's problem-dependent regret bound: the variance/range term is controlled by `M`, so faint targets cost less than easy targets.

The bound I need for each target is that the average reaching probability of the learned policy set is at least half of the optimal reaching probability. EULER gives

`M - (1/N_0) sum_{pi in Phi^(s,h)} P_h^pi(s) <= c sqrt(S A H iota_0 M / N_0) + S^2 A H^4 iota_0^3 / N_0`.

For a `delta`-reachable target, `M >= delta`, so `N_0 = O(S^2 A H^4 iota_0^3 / delta)` makes the right side at most `M/2`. After the uniform action override, the target block covers every action within a factor `2A`. After I mix across all `S H` target blocks, that becomes the coverage ratio

`max_{pi,a} P_h^pi(s,a) / mu_h(s,a) <= 2 S A H`

for every target above the reachability threshold. The factors are not mysterious: `2` from approximate reaching, `A` from spreading target actions, and `S H` from mixing all target blocks.

Now I check whether this coverage is really enough for all later rewards. A reward arrives, I form `Phat` from the logged trajectories, and I solve the empirical MDP. To transfer the empirical solution back to the true MDP, I need a uniform policy-evaluation guarantee: for every policy and every reward, `Vhat^pi` and `V^pi` are close from the initial distribution.

I return to the value-difference lemma. At each step I split the occupancy sum into low-reachability states and covered states. The low-reachability states cost at most `H S delta` per step, hence `H^2 S delta` over the horizon. I set `delta = epsilon/(2 S H^2)`, so this entire part is at most `epsilon/2`.

For the covered states, I use Cauchy-Schwarz and then reduce the current action distribution to a deterministic action selector. This reduction is legitimate because `Vhat_{h+1}^pi` depends on the future policy, not on the current randomized action at time `h`. For a significant state and any action, I can splice the policy so that it reaches the same state and then deterministically plays that action, which lets the coverage ratio replace the target-policy occupancy by `2 S A H` times the data distribution `mu`.

The remaining statistical quantity is

`E_mu |(Phat_h - P_h) G(s,a)|^2 1[a=nu(s)]`

uniformly over value functions `G` in `[0,H]^S` and deterministic selectors `nu:S->A`. I need this uniformity because the reward and the empirical planner can be chosen after the data. The concentration proof uses a self-bounding trick. For each sample, I compare the empirical squared prediction error to the true squared prediction error. The empirical transition estimate is the minimizer of the empirical squared loss for each cell, so the empirical sum of the comparison variables is nonpositive. The expectation of the comparison variable is exactly the squared transition-estimation quantity I want, and its variance is at most `4 H^2` times its mean.

Bernstein then gives a fast self-bounded rate after a covering argument. The covering is where the action dependence improves: I do not cover every possible `Q(s,a)` table. For a fixed deterministic selector, only one action per state is active, so the cover costs `A^S (H/eps)^{2S}` rather than a full table net. The result is

`E_mu |(Phat_h - P_h) G(s,a)|^2 1[a=nu(s)] <= O(H^2 S log(A H N / p) / N)`.

Combining this with the `2 S A H` coverage ratio gives a per-step contribution of order `sqrt(H^3 S^2 A log / N)`. Summing over `H` steps gives `O(sqrt(H^5 S^2 A log / N))`. With

`N = O(H^5 S^2 A iota / epsilon^2)`,

the covered part is at most `epsilon/2`, and the full evaluation error is at most `epsilon`.

Once I have uniform evaluation, the planning theorem is mechanical. Let `pi*` be optimal in the true MDP and `pihat*` optimal in the empirical MDP. The true suboptimality of the returned empirical policy is bounded by evaluation error for `pi*`, the nonpositive empirical optimality gap, the solver's own empirical optimization error, and evaluation error for the returned policy. With an `epsilon`-accurate solver, the total is `3 epsilon`. Because the evaluation event is uniform over policies and rewards, the same dataset handles any number of later reward functions.

I also need to know whether the extra state factor is real. The lower bound says yes. In the one-state hard instance, every action has an unknown near-uniform distribution over `2n` absorbing leaves. If I must be near-optimal for every leaf reward vector, then the returned action must identify which transition distribution has the best inner product with each reward direction. A packing of nearly uncorrelated `+1/-1` perturbations and Fano's inequality force `Omega(n A / epsilon^2)` samples to decode the transition index for each action.

That gives only one factor of `S`, so the construction embeds `n` such instances in a binary tree. A reward can first force the policy to visit a chosen internal copy, then use the terminal rewards to demand correct action choice inside that copy. Since the data-collection phase must support every chosen copy, the learner pays the one-state cost `n` times. Persistent terminal reward contributes the `H^2` scaling, giving `Omega(S^2 A H^2 / epsilon^2)`.

The final picture is now clear to me. Exploration is separable from reward because the first phase is not trying to optimize an absent task; it is estimating the transition geometry in exactly the places any later task can exploit. Synthetic rewards are allowed because they are reachability probes, not commitments to downstream objectives. Confidence comes from a coverage ratio plus a uniform self-bounded transition-evaluation bound. The method is therefore more than optimism with reward set to zero: it first derives the data distribution that arbitrary future rewards require, then uses ordinary reward-aware exploration only as a subroutine for constructing that distribution.
