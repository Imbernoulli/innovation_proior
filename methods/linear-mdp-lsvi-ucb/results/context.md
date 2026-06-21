## Problem Setting

We are in finite-horizon reinforcement learning. Each episode has `H` steps; at step `h`, the learner observes a state `x`, chooses an action `a`, receives a reward in `[0,1]`, and transitions according to an unknown kernel `P_h(.|x,a)`. There is no simulator, no ability to query arbitrary state-action pairs, and no reset inside an episode. The initial state of each episode may be chosen adversarially. Performance is high-probability regret against the optimal policy over `K` episodes, with `T=KH` total decisions.

The hard regime is not the small tabular case. The state space may be extremely large or measurable and infinite, so guarantees that scale with `|S|` are unusable. A useful theorem must exploit a known feature map `phi(x,a) in R^d` and should scale polynomially with `d`, `H`, and `T`, while avoiding any enumeration over states.

## Bellman Structure

For any policy `pi`, action values satisfy `Q_h^pi = r_h + P_h V_{h+1}^pi`, and optimal values satisfy `Q_h^* = r_h + P_h V_{h+1}^*`, `V_h^*(x)=max_a Q_h^*(x,a)`. Since rewards lie in `[0,1]`, every value is bounded by `H`. The Bellman equation is the only route by which a future reward can be assigned to a present action.

In a large state space, the central obstacle is that the term `P_h V_{h+1}` depends on an unknown transition law. A method cannot estimate a full transition table, and if the state space is continuous it cannot even represent one. Any tractable approach has to estimate only the part of the dynamics that matters for Bellman predictions.

## Existing Tools

Least-squares temporal-difference methods and least-squares value iteration fit linear value functions by regressing sampled Bellman targets on features. They give a practical way to update value estimates, but by themselves they are greedy estimation procedures. They do not explain which uncertain state-action directions should be explored, and they do not give online regret guarantees in the large-state setting.

Optimism in the face of uncertainty is the standard exploration principle. In tabular RL, optimistic algorithms place confidence intervals on rewards, transitions, or action values and act according to the most favorable plausible model. In linear bandits, ridge regression plus a self-normalized confidence ellipsoid gives an uncertainty width `sqrt(phi^T Lambda^{-1} phi)`, which is large exactly along feature directions that have little data.

## Failure Modes

A purely tabular analysis cannot be the answer: known tabular lower bounds scale with the number of state-action pairs, so state-space dependence is information-theoretic without extra structure. A pure linear-bandit reduction is also inadequate: when `H>1`, the value of an action includes where it sends the learner, and optimism can compound through the horizon if the Bellman recursion is ignored.

There is also a statistical dependence problem. In a backward value-iteration procedure, the target at step `h` uses a future value function learned from the same dataset. Standard fixed-function regression concentration does not automatically apply to such data-dependent targets. Any proof must control a class of possible future value functions, not just one preselected function.

## Evaluation Criteria

The desired result should give a computationally efficient online algorithm, a regret bound independent of the cardinality of the state space, and a proof that explains how exploration remains optimistic through the Bellman recursion. It should remain meaningful in the one-hot tabular special case, match the linear-bandit geometry when the horizon is one, and make clear which structural mismatch would break the guarantee.

The important scientific test is whether the method learns enough about Bellman predictions without learning the entire transition model. A convincing answer has to identify the structural condition that makes this possible, the uncertainty bonus that drives exploration, and the concentration argument that justifies using that bonus with learned future values.
