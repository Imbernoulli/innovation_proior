## Problem Setting

We are in finite-horizon reinforcement learning. Each episode has `H` steps; at step `h`, the learner observes a state `x`, chooses an action `a`, receives a reward in `[0,1]`, and transitions according to an unknown kernel `P_h(.|x,a)`. There is no simulator, no ability to query arbitrary state-action pairs, and no reset inside an episode. The initial state of each episode may be chosen adversarially. Performance is high-probability regret against the optimal policy over `K` episodes, with `T=KH` total decisions.

The state space may be extremely large or measurable and infinite. A useful algorithm works with a known feature map `phi(x,a) in R^d` and has guarantees that scale polynomially with `d`, `H`, and `T`, without enumerating states.

## Bellman Structure

For any policy `pi`, action values satisfy `Q_h^pi = r_h + P_h V_{h+1}^pi`, and optimal values satisfy `Q_h^* = r_h + P_h V_{h+1}^*`, `V_h^*(x)=max_a Q_h^*(x,a)`. Since rewards lie in `[0,1]`, every value is bounded by `H`. The Bellman equation is the only route by which a future reward can be assigned to a present action.

In a large state space, the term `P_h V_{h+1}` depends on an unknown transition law. A method cannot estimate a full transition table, and if the state space is continuous it cannot even represent one.

## Existing Tools

Least-squares temporal-difference methods and least-squares value iteration fit linear value functions by regressing sampled Bellman targets on features. They give a practical way to update value estimates.

Optimism in the face of uncertainty is the standard exploration principle. In tabular RL, optimistic algorithms place confidence intervals on rewards, transitions, or action values and act according to the most favorable plausible model. In linear bandits, ridge regression plus a self-normalized confidence ellipsoid gives an uncertainty width `sqrt(phi^T Lambda^{-1} phi)`, which is large exactly along feature directions that have little data.

## Research Question

How can an agent exploit a known feature map to learn and plan in a finite-horizon MDP with a large or infinite state space, achieving sublinear regret with a polynomial dependence on the feature dimension?
