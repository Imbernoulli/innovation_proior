## Temporal Prediction Needs Features

A learner is trying to estimate long-run return in a Markov environment from sampled transitions. Temporal-difference learning already gives an incremental rule: make the current prediction agree with the next prediction plus whatever immediate signal has arrived. In a linear value learner, though, the rule can only generalize through the state features it is handed. The open question is therefore not only how to update a scalar value, but what geometry the state features should impose before the update begins.

## The Known Learning Engine

For absorbing Markov chains, the expected return from a state can be written as an immediate contribution plus the contributions that flow through later states. Sutton's temporal-difference analysis shows that linear TD can converge to the ideal predictions under suitable independence and learning-rate conditions. It also shows why bootstrapping is attractive: a later prediction can be a cleaner target than the final sampled outcome, because it averages over future randomness that one particular trajectory has not averaged away.

This leaves a practical weakness. A one-step TD update propagates information backward only through experienced transitions, and its speed depends heavily on whether the representation lets one visit inform many states or only one.

## Fixed Similarity Can Be Wrong

Tabular features avoid bad generalization but give none at all. Coarse coding and CMAC-style tiles make learning faster by letting one update affect a local region, and Watkins's demonstrations make clear why that is useful: compact function representations are needed when storing a separate value for every point is too expensive.

The same compactness creates the failure mode. A fixed spatial metric says that nearby points should share updates. In an open grid that can be sensible; near a barrier it can be false. Two positions can be adjacent in input space while being far apart in the task, because the agent must travel around the obstacle. A representation that cannot change its notion of neighborhood can smear value across exactly the boundary where the decision problem says the states differ.

## Full Models Solve Too Much

The model-based alternative is to learn transition structure and plan from it. Dyna-style architectures learn a one-step world model, generate hypothetical experience, and use dynamic-programming-like updates to speed trial-and-error learning. This gives flexibility: a local change in the world can be reflected in the model and used for planning.

That flexibility has a cost. The learner must store a model, use it to search or update, and still derive values from the transition structure. For a value learner whose immediate bottleneck is representation, a complete world map may be more machinery than is needed.

## The Missing Middle

The desired object must satisfy three constraints at once. It must be learnable from experience like a model-free prediction, because the transition matrix is not handed to the agent. It must generalize according to temporal reachability rather than an a priori input metric. And it should keep reusable structure separate from the particular payoff currently being learned, so that changing the payoff does not force the whole temporal prediction problem to start over.

Before the new construction, the alternatives do not meet all three constraints. Tabular TD is cheap but tied to the current value estimates. Coarse coding generalizes but by the wrong geometry when the task geometry bends. Full models are flexible but expensive. The gap is a representation that is predictive enough to carry task geometry without becoming a full planner.

## Code framework

```python
import numpy as np

def successor_representation(P, gamma):
    """Compute the successor representation M = (I - gamma P)^-1."""
    n = P.shape[0]
    return np.linalg.inv(np.eye(n) - gamma * P)

def value_from_sr(M, R):
    """Return value vector V = M R."""
    return M @ R

def td_update_sr(M, s, s_next, gamma, alpha, n_states):
    """One TD(0) update for the successor representation row of state s."""
    onehot = np.zeros(n_states)
    onehot[s] = 1.0
    delta = onehot + gamma * M[s_next] - M[s]
    M[s] += alpha * delta
    return M
```
