## Temporal Prediction Needs Features

A learner is trying to estimate long-run return in a Markov environment from sampled transitions. Temporal-difference learning already gives an incremental rule: make the current prediction agree with the next prediction plus whatever immediate signal has arrived. In a linear value learner, though, the rule can only generalize through the state features it is handed. The open question is therefore not only how to update a scalar value, but what geometry the state features should impose before the update begins.

## The Known Learning Engine

For absorbing Markov chains, the expected return from a state can be written as an immediate contribution plus the contributions that flow through later states. Sutton's temporal-difference analysis shows that linear TD can converge to the ideal predictions under suitable independence and learning-rate conditions. It also shows why bootstrapping is attractive: a later prediction can be a cleaner target than the final sampled outcome, because it averages over future randomness that one particular trajectory has not averaged away.

## Fixed Similarity Can Be Wrong

Tabular features avoid bad generalization but give none at all. Coarse coding and CMAC-style tiles make learning faster by letting one update affect a local region, and Watkins's demonstrations make clear why that is useful: compact function representations are needed when storing a separate value for every point is too expensive.

Two positions can be adjacent in input space while being far apart in the task, because the agent must travel around an obstacle. A representation that cannot change its notion of neighborhood can smear value across exactly the boundary where the decision problem says the states differ.

## Full Models

The model-based alternative is to learn transition structure and plan from it. Dyna-style architectures learn a one-step world model, generate hypothetical experience, and use dynamic-programming-like updates to speed trial-and-error learning. This gives flexibility: a local change in the world can be reflected in the model and used for planning. The learner stores a model, uses it to search or update, and derives values from the transition structure.

## Code Framework

```python
import numpy as np

def compute_value(P, R, gamma):
    """Compute value function V = (I - gamma P)^-1 R by matrix inversion."""
    n = P.shape[0]
    return np.linalg.solve(np.eye(n) - gamma * P, R)

def td_update(V, s, s_next, r, gamma, alpha):
    """One TD(0) update for the value of state s."""
    delta = r + gamma * V[s_next] - V[s]
    V[s] += alpha * delta
    return V
```
