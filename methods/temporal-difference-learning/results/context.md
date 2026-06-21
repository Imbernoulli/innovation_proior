# Context: Learning Predictions From Sequential Experience

## Research Question

The problem is learning to predict a future-dependent quantity from experience with an incompletely known dynamical system. A game position may forecast a win or loss, a weather pattern may forecast rain, a process state may forecast a later cost, and a random walk state may forecast which boundary will be reached. The data do not arrive as isolated input-label pairs. They arrive as a temporal sequence in which each observation is followed by another observation, and only later by a final outcome or accumulated signal. The open question is how to construct a learning rule that operates on this sequential structure.

## Background

The standard framing is supervised prediction. Given an observation vector `x_t`, a predictor `P(x_t, w)` is trained against the observed outcome `z`. For a linear predictor, `P_t = w^T x_t`, and the Widrow-Hoff update changes weights in proportion to `z - P_t`. This approach is simple and broadly useful.

Sequential prediction also has a temporal credit-assignment dimension. A later event can reveal that an earlier state was better or worse than expected. In animal learning, secondary reinforcement suggests that a predictor of reward can itself acquire reinforcing power. In optimal control, Bellman's dynamic programming uses recursive value equations to relate a state's value to the values of successor states. In engineering learning rules, the delta rule supplies a practical error-correction mechanism.

The statistical structure is visible in a simple game example. Suppose a novel state leads immediately into a state already known to lose most of the time, but the episode happens to end in a lucky win. A supervised update against the win treats the novel state as good. The continuation, meanwhile, reflects the quality of the state more directly than the eventual realized outcome, because the outcome includes randomness that occurs after the state being evaluated.

## Baselines

Widrow-Hoff or LMS updates a linear predictor by `Delta w = alpha (z - w^T x) x`. It is incremental over independent examples. Under repeated presentation of a fixed training set, it minimizes error against the outcomes in that set.

The Rescorla-Wagner model of classical conditioning changes associative strengths by a discrepancy between actual reinforcement and summed prediction. It captures surprise, blocking, and competition among predictors, and operates at the trial level: a whole trial is compressed into one reinforcement term and one aggregate expectation.

Samuel's checker player adjusted an earlier board evaluation toward a later board evaluation. This recognized the value of successor predictions, and was a component of a game-playing system.

Dynamic programming solves recursive value equations by iterating estimates built from successor estimates. It bootstraps from current estimates and is normally applied with a known transition model or a sweep over possible successor states.

Adaptive heuristic critics and related production-rule systems used prediction-like internal reinforcement inside control architectures, showing that prediction changes could guide action learning.

## Evaluation Settings

A bounded random walk is the clean diagnostic setting. The chain has nonterminal states represented by distinct feature vectors, starts in the middle, moves left or right with equal probability, and terminates at a left boundary with outcome 0 or a right boundary with outcome 1. The ideal prediction for each state is the probability of right-side termination, which is computable exactly from the Markov chain.

Performance is measured by root-mean-squared error between learned and ideal predictions, averaged over independently generated training sets. Repeated-presentation experiments test the fixed point of each learning rule on a finite data set. Single-presentation experiments test how quickly a rule learns from limited experience across learning rates. The setting is deliberately simple so the comparison focuses on the use of sequential structure rather than on representation or control.

## Code Framework

The scaffold is a linear prediction harness over transitions. A supervised baseline waits for the final outcome. The streaming transition update is the open slot.

```python
import numpy as np

def predict(w, x):
    return float(w @ x)

def grad_predict(w, x):
    return x

class SupervisedPredictor:
    def update(self, w, observations, outcome, alpha):
        delta_w = np.zeros_like(w)
        for x in observations:
            delta_w += alpha * (outcome - predict(w, x)) * grad_predict(w, x)
        return w + delta_w

class StreamingSequencePredictor:
    def __init__(self, n_features):
        self.state = np.zeros(n_features)

    def step(self, w, x_t, signal_t1, x_t1, alpha):
        raise NotImplementedError

    def end_sequence(self):
        self.state.fill(0.0)
```
