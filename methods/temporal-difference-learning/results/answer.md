# Temporal-Difference Learning

Temporal-difference learning is a family of prediction methods that update current predictions from errors in their own temporal consistency. Instead of waiting for the final outcome and training every earlier prediction against that outcome, it updates as experience arrives by comparing a prediction with the immediate signal plus the next prediction.

For an episodic final-outcome problem, define the terminal outcome as `P_{m+1} = z`. Then:

`z - P_t = sum_{k=t}^m (P_{k+1} - P_k)`.

The supervised error is exactly a sum of successive prediction differences. This identity yields an incremental update family:

`Delta w_t = alpha (P_{t+1} - P_t) e_t`,

where the eligibility trace is:

`e_t = sum_{k=1}^t lambda^{t-k} grad_w P_k`.

The trace is maintained online by:

`e_t = grad_w P_t + lambda e_{t-1}`.

Special cases:

- `lambda = 1`: for a linear predictor in an undiscounted final-outcome sequence, the same per-sequence weight change as Widrow-Hoff/LMS, but computed incrementally.
- `lambda = 0`: pure one-step bootstrapping, `Delta w_t = alpha (P_{t+1} - P_t) grad_w P_t`.

For cumulative discounted prediction, with return:

`z_t = c_{t+1} + gamma c_{t+2} + gamma^2 c_{t+3} + ...`,

correct predictions satisfy:

`P_t = c_{t+1} + gamma P_{t+1}`.

The temporal-difference error is:

`delta_t = c_{t+1} + gamma P_{t+1} - P_t`.

The general linear TD(lambda) update is:

`e_t = gamma lambda e_{t-1} + grad_w P(x_t, w)`

`w <- w + alpha delta_t e_t`.

In tabular value prediction:

`V(s_t) <- V(s_t) + alpha [r_{t+1} + gamma V(s_{t+1}) - V(s_t)]`.

For a terminal transition, use successor prediction `0` when the terminal reward or cost carries the observed outcome.

This combines Monte Carlo sampling with dynamic-programming bootstrapping. Like Monte Carlo learning, it samples actual experience and does not need a transition model. Like dynamic programming, it bootstraps from current successor estimates. Unlike pure Monte Carlo, it can update before the final outcome; unlike ordinary dynamic programming, it learns directly from sampled transitions.

Sutton's 1988 analysis gives two central guarantees for the linear TD(0) case. For absorbing Markov chains with linearly independent observations and finite expected terminal outcomes, the expected predictions converge, for sufficiently small positive learning rate, to the ideal predictions `(I - Q)^{-1} h`. Under repeated presentation of a finite training set with linearly independent state observations, TD(0) converges to the maximum-likelihood or certainty-equivalence predictions of the Markov model implied by the data, while Widrow-Hoff converges to the best fit to the raw observed returns.

Implementation for a linear predictor:

```python
import numpy as np

class TDLambda:
    def __init__(self, n_features, alpha=0.1, gamma=1.0, lam=0.0):
        self.w = np.zeros(n_features)
        self.e = np.zeros(n_features)
        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam

    def predict(self, x):
        return float(self.w @ x)

    def step(self, x, reward, x_next):
        v = self.predict(x)
        v_next = 0.0 if x_next is None else self.predict(x_next)
        delta = reward + self.gamma * v_next - v
        self.e = self.gamma * self.lam * self.e + x
        self.w += self.alpha * delta * self.e
        return delta

    def end_episode(self):
        self.e.fill(0.0)
```

The companion `TDmodel.c` code for Sutton and Barto's conditioning model implements the same signed error with conditioning notation:

```c
new_Vbar = Vbar(V, X);
alpha_beta_error = alpha * beta * (lambda + gamma * new_Vbar - old_Vbar);
V[i] += alpha_beta_error * trace[i];
trace[i] += delta * (X[i] - trace[i]);
old_Vbar = Vbar(V, X);
```

Here `lambda` is the immediate unconditioned stimulus or reinforcement, not the trace parameter in TD(lambda), and the C constant `delta` is the stimulus-trace update rate. The error term is the conditioning version of `reward + gamma * next_prediction - current_prediction`, multiplied by the existing trace to change predictive weights.
