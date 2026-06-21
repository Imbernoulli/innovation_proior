Policy gradient methods directly maximize expected return by estimating the gradient of the policy's performance. The basic likelihood-ratio estimator multiplies the score of each action by some measure of the return that follows it. The obvious choice, the total trajectory reward, is catastrophically noisy: every action gets credited with rewards that occurred before it was taken and with all the randomness introduced by later actions and the environment. Even after restricting to reward-to-go, the multiplier for an action still contains the effects of every future action, so variance grows with the horizon and makes learning in continuous control impractical. The theoretically ideal multiplier is the advantage, the difference between the action's value and the state's average value, because it is centered at zero and removes everything predictable from the state alone. The advantage is unknown, so the real problem is to estimate it from sampled rewards and a learned value function without adding so much bias that the policy gradient points the wrong way.

Existing estimators sit at unhappy extremes. REINFORCE with a state-independent baseline is unbiased but remains high variance over long episodes. A standard actor-critic that uses a one-step temporal-difference residual is low variance, but it is heavily biased whenever the learned value function is imperfect, which it always is for a neural network. The full Monte-Carlo return minus a state baseline is unbiased for any value function, but it carries the entire horizon's variance and provides no smooth way to interpolate between the two extremes. What is missing is a family of advantage estimators parameterized by a single knob that slides continuously between low-variance bootstrapping and low-bias Monte-Carlo credit assignment.

The method I propose is Generalized Advantage Estimation, or GAE. It is built around the one-step TD residual with respect to an approximate state-value function V: delta_t = r_t + gamma V(s_{t+1}) - V(s_t). If V were exact, the conditional expectation of this residual would be the true discounted advantage, but with an imperfect V it is a biased but low-variance one-step estimate. Summing k of these residuals causes the interior value predictions to telescope, yielding a k-step return minus a baseline: the result equals -V(s_t) plus k real rewards plus a discounted bootstrap value at step t+k. As k grows, the dependence on V shifts farther into the future and is more heavily discounted, reducing bias at the cost of variance. Rather than committing to a single k, GAE takes a geometrically weighted average over all k-step estimators with weights (1-lambda) lambda^{k-1}. This average collapses into an elegantly simple expression: the estimated advantage is just a discounted sum of the same one-step residuals, A_hat_t = sum_{l>=0} (gamma lambda)^l delta_{t+l}. The effective discount is the product gamma lambda, and lambda itself becomes the bias-variance knob.

At lambda=0 the estimator reduces to the one-step TD residual, which has low variance but is biased unless V is exact. At lambda=1 it becomes the Monte-Carlo discounted return minus the state baseline, which is unbiased for any V but has high variance. For intermediate lambda, the bias comes only from value-function error, while the variance is reduced by the steeper discount gamma lambda. The discount gamma and the averaging parameter lambda serve different purposes and should not be collapsed into one number: gamma biases the target even with a perfect value function by focusing credit on a finite horizon, whereas lambda introduces bias only through imperfections in V. There is also a useful reinterpretation through reward shaping. If we view V as a potential function, the TD residual is the shaped reward, and GAE is simply the gamma-lambda-discounted return in this shaped MDP. Shaping with V concentrates delayed credit into the immediate residual, and the extra lambda discount then trims the mostly-noise tail.

In practice the infinite sum is computed by a backward recursion, A_hat_t = delta_t + gamma lambda A_hat_{t+1}, resetting at episode terminals. The advantages are used inside a KL-constrained policy update, which to second order yields the natural policy gradient direction. The value function itself is fit to Monte-Carlo discounted returns and updated only after the policy step, because computing advantages with a freshly fitted V would drive the residuals toward zero and wipe out the policy gradient. GAE thus provides a single, cheap, principled estimator that makes policy gradient methods practical for high-dimensional continuous control.

```python
import numpy as np


def discount_cumsum(x, discount):
    """Backward discounted cumulative sum.

    out[t] = x[t] + discount * x[t+1] + discount^2 * x[t+2] + ...
    """
    out = np.zeros_like(x, dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(x))):
        running = x[t] + discount * running
        out[t] = running
    return out


def compute_gae_path(rewards, values, gamma, lam, last_value=0.0):
    """Generalized Advantage Estimation for one path.

    rewards:  r_0, ..., r_{T-1}
    values:   V(s_0), ..., V(s_{T-1}) from the old value function
    last_value: V(s_T); use 0.0 for terminal paths

    Returns advantages and Monte-Carlo discounted value targets.
    """
    values_ext = np.append(values, last_value)
    deltas = rewards + gamma * values_ext[1:] - values_ext[:-1]
    advantages = discount_cumsum(deltas, gamma * lam)

    rewards_ext = np.append(rewards, last_value)
    value_targets = discount_cumsum(rewards_ext, gamma)[:-1]
    return advantages, value_targets


def compute_gae_batch(rewards, values, dones, last_value, gamma, lam):
    """GAE for a rollout array that may span several episodes.

    dones[t] == 1 if s_{t+1} is terminal, blocking bootstrap across it.
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    value_targets = np.zeros(T, dtype=np.float32)

    lastgaelam = 0.0
    running_return = last_value
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        next_value = last_value if t == T - 1 else values[t + 1]

        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        advantages[t] = lastgaelam

        running_return = rewards[t] + gamma * nonterminal * running_return
        value_targets[t] = running_return

    return advantages, value_targets
```
