# Q-Learning

## Problem

Learn optimal control in a finite Markov decision process from sampled transitions only. The learner does not know the reward means `rho(x,a)` or transition probabilities `P_xy[a]`, but it must converge to the same optimal action values that dynamic programming would compute from the model.

## Method

Store a tabular action-value estimate `Q(x,a)`. On each observed transition `(x_t, a_t, r_t, x_{t+1})`, update only the experienced state-action entry:

```text
Q_{t+1}(x_t,a_t)
  = (1 - alpha_t) Q_t(x_t,a_t)
    + alpha_t (r_t + gamma max_b Q_t(x_{t+1},b)).
```

Equivalently:

```text
Q(x_t,a_t) <- Q(x_t,a_t)
              + alpha_t [r_t + gamma max_b Q(x_{t+1},b) - Q(x_t,a_t)].
```

All other entries are unchanged. The greedy policy implied by the table is:

```text
pi_Q(x) in argmax_a Q(x,a).
```

The behavior policy that chooses `a_t` can be exploratory and need not equal `pi_Q`. It only has to keep visiting every state-action pair often enough.

## Why It Is Off-Policy

The target uses `max_b Q(x_{t+1},b)`, not the value of the action that the behavior policy actually takes next. The sampled transition provides a reward and next state for the experienced pair `(x_t,a_t)`, while the continuation value is the greedy continuation under the current table.

That makes the update a sampled Bellman optimality backup. The behavior policy controls which entries are sampled; the target being learned is still the optimal action-value equation. This is the key difference from an on-policy temporal-difference control update, which would bootstrap from the next action actually selected by the behavior policy.

## Convergence Theorem

Let the state and action sets be finite, rewards be bounded, and `0 <= alpha_t < 1`. For each state-action pair `(x,a)`, let `n_i(x,a)` be the time of the `i`-th update to that pair. Assume:

```text
sum_i alpha_{n_i(x,a)} = infinity
sum_i alpha_{n_i(x,a)}^2 < infinity
```

for every `(x,a)`. These assumptions imply that every pair is sampled infinitely often and that the per-pair noise is averaged away. With discount `0 < gamma < 1`, the Q-learning iterates converge with probability one:

```text
Q_t(x,a) -> Q*(x,a)    for all x,a.
```

The same proof strategy also covers the undiscounted absorbing case when values remain bounded by guaranteed absorption.

## Proof Skeleton

Watkins and Dayan prove convergence with the action-replay process (ARP).

The ARP is an artificial controlled Markov process built from the learner's experience. Each observed transition `(x_t,a_t,y_t,r_t,alpha_t)` is a card. From a replay state `(x,n)`, action `a` searches downward through cards at levels at most `n` for matching `(x,a)` cards. A matching card is replayed with probability `alpha_t`; otherwise it is skipped. Replaying emits the recorded reward and moves to the recorded next state at the lower level. The bottom card pays the initial table value.

Lemma A: the optimal action value of the ARP at level `n` equals the Q-learning table after `n` real updates. The induction is exact: replay probability `alpha_t` produces the new sampled target, and skip probability `1 - alpha_t` preserves the old estimate.

Lemma B: as the replay level grows, the ARP converges to the real Markov process. Discounting bounds the tail of long returns. The condition `sum alpha = infinity` prevents the replay process from falling too far down the deck with high probability. The condition `sum alpha^2 < infinity` gives stochastic-approximation convergence of replay reward and transition averages to the true reward means and transition probabilities. Close finite-horizon models have close finite-horizon action values.

Combining the lemmas: the learned table equals the ARP's optimal action values, and the ARP's optimal action values converge to the real process's optimal action values. Therefore the learned table converges to `Q*`.

## Reference Implementation

```python
import numpy as np
from collections import defaultdict


def q_learning(env, n_actions, gamma, alpha_schedule, epsilon, n_steps):
    Q = defaultdict(lambda: np.zeros(n_actions))

    def choose_action(state):
        if np.random.random() < epsilon:
            return np.random.randint(n_actions)
        return int(np.argmax(Q[state]))

    state = env.reset()

    for _ in range(n_steps):
        action = choose_action(state)
        next_state, reward, done = env.step(action)

        continuation = 0.0 if done else gamma * np.max(Q[next_state])
        target = reward + continuation

        alpha = alpha_schedule(state, action)
        Q[state][action] += alpha * (target - Q[state][action])

        state = env.reset() if done else next_state

    return Q
```
