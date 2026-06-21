# The Successor Representation

For a fixed policy on a finite Markov environment with transition matrix `P`, discount `0 <= gamma < 1`, and immediate expected reward vector `R`, define

```text
M(s,s') = E[ sum_{t>=0} gamma^t 1{s_t = s'} | s_0 = s ].
```

Equivalently,

```text
M = I + gamma P + gamma^2 P^2 + ... = (I - gamma P)^-1.
```

The value function factorizes:

```text
V(s) = sum_{s'} M(s,s') R(s')
V = M R.
```

Proof: Bellman consistency gives `V = R + gamma P V`, hence `(I - gamma P)V = R` and `V = (I - gamma P)^-1 R = M R`. Expanding the return gives the same identity:

```text
V(s) = E[sum_t gamma^t R(s_t)]
     = sum_{s'} (sum_t gamma^t [P^t]_{s,s'}) R(s')
     = sum_{s'} M(s,s') R(s').
```

The distinctive move is to use `M` as the state-feature matrix. With row features `M(s,.)`, a linear value approximation predicts `M w`. Since the true value is `M R`, the exact weights are `w* = R`. In Dayan's absorbing-chain notation this is `M = (I - Q)^-1`, `V = M h`, and `w* = h`. The temporal transition structure is compiled into the representation; the weights carry immediate reward.

`M` is learnable without a supplied model because every entry is a discounted prediction. On transition `s_t -> s_{t+1}`:

```text
delta(s') = 1{s_t = s'} + gamma M_hat(s_{t+1},s') - M_hat(s_t,s')
M_hat(s_t,s') <- M_hat(s_t,s') + alpha delta(s')
```

This is TD(0) with reward replaced by an occupancy indicator. The error is vector-valued, one component per possible successor state.

The method sits between model-free value learning and model-based planning. It is learned from experience like a model-free prediction, but it separates predictive dynamics from reward, so a reward change only requires recomputing `M R`. It is less flexible than a full model because transition changes alter the compiled occupancy matrix itself, so `M` must be relearned or repaired.

```python
import numpy as np

def successor_representation(P, gamma):
    n = P.shape[0]
    return np.linalg.inv(np.eye(n) - gamma * P)

def value(M, R):
    return M @ R

def td_learn_sr(sample_next_state, n_states, gamma, alpha, n_steps, rng):
    M_hat = np.zeros((n_states, n_states))
    s = rng.integers(n_states)
    for _ in range(n_steps):
        s_next = sample_next_state(s)
        onehot = np.zeros(n_states)
        onehot[s] = 1.0
        delta = onehot + gamma * M_hat[s_next] - M_hat[s]
        M_hat[s] += alpha * delta
        s = s_next
    return M_hat
```
