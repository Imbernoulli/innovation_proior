# Maximum Entropy Inverse Reinforcement Learning

## Problem

Given demonstrations of a purposeful agent acting in a known-structure MDP (states `s_i`,
actions `a_i`, transition dynamics, state features `f_{s_j} ∈ ℝ^k`), recover the reward the
agent optimizes — assumed linear in features, `reward(f_ζ) = θ·f_ζ` with path feature counts
`f_ζ = Σ_{s_j ∈ ζ} f_{s_j}` — so the behavior can be predicted, completed, and generalized.

The obstruction: reward recovery is **ill-posed**. Many rewards (including the all-zero
reward, and any constant) make the demonstrations optimal; and feature matching alone,
`Σ_ζ P(ζ) f_ζ = f̄`, is satisfied by many different behaviors. Prior approaches break the tie
with arbitrary heuristics (max-margin) and return a *mixture* of policies, not a single model.

## Key idea

Among all distributions over trajectories that match the demonstrated feature expectations,
choose the one of **maximum entropy** (Jaynes 1957) — maximally noncommittal about everything
the feature constraints do not pin down. Solving

> maximize `H(P) = −Σ_ζ P(ζ) log P(ζ)` s.t. `Σ_ζ P(ζ) f_ζ = f̄`, `Σ_ζ P(ζ) = 1`

by Lagrange multipliers gives a Boltzmann / exponential-family distribution in which the
reward weights `θ` are the dual variables of feature matching:

> `P(ζ | θ) = (1/Z(θ)) exp(θ·f_ζ)`,  `Z(θ) = Σ_ζ exp(θ·f_ζ)`.

The sign is fixed by writing
`J = H(P) + θ·(Σ_ζ P(ζ)f_ζ − f̄) + μ(Σ_ζ P(ζ) − 1)`, so stationarity gives
`−log P(ζ) − 1 + θ·f_ζ + μ = 0` and therefore the positive-score form above.

Higher-reward paths are exponentially more probable; equal-reward paths are equally probable.
Global normalization avoids the **label bias** of locally normalized (action-based) models.
For stochastic dynamics `T`, conditioning on transition outcomes and assuming the randomness
has limited effect on behavior yields the tractable approximation

> `P(ζ | θ, T) ≈ (exp(θ·f_ζ)/Z(θ,T)) · Π_{(s_{t+1}, a_t, s_t) ∈ ζ} P_T(s_{t+1} | a_t, s_t)`,

and a stochastic policy `P(action a | θ,T) ∝ Σ_{ζ: a∈ζ_{t=0}} P(ζ | θ,T)`.

## Learning: convex maximum likelihood

Fitting `θ` by maximizing the average log-likelihood of the demonstrations,
`θ* = argmax_θ (1/m)Σ_examples log P(ζ̃ | θ, T)`, is the same problem as feature-matching /
max-entropy. With `b_T(ζ)=Π P_T(s_{t+1}|a_t,s_t)` as the transition-product base measure,
`Z(θ,T)=Σ_ζ b_T(ζ)exp(θ·f_ζ)` and the θ-independent `log b_T(ζ̃)` term drops from the
gradient. Therefore `∇_θ log Z(θ,T) = Σ_ζ P(ζ|θ,T) f_ζ`, giving

> `∇_θ L(θ) = f̄ − Σ_ζ P(ζ|θ,T) f_ζ = f̄ − Σ_{s_i} D_{s_i} f_{s_i}`  — **demonstrated − expected feature counts**,

with `D_{s_i}` the expected state-visitation frequency. The Hessian is
`∇²L = −Cov_{P(ζ|θ,T)}[f_ζ] ⪯ 0`, so `L` is **concave** and maximizing it is a convex
optimization problem. At the optimum the gradient vanishes ⟹ feature expectations match ⟹ the learned
behavior has the same value as the demonstrations under the true reward (Abbeel–Ng guarantee),
realized by one coherent stochastic policy. Convergence of `Z(θ)` holds for finite-horizon
and infinite-horizon discounted problems; for trajectories absorbed in finite steps the
entropy-maximizing weights are convergent. Plain gradient ascent fits the objective below;
an exponentiated-gradient optimizer can be substituted to obtain the `ℓ_1`-type regularization
associated with bounded-uncertainty max-entropy estimation (Dudík–Schapire 2006).

## Expected state-visitation frequencies

The only hard quantity, `Σ_ζ P(ζ|θ,T) f_ζ`, reduces to `Σ_{s_i} D_{s_i} f_{s_i}`; naive path
enumeration is exponential in the horizon, so compute `D_{s_i}` by dynamic programming.

**Backward pass** (finite horizon; zero-step boundary `Z^{(0)}_{s_i}=1`, recurse for remaining horizon `h=1…N`):

> `Z^{(h)}_{a_{i,j}} = Σ_k P(s_k | s_i, a_{i,j}) · exp(reward(s_i|θ)) · Z^{(h-1)}_{s_k}`,  `Z^{(h)}_{s_i} = Σ_{a_{i,j}} Z^{(h)}_{a_{i,j}}`.

**Local action probability:** for finite horizon, `P_t(a_{i,j} | s_i) = Z^{(N-t)}_{a_{i,j}} / Z^{(N-t)}_{s_i}`. For a large-horizon absorbed or discounted problem, iterating until this ratio stabilizes gives the stationary approximation.

**Forward pass** (`D_{s_i,0} = p_0(s_i)`; for `t = 0…N−2` when `N` state positions are counted):

> `D_{s_k, t+1} = Σ_{s_i, a_{i,j}} D_{s_i, t} · P_t(a_{i,j} | s_i) · P(s_k | s_i, a_{i,j})`,  then  `D_{s_i} = Σ_{t=0}^{N-1} D_{s_i, t}`.

## Implementation

```python
import numpy as np

def _logsumexp(x, axis):
    m = np.max(x, axis=axis, keepdims=True)
    shifted = np.exp(x - m)
    shifted[~np.isfinite(shifted)] = 0.0
    out = m + np.log(np.sum(shifted, axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)

def _log_transition(transition):
    log_t = np.full_like(transition, -np.inf, dtype=float)
    positive = transition > 0
    log_t[positive] = np.log(transition[positive])
    return log_t

def find_feature_expectations(feature_matrix, trajectories):
    """f̄ : average demonstrated feature counts (the matching target)."""
    fe = np.zeros(feature_matrix.shape[1])
    for traj in trajectories:
        for s in traj:
            fe += feature_matrix[s]
    return fe / len(trajectories)

def find_time_indexed_policy(n_states, reward, n_actions, transition, horizon):
    """Backward pass: Z_s^(0)=1, then P_t(a|s)=Z_a^(N-t)/Z_s^(N-t)."""
    log_t = _log_transition(transition)
    log_z_s = np.zeros(n_states)
    policies_by_remaining = []
    for _ in range(1, horizon + 1):
        log_z_a = np.empty((n_states, n_actions))
        for a in range(n_actions):
            downstream = _logsumexp(log_t[:, a, :] + log_z_s[None, :], axis=1)
            log_z_a[:, a] = reward + downstream
        log_z_s = _logsumexp(log_z_a, axis=1)
        policy = np.exp(log_z_a - log_z_s[:, None])
        policies_by_remaining.append(policy / policy.sum(axis=1, keepdims=True))
    return np.asarray(policies_by_remaining)

def find_expected_svf(n_states, reward, n_actions, transition,
                      trajectories, horizon):
    """Forward pass -> expected state-visitation frequencies D_{s} = Σ_t D_{s,t}."""
    policies = find_time_indexed_policy(n_states, reward, n_actions,
                                         transition, horizon)
    n_traj = len(trajectories)
    start = np.zeros(n_states)
    for traj in trajectories:
        start[traj[0]] += 1.0
    start /= n_traj                                              # p_0(s) = D_{s,0}
    D = np.zeros((n_states, horizon))
    D[:, 0] = start
    for t in range(horizon - 1):
        policy = policies[horizon - t - 1]
        for i in range(n_states):
            for a in range(n_actions):
                D[:, t + 1] += D[i, t] * policy[i, a] * transition[i, a, :]
    return D.sum(axis=1)

def irl(feature_matrix, n_actions, transition, trajectories,
        epochs, lr, horizon=None):
    """Convex MLE fit; gradient = demonstrated − expected feature counts."""
    n_states, d = feature_matrix.shape
    horizon = horizon or max(len(traj) for traj in trajectories)
    theta = np.random.uniform(size=(d,))
    f_bar = find_feature_expectations(feature_matrix, trajectories)
    for _ in range(epochs):
        reward = feature_matrix.dot(theta)                      # r(s) = θ·f_s
        D = find_expected_svf(n_states, reward, n_actions,
                              transition, trajectories, horizon)
        expected = feature_matrix.T.dot(D)                      # Σ_s D_s f_s
        grad = f_bar - expected                                 # ∇L(θ)
        theta += lr * grad                                      # gradient ascent
    return feature_matrix.dot(theta)
```

## Why each choice

- **Distribution over trajectories, not policies/mixtures** — one principled object on which
  max-entropy can act; yields a single coherent stochastic policy.
- **Maximum entropy, not max-margin** — the margin tie-break is arbitrary and needs a single
  optimal demonstration; max-entropy is the unique principled resolution of leftover ambiguity.
- **Global normalization, not local (action-based) softmax** — avoids label bias; in the
  deterministic path distribution, equal reward gives equal probability and higher reward gives
  exponentially higher probability.
- **`P(ζ) ∝ exp(θ·f)`** — not chosen, forced by max-entropy + linear feature constraints.
- **Linear reward in features** — makes feature expectations sufficient statistics for value
  and the constraints linear, giving the exponential family.
- **Forward/backward DP for `D_{s_i}`** — enumeration is exponential in horizon; the
  remaining-horizon backup is polynomial and yields the time-indexed policy needed for exact
  finite-horizon counts, mirroring CRF forward-backward / soft value iteration.
