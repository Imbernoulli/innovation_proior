# The Successor Representation

## Problem

Temporal-difference (TD) value learning is fast or slow depending almost entirely on how
states are represented to a linear value approximator. Tie features directly to states and
value generalizes badly: a tabular/punctate code gives no generalization at all, and a code
with a fixed spatial metric (coarse coding / CMAC) generalizes between states that share an
a-priori metric even when their task-relevant futures diverge — e.g. two grid cells on
opposite sides of a barrier. What is the *right* representation for a temporal task, one
whose generalization is temporal and task-derived, learnable from experience without a
model, and reusable when the reward changes but the dynamics do not?

## Key idea

Value of a state is a discounted sum over its futures, so two states should share features
to exactly the extent that they share successors. The representation that encodes this
literally is the **successor representation (SR)**: the expected discounted future occupancy
of every state. This *factors* value into a dynamics part and a reward part — V = M R —
separating what is learnable by TD (the occupancy matrix M) from the reward R, and placing
the method between model-free and model-based reinforcement learning.

## The result, stated cleanly

For a fixed policy on a finite Markov environment with stochastic transition matrix P,
discount γ ∈ [0, 1), and immediate expected-reward vector R, define the successor
representation

  M(s, s′) = E[ Σ_{t≥0} γ^t · 1(s_t = s′) | s_0 = s ].

Because this is the geometric (Neumann) series in γP,

  M = I + γP + γ²P² + γ³P³ + … = (I − γP)⁻¹.

**Value–SR identity.** The value function factorizes as

  V(s) = Σ_{s′} M(s, s′) R(s′),   i.e.   V = M R.

*Proof.* The value vector satisfies the Bellman consistency V = R + γP V, so
(I − γP) V = R, hence V = (I − γP)⁻¹ R = M R. Equivalently, expand the discounted return:
V(s) = E[Σ_t γ^t R(s_t)] = Σ_t γ^t (P^t R)(s) = Σ_{s′} (Σ_t γ^t [P^t]_{s,s′}) R(s′) =
Σ_{s′} M(s, s′) R(s′). ∎

**Why factoring helps (the generalization argument).** With M as the row-feature matrix, the
optimal weights of a linear value approximator are just the immediate rewards. Take each
state's feature vector to be its row of M, so the linear prediction is the vector M w;
requiring M w* = V = M R forces w* = R, so the entire temporal component is compiled into
the features and the weights carry only the memoryless reward. Concretely, the transition
operator drops out of the learning dynamics. The batch TD proof writes state features as
columns in X, so its predicted-value vector is Xᵀw̄; in the row-feature convention used here,
Xᵀ is the feature matrix. Under a general representation, the value-space contraction is

  (Xᵀ w̄_{n+1} − V) = [I − α Xᵀ X D (I − γP)] (Xᵀ w̄_n − V),

(D the diagonal of state visitation counts). If the row-feature matrix is the SR
M = (I − γP)⁻¹, then the corresponding column-feature matrix is X̄ = Mᵀ and X̄ᵀw̄ = M w̄. Since
(I − γP)M = I, the mean-weight update becomes w̄_{n+1} = w̄_n + α X̄D(R − w̄_n), or

  (w̄_{n+1} − R) = (I − α X̄D)(w̄_n − R).

Equivalently, written only with the row-feature matrix, the last factor is MᵀD. The
transition factor is gone either way, giving direct relaxation of the weights toward R with
no temporal threading, and convergence is inherited from the invertibility of M. A reward
change is then a single recomputation V = M R; it never requires re-running value TD,
because R was never entangled with the dynamics. This is what transfers across tasks that
share dynamics but change reward.

**Learning M without a model.** Each occupancy is itself a discounted-sum-over-the-future,
so M is learned by the *same* TD machinery used for value, with the reward target replaced by
an occupancy indicator. On a transition s_t → s_{t+1}, with e(s_t) the one-hot of s_t,

  δ(s′) = 1(s_t = s′) + γ M̂(s_{t+1}, s′) − M̂(s_t, s′),    M̂(s_t, s′) ← M̂(s_t, s′) + α δ(s′).

This is TD(0) verbatim; the only structural difference from value learning is that the error
is **vector-valued** — one prediction error per successor state s′ — because each row predicts
a whole occupancy vector. Since M depends only on the policy and dynamics (not the reward), it
can be learned during reward-free exploration (latent learning).

**Position in the spectrum.** M is a *compiled* form of the transition statistics — it keeps
discounted visitation counts but discards the one-step structure — just as the value function
is a compiled form of the reward statistics. So reward revaluation is free (recompute M R)
but transition revaluation is costly (M must be relearned). The SR therefore sits between
model-free value learning (efficient, but value tied to states and reward changes propagate
slowly) and model-based planning (flexible, with a veridical goal-independent map, but
expensive to learn and to compute value from): model-free efficiency with the reward-side
flexibility of a model.

## Implementation

```python
import numpy as np

def successor_representation(P, gamma):
    """Closed form when dynamics are known: M = (I - gamma P)^-1."""
    n = P.shape[0]
    return np.linalg.inv(np.eye(n) - gamma * P)

def value(M, R):
    """Value factorizes as dynamics x reward: V = M R."""
    return M @ R                       # equals (I - gamma P)^-1 R, the Bellman value

def td_learn_sr(sample_next_state, n_states, gamma, alpha, n_steps, rng):
    """Learn M from experience by TD(0): occupancy indicator in place of reward.
    The TD error is vector-valued -- one error per successor state."""
    M_hat = np.zeros((n_states, n_states))
    s = rng.integers(n_states)
    for _ in range(n_steps):
        s_next = sample_next_state(s)
        onehot = np.zeros(n_states); onehot[s] = 1.0
        delta = onehot + gamma * M_hat[s_next] - M_hat[s]   # vector-valued TD error
        M_hat[s] += alpha * delta
        s = s_next
    return M_hat

# Reward revaluation is free: change R, recompute value(M, R) -- M is untouched,
# because the successor representation never depended on the reward. A change to the
# transition structure P, by contrast, requires relearning M.
```
