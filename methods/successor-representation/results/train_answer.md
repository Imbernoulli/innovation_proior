I will present the method as the successor representation, the representation that makes a state's features equal to the discounted distribution of states it is expected to visit in the future. Consider a fixed policy acting in a finite Markov environment whose one-step transition matrix is P and whose discount factor is gamma with 0 less than or equal to gamma less than one. Let R be the vector of immediate expected rewards, so R(s) is the expected reward received the first step after leaving state s. The question is how to represent each state so that long-run value estimation becomes a simple, local supervised problem rather than a slow backup through the chain.

The key object is the successor matrix M. For each pair of states s and s', the entry M(s,s') is defined as the expected discounted number of future visits to s' when the process starts from s. In symbols, M(s,s') equals the expectation of the sum over t greater than or equal to zero of gamma to the power t times the indicator that s_t equals s', given that s_0 equals s. Because the process is Markov, this geometric series collapses to the matrix equation M equals I plus gamma P plus gamma squared P squared and so on, which is exactly the inverse of I minus gamma P. So M can be written compactly as the resolvent M equals (I minus gamma P) inverse.

Once M is in hand, value estimation factorizes cleanly. Bellman consistency tells us that V equals R plus gamma P V, which rearranges to (I minus gamma P) V equals R and therefore V equals (I minus gamma P) inverse R, which is M R. Expanding the return directly gives the same factorization: the expected discounted return from s is the sum over successor states s' of M(s,s') times R(s'). Thus the hard temporal summation is encoded in M, while R remains a simple per-state signal.

The distinctive move is to use each row of M as the feature vector for the corresponding state. With row features M(s,.), a linear value approximator predicts M(s,.) times w. Since the true value vector is M R, and M is invertible in the standard discounted setting, the optimal weights are simply w star equals R. The weights no longer have to learn multi-step consequences; those consequences live in the representation itself. This is a different division of labor from ordinary temporal-difference learning, where features are usually fixed and weights must encode both reward structure and temporal propagation.

The successor representation is learnable from experience without a supplied transition matrix. Each entry M(s,s') is itself a scalar prediction about future occupancy, which is exactly the kind of discounted prediction that temporal-difference methods can estimate. On a transition from s_t to s_{t+1}, the target for the row associated with s_t is the one-hot vector that marks the current state plus gamma times the row predicted for the next state. The vector-valued prediction error is delta equals e(s_t) plus gamma M_hat(s_{t+1},.) minus M_hat(s_t,.), and the row is updated by adding alpha times delta. This is TD(0) with the usual scalar reward replaced by an occupancy indicator. There is one prediction per possible successor state, so the representation is more expensive than a single value function, but it is far less expensive than storing and planning with a full one-step world model.

This construction occupies a useful middle ground. A purely model-free value function is compact, but reward information and transition information are entangled; if the reward changes, new values must be backed up through experience. A full model is flexible, but it stores one-step dynamics and needs planning or repeated dynamic-programming updates to turn those dynamics into values. The successor representation compiles multi-step occupancy statistics instead. If rewards change while transitions stay fixed, one only has to multiply the same M by the new R. If transitions themselves change, M must be relearned, so the method trades transition flexibility for reward flexibility. This separation also explains latent learning: an agent can wander through an environment before any reward is present and still build a useful M, because the predictive occupancy structure does not depend on what the rewards will eventually be.

A concrete verification on a small random chain confirms both the analytic factorization and the TD learning rule. The code below constructs a five-state Markov chain, computes M from the transition matrix directly, compares it with M learned from sampled transitions, and checks that the value vector obtained from M R agrees with the fixed-point value. Running it should show a small finite-sample error that shrinks as the number of sampled transitions grows.

```python
import numpy as np

rng = np.random.default_rng(0)
n = 5

"""Build a random stochastic transition matrix."""
P = rng.random((n, n))
P = P / P.sum(axis=1, keepdims=True)
gamma = 0.9
R = rng.random(n)

"""Analytic successor representation and value."""
M_exact = np.linalg.inv(np.eye(n) - gamma * P)
V_exact = M_exact @ R
print("V from analytic SR:", V_exact)

"""Learn M by TD(0) over successor-state occupancy indicators."""
alpha = 0.05
n_steps = 50_000
M_hat = np.zeros((n, n))
s = rng.integers(n)
for _ in range(n_steps):
    s_next = rng.choice(n, p=P[s])
    onehot = np.zeros(n)
    onehot[s] = 1.0
    delta = onehot + gamma * M_hat[s_next] - M_hat[s]
    M_hat[s] += alpha * delta
    s = s_next

V_learned = M_hat @ R
print("V from learned SR:", V_learned)
print("Frobenius error ||M_exact - M_hat||:", np.linalg.norm(M_exact - M_hat, 'fro'))
print("Value error ||V_exact - V_learned||:", np.linalg.norm(V_exact - V_learned))
```

In summary, the canonical name for this approach is the successor representation. It replaces the usual state-feature map with a predictive occupancy matrix, turning long-run value estimation into the product of a learned temporal structure and a per-state reward vector. The representation is trained with the same TD machinery used for scalar values, but the reward signal is replaced by the identity of the state that is currently visited. Once learned, the successor representation supports fast reward reevaluation and encodes the environment's temporal geometry directly in the feature space.
