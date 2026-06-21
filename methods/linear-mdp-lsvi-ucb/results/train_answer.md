The canonical method I am presenting is LSVI-UCB, which stands for Least-Squares Value Iteration with Upper Confidence Bound, designed specifically for linear Markov Decision Processes. It addresses the fundamental challenge of exploring and planning in finite-horizon reinforcement learning when the state space is so large or even infinite that any algorithm whose regret scales with the number of states is hopeless. Instead, LSVI-UCB exploits a known feature map and a structural linearity assumption on the transition kernel and reward function to obtain a regret bound that depends only polynomially on the feature dimension, the horizon, and the total number of decisions.

The setting is a finite-horizon episodic MDP with horizon H. At each step h, the learner observes a state x, chooses an action a, receives a reward, and transitions to a next state according to an unknown kernel P_h. The state space may be continuous and arbitrarily complex, so we cannot maintain a table of states or learn a transition matrix. What saves us is the linear MDP assumption. We are given a feature map phi(x,a) taking values in R^d with bounded norm, and we assume that for every step h there exist an unknown weight vector theta_h in R^d and an unknown vector of signed measures mu_h such that the reward satisfies r_h(x,a) equals the inner product of phi(x,a) and theta_h, and the transition kernel satisfies P_h(.|x,a) equals the inner product of phi(x,a) and mu_h(.). This is not an assumption that the policy is linear or that the value function happens to be linear; it is a structural assumption on the environment itself. Because the transition kernel factors through the same features as the reward, the Bellman backup preserves linearity. For any policy pi, the action-value function Q_h^pi can be written as the inner product of phi(x,a) with a coefficient vector that combines theta_h with the integral of the future value function against mu_h. This closure under Bellman recursion is the entire reason the problem becomes tractable.

The algorithm itself is a marriage of least-squares value iteration and linear-bandit optimism. At the beginning of each episode k, LSVI-UCB performs a backward sweep over the horizon. For each step h, it forms the regularized design matrix Lambda_h as the sum of outer products of observed feature vectors from all previous episodes plus a ridge term lambda times the identity. It then computes a weight vector w_h by ridge regression, where the regression target for each previous transition is the immediate reward plus the optimistic estimate of the next state's best action-value. The optimistic action-value function is defined by adding to the linear prediction a bonus term beta times the square root of phi(x,a)^T Lambda_h^{-1} phi(x,a), and clipping the result at H so that values never exceed the horizon bound. The bonus is the familiar elliptical uncertainty width from linear bandits: it is large along feature directions where little data has been collected and small along well-explored directions. During the episode, the algorithm simply acts greedily with respect to these optimistic Q-estimates.

The theoretical guarantee is a high-probability regret bound of order tilde O(sqrt(d^3 H^3 T)), where T equals K H is the total number of decisions across K episodes. This bound has no dependence on the cardinality of the state space and no exponential dependence on the horizon. The proof rests on three pillars. First, the linear transition structure turns every Bellman expectation into a linear function of features, so ridge regression can estimate backed-up values without ever representing the full transition kernel. Second, the regression targets contain the learned future value function V_{h+1}^k, which is itself a function of the data. Standard fixed-function concentration does not apply, so the analysis uses uniform self-normalized concentration over the class of value functions the algorithm can possibly produce. This uniformity introduces a d^2 covering cost over the bonus matrix, which is why the confidence radius scales as d H rather than sqrt(d) H. Third, optimism propagates backward by Bellman induction. If the optimistic value at step h plus one dominates the optimal value, then the extra future value contributes nonnegatively to the Bellman backup at step h, so the same bonus suffices to maintain optimism at the earlier step. This recursive optimism is what keeps the horizon dependence polynomial.

For regret accounting, the per-episode optimistic gap telescopes into a martingale difference sequence plus cumulative bonus terms. Azuma's inequality controls the martingale part, while the elliptical-potential lemma bounds the total bonus paid across episodes by order d log T per step. Multiplying the cumulative width sum by the confidence radius yields the final sqrt(d^3 H^3 T) rate. The method also extends gracefully to approximate linear MDPs, where the transition and reward are within zeta of a linear model. In that case the regret picks up an additional term linear in T times zeta, reflecting the persistent bias of model misspecification, while the square-root exploration term remains unchanged.

What makes LSVI-UCB more than regression plus UCB is the precise way these ingredients fit together. The linear MDP factorization closes the Bellman backup under the feature class. The uniform concentration argument justifies using the learned future value as a regression target. Recursive optimism turns local confidence intervals into a global exploration policy. And the elliptical-potential lemma converts the geometry of feature visits into a sublinear regret bound. Without any one of these pieces, the analysis collapses either statistically or computationally.

The following Python script gives a small, self-contained illustration of the core mechanics on a synthetic linear MDP. It constructs an environment where transitions and rewards are linear in a known feature map, then runs the LSVI-UCB backward sweep and greedy rollouts for several episodes. The code also computes the realized regret against the optimal policy and verifies that the optimistic value estimates indeed upper-bound the optimal values at the initial states, giving a concrete sanity check of the algorithm's design.

```python
import numpy as np

np.random.seed(0)

H = 5          #horizon
A = 3          #number of actions
d = 4          #feature dimension
K = 200        #episodes
LAMBDA = 1.0
BETA = 2.0 * d * H * np.sqrt(np.log(2 * d * H * K / 0.05))

def phi(s, a):
    x = np.zeros(d)
    x[0] = 1.0
    x[1] = (s - 4.5) / 4.5
    x[2] = np.sin(2 * np.pi * a / A)
    x[3] = np.cos(2 * np.pi * a / A)
    return x / np.linalg.norm(x)

S = list(range(10))
true_theta = np.random.randn(H, d) * 0.1
true_mu = np.random.randn(H, d, len(S))
for h in range(H):
    for i in range(d):
        true_mu[h, i] = np.abs(true_mu[h, i])
        true_mu[h, i] /= true_mu[h, i].sum()

def true_reward(h, s, a):
    return np.clip(phi(s, a) @ true_theta[h], 0.0, 1.0)

def sample_next(h, s, a):
    probs = phi(s, a) @ true_mu[h]
    probs = np.clip(probs, 0, None)
    probs /= probs.sum()
    return np.random.choice(S, p=probs)

def compute_optimal():
    Q = np.zeros((H, len(S), A))
    V = np.zeros((H + 1, len(S)))
    for h in range(H - 1, -1, -1):
        for s in S:
            for a in range(A):
                r = true_reward(h, s, a)
                exp_v = sum((phi(s, a) @ true_mu[h])[sp] * V[h + 1, sp] for sp in range(len(S)))
                Q[h, s, a] = r + exp_v
            V[h, s] = Q[h, s].max()
    return Q, V

Q_opt, V_opt = compute_optimal()

Lambda = [LAMBDA * np.eye(d) for _ in range(H)]
Phi_sum = [np.zeros(d) for _ in range(H)]
target_sum = [np.zeros(d) for _ in range(H)]  #sum of phi * (r + V_next)
W = [np.zeros(d) for _ in range(H)]

def optimistic_value(h, s, a):
    f = phi(s, a)
    return min(f @ W[h] + BETA * np.sqrt(f @ np.linalg.inv(Lambda[h]) @ f), H)

def optimistic_V(h, s):
    return max(optimistic_value(h, s, a) for a in range(A))

total_regret = 0.0
regrets = []

for ep in range(K):
    for h in range(H - 1, -1, -1):
        W[h] = np.linalg.solve(Lambda[h], target_sum[h])

    s = np.random.choice(S)
    episode_value_opt = V_opt[0, s]
    episode_value_alg = 0.0
    trajectory = []

    for h in range(H):
        a = max(range(A), key=lambda a: optimistic_value(h, s, a))
        r = true_reward(h, s, a)
        sp = sample_next(h, s, a)
        trajectory.append((h, s, a, r, sp))
        episode_value_alg += r
        s = sp

    for h, s, a, r, sp in trajectory:
        f = phi(s, a)
        Lambda[h] += np.outer(f, f)
        bonus_next = optimistic_V(h + 1, sp) if h + 1 < H else 0.0
        target_sum[h] += f * (r + bonus_next)

    total_regret += episode_value_opt - episode_value_alg
    regrets.append(total_regret)

W = [np.linalg.solve(Lambda[h], target_sum[h]) for h in range(H)]
violations = 0
for s in S:
    if optimistic_V(0, s) < V_opt[0, s] - 1e-6:
        violations += 1

print(f"Total regret after {K} episodes: {total_regret:.3f}")
print(f"Optimism violations at h=0: {violations} out of {len(S)} states")
print(f"Final regret / sqrt(K): {total_regret / np.sqrt(K):.3f}")
```
