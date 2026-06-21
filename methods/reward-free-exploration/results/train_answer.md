I present reward-free exploration, a two-phase algorithm for episodic tabular Markov decision processes. In the first phase the learner interacts with the environment without seeing any task reward. In the second phase a reward function is revealed, and a near-optimal policy must be produced using only the previously collected data. The canonical name is reward-free exploration because the exploration objective is decoupled from the reward that later defines the planning task.

The setting is a finite-horizon tabular MDP with S states, A actions, horizon H, unknown transitions, and rewards revealed only after data collection. If the reward were known during exploration, standard methods such as UCBVI or EULER could be used directly. But rewards are often tuned, shaped, or changed while sweeping constraints, so rerunning a reward-aware exploration algorithm for every candidate would waste samples. The goal is to collect a single reusable dataset that supports many downstream reward functions.

The key observation is that a dataset is useful only if it estimates transitions where later policies will go. The value-difference lemma makes this precise: the value gap between two MDPs that share a reward and policy but differ in transitions is a sum of transition-estimation errors weighted by the policy occupancy. Since the reward and later policy are unknown during data collection, the dataset must dominate the occupancy of every policy that might later be optimal for some reward.

A naive approach is to run standard exploration with the reward set to zero or with bonuses for unknown states. This is insufficient. Zero reward provides no adversarial value function that exposes what the planner will later need, and unknown-state bonuses chase a coarse known-set objective rather than the geometric coverage required by arbitrary future rewards. The right object is a data distribution strong enough for all later planning tasks but not demanding impossible samples from nearly unreachable states.

For each target state-time pair (s, h), the algorithm constructs a synthetic reachability reward that is one when the trajectory visits s at step h and zero otherwise. Under this probe, the value of a policy equals its probability of reaching (s, h), and the optimal value is the maximum reaching probability. This synthetic reward is not a downstream task; it is an instrument for measuring a cell of the transition geometry.

For every target (s, h), the method runs EULER for N_0 episodes on the corresponding reachability reward, obtaining a multiset of policies Phi^(s, h) whose average reaching probability is at least half of the optimal reaching probability. Because the indicator reward is at most one along any trajectory, EULER's problem-dependent regret bound gives a sample cost that scales with the maximum reaching probability, so faint targets cost less than easy targets. Each learned policy is then modified at the target state and time by replacing the action distribution with a uniform distribution over actions. This converts state coverage into state-action coverage without reducing the reaching probability, because reaching (s, h) is determined before the action at time h is taken.

All modified policy multisets are merged into a single set Psi. The exploration dataset D is collected by sampling a policy uniformly from Psi and rolling it out. The induced exploration distribution mu satisfies a coverage invariant: for every state-time pair whose maximum reaching probability is at least delta, the ratio between the best policy-action occupancy and the data occupancy is bounded by 2 S A H. The factor two comes from approximate reaching, A from spreading actions at the target, and S H from mixing over all target blocks. States below the reachability threshold are ignored, but their total contribution to any value function is small.

Set delta = epsilon / (2 S H^2). Low-reachability states contribute at most H S delta per step, hence H^2 S delta over the horizon, which is at most epsilon / 2. For covered states, the value-difference lemma and the coverage ratio reduce uniform policy evaluation to a self-bounded transition-estimation problem. A covering argument over deterministic action selectors yields a fast Bernstein rate. The per-step error is O(sqrt(H^3 S^2 A log terms / N)), and summing over H steps gives total evaluation error O(sqrt(H^5 S^2 A log terms / N)). Choosing N = O(H^5 S^2 A iota / epsilon^2) makes the covered part at most epsilon / 2, so the full evaluation error is at most epsilon.

Planning is then mechanical. Form the empirical transition model Phat from D and run any planner that returns an epsilon-suboptimal policy for the known MDP (Phat, r), such as value iteration or natural policy gradient. Let pi* be optimal in the true MDP and pihat* the empirical policy. The true suboptimality is bounded by evaluation error for pi*, the nonpositive empirical optimality gap, the solver error, and evaluation error for the returned policy. With an epsilon-accurate solver, the total suboptimality is at most 3 epsilon. Because the evaluation event is uniform over policies and rewards, the same dataset supports any number of later reward functions, even chosen adaptively.

The sample complexity has leading term O_tilde(H^5 S^2 A / epsilon^2), plus lower-order term O_tilde(S^4 A H^7 / epsilon). The extra factor of S over fixed-reward tabular RL is unavoidable. The lower bound constructs a binary tree of hard one-state instances. In each instance every action has an unknown near-uniform distribution over absorbing leaves, and a reward vector forces identification of the best transition distribution. Packing nearly uncorrelated perturbations and applying Fano's inequality gives Omega(n A / epsilon^2) samples per instance, and embedding Omega(S) copies multiplies the cost by S. Persistent terminal reward contributes the H^2 scaling, yielding Omega(S^2 A H^2 / epsilon^2). Thus the additional S factor is the genuine price of coverage for all possible downstream rewards.

The following Python script illustrates the core mechanism on a tiny synthetic MDP. It builds a random tabular MDP, computes reachability probes with synthetic rewards, mixes the resulting policies, collects a dataset, forms an empirical transition model, and compares the value of an empirical planner's policy to the true optimal value for a randomly chosen downstream reward.

```python
import numpy as np

def random_mdp(S, A, H, seed=0):
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.ones(S), size=(H, S, A))

def reachability_reward(S, H, target_s, target_h):
    r = np.zeros((H, S))
    r[target_h, target_s] = 1.0
    return r

def value_iteration(P, r, H):
    S, A = P.shape[1], P.shape[2]
    V = np.zeros((H + 1, S))
    pi = np.zeros((H, S, A))
    for h in range(H - 1, -1, -1):
        for s in range(S):
            q = np.array([r[h, s] + P[h, s, a] @ V[h + 1] for a in range(A)])
            a_best = np.argmax(q)
            V[h, s] = q[a_best]
            pi[h, s, a_best] = 1.0
    return V, pi

def policy_value(P, r, pi, s0):
    H, S, A = pi.shape
    V = np.zeros((H + 1, S))
    for h in range(H - 1, -1, -1):
        for s in range(S):
            V[h, s] = sum(pi[h, s, a] * (r[h, s] + P[h, s, a] @ V[h + 1]) for a in range(A))
    return V[0, s0]

def sample_trajectory(P, pi, s0):
    H, S, A = pi.shape
    traj = []
    s = s0
    for h in range(H):
        a = np.random.choice(A, p=pi[h, s])
        s_next = np.random.choice(S, p=P[h, s, a])
        traj.append((h, s, a, s_next))
        s = s_next
    return traj

def build_empirical_model(trajectories, S, A, H):
    counts = np.zeros((H, S, A, S))
    for traj in trajectories:
        for h, s, a, s_next in traj:
            counts[h, s, a, s_next] += 1
    P_hat = np.zeros((H, S, A, S))
    for h in range(H):
        for s in range(S):
            for a in range(A):
                total = counts[h, s, a].sum()
                P_hat[h, s, a] = counts[h, s, a] / total if total > 0 else np.ones(S) / S
    return P_hat

S, A, H = 5, 3, 4
P = random_mdp(S, A, H, seed=1)
s0 = 0
n_eps_per_target = 200

policies = []
for target_h in range(H):
    for target_s in range(S):
        r = reachability_reward(S, H, target_s, target_h)
        _, pi = value_iteration(P, r, H)
        pi_mod = pi.copy()
        pi_mod[target_h, target_s] = np.ones(A) / A
        policies.append(pi_mod)

trajectories = []
for _ in range(n_eps_per_target * S * H):
    pi = policies[np.random.randint(len(policies))]
    trajectories.append(sample_trajectory(P, pi, s0))

P_hat = build_empirical_model(trajectories, S, A, H)
r = np.random.default_rng(7).random((H, S))
_, pi_true = value_iteration(P, r, H)
_, pi_emp = value_iteration(P_hat, r, H)

print(f"true optimal value: {policy_value(P, r, pi_true, s0):.4f}")
print(f"empirical planner value: {policy_value(P, r, pi_emp, s0):.4f}")
print(f"dataset size: {len(trajectories)} episodes")
```

In summary, reward-free exploration separates the data-collection objective from the downstream reward by first constructing a coverage distribution that dominates every policy occupancy a future reward might exploit. Synthetic reachability rewards turn the geometric coverage problem into a sequence of ordinary planning tasks solvable by standard regret-minimizing methods. The resulting dataset supports arbitrarily many later rewards with a sample complexity carrying one extra factor of S compared with fixed-reward tabular RL, and this extra factor is necessary. The method is not merely optimism with the reward removed; it is a principled reduction from reward-agnostic exploration to the construction of a transition-coverage object.
