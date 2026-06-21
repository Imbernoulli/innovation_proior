I have a pile of GPS traces — taxi drivers threading the same road network day after day — and I want to learn what they are actually trying to do, so that I can predict the next turn or infer a destination from half a trip. Cloning the behavior is the obvious move: at each road segment, copy the turn the driver tended to take. But that only memorizes the segments I happened to observe; it says nothing about a segment I never saw, or about a slightly different goal. What transfers is the *reason* — the trade-off the driver makes between distance, speed, road type, and number of turns. So I model the network as a Markov Decision Process: states are road segments, actions are intersection transitions, the destination is an absorbing state, and the driver is assumed to optimize some reward. Forward reinforcement learning takes a reward and produces behavior; I have the behavior and want the reward. This is the inverse problem. Concretely, each segment $s_j$ carries a feature vector $f_{s_j}\in\mathbb{R}^k$ (road type, speed, lanes, turn taken), the driver scores a state by weights $\theta$ on those features and a whole path $\zeta$ by the sum along it, so $\mathrm{reward}(f_\zeta)=\theta\cdot f_\zeta$ with path feature counts $f_\zeta=\sum_{s_j\in\zeta} f_{s_j}$, and the demonstrations give the empirical expected feature counts $\bar f=\tfrac{1}{m}\sum_i f_{\tilde\zeta_i}$.

The trouble is that recovering $\theta$ is ill-posed. The first instinct — find $\theta$ that makes the demonstrated paths optimal — is exactly how the inverse problem was first posed (Ng & Russell 2000), and the solution set is degenerate. Their characterization says the policy that always picks $a_1$ is optimal iff $(P_{a_1}-P_a)(I-\gamma P_{a_1})^{-1}R\succeq 0$ for every other action $a$; the all-zero reward satisfies this trivially, because with zero reward *every* policy is optimal, and so does any constant reward, and beyond those a whole cone of weights works. "Make the demonstration optimal" carries almost no information. Ng & Russell patch this with a margin tie-break, maximizing $\sum_s(Q^*(s,a_1)-\max_{a\neq a_1}Q^*(s,a))$ via a linear program — but the margin is an arbitrary objective with no probabilistic meaning, it demands the full optimal policy as input, and it collapses entirely when the demonstrator is imperfect, which GPS-tracked humans always are. Relaxing the demand to merely *match feature counts* has a beautiful justification — since reward is linear in features, any behavior whose expected counts equal $\bar f$ has the same value as the demonstrations under *every* linear reward, including the true one (Abbeel & Ng 2004) — but it does not resolve the ambiguity, it relocates it: $\sum_\zeta P(\zeta)f_\zeta=\bar f$ is a handful of linear equations on a distribution over astronomically many paths, satisfied by countless policies, and Abbeel & Ng return a *mixture* of policies (flip a coin to choose which to follow) selected by another max-margin tie-break. A mixture is not a coherent model of one driver, and there is still no principled rule for which feature-matching distribution to commit to. Maximum margin planning (Ratliff, Bagnell & Zinkevich 2006) inherits the same fragility: when no single reward makes the behavior cleanly optimal, the margin has nothing to latch onto.

The wrongness is the clue, and it is the oldest problem in statistical inference: assign a distribution when your knowledge fixes only some expectation values and leaves the rest open. The clean resolution is Jaynes's — among all distributions consistent with what you know, choose the one of maximum entropy, $H(P)=-\sum_\zeta P(\zeta)\log P(\zeta)$, the unique distribution that is maximally noncommittal about everything the constraints do not pin down. I propose **Maximum Entropy Inverse Reinforcement Learning**: stop reasoning about policies and mixtures, reason directly about a distribution $P(\zeta)$ over entire paths, and pick the maximum-entropy one subject to feature-matching and normalization. I maximize $H(P)$ subject to $\sum_\zeta P(\zeta)f_\zeta=\bar f$ and $\sum_\zeta P(\zeta)=1$. Turning Jaynes's crank, I form the Lagrangian with multipliers $\theta$ (one per feature) and $\mu$, attaching $\theta$ to model-minus-demonstrated counts so a positive reward raises probability,
$$J=-\sum_\zeta P(\zeta)\log P(\zeta)+\theta\cdot\Big(\sum_\zeta P(\zeta)f_\zeta-\bar f\Big)+\mu\Big(\sum_\zeta P(\zeta)-1\Big),$$
and setting $\partial J/\partial P(\zeta)=0$ gives $-\log P(\zeta)-1+\theta\cdot f_\zeta+\mu=0$, so $\log P(\zeta)=\theta\cdot f_\zeta+\mu-1$. The $\mu$ term is a path-independent constant fixed by normalization, $\exp(\mu-1)=1/Z(\theta)$ with $Z(\theta)=\sum_\zeta\exp(\theta\cdot f_\zeta)$, and therefore
$$P(\zeta\mid\theta)=\frac{1}{Z(\theta)}\exp(\theta\cdot f_\zeta)=\frac{1}{Z(\theta)}\exp\Big(\sum_{s_j\in\zeta}\theta\cdot f_{s_j}\Big).$$
I did not choose this exponential form — it fell out. The probability of a path is exponential in its reward; higher-reward paths are exponentially more likely; two paths of equal reward get equal probability; and the Lagrange multipliers $\theta$ that enforce feature-matching *are* the reward weights. The reward is recovered not by fiat or by a margin heuristic but as the dual variable of "match the demonstrated feature counts while assuming nothing else." This is a Boltzmann distribution over behaviors, a single globally normalized object — not a mixture, not a per-state patchwork.

The global normalization is load-bearing, not decoration. The obvious alternative is to normalize locally, setting each action's probability proportional to $\exp(Q^*(s,a))$, a softmax at each branch. That family suffers label bias, the pathology of locally normalized sequence models: probability mass is conserved locally, so a segment with two outgoing turns splits its mass two ways and one with five splits five ways, a split forced by branching structure rather than reward. A path through low-branching regions then hoards probability for purely structural reasons, and a higher-reward path can end up *less* probable than a lower-reward one — incoherent for a model meant to say "high reward means likely." The globally normalized $P(\zeta)\propto\exp(\theta\cdot f_\zeta)$ compares all paths in one shared pool, so equal-reward paths tie exactly and higher-reward paths get exponentially more probability. For stochastic dynamics the driver should not be rewarded or blamed for the dice the environment rolls, so I condition on the transitions: with $T$ the transition distribution, and assuming the randomness has limited effect on which behavior the driver favors (so the partition function is roughly constant across outcomes), the exact constrained distribution reduces to the tractable approximation
$$P(\zeta\mid\theta,T)\approx\frac{\exp(\theta\cdot f_\zeta)}{Z(\theta,T)}\prod_{(s_{t+1},a_t,s_t)\in\zeta}P_T(s_{t+1}\mid a_t,s_t),$$
where the reward still drives an exponential preference and the transition product folds in the chance the world realizes that path. This immediately yields a coherent stochastic policy, $P(\text{action }a\mid\theta,T)\propto\sum_{\zeta:\,a\in\zeta_{t=0}}P(\zeta\mid\theta,T)$ — no coin flip, no mixture.

To fit $\theta$ I maximize the average log-likelihood of the demonstrations, $\theta^*=\arg\max_\theta\tfrac{1}{m}\sum\log P(\tilde\zeta\mid\theta,T)$ — which, satisfyingly, is the *same* problem as feature-matching and as maximum entropy in this family. Writing the transition product as a base measure $b_T(\zeta)=\prod P_T(s_{t+1}\mid a_t,s_t)$, the normalizer is $Z(\theta,T)=\sum_\zeta b_T(\zeta)\exp(\theta\cdot f_\zeta)$ and the $\theta$-independent $\log b_T(\tilde\zeta)$ term drops from the gradient. Differentiating the log-partition gives $\nabla_\theta\log Z(\theta,T)=\sum_\zeta P(\zeta\mid\theta,T)f_\zeta$, the model's expected feature counts, so
$$\nabla_\theta L(\theta)=\bar f-\sum_\zeta P(\zeta\mid\theta,T)f_\zeta=\bar f-\sum_{s_i}D_{s_i}f_{s_i},$$
demonstrated minus expected feature counts, with $D_{s_i}$ the expected state-visitation frequency. This says exactly what learning *is*: when the model under-visits highways relative to the drivers, the highway weight rises; when it over-visits, it falls; and the process stops precisely when expected counts equal demonstrated counts, at which point $\nabla L=0$ recovers the Abbeel–Ng performance guarantee, now realized by a single stochastic policy chosen by principle rather than tie-break. The optimization is well-behaved: differentiating once more, $\nabla^2_\theta L=-\mathrm{Cov}_{P(\zeta\mid\theta,T)}[f_\zeta]\preceq 0$ — the second derivative of a log-partition is a variance, Jaynes's identity reappearing as curvature — so $L$ is concave and the fit is convex, no local maxima. ($Z(\theta)$ converges for finite-horizon and discounted infinite-horizon problems, and the demonstrated trips are absorbed in finitely many steps, so the entropy-maximizing weights produce finite-time paths and the relevant $Z$ converges.) Plain gradient ascent suffices; swapping in exponentiated-gradient updates yields the $\ell_1$-flavored regularization that "match the counts only to within sampling noise" turns into (Dudík & Schapire 2006).

The only hard quantity is the expected feature counts $\sum_\zeta P(\zeta\mid\theta,T)f_\zeta=\sum_{s_i}D_{s_i}f_{s_i}$, and naive path enumeration is exponential in the horizon — hopeless on a 300,000-state network. But the expected counts depend on paths only through $D_{s_i}$, the expected number of visits to each state, which dynamic programming computes in two passes without ever enumerating a path. The backward pass builds remaining-horizon partition functions: initialize $Z^{(0)}_{s_i}=1$ and recurse for $h=1,\dots,N$ with
$$Z^{(h)}_{a_{i,j}}=\sum_k P(s_k\mid s_i,a_{i,j})\,\exp(\mathrm{reward}(s_i\mid\theta))\,Z^{(h-1)}_{s_k},\qquad Z^{(h)}_{s_i}=\sum_{a_{i,j}}Z^{(h)}_{a_{i,j}},$$
which is value iteration's backup but with a soft sum over actions in place of a hard max — exactly what the exponential-family model asks for — and gives the time-indexed local policy $P_t(a_{i,j}\mid s_i)=Z^{(N-t)}_{a_{i,j}}/Z^{(N-t)}_{s_i}$, the fraction of a state's exponentiated path-mass flowing through each action. The forward pass then pushes visitation mass through that policy: seed $D_{s_i,0}=p_0(s_i)$ with the empirical start-state distribution and propagate
$$D_{s_k,t+1}=\sum_{s_i,a_{i,j}}D_{s_i,t}\,P_t(a_{i,j}\mid s_i)\,P(s_k\mid s_i,a_{i,j}),$$
finally summing $D_{s_i}=\sum_{t=0}^{N-1}D_{s_i,t}$. Now $\sum_{s_i}D_{s_i}f_{s_i}$ is the model's expected feature counts in time polynomial in states and horizon, the gradient $\bar f-\sum_{s_i}D_{s_i}f_{s_i}$ is in hand, and I step $\theta$ along it and repeat. The reward drops out of the dual, the behavior model drops out of the principle, and the algorithm drops out of the gradient.

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
