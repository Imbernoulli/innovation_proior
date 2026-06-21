An agent is dropped into an unknown finite MDP — it picks actions, collects bounded rewards, and is shoved between states by transition probabilities it cannot see — and the question I care about is sharper than convergence: is there an algorithm that, with probability at least $1-\delta$, behaves within $\epsilon$ of optimal after only a *polynomial* number of actions and amount of computation, polynomial in the state count, the action count, $1/\epsilon$, $1/\delta$, and the inherent time-scale of the problem? The two standard families both fail in exactly the way that matters. Q-learning converges to $Q^*$, but its theorem assumes every state-action pair is visited infinitely often and supplies no strategy that achieves that — the hard half of the problem, guaranteeing the visitation, is shoved into an assumption. Model-based value iteration on empirical counts has the same defect: it converges *if you explore enough*, and nobody bounds "enough." The obstacle behind both is the exploration–exploitation dilemma, where every action does two jobs at once: it earns reward and it reveals information. Always exploit and the agent may stay forever wrong about the action that opens the door to a high-reward region; always explore and it gathers information it never cashes in. Practitioners knew that optimistic initialization — set value estimates high so a greedy agent is dragged toward under-explored options — tends to help, but it was filed under ad-hoc tricks: no theorem said why optimism resolves exploration, how high to inflate, or for how long.

Before stating what to do, I have to fix what "near-optimal in finite time" can even mean, because in the undiscounted average-reward case it is subtle. A two-state example settles it: from state $0$ with self-loop probability $1-\rho$ and probability $\rho$ of slipping into an absorbing high-payoff state, the asymptotic return is large but reaching it genuinely takes on the order of $1/\rho$ steps — the *optimal policy itself* needs that long to mix. So I cannot promise to beat the optimal policy in arbitrary time. What I can promise is to compete, in time polynomial in $T$, with the best policy whose own return settles within $T$ steps. The $\epsilon$-return mixing time of $\pi$ is the smallest $T$ with $|U^\pi(i,t)-U^\pi|\le\epsilon$ for all $t\ge T$ and all start states $i$; let $\Pi^{T,\epsilon}$ be the policies that mix within $T$, and $\mathrm{opt}(\Pi^{T,\epsilon})$ their best return. In the discounted case the same $T$ appears as a horizon time: bounding any path by $\sum_{k=1}^{T}\gamma^{k-1}R_k + R_{\max}\gamma^T/(1-\gamma)$ and forcing the tail below $\epsilon$ gives $T\ge \frac{1}{1-\gamma}\log\frac{R_{\max}}{\epsilon(1-\gamma)}$, beyond which the future is negligible. One parameter $T$ tames both cases.

I propose R-MAX. The route to it runs through three load-bearing pieces. The first is the simulation lemma: how accurate must a model be before planning on it is safe? Fix a policy $\pi$, a start state $i$, and the $T$-step average return. If two models share the path-reward function, the value gap is controlled by how differently they distribute probability over $T$-paths, $|U_M-U_{\hat M}|\le R_{\max}\sum_p|\Pr_M(p)-\Pr_{\hat M}(p)|$. A path probability is a *product* of $T$ transitions, which could drift, so I telescope through hybrid processes $h_i$ that use one model for the first $i$ steps and the other afterward; consecutive hybrids differ in exactly one transition. Factoring each path into prefix, that single transition, and suffix, the prefix probabilities sum to $1$, each transition gap is at most $\eta$, and summing the suffix over the $N$ possible next states contributes a factor $N$, so each of the $T$ telescope terms is at most $N\eta$ and

$$\sum_p \big|\Pr_M(p)-\Pr_{\hat M}(p)\big| \;\le\; N\,T\,\eta,\qquad |U_M-U_{\hat M}|\;\le\; R_{\max}\,N\,T\,\eta.$$

The decisive feature is that the accuracy I need depends only *polynomially* on $1/T$, $1/\epsilon$, $1/N$: taking $\eta = \epsilon/(2NTR_{\max})$ spends half the value budget on model error and keeps the other half for the induced-model slack. The second piece is how many samples deliver $\eta$-accuracy, which is pure concentration. For a Bernoulli slot indicator, Hoeffding gives $\Pr(|K^{-1}\sum_j X_j - p|>b)\le 2\exp(-2Kb^2)$; with $b=K^{-1/3}$ the failure probability is $2\exp(-2K^{1/3})$, exponentially small. Pushing $K^{-1/3}$ below $\eta$ with margin and surviving the slot union bound yields the hard known threshold

$$K_1 \;=\; \max\!\Big(\big\lceil (4NTR_{\max}/\epsilon)^3\big\rceil,\; \big\lceil -6\ln^3(\delta/(6M_{\text{slots}}))\big\rceil\Big)+1,$$

with $M_{\text{slots}}=NA$ for a single-agent MDP and $Nk^2$ for the zero-sum stochastic-game version. Once a slot has $K_1$ transition samples I may *declare it known*, freeze its empirical next-state frequencies, and treat them as real; using only the first $K_1$ samples keeps the frozen estimate from drifting and keeps the count clean.

The third piece is what to do about the slots I do not yet trust, and here is the design choice that is the whole point. The explicit route restricts the model to the set $S$ of currently known states, keeps the true transitions among them, and redirects every transition that would leave $S$ into one absorbing drain $s_0$ that pays $0$ and self-loops. Running the real optimal $T$-step policy $\pi^*$ on this $M_S$ and splitting its paths into $q$-paths that stay inside $S$ (identical in $M$ and $M_S$) and $r$-paths that escape gives a clean dichotomy: either the known region already contains a near-optimal policy I can find by planning and exploit, or $\sum_r \Pr_M[r]\ge \alpha/G_{\max}^T$, so some policy reaches an unknown state with non-trivial probability and I can find *that* by rewarding $s_0$. Explore or exploit, both computable by value iteration — this is Explicit Explore or Exploit, $E^3$. But $E^3$ carries an explicit branch and, to decide it, leans on comparing against the unknown $\mathrm{opt}(\Pi^{T,\epsilon})$. What makes R-MAX is collapsing that branch. The two computations differ only in the reward assigned to the unknown region: real rewards versus $R_{\max}$ at the drain. So bake the optimism into a *single* model and plan once. Keep known slots at their empirical estimates, but route every unknown transition with probability $1$ into a fictitious absorbing state $G_0$ that yields $R_{\max}$ and self-loops, and give an untried slot's immediate reward $R_{\max}$ until its payoff is first observed. An untrusted entry is thereby worth the best continuation that could possibly happen — $R_{\max}$ forever in the average view, the discounted tail in the discounted view. Compute the single optimal policy of this optimistic model $M_L$ and execute it. No branch, no comparison against opt, no exploration schedule: optimism *is* the exploration mechanism and the planner *is* the decision-maker.

Why the fiction never hurts is the lever of the proof. Any policy is worth at least as much in $M_L$ as in $M$ — paths avoiding $L$ are identical, unseen rewards are never below the real bounded payoff, and after the first unknown entry the fiction gives the largest payoff any real continuation could — so $V_{M_L}(\text{R-MAX})\ge U_M(\pi^*)$. Now compare what the R-MAX policy earns in reality versus in the fiction, splitting its $T$-paths into $q$-paths that never touch an unknown entry (where $M$ and $M_L$ agree exactly, contributing zero difference) and $r$-paths that hit one. Since the $T$-step average payoff lies in $[0,R_{\max}]$,

$$\big|U_M(\text{R-MAX}) - V_{M_L}(\text{R-MAX})\big| \;\le\; R_{\max}\sum_r \Pr_M[r].$$

The gap is exactly (probability of hitting an unknown entry) $\times R_{\max}$, and the dichotomy falls out for free. If $\sum_r\Pr_M[r]\ge \alpha/R_{\max}$, the agent explores: with probability at least $\alpha/R_{\max}$ per phase it touches an unknown entry and collects a sample. Otherwise the gap is below $\alpha$, and $U_M(\text{R-MAX}) > V_{M_L}(\text{R-MAX}) - \alpha \ge U_M(\pi^*) - \alpha = \mathrm{opt}-\alpha$. The suboptimality and the exploration probability are two readings of one inequality, and I never have to know which case I am in or what opt is. The remaining subtlety is that I plan on the empirical $\hat M_L$, not on true known-slot probabilities; the simulation lemma absorbs this by running the argument across the real $M$, the empirical $\hat M_L$ I plan on, and an idealized $M'_L$ with true known-slot probabilities, the $\epsilon/2$ of model error plus $\epsilon/2$ of induced-model slack combining to $\epsilon$.

Each of the four design choices beats its obvious alternative. $R_{\max}$ optimism beats zero or pessimistic initialization, which can make a greedy planner ignore an unknown action forever. The fictitious absorbing state gives the unknown a *stable maximal* continuation value, so the planner values reaching it correctly for the optimism argument rather than chasing a moving target. The hard known threshold beats letting estimates float, because the simulation lemma needs fixed accuracy and freezing at $K_1$ samples makes the counting and union bounds finite. And the single optimistic plan beats the explicit $E^3$ branch by folding explore and exploit into one planning call. Assembling and counting: initialize every slot unknown and optimistic; repeat, computing the optimal $T$-step policy of the current optimistic model, executing it until a slot crosses $K_1$, recording transitions, replacing a slot's immediate reward by its observed payoff on first visit, and marking a slot known once it has $K_1$ transition samples. There are $M_{\text{slots}}$ slots, each known after $K_1=\mathrm{poly}(N,k\text{ or }A,T,1/\epsilon,\log(1/\delta))$ samples, so pigeonhole caps useful unknown-slot visits at $M_{\text{slots}}K_1$; in the exploration branch a $T$-step attempt hits an unknown slot with probability at least $\alpha/R_{\max}$, and choosing $K_2$ attempts polynomially large so that $K_2\alpha/R_{\max}-K_2^{2/3}$ exceeds $M_{\text{slots}}K_1$ delivers the needed samples with failure below $\delta/3$. The total $\delta$ budget splits three ways — a known slot secretly mis-estimated (controlled by the $K_1$ union bound), unlucky exploration attempts (the $K_2$ bound), and realized exploitation return below its expectation (concentration of bounded $T$-step averages) — and unioning gives the guarantee: with probability at least $1-\delta$, R-MAX attains expected $T$-step average return at least $\mathrm{Opt}(\Pi(\epsilon,T))-2\epsilon$ in a number of actions and amount of computation polynomial in $N$, $k$ or $A$, $T$, $1/\epsilon$, $1/\delta$, and $R_{\max}$, with the internal tolerance rescaled for any desired final $\epsilon$. The mixing time need not even be known: the whole bound is a polynomial $P(T)$, so run with $T=1$ for $P(1)$ steps, $T=2$ for $P(2)$, and so on; reaching the genuine $P(T_0)$ phase costs at most $T_0\,P(T_0)$, still polynomial, and every over-estimate past $T_0$ is near-optimal and amortizes the early losses. The concrete planner is $T$-step backward value iteration on the optimistic model, with unknown transitions pointing into the high-value fictitious state and immediate reward $R_{\max}$ until the slot payoff is seen.

```python
import numpy as np

def rmax_known_threshold(n_states, n_actions, R_max, eps, delta, T, *, joint_action=False):
    """R-MAX K1 threshold.

    joint_action=False gives the single-agent MDP slot count N*A.
    joint_action=True gives the stochastic-game slot count N*k^2.
    """
    slot_count = n_states * (n_actions**2 if joint_action else n_actions)
    log_arg = delta / (6.0 * slot_count)
    transition_term = (4.0 * n_states * T * R_max / eps) ** 3
    confidence_term = -6.0 * (np.log(log_arg) ** 3)
    return int(max(np.ceil(transition_term), np.ceil(confidence_term))) + 1

def value_iteration_optimistic(known, Phat, Rmodel, R_max, T, gamma, n_states, n_actions):
    """T-step planning in the optimistic induced model."""
    def absorber_tail(steps):
        if steps <= 0:
            return 0.0
        if np.isclose(gamma, 1.0):
            return R_max * steps
        return R_max * (1.0 - gamma**steps) / (1.0 - gamma)

    U = np.zeros(n_states)
    pi = [[0] * n_states for _ in range(T)]
    for t in range(T - 1, -1, -1):
        tail = absorber_tail(T - t - 1)
        Q = np.empty((n_states, n_actions))
        for s in range(n_states):
            for a in range(n_actions):
                if known[s][a]:
                    Q[s, a] = Rmodel[s, a] + gamma * Phat[s][a].dot(U)
                else:
                    Q[s, a] = Rmodel[s, a] + gamma * tail
            pi[t][s] = int(np.argmax(Q[s]))
        U = Q.max(axis=1)
    return pi

def r_max(env, n_states, n_actions, R_max, eps, delta, T, gamma):
    K1 = rmax_known_threshold(n_states, n_actions, R_max, eps, delta, T)

    counts = np.zeros((n_states, n_actions, n_states))
    n_sa = np.zeros((n_states, n_actions), dtype=int)
    reward_seen = np.zeros((n_states, n_actions), dtype=bool)
    known = [[False] * n_actions for _ in range(n_states)]
    Phat = [[np.zeros(n_states) for _ in range(n_actions)] for _ in range(n_states)]
    Rmodel = np.full((n_states, n_actions), R_max, dtype=float)

    s = env.current_state()
    while True:
        pi = value_iteration_optimistic(known, Phat, Rmodel, R_max, T, gamma,
                                        n_states, n_actions)
        for t in range(T):
            a = pi[t][s]
            r, s_next = env.step(a)
            if not reward_seen[s, a]:
                Rmodel[s, a] = r
                reward_seen[s, a] = True
            if n_sa[s, a] < K1:
                counts[s, a, s_next] += 1
                n_sa[s, a] += 1
                if n_sa[s, a] == K1:
                    known[s][a] = True
                    Phat[s][a] = counts[s, a] / K1
                    s = s_next
                    break
            s = s_next
```
