We are dropped into an unknown finite communicating MDP — at each step we sit in a state $s$, pick an action $a$, collect a random reward in $[0,1]$ with unknown mean $\bar r(s,a)$, and are carried to a new state drawn from an unknown transition law $p(\cdot\,|\,s,a)$ over the $S$ states. There is no discount and there are no resets: we wander continuously, and what we must keep small is the regret accumulated *while learning*,

$$\Delta(M,A,s,T) \;=\; T\,\rho^*(M) \;-\; \sum_{t=1}^{T} r_t,$$

against the optimal average reward $\rho^*$. This charges us for every reward we miss step by step, including each step we waste stranded somewhere unrewarding because of a bad exploratory move; exploration is not free, it is billed at full price. We want regret that grows like $\sqrt{T}$, knowing only $S$ and $A$, with the MDP's difficulty captured by a quantity that depends on its transition structure alone rather than on a mixing time of an unknown optimal policy that we would have to be told in advance.

The tools already on the table do not reach this. E3 partitions states into "known" and "unknown" and carries two model MDPs — an exploitation one and an exploration one with an absorbing bonus on the unknown frontier — explicitly deciding at each known state whether a near-optimal policy exists in the known-MDP (exploit) or it should run the exploration policy to fetch a new sample. R-max is cleaner: initialize every unknown $(s,a)$ optimistically to $R_{\max}$ with a self-loop, solve the fictitious model, follow it, and you are guaranteed either to be near-optimal or to hit an unknown $(s,a)$ and learn — explore-or-exploit made implicit. Both are correct, but both deliver an $\varepsilon$-optimal average reward with a $1/\varepsilon^3$ sample complexity, which when converted to regret over a horizon stalls at $T^{2/3}$, not $\sqrt{T}$; and both require the $\varepsilon$-return mixing time of an optimal policy as an *input*, a number we do not know and which, guessed too small, blows the bound up to exponential in the true one. MBIE computes an optimistic policy from confidence intervals but only for the discounted setting with a trajectory-based regret that can be trivially zero. Index policies for ergodic MDPs apply confidence per current state but assume every policy visits every state and hide an additive term exponential in the number of states. And UCRL, the direct predecessor, already implements optimism over a confidence set of MDPs but carries large exponents, leans on a mixing time, and assumes full ergodicity rather than the bare communicating property the regret notion actually needs. Two shortcomings stand out before any new method: the wrong exponent in $T$, and dependence on a hard-to-know mixing parameter instead of on transition structure.

I propose UCRL2. The single move it is built on is the one that already gives $\sqrt{T}$ regret with no oracle parameters in the one-state world — the multi-armed bandit. UCB1 keeps for each arm an empirical mean $\bar x_i$ and pulls the arm maximizing $\bar x_i + \sqrt{2\ln n / T_i}$, where the second term is exactly the Chernoff–Hoeffding half-width. The magic is that no exploration schedule is ever written down: the confidence interval *is* the exploration rule, since an under-pulled arm has a fat radius, hence a high index, hence gets tried, and once pulled enough the radius drops below the gap and it is abandoned. I lift precisely this — not "add a bonus sometimes," not "flag states known/unknown," but: keep a confidence region around the unknown parameters and behave as if the truth is the most favorable point in that region. So I maintain a set $M_k$ of MDPs that are statistically plausible given the data, pick out of it the MDP whose *optimal average reward is the largest* — the most optimistic plausible world — and follow its optimal policy. An under-visited $(s,a)$ has wide confidence, so the optimistic world can assign it a flattering reward and a convenient transition, which makes a good policy there want to go try it; exploration is again a side effect of optimism, and both the explore/exploit branch and the mixing-time input vanish.

Two costs that bandits never face must be absorbed. First, an MDP's unknown includes a whole distribution $p(\cdot\,|\,s,a)$ over $S$ next-states, and the $L_1$ deviation bound — the empirical and true distributions differ in $L_1$ by $\varepsilon$ with probability at most $(2^S-2)\exp(-n\varepsilon^2/2)$ — solves to a radius $\varepsilon \sim \sqrt{S/n}$, a factor $\sqrt{S}$ worse than a scalar mean. That $\sqrt{S}$ is not removable; it is the price of an $S$-outcome distribution and will appear in the bound. Second, a wrong action can *transport* us into a corner from which it takes many steps to return to reward. The number that measures this, depending only on transitions, is the diameter

$$D(M) \;=\; \max_{s\neq s'} \, \min_{\pi} \, \mathbb{E}\big[\,T(s'\,|\,M,\pi,s)\,\big],$$

finite exactly when the MDP is communicating, and never smaller than about $\log_A S - 3$. I expect $D$ to multiply the exploration price.

Planning in the optimistic world is the first real wall: optimism is now not just about rewards but about the *transition law* too, and a good policy will want transitions that funnel probability toward high-value states. So I collapse the entire plausible set into one MDP $\tilde M^+$ with an enriched action set — for each real $(s,a)$, each admissible transition vector $\tilde p(\cdot\,|\,s,a)$ inside the $L_1$ ball and each admissible mean $\tilde r(s,a)$ inside the reward interval becomes a separate extended action — so a policy on $\tilde M^+$ is exactly "a plausible MDP plus a policy on it," and maximizing average reward over plausible MDPs *is* solving $\tilde M^+$. Undiscounted value iteration on $\tilde M^+$ reads

$$u_{i+1}(s) \;=\; \max_a \Big\{\, \hat r_k(s,a) + d_r(s,a) \;+\; \max_{p \in \mathcal{P}(s,a)} \sum_{s'} p(s')\,u_i(s') \,\Big\},\qquad u_0 \equiv 0,$$

pushing each reward to the top of its interval. The outer $\max_a$ is ordinary; the inner $\max_p$ is new but cheap. It maximizes a *linear* function over the $L_1$-ball of distributions, so its optimum sits at a vertex I can read off without an LP: sort states by $u_i$ descending, start from the empirical $\hat p$, add the whole $L_1$ slack to the top-value state — $p(s'_1) = \min\{1,\ \hat p(s'_1) + d_p/2\}$ — and drain that same mass from the lowest-value states first until $p$ is a distribution again; shifting $d_p/2$ up and $d_p/2$ down moves exactly $d_p$ of $L_1$ mass, and one sort makes this $O(S)$ per $(s,a)$, hence $O(S^2A)$ per iteration. The reason to read off the vertex rather than call a solver is that the structure of the $L_1$ ball makes the optimum explicit; an LP would only rediscover it.

Convergence is the second subtlety. Undiscounted value iteration can oscillate forever if an optimal policy is periodic. But notice that in every iteration a single state $s'_1$ holds the maximal $u_i$, and because $d_p>0$ the inner step puts positive probability on $s'_1$ for *every* chosen row — including the row starting at $s'_1$ itself, a positive self-loop at the common target. That makes the selected transition matrix aperiodic, exactly the condition Puterman's proof needs; and when the plausible set contains the communicating true MDP, $\tilde M^+$ has a state-independent gain, so the iterates converge: $u_{i+1}-u_i \to \tilde\rho_k\,\mathbf 1$ and the recentred iterate converges to the bias. I stop when the increments have nearly equalized, $\mathrm{span}(u_{i+1}-u_i) < \varepsilon$, with $\varepsilon = 1/\sqrt{t_k}$ at episode $k$ — accurate enough that planning error contributes only $O(\sqrt{T})$ in total, cheap enough to compute — and the greedy policy $\tilde\pi_k$ is then $1/\sqrt{t_k}$-optimal in $\tilde M_k$, with $\tilde\rho_k \ge \rho^* - 1/\sqrt{t_k}$ because the true MDP is in the set so the most-optimistic plausible reward exceeds the true optimum up to the stopping slack.

The load-bearing fact about these iterates is that on a good episode the span of $u_i$ is at most $D$, hence the recentred $w_k(s) = u_i(s) - \tfrac12(\max u_i + \min u_i)$ satisfies $\|w_k\|_\infty \le D/2$. The argument: $u_i(s)$ is the best $i$-step value from $s$ in $\tilde M^+$, and if $u_i(s'') - u_i(s') > D$ I could beat $u_i(s')$ by first driving $s' \to s''$ in $\le D$ expected steps and then running the optimal $i$-step policy for $s''$, forfeiting at most $D$ rewards — a contradiction. This is where the diameter enters the bound: the recentred value vector is what gets multiplied by transition errors in the regret, so $D$ rides in through $\|w_k\|_\infty$.

The episode structure is chosen to make the accounting close. Recomputing the optimistic policy every step is the bandit habit, affordable there because the $\mathrm{argmax}$ is free, but extended value iteration is $O(S^2A)$ and the estimates change meaningfully only when much more data arrives. So I replan only at episode starts and end an episode the instant the count of the action currently being taken has *doubled* relative to its value at the episode start — the doubling rule. It does three jobs at once, which is why it beats fixed-length episodes. It caps the number of episodes at $m \le SA\log_2(8T/(SA))$, since a count can double only logarithmically often, keeping planning cost and the additive $D\cdot m$ terms logarithmic. It forces within-episode counts to satisfy $N_k \le N(s,a) \le 2N_k$, exactly what lets sums of confidence widths telescope. And it is the cheapest schedule that keeps the policy roughly current. The radii themselves are fixed by demanding that the truth almost never escape the set: with the $L_1$ tail and $S$ outcomes, $\varepsilon = \sqrt{14\,S\log(2At/\delta)/n}$ drives the fixed-$n$ transition failure below $\delta/(20\,t^7 SA)$, and Hoeffding's $\varepsilon_r = \sqrt{7\log(2SAt/\delta)/(2n)}$ does the same for rewards, so after unioning over counts and pairs $P(M\notin M_k) < \delta/(15\,t_k^6)$ — fast enough that even multiplying by the episode length and summing leaves the whole bad-episode contribution at $O(\sqrt{T})$.

The regret on good episodes is the real bound. Optimism plus the value-iteration relation give $\Delta_k \le v_k(\tilde P_k - I)w_k + 2\sum v_k\cdot(\text{reward radius}) + 2\sum v_k/\sqrt{t_k}$, and splitting $\tilde P_k - I = (\tilde P_k - P_k) + (P_k - I)$ separates the optimism-vs-reality transition gap from a near-zero stationary term. By Hölder the first piece is bounded by $\sum_s v_k(s,\tilde\pi_k(s))\,\|\tilde p_k - p_k\|_1\,\|w_k\|_\infty$, and since both transition vectors lie in the plausible set their $L_1$ difference is at most twice the radius while $\|w_k\|_\infty \le D/2$, so the factors of two cancel and this dominant term is $D\sqrt{14\,S\log(2AT/\delta)}\cdot\sum v_k/\sqrt{N_k}$ — carrying $D$ (from the span), $\sqrt{S}$ (from the $L_1$ radius), and the visit-count sum. The second piece is controlled by a martingale $X_t = (p(\cdot\,|\,s_t,a_t) - e_{s_{t+1}})w_k$ with $|X_t| \le D$, so Azuma gives $\sum_k v_k(P_k-I)w_k \le D\sqrt{(5/2)T\log(8T/\delta)} + D\cdot m$. Everything reduces to $\sum_k\sum_{s,a} v_k/\sqrt{N_k}$, where the doubling rule pays off a second time: since $v_k \le N_k$ within an episode, an induction gives $\sum_k v_k/\sqrt{N_k} \le (\sqrt2+1)\sqrt{N(s,a)}$, and Jensen sums this over pairs to $(\sqrt2+1)\sqrt{SAT}$. Assembling, the dominant $D\sqrt{S\log}\cdot\sqrt{SAT} = DS\sqrt{AT\log}$ swallows the rest, and with probability at least $1-\delta$,

$$\Delta(M,\text{UCRL2},s,T) \;\le\; 34\,DS\sqrt{AT\log(T/\delta)} \;=\; \tilde O\big(DS\sqrt{AT}\big).$$

A two-state-gadget construction embeds a bandit with $\approx SA$ arms inside the MDP, whose rewarding state returns with probability $\delta = \Theta(1/D)$ so each successful trip lasts about $D$ steps; the KL/Pinsker argument on the special arm forces $\Omega(\sqrt{DSAT})$ regret, matching the upper bound in $A$ and $T$ up to a $\sqrt{DS}\,\sqrt{\log}$ factor — so optimism did not cost the exponent in $T$, and the diameter is genuinely necessary. The same machinery, read three ways, yields a sample complexity of $\tilde O(D^2S^2A/\varepsilon^2)$ steps to per-step regret below $\varepsilon$, a gap-dependent $O(D^2S^2A\log T/g)$ logarithmic bound with $g$ the average-reward gap, and, by restarting with confidence $\delta/\ell^2$ at steps $t_i = \lceil i^3/\ell^2\rceil$, an $\tilde O(\ell^{1/3}T^{2/3}DS\sqrt A)$ bound against per-segment optima when the MDP changes $\ell$ times.

```python
import numpy as np

def ucrl2(mdp, T, delta):
    S, A = mdp.S, mdp.A
    Nsa  = np.zeros((S, A)); Rsa = np.zeros((S, A)); Psas = np.zeros((S, A, S))
    s = mdp.reset(); t = 1
    while t <= T:
        tk = t; vk = np.zeros((S, A))
        rhat = Rsa  / np.maximum(1, Nsa)
        phat = Psas / np.maximum(1, Nsa)[:, :, None]
        dr = np.sqrt(7 * np.log(2*S*A*tk/delta) / (2*np.maximum(1, Nsa)))     # reward radius
        dp = np.sqrt(14 * S * np.log(2*A*tk/delta) / np.maximum(1, Nsa))      # L1 transition radius
        policy = extended_value_iteration(rhat, dr, phat, dp, S, A, 1/np.sqrt(tk))
        while t <= T and vk[s, policy[s]] < max(1, Nsa[s, policy[s]]):
            a = policy[s]; r, s2 = mdp.step(s, a)
            vk[s, a] += 1; Rsa[s, a] += r; Psas[s, a, s2] += 1
            s = s2; t += 1
        Nsa += vk

def extended_value_iteration(rhat, dr, phat, dp, S, A, eps):
    r_opt = np.minimum(1.0, rhat + dr)
    u = np.zeros(S)
    while True:
        order = np.argsort(-u)
        q = np.empty((S, A))
        for s in range(S):
            for a in range(A):
                p = max_l1_transition(phat[s, a], dp[s, a], order)
                q[s, a] = r_opt[s, a] + p @ u
        u_next = q.max(axis=1); d = u_next - u
        if d.max() - d.min() < eps:
            return q.argmax(axis=1)
        u = u_next

def max_l1_transition(p_hat, radius, order):
    p = p_hat.copy()
    top = order[0]
    p[top] = min(1.0, p[top] + radius / 2.0)
    total = p.sum()
    j = len(order) - 1
    while total > 1.0 and j > 0:
        low = order[j]
        cut = min(p[low], total - 1.0)
        p[low] -= cut
        total -= cut
        j -= 1
    return p
```
