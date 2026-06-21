The problem is to allocate one unit of effort at each discrete time step among several competing Markov reward processes, often called arms, jobs, or projects. Whichever arm we advance pays a random reward and moves to a new state according to its own transition law; every other arm is frozen, earning nothing and remaining exactly where it was. Future rewards are discounted by a factor beta between zero and one, and the goal is to maximize total expected discounted reward over an infinite horizon. The exact dynamic-programming equation exists and an optimal stationary policy is guaranteed by standard discounted-MDP theory, but its state is the joint state of all arms. If each arm has k states, the joint space has k^n states, and for Bayesian bandit arms the per-arm state is already an infinite-dimensional posterior. Solving the Bellman equation directly over this product space is therefore exponential in the number of arms and quickly becomes intractable. Myopic priority rules such as one-step-lookahead or cost-rate heuristics are cheap and per-arm, but they are provably optimal only in special deteriorating or monotone regimes and undervalue arms whose prospects improve with work, so they fail in the general discounted case.

The right way forward is the Gittins index, also called the dynamic allocation index. It assigns a single scalar to each arm based only on that arm's own current state and stochastic law, ignoring all other arms entirely. The optimal policy is then simple: at every step, advance any arm whose current index is maximal. Because the index decouples the problem, the intractable joint-state dynamic program collapses into n independent one-dimensional computations.

The index can be understood in three equivalent ways, each useful for a different purpose. First, it is the best discounted reward-rate the arm can deliver over a stopping time chosen by the player: for an arm in state x, it is the supremum over stopping rules tau of the expected discounted reward collected up to tau divided by the expected discounted time spent up to tau. The denominator is discounted time rather than step count, which is the only normalization that makes a constant-reward arm have an index equal to its constant reward. The supremum is attained by continuing to play the arm until its index first drops below the value it had when started. Second, the index is the largest constant per-pull charge gamma at which optimal play of the arm still breaks even; this fair-charge interpretation makes the optimality proof transparent. Third, it is the calibration value lambda at which one is indifferent between beginning with the arm and retiring forever to a standard arm that pays lambda each period. Because the standard arm has only one frozen state, the pair consisting of the original arm and the standard arm has the same state space as the original arm alone, yielding a one-dimensional Bellman equation: the value in state x is the maximum of retiring to lambda over one minus beta, or pulling the arm once and continuing. The index is the largest lambda for which the play-on branch is at least as good as retirement.

Why this decoupling is legitimate is the central insight of the problem. In a generic Markov decision process, unchosen actions disappear into the past and the state evolves regardless of what you do. Here, the arms you do not touch stay frozen. Postponing an arm therefore costs only the extra discounting, not any foregone future opportunity. That non-irrevocability means the arms can be ranked one at a time. The fair-charge view makes the optimality argument precise. For a single arm, let the fair charge after t plays be the index gamma(x(t)) of the current state, and let the prevailing charge be the running minimum of these fair charges over the plays so far. This prevailing-charge stream is nonincreasing, random, and policy-independent, because it depends only on the arm's own transitions, not on when other arms are interleaved. For any way of playing the arm up to a stopping time, the expected discounted reward is at most the expected discounted prevailing charges, with equality when the arm is stopped exactly at times where the fair charge has come down to the prevailing charge. Summing this inequality over all arms gives a universal upper bound: every policy's expected total discounted reward is at most the expected total discounted charges it pays. The greatest-index policy achieves this bound. It never leaves an arm idle while its fair charge is strictly above its prevailing charge, and it switches away from an arm only when that arm's fair charge has fallen to its prevailing charge, so the sequence of paid charges is nonincreasing. Because discounting weights earlier payments more heavily, this nonincreasing rearrangement is the largest possible charge total, and the policy simultaneously earns exactly that total. Hence the greatest-index policy is optimal.

Several special cases confirm the structure. For a deteriorating arm whose index never rises, the best stopping time is one step, so the index reduces to the immediate expected reward and one-step lookahead is optimal; this recovers classical priority rules such as the c-mu rule in scheduling. For an improving arm, the optimal stopping rule never cuts the run early. For the Bayesian Bernoulli bandit with Beta posterior exponents (a,b), the index exceeds the immediate posterior mean because continuing preserves the option value of learning; it can be computed by truncating the horizon and sweeping backward over the single arm's two-parameter state. For the Normal-mean bandit with posterior normal distribution, adding a constant shifts the index and multiplying rewards scales it, giving a shift-scale law that reduces tabulation to a single one-variable function. In every case the joint state space is avoided.

```python
def gittins_index_by_calibration(states, transition, reward, beta, lo, hi, tol=1e-9):
    """Compute the Gittins index for every state of ONE Markov reward process.

    The index is calibrated against a standard arm that pays a constant lam
    forever. For each candidate lam, value iteration is performed over the
    process's own states using the one-dimensional Bellman equation
        V(s) = max(lam/(1-beta), reward(s) + beta * E[V(next_state)]).
    The index of state x is the largest lam for which playing on is still at
    least as good as retiring to the standard arm.
    """
    def plays_on(x0, lam):
        retire = lam / (1 - beta)
        V = {s: reward(s) / (1 - beta) for s in states}
        while True:
            newV, delta = {}, 0.0
            for s in states:
                cont = reward(s) + beta * sum(p * V[s2]
                                              for s2, p in transition(s).items())
                newV[s] = max(retire, cont)
                delta = max(delta, abs(newV[s] - V[s]))
            V = newV
            if delta < tol:
                break
        cont0 = reward(x0) + beta * sum(p * V[s2]
                                        for s2, p in transition(x0).items())
        return cont0 >= retire

    index = {}
    for x0 in states:
        a, b = lo, hi
        while b - a > tol:
            m = 0.5 * (a + b)
            if plays_on(x0, m):
                a = m
            else:
                b = m
        index[x0] = a
    return index


def greatest_index_policy(processes, states, indices):
    """Advance the process whose current state has the largest Gittins index."""
    return max(range(len(processes)), key=lambda j: indices[j][states[j]])
```

The Gittins index theorem thus replaces an exponential joint-state dynamic program with n separate scalar index computations, and the resulting greatest-index policy is provably optimal for the whole family of independent Markov reward processes.
