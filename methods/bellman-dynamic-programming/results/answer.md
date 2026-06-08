# Dynamic programming: the principle of optimality and the Bellman equation

## Problem

Maximize a return that accumulates over a sequence of stages, where at each stage a decision changes the system's state and earns a return. Direct enumeration of all decision sequences costs exponentially in the number of stages and is undefined when transitions are random. Dynamic programming solves this whole class of multistage decision problems through a single recursive object — the value function — and the equation it satisfies.

## Key idea

Stop solving for a fixed sequence of decisions (open-loop) and solve instead for a feedback rule on the state (closed-loop). The right object to optimize is the **value function**

$$f_N(p) = \text{optimal (expected) return over } N \text{ stages, starting from state } p,$$

a function of state and stages-to-go only. The optimal value is unique even when the optimal rule is not, and the rule recovers from the value function.

## The principle of optimality

> An optimal policy has the property that, whatever the initial state and initial decision are, the remaining decisions must constitute an optimal policy with regard to the state resulting from the first decision.

*Proof (exchange argument).* If, following an optimal policy, the tail after the first decision were suboptimal for the subproblem starting at the resulting state, replacing that tail with the optimal continuation from that state would strictly raise the total return while leaving the first decision fixed — contradicting optimality of the whole. ∎

## The Bellman equation

Applying the principle: take the first decision, land in the resulting state, then act optimally for the remaining stages — whose value is, by definition, the value function one stage shorter. Maximize over the first decision.

- **Discrete deterministic** (decision $k$ sends $p\mapsto T_k(p)$):
$$f_N(p) = \max_k\, f_{N-1}\big(T_k(p)\big), \qquad N=2,3,\dots$$

- **Discrete stochastic** (decision $k$ gives next-state distribution $dG_k(p,z)$):
$$f_N(p) = \max_k \int f_{N-1}(z)\, dG_k(p,z), \qquad N=2,3,\dots$$
The deterministic case is the degenerate case where $dG_k$ concentrates on $T_k(p)$.

- **With a per-stage reward** (general "reward-now plus value-of-next-state" form; allocation example):
$$f(x) = \sup_{0\le y\le x}\Big[\, g(y)+h(x-y)+f\big(ay+b(x-y)\big)\,\Big], \qquad f(0)=0.$$

- **Infinite-horizon stationary** (discount/contraction factor $\alpha\in(0,1)$):
$$f(p) = \max_k\Big[\, r(p,k) + \alpha\!\int f(z)\, dG_k(p,z)\,\Big].$$

- **Continuous / calculus of variations** (maximize $\int_0^T F(x,y)\,dt$ s.t. $\dot x = G(x,y)$, $x(0)=c$): with $f(c,T)=\max_y\int_0^T F\,dt$, the principle over an initial interval of length $S$ and then $S\to0$ gives
$$f_T = \max_v\big[\, F(c,v) + G(c,v)\,f_c\,\big],$$
with interior stationarity $F_v+G_v f_c=0$. When $G_v\ne0$, $f_c=-F_v/G_v$, so
$$f_T=\frac{F\,G_v-G\,F_v}{G_v}.$$
With elapsed time $t$, remaining time $\tau$, and costate $\lambda(t)=f_c(x(t),\tau)$, differentiating the PDE along $\dot x=G$ gives $\dot\lambda=-F_x-G_x\lambda$, while stationarity gives $F_v+G_v\lambda=0$. These are the Euler equations for the augmented integrand $F(x,v)+\lambda(G(x,v)-\dot x)$. Dynamic programming thus contains the calculus of variations and, unlike it, handles inequality constraints (the max is simply over the feasible set) and stochastic transitions (the integral becomes an expectation).

## The curse of dimensionality

The value function lives on the state space; tabulating it over a grid in an $M$-dimensional state costs exponentially in $M$. Dynamic programming trades the exponential blow-up in the *horizon* (the $k^N$ of policy enumeration) for an exponential in the *state dimension* — a decisive win whenever the state is low-dimensional.

## Constructive consequence: solving $f = T(f)$

Write the operator $T(W)(p) = \max_k\big[r(p,k) + \alpha\int W(z)\,dG_k(p,z)\big]$.

- **Contraction.** For $A_k=r(p,k)+\alpha\int W\,dG_k$ and $B_k=r(p,k)+\alpha\int V\,dG_k$, each $|A_k-B_k|\le\alpha\|W-V\|_\infty$; $\max A-\max B\le\max(A-B)$, and the reverse bound follows by swapping $W,V$. Hence $\|T W - T V\|_\infty \le \alpha\,\|W-V\|_\infty$ with $\alpha\in(0,1)$. By the contraction-mapping theorem, $T$ has a unique fixed point $f$, and the iterates converge to it geometrically from any start.

- **Value iteration (approximation in function space):** $f_{n+1}=T(f_n)$, converging geometrically.

- **Policy iteration (approximation in policy space):** evaluate the current policy $\pi$, then choose a greedy policy $\pi'$ with respect to $V_\pi$. Since $T_{\pi'}V_\pi=TV_\pi\ge T_\pi V_\pi=V_\pi$ and $T_{\pi'}$ is monotone, iterating $T_{\pi'}$ to its fixed point gives $V_{\pi'}\ge V_\pi$. Repeating this gives a monotone improving sequence to the optimum.

- **Finite horizon = backward induction:** terminal $f_1(p)=\max_k r(p,k)$, then $f_2$ from $f_1$, and so on back to the start; the constructed rule is optimal in every reachable subproblem.

## Implementation

```python
# The Bellman recursion in its operational finite-state forms.

def _best_action(actions, score, prefer=None, tol=0.0):
    best_action, best_value = None, float("-inf")
    preferred_value = None
    for action in actions:
        value = score(action)
        if prefer is not None and action == prefer:
            preferred_value = value
        if value > best_value:
            best_action, best_value = action, value
    if best_action is None:
        raise ValueError("each state must have at least one feasible decision")
    if preferred_value is not None and preferred_value >= best_value - tol:
        return prefer, preferred_value
    return best_action, best_value


def _expected(values, distribution):
    return sum(prob * values[state] for state, prob in distribution.items())


def value_iteration_finite(states, decisions, transition, reward, terminal, horizon):
    states = list(states)
    f_next = {p: terminal(p) for p in states}
    policy = [dict() for _ in range(horizon)]
    for stage in range(horizon - 1, -1, -1):
        f = {}
        for p in states:
            best_d, best_v = _best_action(
                decisions(p, stage),
                lambda d: reward(p, d, stage) + f_next[transition(p, d, stage)],
            )
            f[p], policy[stage][p] = best_v, best_d
        f_next = f
    return f_next, policy


def value_iteration_stochastic(states, decisions, next_dist, reward, terminal, horizon):
    states = list(states)
    f_next = {p: terminal(p) for p in states}
    policy = [dict() for _ in range(horizon)]
    for stage in range(horizon - 1, -1, -1):
        f = {}
        for p in states:
            best_d, best_v = _best_action(
                decisions(p, stage),
                lambda d: reward(p, d, stage) + _expected(f_next, next_dist(p, d, stage)),
            )
            f[p], policy[stage][p] = best_v, best_d
        f_next = f
    return f_next, policy


def value_iteration(states, decisions, next_dist, reward, alpha, tol=1e-10, max_iters=100000):
    if not 0 <= alpha < 1:
        raise ValueError("alpha must be in [0, 1)")
    states = list(states)
    f = {p: 0.0 for p in states}
    for _ in range(max_iters):
        f_new = {}
        delta = 0.0
        for p in states:
            _, best_v = _best_action(
                decisions(p),
                lambda d: reward(p, d) + alpha * _expected(f, next_dist(p, d)),
            )
            f_new[p] = best_v
            delta = max(delta, abs(best_v - f[p]))
        f = f_new
        if delta < tol:
            policy = {
                p: _best_action(
                    decisions(p),
                    lambda d: reward(p, d) + alpha * _expected(f, next_dist(p, d)),
                )[0]
                for p in states
            }
            return f, policy
    raise RuntimeError("value iteration did not converge before max_iters")


def _solve_linear_system(matrix, rhs):
    n = len(rhs)
    a = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            raise ValueError("singular policy-evaluation system")
        a[col], a[pivot] = a[pivot], a[col]
        scale = a[col][col]
        for j in range(col, n + 1):
            a[col][j] /= scale
        for r in range(n):
            if r == col:
                continue
            factor = a[r][col]
            if factor == 0.0:
                continue
            for j in range(col, n + 1):
                a[r][j] -= factor * a[col][j]
    return [a[i][n] for i in range(n)]


def _evaluate_policy(states, policy, next_dist, reward, alpha):
    index = {p: i for i, p in enumerate(states)}
    matrix = [[0.0 for _ in states] for _ in states]
    rhs = [0.0 for _ in states]
    for p, i in index.items():
        d = policy[p]
        matrix[i][i] = 1.0
        rhs[i] = reward(p, d)
        for z, prob in next_dist(p, d).items():
            matrix[i][index[z]] -= alpha * prob
    solution = _solve_linear_system(matrix, rhs)
    return {p: solution[index[p]] for p in states}


def policy_iteration(states, decisions, next_dist, reward, alpha):
    if not 0 <= alpha < 1:
        raise ValueError("alpha must be in [0, 1)")
    states = list(states)
    policy = {p: _best_action(decisions(p), lambda d: 0.0)[0] for p in states}
    while True:
        f = _evaluate_policy(states, policy, next_dist, reward, alpha)
        new_policy = {}
        stable = True
        for p in states:
            best_d, _ = _best_action(
                decisions(p),
                lambda d: reward(p, d) + alpha * _expected(f, next_dist(p, d)),
                prefer=policy[p],
                tol=1e-12,
            )
            new_policy[p] = best_d
            stable = stable and best_d == policy[p]
        if stable:
            return policy, f
        policy = new_policy
```
