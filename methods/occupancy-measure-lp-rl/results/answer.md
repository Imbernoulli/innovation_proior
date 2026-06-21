# The Occupancy-Measure LP for MDPs

## Final Method

For a finite discounted MDP with states `S`, actions `A`, transition probabilities `P(t|s,a)`, rewards `r(s,a)`, start distribution `mu`, and `0 <= gamma < 1`, the control problem can be written exactly as a linear program by changing variables from policies to state-action flow.

The decision variable is the normalized discounted occupancy measure

```text
lambda^pi(s,a) = (1 - gamma) sum_{t >= 0} gamma^t Pr_pi[s_t = s, a_t = a | s_0 ~ mu].
```

It is joint mass on `(state, action)`, not merely a policy parameter. This is the key recasting: the controlled dynamics become linear conservation constraints over state-action occupancy flow.

## The LP Pair

The value-function primal starts from the Bellman inequalities:

```text
minimize_V    (1 - gamma) sum_s mu(s) V(s)

subject to    V(s) >= r(s,a) + gamma sum_t P(t|s,a) V(t)
              for every state s and action a.
```

The inequalities are linear because the Bellman max has been split into one constraint per action. If `V >= T V`, monotonicity and contraction imply `V >= V*`; since `V*` itself is feasible, strictly positive state weights recover `V*` everywhere. With the start distribution `mu` as the weight, `V*` is still optimal and the objective is the optimal start-distribution value, though states irrelevant to `mu` need not be uniquely pinned down.

Dualizing gives one nonnegative variable `lambda(s,a)` per Bellman inequality:

```text
maximize_lambda    sum_{s,a} lambda(s,a) r(s,a)

subject to         sum_a lambda(s,a)
                   = (1 - gamma) mu(s)
                     + gamma sum_{s',a'} P(s | s',a') lambda(s',a')
                   for every state s,

                   lambda(s,a) >= 0.
```

The equality is the discounted Bellman-flow equation. The left side is total mass available at state `s`. The right side is new mass injected from the start distribution plus discounted mass arriving from predecessor state-action pairs. This is the linear structure that replaces the nonlinear policy-to-value map.

## Occupancy Theorem

Every stationary policy produces a feasible `lambda`. Summing the discounted visitation definition over actions at state `s` gives

```text
sum_a lambda^pi(s,a)
  = (1 - gamma) mu(s)
    + gamma sum_{s',a'} P(s | s',a') lambda^pi(s',a').
```

Conversely, any nonnegative `lambda` satisfying the flow constraints is a genuine occupancy measure. Summing all state constraints forces `sum_{s,a} lambda(s,a) = 1`. Then recover a policy by conditioning the joint mass:

```text
pi_lambda(a|s) = lambda(s,a) / sum_b lambda(s,b)
```

for states with positive marginal mass, with arbitrary actions on zero-mass states. Under this policy, `lambda` satisfies the same discounted occupancy fixed-point equation as `lambda^{pi_lambda}`; uniqueness follows from the `gamma`-discounted transition operator. Therefore the feasible region of the dual LP is exactly the occupancy polytope of stationary policies. There are no extra flow points introduced by the relaxation.

The objective is also exact:

```text
sum_{s,a} lambda^pi(s,a) r(s,a)
  = (1 - gamma) E_pi[sum_{t >= 0} gamma^t r(s_t,a_t)].
```

Thus maximizing discounted return is equivalent to maximizing a linear reward functional over the occupancy-flow polytope.

## Policy Recovery And Certificates

Solve the occupancy LP for `lambda*`, then return

```text
pi*(a|s) = lambda*(s,a) / sum_b lambda*(s,b).
```

This normalization is used on states with positive marginal mass. On zero-mass states, any action distribution may be chosen because those choices do not change the occupancy or the `mu`-start objective. Strong LP duality gives

```text
sum_{s,a} lambda*(s,a) r(s,a) = (1 - gamma) mu^T V*.
```

Because `lambda*` is the occupancy of the recovered policy, that policy is optimal for the specified start distribution. Complementary slackness supplies the Bellman certificate: if `lambda*(s,a) > 0`, then the Bellman inequality for `(s,a)` is tight, so positive flow is placed only on actions greedy with respect to an optimal value function. Since the objective is linear over a polytope, an optimum can be chosen at a vertex; in the finite MDP case this corresponds to a deterministic optimal policy.

## Historical And Code Artifact

Manne's 1958 Cowles discussion paper is the average-cost predecessor. It uses variables `x_ij` for the joint probability of being in inventory state `i` and taking decision `j`, imposes nonnegativity, normalization, and statistical-equilibrium flow constraints, and minimizes the linear cost `sum c_ij x_ij`. The discounted formulation above keeps the same state-action-flow idea but replaces steady-state equilibrium with injected and discounted flow:

```text
state-action mass at s = start-source mass + discounted predecessor inflow.
```

The implementation artifact is `code/occupancy_lp.py`. It follows the same signs and normalization:

- `solve_value_primal(mdp)` builds each inequality as `gamma P[s,a,:] V - V[s] <= -r[s,a]`, exactly equivalent to `V(s) >= r(s,a) + gamma sum_t P(t|s,a)V(t)`.
- `solve_occupancy_dual(mdp)` calls `linprog` with objective `-r.reshape(-1)` and equality `sum_a lambda(s,a) - gamma sum_{s',a'} P(s|s',a')lambda(s',a') = (1 - gamma)mu(s)`, so the maximization sign and predecessor-flow sign match the dual above.
- `policy_from_occupancy(occupancy)` normalizes each positive state marginal and uses a uniform arbitrary policy on zero-mass states.
- `solve_optimal_policy(mdp)` returning both the optimal policy and occupancy.
