I start with the part that refuses to look like linear optimization. For each state I want to choose an action, but the dynamic-programming equation says `V*(s) = max_a [r(s,a) + gamma sum_{s'} P(s'|s,a)V*(s')]`. The max is doing all the control work, and it is exactly what prevents the equation from being linear. If I optimize directly over the policy, I do not escape the problem. A policy changes the transition matrix `P_pi`, and the value is `(I - gamma P_pi)^-1 r_pi`; the inverse depends on the policy. So the obvious policy coordinates make the objective curved and global.

The first move I want to try is to stop treating the max as an operator and treat it as a set of inequalities. Saying that `V(s)` dominates the best one-step lookahead is the same as saying it dominates every action's lookahead:

`V(s) >= r(s,a) + gamma sum_{s'} P(s'|s,a)V(s')` for every `(s,a)`.

These constraints are linear in `V`. But I should be careful: relaxing one max equation into many inequalities can enlarge the feasible set, and I have no right yet to assume the relaxation has the same solution. So I need to check what this feasible region actually looks like. If `V >= T V`, monotonicity gives `V >= T V >= T^2 V >= ...`, and contraction gives `T^k V -> V*`. Therefore every feasible `V` lies above `V*` coordinatewise. Since `V*` itself satisfies all of the inequalities, it is the least feasible value vector. Minimizing a strictly positive weighted sum of the coordinates over this feasible region should then pull the answer down to `V*` everywhere.

I want to see this on something concrete before I trust it. Take two states and two actions with

```text
P(.|s0,a0) = (0.8, 0.2)   r(s0,a0) = 1
P(.|s0,a1) = (0.1, 0.9)   r(s0,a1) = 0
P(.|s1,a0) = (0.0, 1.0)   r(s1,a0) = 0
P(.|s1,a1) = (0.6, 0.4)   r(s1,a1) = 2
```

with `mu = (1, 0)` and `gamma = 0.9`. If the optimal policy is `a0` at `s0` and `a1` at `s1`, the value satisfies

```text
V0 = 1 + 0.9(0.8 V0 + 0.2 V1)
V1 = 2 + 0.9(0.6 V0 + 0.4 V1).
```

Solving the 2x2 system gives `V = (12.195, 13.415)`. Now I test all four Bellman inequalities at this `V`. At `s0`: `a0` gives `1 + 0.9(0.8*12.195 + 0.2*13.415) = 12.195`, which equals `V0` exactly, while `a1` gives `11.963 < 12.195`. At `s1`: `a1` gives `13.415 = V1`, while `a0` gives `12.073 < 13.415`. So all four inequalities hold, the chosen action is tight in each state, and the other action is slack. That is the picture the relaxation argument predicted: `V*` sits on the lower boundary, touching exactly the greedy constraints. The relaxation did not introduce a spurious lower surface here.

For a start distribution `mu` that has zero entries, `V*` is still an optimal solution and the objective is the `mu`-start value, but irrelevant coordinates need not be uniquely selected in general. With the start weight on `s0` only, I would still expect the inequality coupling to constrain `V1`, because `s0` reaches `s1` with probability `0.2`. So the weight `(1 - gamma) mu` gives the primal program

`min_V (1 - gamma) sum_s mu(s)V(s)`

subject to

`V(s) >= r(s,a) + gamma sum_{s'} P(s'|s,a)V(s')`.

This already gives a linear program, but it is still a program over values. The policy is recovered afterward by a greedy action, which means I have not yet made the controlled motion itself linear. I would like variables whose feasible set is the set of possible long-run controlled behaviors, so that the constraints describe the dynamics directly rather than the value of the dynamics.

The dual of the value LP is where I look next, because dual variables sit on constraints, and here there is one Bellman inequality per state-action pair. So the dual has one nonnegative variable per state-action pair. I call it `lambda(s,a)`. Forming the Lagrangian with constraints written as `r(s,a) + gamma P_{s,a}V - V(s) <= 0`, I collect the coefficient of a fixed `V(s)`. It contains the objective mass `(1 - gamma)mu(s)`, a negative contribution from all dual variables attached to actions taken at state `s`, and a positive contribution from predecessor state-action pairs that can transition into `s`. Because `V(s)` is a free variable, its coefficient must vanish, which forces

`sum_a lambda(s,a) = (1 - gamma)mu(s) + gamma sum_{s',a'} P(s|s',a')lambda(s',a')`.

The remaining part of the Lagrangian is the linear reward term `sum_{s,a} lambda(s,a)r(s,a)`. So the dual is

`max_lambda sum_{s,a} lambda(s,a)r(s,a)`

subject to the equality above and `lambda >= 0`.

The dual constraint reads like a conservation law. The left side is all mass present at state `s`, summed over the actions chosen there. The right side has two sources: new mass injected from the initial distribution, `(1 - gamma)mu(s)`, and old mass arriving from predecessor pairs `(s',a')`, discounted by `gamma` and routed by `P(s|s',a')`. That looks like a continuity equation for discounted state-action flow, but a structural resemblance in the dual is not yet a theorem about policies. I need to know whether these `lambda` variables actually correspond to controlled trajectories.

So I propose a candidate object. For a fixed policy `pi`, define

`lambda^pi(s,a) = (1 - gamma) sum_{t>=0} gamma^t Pr[s_t=s, a_t=a | s_0 ~ mu, pi]`.

The factor `(1 - gamma)` is meant to normalize the total mass: summing over all states and actions gives `(1 - gamma) sum_t gamma^t = (1 - gamma)/(1 - gamma) = 1`. I check the flow equation directly. At time zero, mass at `(s,a)` is `(1 - gamma)mu(s)pi(a|s)`. At later times, mass first arrives into `s` from all predecessor pairs and then chooses action `a` with probability `pi(a|s)`. Collecting the discounted sum,

`lambda^pi(s,a) = pi(a|s)[(1 - gamma)mu(s) + gamma sum_{s',a'} P(s|s',a')lambda^pi(s',a')]`.

If I sum this over actions, the policy probabilities at state `s` sum to one, and the bracket is exactly the right-hand side of the dual equality. So every policy gives a feasible nonnegative flow.

The other direction is the one I am less sure of, and it is the real test: a linear program over flows is only useful if its feasible points are genuine controlled trajectories, not artifacts of the relaxation. So I take an arbitrary `lambda >= 0` satisfying the flow equalities and ask whether it comes from some policy. First I sum those equalities over `s`. The left side becomes `sum_{s,a} lambda(s,a)`. The right side becomes `(1 - gamma)sum_s mu(s) + gamma sum_{s',a'}lambda(s',a') sum_s P(s|s',a')`, which is `(1 - gamma) + gamma sum lambda` since `mu` and each row of `P` sum to one. Setting `m = sum lambda`, this says `m = (1 - gamma) + gamma m`, so `(1 - gamma)m = (1 - gamma)` and `m = 1`. Any feasible flow has total mass exactly one.

Now I condition the joint mass to recover an action rule:

`pi_lambda(a|s) = lambda(s,a) / sum_b lambda(s,b)`

whenever the state marginal is positive, and I choose anything on states with zero marginal. If `lambda` is to be read as joint state-action mass, this conditional is forced; there is no other consistent way to extract a policy. But I still have to prove that this recovered policy actually regenerates `lambda`, otherwise the read-off is meaningless.

The flow equation says

`sum_a lambda(s,a) = (1 - gamma)mu(s) + gamma sum_{s',a'} P(s|s',a')lambda(s',a')`.

On states with positive marginal, multiply both sides by `pi_lambda(a|s)`. The left side becomes `pi_lambda(a|s) sum_b lambda(s,b) = lambda(s,a)` by definition of the conditional, so

`lambda(s,a) = pi_lambda(a|s)[(1 - gamma)mu(s) + gamma sum_{s',a'} P(s|s',a')lambda(s',a')]`.

On states with zero marginal, the left side is zero, so the bracketed source-plus-inflow term is zero, and any arbitrary action distribution there still gives zero state-action mass. Together these are exactly the fixed-point equation that `lambda^{pi_lambda}` satisfies. For a fixed policy that fixed point is unique: it is an affine system `x = b + gamma M x` with `M` a substochastic transition operator, and `(I - gamma M)` is invertible because the Neumann series `sum gamma^k M^k` converges for `gamma < 1`. So `lambda` and `lambda^{pi_lambda}` solve the same nonsingular system and must be equal. The feasible `lambda` is not a phantom point.

I want to see this closure numerically, not just believe the algebra. Running the dual LP on the two-state instance returns `lambda(s0,a0) = 0.780`, `lambda(s1,a1) = 0.220`, with the other two entries zero. Their sum is `1.000`, matching the mass argument. Conditioning gives `pi(a0|s0) = 1`, `pi(a1|s1) = 1`, the deterministic policy I guessed earlier. And feeding that flow back through the conservation equation, state by state: at `s0`, `0.780` versus `(1-0.9)*1 + 0.9*(0.8*0.780 + 0.6*0.220) = 0.1 + 0.9*0.756 = 0.780`; at `s1`, `0.220` versus `0 + 0.9*(0.2*0.780 + 0.4*0.220) = 0.9*0.244 = 0.220`. Both balance to the digits shown, so the LP solution really is the occupancy of the recovered policy.

That closes the change of variables: the feasible set of the dual LP is the set of normalized discounted occupancies of stationary policies. The return is also linear in this variable,

`sum_{s,a} lambda^pi(s,a)r(s,a) = (1 - gamma) E[sum_{t>=0} gamma^t r(s_t,a_t)]`,

since `lambda^pi` is exactly the discounted-and-normalized visitation. I check this against an independent computation rather than reusing the LP. For the recovered deterministic policy I form `P_pi` and `r_pi` and solve `V^pi = (I - gamma P_pi)^-1 r_pi` directly; this gives `V^pi = (12.195, 13.415)`, so `J(pi) = mu^T V^pi = 12.195`. Then `(1 - gamma) J(pi) = 0.1 * 12.195 = 1.2195`. The dual reward `sum lambda r = 0.780*1 + 0.220*2 = 1.2195`. The two agree, so the linear reward functional over flows really is the discounted return up to the `(1 - gamma)` scaling. The nonlinearity in the policy has not been approximated away; it has been moved into a larger variable whose valid values are described by linear flow constraints. The controlled dynamics are no longer hidden in `(I - gamma P_pi)^-1`; they are written as mass conservation over state-action pairs.

With both LPs in hand I can also test strong duality numerically. The primal objective is `(1 - gamma) mu^T V* = 0.1 * 12.195 = 1.2195`, and I just computed the dual objective as `1.2195`. They coincide, as duality requires for a feasible bounded pair. So if `lambda*` maximizes the dual, the policy recovered by conditioning has occupancy `lambda*`, its scaled return equals the dual objective, and that equals the primal objective `(1 - gamma)mu^T V*`. Therefore the recovered policy's return from `mu` equals the optimal dynamic-programming value from `mu`.

Complementary slackness should then say something about which actions carry flow. If `lambda*(s,a) > 0`, the corresponding dual constraint pairs with a tight primal inequality:

`V*(s) = r(s,a) + gamma sum_{s'} P(s'|s,a)V*(s')`.

In the numeric instance this is exactly what I saw: flow landed only on `(s0,a0)` and `(s1,a1)`, and those were precisely the two Bellman inequalities that were tight at `V*`, while the slack inequalities `(s0,a1)` and `(s1,a0)` got zero flow. So positive flow is placed only on actions that are greedy with respect to the optimal value. The value function and the state-action flow are dual certificates for the same control problem.

One caveat I should not overstate. When `mu` has zero entries, the primal objective puts no weight on those coordinates, so I worried the value LP might leave them unpinned. Testing a variant with an unreachable absorbing decoy state of weight zero, the LP still returned its exact value, because each state sits in its own constraint `V(s) >= r(s,a) + gamma P(s|s,a)V(s)` which alone determines an absorbing state's value. So in these small cases nothing is left free, even though in general the value on `mu`-irrelevant coordinates is only required to be optimal, not unique. The flow side is cleaner: the occupancy and the `mu`-start objective do not depend on the action choice at zero-mass states.

The average-cost version fits the same picture with a different source term. Instead of discounted injection and discounted recirculation, the state-action variables are joint probabilities in statistical equilibrium. The balance equation says that the probability of starting a period in state `i` equals the probability of ending the previous period in that same state after all predecessor state-action transitions. The objective is a linear average cost in those joint probabilities, and conditioning the joint probabilities recovers the decision rule. That older equilibrium form and the discounted form share the same structure: optimize over state-action visitation mass, constrained by flow conservation, rather than over policies directly.

So the program I arrive at is not just "write an LP for an MDP." It is to make state-action occupancy the decision variable, characterize the exact set of possible occupancies by linear Bellman-flow constraints, optimize the linear reward over that flow polytope, and recover the policy by statewise normalization.
