I start with the part that refuses to look like linear optimization. For each state I want to choose an action, but the dynamic-programming equation says `V*(s) = max_a [r(s,a) + gamma sum_{s'} P(s'|s,a)V*(s')]`. The max is doing all the control work, and it is exactly what prevents the equation from being linear. If I optimize directly over the policy, I do not escape the problem. A policy changes the transition matrix `P_pi`, and the value is `(I - gamma P_pi)^-1 r_pi`; the inverse depends on the policy. So the obvious policy coordinates make the objective curved and global.

The first useful move is to stop treating the max as an operator and treat it as a set of inequalities. Saying that `V(s)` dominates the best one-step lookahead is the same as saying it dominates every action's lookahead:

`V(s) >= r(s,a) + gamma sum_{s'} P(s'|s,a)V(s')` for every `(s,a)`.

These constraints are linear in `V`. I need to check that this relaxation does not create the wrong lower surface. If `V >= T V`, monotonicity gives `V >= T V >= T^2 V >= ...`, and contraction gives `T^k V -> V*`. Therefore every feasible `V` lies above `V*` coordinatewise. Since `V*` itself satisfies all of the inequalities, it is the least feasible value vector. Minimizing a strictly positive weighted sum of the coordinates over this feasible region recovers `V*` everywhere. For a start distribution `mu` that has zero entries, `V*` is still an optimal solution and the objective is the `mu`-start value, but irrelevant coordinates need not be uniquely selected. For the start-specific dual I use `(1 - gamma) mu` as the weight, giving the primal program

`min_V (1 - gamma) sum_s mu(s)V(s)`

subject to

`V(s) >= r(s,a) + gamma sum_{s'} P(s'|s,a)V(s')`.

This already gives a linear program, but it is still not the conceptual end. It is a program over values. The policy is recovered afterward by a greedy action, which means I have not yet made the controlled motion itself linear. I need variables whose feasible set is the set of possible long-run controlled behaviors.

The dual of the value LP points to the right variables. There is one Bellman inequality per state-action pair, so the dual has one nonnegative variable per state-action pair. I call it `lambda(s,a)`. Forming the Lagrangian with constraints written as `r(s,a) + gamma P_{s,a}V - V(s) <= 0`, I collect the coefficient of a fixed `V(s)`. It contains the objective mass `(1 - gamma)mu(s)`, a negative contribution from all dual variables attached to actions taken at state `s`, and a positive contribution from predecessor state-action pairs that can transition into `s`. Boundedness in the free variable `V(s)` forces

`sum_a lambda(s,a) = (1 - gamma)mu(s) + gamma sum_{s',a'} P(s|s',a')lambda(s',a')`.

The remaining part of the Lagrangian is the linear reward term `sum_{s,a} lambda(s,a)r(s,a)`. So the dual is

`max_lambda sum_{s,a} lambda(s,a)r(s,a)`

subject to the equality above and `lambda >= 0`.

Now I read the dual constraint as a conservation law. The left side is all mass present at state `s`, summed over the actions chosen there. The right side has two sources: new mass injected from the initial distribution, `(1 - gamma)mu(s)`, and old mass arriving from predecessor pairs `(s',a')`, discounted by `gamma` and routed by `P(s|s',a')`. This is not merely a dual algebraic accident. It is the continuity equation for discounted state-action flow.

This suggests the right object for a policy. For a fixed policy `pi`, define

`lambda^pi(s,a) = (1 - gamma) sum_{t>=0} gamma^t Pr[s_t=s, a_t=a | s_0 ~ mu, pi]`.

The factor `(1 - gamma)` normalizes the total mass: summing over all states and actions gives `(1 - gamma) sum_t gamma^t = 1`. I check the flow equation directly. At time zero, mass at `(s,a)` is `(1 - gamma)mu(s)pi(a|s)`. At later times, mass first arrives into `s` from all predecessor pairs and then chooses action `a` with probability `pi(a|s)`. Therefore

`lambda^pi(s,a) = pi(a|s)[(1 - gamma)mu(s) + gamma sum_{s',a'} P(s|s',a')lambda^pi(s',a')]`.

If I sum this over actions, the policy probabilities at state `s` sum to one, and I get exactly the dual equality. Every policy gives a feasible nonnegative flow.

The other direction is the real test. A linear program over flows is useful only if its feasible points are genuine controlled trajectories, not extra artifacts introduced by relaxation. So I take an arbitrary `lambda >= 0` satisfying the flow equalities. First I sum those equalities over `s`. The left side becomes `sum_{s,a} lambda(s,a)`. The right side becomes `(1 - gamma)sum_s mu(s) + gamma sum_{s',a'}lambda(s',a') sum_s P(s|s',a')`, which is `(1 - gamma) + gamma sum lambda`. Since `gamma < 1`, this forces `sum lambda = 1`.

Now I condition the joint mass to recover an action rule:

`pi_lambda(a|s) = lambda(s,a) / sum_b lambda(s,b)`

whenever the state marginal is positive, and I choose anything on states with zero marginal. This is the only reasonable read-off rule because if `lambda` is joint state-action mass, the policy must be the conditional action distribution given the state. I still need to prove that this recovered policy generates the same `lambda`.

The flow equation says

`sum_a lambda(s,a) = (1 - gamma)mu(s) + gamma sum_{s',a'} P(s|s',a')lambda(s',a')`.

On states with positive marginal, multiplying both sides by `pi_lambda(a|s)` gives

`lambda(s,a) = pi_lambda(a|s)[(1 - gamma)mu(s) + gamma sum_{s',a'} P(s|s',a')lambda(s',a')]`,

because `pi_lambda(a|s) sum_b lambda(s,b) = lambda(s,a)`. On states with zero marginal, the flow equation makes the bracketed source-plus-inflow term zero, so any arbitrary action distribution still gives zero state-action mass. This is exactly the fixed-point equation for the discounted state-action occupancy of `pi_lambda`. For a fixed policy, the occupancy equation has a unique solution: it is an affine equation with a transition operator multiplied by `gamma`, so the inverse exists or, equivalently, repeated substitution is a contraction. Hence the feasible `lambda` is not a phantom point. It is precisely `lambda^{pi_lambda}`.

That closes the change of variables. The feasible set of the dual LP is exactly the set of normalized discounted occupancies of stationary policies. The return is also linear in this variable:

`sum_{s,a} lambda^pi(s,a)r(s,a) = (1 - gamma) E[sum_{t>=0} gamma^t r(s_t,a_t)]`.

The nonlinearity in the policy has not been approximated away; it has been moved into a larger variable whose valid values are described by linear flow constraints. The controlled dynamics are no longer hidden in `(I - gamma P_pi)^-1`. They are written as mass conservation over state-action pairs.

Strong LP duality now gives the optimality certificate. The primal value LP has `V*` as an optimal solution and objective `(1 - gamma)mu^T V*`. The dual occupancy LP has the same optimal value. If `lambda*` maximizes the dual, then the policy recovered by conditioning, `pi*(a|s) = lambda*(s,a)/sum_b lambda*(s,b)`, has occupancy `lambda*`, so its scaled return is the dual objective. Therefore its return from `mu` equals the optimal dynamic-programming value from `mu`.

Complementary slackness gives the interpretation I want. If `lambda*(s,a) > 0`, then the Bellman inequality for `(s,a)` must be tight at an optimal value solution:

`V*(s) = r(s,a) + gamma sum_{s'} P(s'|s,a)V*(s')`.

So positive flow is placed only on actions that are greedy with respect to the optimal value. The value function and the state-action flow are dual certificates for the same control problem.

The average-cost version fits the same picture with a different source term. Instead of discounted injection and discounted recirculation, the state-action variables are joint probabilities in statistical equilibrium. The balance equation says that the probability of starting a period in state `i` equals the probability of ending the previous period in that same state after all predecessor state-action transitions. The objective is a linear average cost in those joint probabilities, and conditioning the joint probabilities recovers the decision rule. That older equilibrium form and the discounted form share the same structural discovery: optimize over state-action visitation mass, constrained by flow conservation, rather than over policies directly.

So the method is not just "write an LP for an MDP." The method is to make state-action occupancy the decision variable, characterize the exact set of possible occupancies by linear Bellman-flow constraints, optimize the linear reward over that flow polytope, and recover the policy by statewise normalization.
