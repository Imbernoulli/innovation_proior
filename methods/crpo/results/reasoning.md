I start with the constrained objective `max_pi J_0(pi)` subject to `J_i(pi) <= d_i`, and I keep returning
to the same practical mismatch. The Lagrangian method turns safety into a weighted blend,
`J_0 - sum_i lambda_i (J_i - d_i)`, then learns the weights from constraint violation. That is principled
at the saddle-point level, but the policy update I execute at a particular iteration is controlled by a
dual variable that remembers past violations. If the current policy is newly unsafe, the multiplier may
still be too small; if the policy is already back under budget, the multiplier may still be too large.
So the problem is not that duality is mathematically useless. The problem is that the dual variable is a
slow actuator for a decision that I can read from the current policy estimate.

I ask what I actually want from the next policy step. If all estimated costs are within their budgets,
then using the step to reduce cost is wasteful because the only unsolved objective is reward. If at least
one estimated cost is beyond its budget by the allowed margin, then improving reward is the wrong
priority because the policy is outside the acceptable region. This turns the apparent reward-versus-cost
trade-off into a choice of target for a single ordinary policy-optimization step. I do not need to form a
weighted mixture; I need to choose the reward objective when the policy is approximately feasible, and a
violated cost objective when it is not.

The resulting iteration is direct. Under the current policy `pi_{w_t}`, I estimate action-values
`\bar Q_t^i` for the reward and every cost. I then estimate each constraint return as a weighted average,
`\bar J_{i,B_t} = sum_{j in B_t} rho_{j,t} \bar Q_t^i(s_j,a_j)`, reusing the same distributional
information the policy-evaluation step already provides. If `\bar J_{i,B_t} <= d_i + eta` for every
cost constraint, I add the current parameter to the approximately feasible set `N_0` and take one update
toward maximizing `J_0`. If some constraint has `\bar J_{i,B_t} > d_i + eta`, I choose one violated
constraint and take one update toward minimizing that `J_i`. In natural-gradient form this means a plus
step for reward and a minus step for the selected cost: `w_{t+1} = w_t + alpha \bar Delta_t` in the
reward branch, or `w_{t+1} = w_t - alpha \bar Delta_t` in the cost-repair branch, where
`\bar Delta_t = (1 - gamma)^(-1) \bar Q_t^i` for the chosen reward or cost objective.

The tolerance `eta` is not a decorative hyperparameter, and I should not describe it as if it alone were
the final feasibility guarantee. It defines the approximately feasible set used by the switch and by the
output rule. If it is too large, the method can ignore meaningful constraint violation; if it is too
small, the proof has too little room around the boundary, where the optimum often lies and where
estimation error affects the switch. The analysis chooses `eta` on the same scale as the desired error:
in the tabular theorem, `eta = Theta(sqrt(|S||A|) / ((1 - gamma)^1.5 sqrt(T)))`, with natural-gradient
stepsize `alpha = (1 - gamma)^1.5 / sqrt(|S||A| T)`. The output is not simply the last iterate; it is
chosen uniformly from `N_0`, the iterations whose estimated constraints passed the `d_i + eta` test.
That is why the feasibility proof can bound true violation by the tolerance plus estimation error.

Now I check the two update cases carefully. In the reward case, maximizing `J_0` means ascending the
reward direction, so the sign is positive. In the violated-cost case, the goal is to minimize a cost
return, so the policy step descends that cost; equivalently, a policy-gradient implementation can ascend
on the negative cost advantage. If several constraints are violated, the algorithm may choose any one of
them. Random choice is a clean default, but the important condition is simply to select a currently
violated constraint. That keeps the policy update a single unconstrained optimization step rather than a
combined multi-constraint subproblem.

The convergence argument has to account for the fact that I keep changing objectives. Standard NPG
analyses assume one fixed function, so they do not directly apply. I condition on the high-probability
event that policy evaluation is accurate enough. On that event, the switch partitions iterations into the
approximately feasible set `N_0` and the constraint-repair sets `N_i`. The proof then shows that either
there are enough approximately feasible iterations to get the usual reward-progress bound, or their
average reward is already no worse than the optimal feasible policy. For constraints, because the
reported output is sampled from `N_0`, the violation is controlled by `eta` and the critic or sampling
error. This yields, in the tabular softmax setting, both
`J_0(pi*) - E[J_0(w_out)] <= Theta(sqrt(|S||A|) / ((1 - gamma)^1.5 sqrt(T)))` and
`E[J_i(w_out)] - d_i <= Theta(sqrt(|S||A|) / ((1 - gamma)^1.5 sqrt(T)))` for every constraint. If I move
to the neural softmax setting, I also carry the temperature-scaled parameter `tau_t W_t` and use the same
plus/minus branch signs there. The rate is still a `1/sqrt(T)` term plus a finite-width term, but the
parameters are different: `alpha = Theta(1/sqrt(T))`, `K_in = Theta((1 - gamma)^2 sqrt(m))`, and
`eta = Theta(m / ((1 - gamma) sqrt(T)) + 1 / ((1 - gamma)^2.5 m^(1/8)))`, with the high-probability
bound adding a fourth-root logarithmic factor.

So the method I arrive at is constraint-rectified policy optimization. It removes dual variables not by
relaxing the constraint but by replacing the dual-weighted blend with a current-feasibility switch:
approximately feasible policies spend the next step on reward, and policies with an estimated violated
cost spend the next step reducing one violated cost. The proof is built around the same switch, the same
signs, the same tolerance scale, and the same output-from-`N_0` rule.
