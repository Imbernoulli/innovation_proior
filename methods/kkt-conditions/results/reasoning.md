The starting problem is a mismatch between the unconstrained optimality rule and what actually happens at a constrained boundary. If there are no constraints, a local minimizer has no first-order downhill direction, so the gradient must be zero. With constraints, that condition is too strong. A point can be optimal even while the objective still has a nonzero slope, because the directions that would improve the objective may leave the feasible set.

The older boundary-search instinct is to guess where the optimum sits: maybe on one constraint, maybe on the intersection of several constraints, maybe in the interior. That works for small examples but it does not scale as a method. It treats the active boundary as something discovered by enumeration, and it separates the geometry of feasibility from the calculus of the objective.

KKT's move is to encode the active-boundary logic directly in a single local certificate. Equalities always constrain motion, so their gradients can enter stationarity with free multipliers. Inequalities constrain motion only when they are tight, so their multipliers must be nonnegative and must vanish whenever the inequality has slack. The equation

$$
\nabla f(x^\star)+\sum_i\nu_i\nabla h_i(x^\star)+\sum_j\lambda_j\nabla g_j(x^\star)=0
$$

then says that the objective gradient is not required to disappear by itself. It is required to be balanced by the normals of the constraints that can actually push back.

That is the geometric-mechanics view. Think of the objective as applying a local force that wants to move the point downhill. Equality constraints are rails: they can push in both normal directions, so their multipliers have either sign. Inequality constraints are one-sided walls: they can push only when touched, and only in the direction that keeps the point feasible, so their multipliers are nonnegative. If a wall is not touched, it cannot push; complementary slackness records exactly that with $\lambda_j g_j(x^\star)=0$.

This is why KKT is more than a checklist. Stationarity alone would miss feasibility. Feasibility alone would not say the point is locally best. Nonnegative multipliers alone would not identify active constraints. Complementary slackness alone would not balance the objective. Together they form a unified certificate: the point is feasible, the prices/forces have the right sign, inactive constraints exert no force, and the remaining local forces exactly cancel.

The same certificate also explains the economic reading. A multiplier is a shadow price because it measures how valuable it would be to relax the corresponding constraint at the optimum. If a constraint is slack, relaxing it has no first-order value, so the price is zero. If a constraint binds, its price may be positive because the optimum is pressing against it.

The caveat is that KKT is a first-order local statement unless convexity upgrades it. In nonconvex problems, satisfying KKT can mean a local minimum, a saddle-like candidate, or another stationary boundary point depending on second-order behavior and global shape. Constraint qualifications are also essential for the necessity theorem: if the feasible set has a degenerate cusp or badly dependent active gradients, the normal-balance description may fail even at a true local optimum.

The core insight remains stable across those caveats: constrained optimality is not "find where the boundary happens to stop descent" by case guessing. It is "find a feasible point where objective descent is locally neutralized by admissible constraint reactions, with inactive constraints carrying zero load." KKT made that balance algebraic, checkable, and reusable.
