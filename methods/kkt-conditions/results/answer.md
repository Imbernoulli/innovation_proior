# KKT Conditions

KKT conditions give the standard first-order local certificate for constrained optimization. For

$$
\min_x f(x)\quad\text{s.t.}\quad h_i(x)=0,\qquad g_j(x)\le 0,
$$

they assert, under appropriate constraint qualifications, that a local minimizer $x^\star$ must admit multipliers $\nu_i$ and $\lambda_j\ge 0$ satisfying:

- stationarity: $\nabla f(x^\star)+\sum_i\nu_i\nabla h_i(x^\star)+\sum_j\lambda_j\nabla g_j(x^\star)=0$;
- primal feasibility: $h_i(x^\star)=0$ and $g_j(x^\star)\le 0$;
- dual feasibility: $\lambda_j\ge 0$;
- complementary slackness: $\lambda_j g_j(x^\star)=0$.

The unique insight is the unification. KKT does not merely say "check the boundary." It says the objective gradient, equality-constraint normals, and active inequality-constraint normals must form a local balance of forces. Inactive inequalities have slack, so complementary slackness forces their multipliers to zero; they exert no force. Active inequalities can carry positive multipliers, meaning the optimizer is pressing against them.

This changes the mental model from boundary guessing to multiplier balance. Instead of enumerating possible active constraint sets and hoping one contains the optimum, KKT represents active-set selection inside the equations: a constraint either binds and can carry load, or it is slack and has zero multiplier. The candidate optimum is certified by simultaneous feasibility, sign consistency, and gradient cancellation.

For convex problems, the certificate becomes especially powerful: with convex objective and inequality constraints, affine equalities, and standard regularity such as Slater's condition, KKT conditions are not only necessary but sufficient for global optimality. In nonconvex settings, they remain a necessary local test under constraint qualifications, but a KKT point still needs further analysis before being treated as globally optimal.
