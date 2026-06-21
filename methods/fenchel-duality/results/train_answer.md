The problem is to understand and solve a convex minimization problem of the form inf_x f(x) + g(Ax), where f and g are closed proper convex functions and A is a linear map. The usual first instinct is to search directly over x for the point that makes the objective as small as possible. That works in simple cases, but it hides the geometry of the problem: nonsmooth terms, indicator functions, norms, and linear compositions can make the objective look complicated even though its pieces are simple. Another common idea is to write down stationarity or subgradient conditions, but those describe an optimal point after the fact; they do not explain why a particular dual vector certifies a lower bound on every primal point. Lagrange duality does produce lower bounds through multipliers, yet it treats constraints as the source of duality rather than the affine lower-bound structure of convex functions themselves. What is missing is a viewpoint that turns the convex pieces into a catalog of global supporting hyperplanes and then asks for the strongest one.

The method that captures this viewpoint is Fenchel duality. At its heart is the convex conjugate f^*(y) = sup_x { <y,x> - f(x) }. This is not merely a transform; it records, for every possible slope y, the best intercept that keeps the affine function x |-> <y,x> - f^*(y) below f everywhere. Rearranging the definition gives the Fenchel-Young inequality f(x) + f^*(y) >= <y,x>, which is the same as f(x) >= <y,x> - f^*(y). So a dual variable is a lower-bound certificate: it picks a slope, and the conjugate supplies the largest intercept that still produces a global affine minorant.

For the composite objective f(x) + g(Ax), a single minorant is not enough because its linear part still depends on x. The key trick is to choose two conjugate bounds whose linear parts cancel. Use slope -A^T u for f and slope u for g, where u lives in the range space of A. The two Fenchel-Young inequalities are f(x) >= < -A^T u, x > - f^*(-A^T u) and g(Ax) >= <u, Ax> - g^*(u). When they are added, the terms -<A^T u, x> and <u, Ax> cancel exactly, leaving f(x) + g(Ax) >= -f^*(-A^T u) - g^*(u). The right-hand side is a constant that does not depend on x, so every u gives a certified global lower bound on the primal minimum. The Fenchel dual problem is therefore d^* = sup_u { -f^*(-A^T u) - g^*(u) }, and weak duality d^* <= p^* is automatic because each dual candidate is constructed from valid inequalities.

The primal-dual gap has a clean interpretation. For any primal x and dual u, the gap p(x,u) - d(u) is exactly the sum of the two nonnegative Fenchel-Young slacks: f(x) + f^*(-A^T u) - < -A^T u, x > plus g(Ax) + g^*(u) - <u, Ax>. Zero gap means both inequalities are tight, which happens precisely when -A^T u is a subgradient of f at x and u is a subgradient of g at Ax. These conditions say that the chosen supporting hyperplanes actually touch the two convex pieces at compatible points. Under standard Fenchel-Rockafellar regularity, such as continuity of one term at a point where the other is finite or a relative-interior domain qualification, the best lower bound reaches the true minimum and strong duality holds. The entire method is therefore a reversal of perspective: instead of asking where the lowest point is, Fenchel duality asks what the highest globally valid floor beneath the objective can be.

```python
import numpy as np

# A simple finite-dimensional illustration of Fenchel duality.
# Primal:  minimize  (1/2)||x||^2 + (1/2)||A @ x - b||^2
# f(x) = (1/2)||x||^2,  g(z) = (1/2)||z - b||^2,  linear map A.
# Conjugates: f*(y) = (1/2)||y||^2,  g*(u) = (1/2)||u||^2 + u^T b.
# Dual:    maximize  -(1/2)||A^T u||^2 - (1/2)||u||^2 - b^T u.

np.random.seed(0)
n, m = 5, 4
A = np.random.randn(m, n)
b = np.random.randn(m)

# Solve the dual: concave quadratic with gradient -(A A^T + I) u - b = 0.
u_star = -np.linalg.solve(A @ A.T + np.eye(m), b)

# Recover primal candidate from the optimality condition x = -A^T u.
x_star = -A.T @ u_star

# Primal and dual optimal values.
primal_value = 0.5 * np.dot(x_star, x_star) + 0.5 * np.dot(A @ x_star - b, A @ x_star - b)
dual_value = -0.5 * np.dot(A.T @ u_star, A.T @ u_star) - 0.5 * np.dot(u_star, u_star) - np.dot(b, u_star)

# Verify the subgradient conditions.
subgrad_f = x_star
subgrad_g = A @ x_star - b
print("primal value:", primal_value)
print("dual value:  ", dual_value)
print("duality gap: ", primal_value - dual_value)
print("|| -A^T u - grad f(x) ||:", np.linalg.norm(-A.T @ u_star - subgrad_f))
print("|| u - grad g(Ax) ||:     ", np.linalg.norm(u_star - subgrad_g))
```
