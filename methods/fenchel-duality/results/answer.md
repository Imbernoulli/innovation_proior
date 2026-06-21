# Fenchel Duality

Fenchel duality turns convex minimization into a search for the strongest global lower bound. For a
closed proper convex function $f$, the conjugate

$$
f^*(y)=\sup_x\{\langle y,x\rangle-f(x)\}
$$

is the price of using slope $y$ as a global affine minorant. The Fenchel-Young inequality

$$
f(x)+f^*(y)\ge \langle y,x\rangle
$$

is equivalent to

$$
f(x)\ge \langle y,x\rangle-f^*(y).
$$

So a dual variable is a lower-bound certificate: it chooses a supporting slope and receives the best
intercept that still keeps the affine function below $f$ everywhere.

For the primal problem

$$
p^\star=\inf_x\{f(x)+g(Ax)\},
$$

choose paired conjugate bounds with slopes $-A^T u$ and $u$:

$$
f(x)\ge \langle -A^T u,x\rangle-f^*(-A^T u),
$$

$$
g(Ax)\ge \langle u,Ax\rangle-g^*(u).
$$

Adding them cancels the linear terms, giving a constant lower bound valid for every $x$:

$$
f(x)+g(Ax)\ge -f^*(-A^T u)-g^*(u).
$$

The Fenchel dual is therefore

$$
d^\star=\sup_u\{-f^*(-A^T u)-g^*(u)\}.
$$

This is the central shift in viewpoint. The primal searches over points $x$ for the minimum value.
The dual searches over slopes $u$ for the highest certified floor beneath the whole objective.
Weak duality, $d^\star\le p^\star$, is immediate because every dual candidate is constructed as a
global lower bound.

The primal-dual gap is controlled exactly by the conjugate inequalities. For any $x,u$,

$$
f(x)+g(Ax)+f^*(-A^T u)+g^*(u)
$$

equals the sum of two nonnegative Fenchel-Young slacks:

$$
\big[f(x)+f^*(-A^T u)-\langle -A^T u,x\rangle\big]
+\big[g(Ax)+g^*(u)-\langle u,Ax\rangle\big].
$$

Zero gap means both supporting inequalities are tight:

$$
-A^T u\in\partial f(x),\qquad u\in\partial g(Ax).
$$

Under standard Fenchel-Rockafellar regularity conditions, such as an appropriate domain-interior
qualification or continuity of one term at a feasible finite point, the best lower bound reaches the
true minimum. Fenchel duality is powerful because it recasts "find the minimum" as "find the
strongest globally valid support bound," with the conjugate providing the dictionary between slopes,
supports, and certificates.
