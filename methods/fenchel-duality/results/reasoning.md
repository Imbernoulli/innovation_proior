Start with the usual primal reflex: minimize an objective by looking for the point where it is
smallest. For a convex objective this is natural, but it hides the other half of convexity. A convex
function is not only something with minimizers; it is also the envelope of its affine lower
supports. Fenchel duality is the technique that makes that second description operational.

The conjugate packages all possible affine lower bounds. By definition,
$f^*(y)=\sup_x(\langle y,x\rangle-f(x))$. This is the largest violation that a line of slope $y$
would have if its intercept were too high. Rearranged, the same definition says
$f(x)\ge \langle y,x\rangle-f^*(y)$ for every $x$. So choosing $y$ is choosing a slope, and the
conjugate tells us the best intercept that keeps the affine function below $f$ globally. This is the
important change of viewpoint: a dual variable is not merely a multiplier; it is a proposed
supporting lower-bound direction.

For a composite objective $f(x)+g(Ax)$, one lower support for $f$ alone is not enough because it
still depends on $x$. The trick is to choose paired supports whose linear parts cancel. Use slope
$-A^T u$ for $f$ and slope $u$ for $g$:

$$
f(x)\ge \langle -A^T u,x\rangle-f^*(-A^T u),
$$

$$
g(Ax)\ge \langle u,Ax\rangle-g^*(u).
$$

After adding them, the terms $-\langle A^T u,x\rangle$ and $\langle u,Ax\rangle$ cancel exactly.
What remains is a number that is below the primal objective for every $x$:

$$
f(x)+g(Ax)\ge -f^*(-A^T u)-g^*(u).
$$

Thus each $u$ gives a certified global lower bound on the minimum. The dual problem is simply:
choose the $u$ whose certificate is strongest.

This also explains why weak duality is almost automatic. The dual value can never exceed the primal
value because it was constructed by adding two inequalities that hold pointwise for every primal
$x$. There is no magic in the gap; it is the leftover slack in those two Fenchel-Young inequalities.
For any primal-dual pair $(x,u)$, the relevant nonnegative quantities are

$$
f(x)+f^*(-A^T u)-\langle -A^T u,x\rangle
$$

and

$$
g(Ax)+g^*(u)-\langle u,Ax\rangle.
$$

Their sum is exactly the primal objective minus the dual lower bound. So the primal-dual gap is not
an abstract discrepancy between two unrelated optimization problems. It is the amount by which the
chosen supporting hyperplanes fail to touch the two convex pieces at compatible points.

At optimality those slacks vanish. Fenchel-Young is tight exactly when a slope is a subgradient:
$y\in\partial f(x)$ iff $f(x)+f^*(y)=\langle y,x\rangle$. Therefore a zero-gap pair must satisfy

$$
-A^T u\in\partial f(x), \qquad u\in\partial g(Ax).
$$

These conditions say that the lower support chosen for $f$ touches $f$ at $x$, and the lower support
chosen for $g$ touches $g$ at $Ax$, with slopes matched by the linear map so the affine terms cancel.
The minimizer and the best certificate meet at the same value.

Strong duality is the statement that, under the usual convex regularity assumptions, the best global
lower bound is not merely a bound but the exact minimum. In the Fenchel-Rockafellar theorem, a common
finite-dimensional qualification is that the relative interiors of the relevant domains meet after
the linear map, or more simply that one term is continuous at a point where the other is finite. This
condition rules out pathological separation failures. Once it holds, the supremum over certificates
attains the primal infimum, and often the dual optimizer exists as well.

The distinctive insight is therefore a mental reversal. Direct optimization asks, "Where is the
lowest point?" Fenchel duality asks, "What is the highest floor that convexity allows me to place
under the whole objective?" The conjugate is the catalog of legal floors for each slope. The dual
problem searches that catalog for the strongest floor whose linear parts cancel through $A$. The
Fenchel-Young inequality controls the entire primal-dual gap because it is exactly the rule that
keeps every proposed floor below the function.
