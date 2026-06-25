Start with the usual primal reflex: minimize an objective by looking for the point where it is
smallest. For a convex objective this is natural, but it leans on only one description of convexity.
A convex function is not only something with minimizers; it is also the upper envelope of its affine
lower supports. If I take that second description seriously, maybe I can certify the minimum from
below without ever locating the minimizer — by producing the highest constant that provably sits
under the whole objective. Let me see how far that idea can be pushed for $f(x)+g(Ax)$.

The conjugate is the natural bookkeeping device for affine lower bounds. By definition,
$f^*(y)=\sup_x(\langle y,x\rangle-f(x))$. Read the other way, this says
$f(x)\ge \langle y,x\rangle-f^*(y)$ for every $x$: choosing $y$ is choosing a slope, and the
conjugate hands back the best intercept that keeps the affine function of slope $y$ below $f$
globally. So a candidate dual variable is, concretely, a proposed supporting slope together with the
largest intercept convexity will allow it.

For the composite objective $f(x)+g(Ax)$, a single lower support for $f$ alone is no good as a
certificate because $\langle y,x\rangle$ still depends on $x$ — it is a sloped line, not a floor. I
need the $x$-dependence to disappear after I combine the pieces. The two pieces are coupled only
through $A$, which sits inside $g$'s argument, so the linear parts can only cancel if the slope I use
for $g$ is fed back through $A^T$ into the slope I use for $f$. Try slope $u$ for $g$ and slope
$-A^T u$ for $f$:

$$
f(x)\ge \langle -A^T u,x\rangle-f^*(-A^T u),
$$

$$
g(Ax)\ge \langle u,Ax\rangle-g^*(u).
$$

Add them. The first slope term is $\langle -A^T u,x\rangle=-\langle u,Ax\rangle$ by definition of the
adjoint, and the second is $+\langle u,Ax\rangle$, so they sum to zero — the $x$-dependence is gone.
What survives is a number that lies below the primal objective for every $x$:

$$
f(x)+g(Ax)\ge -f^*(-A^T u)-g^*(u).
$$

Each $u$ therefore yields a certified global lower bound, and the strongest certificate is found by
maximizing the right-hand side:

$$
d^\star=\sup_u\{-f^*(-A^T u)-g^*(u)\}.
$$

Before trusting any of this, I want to run one fully explicit case where every conjugate is known in
closed form and check that the numbers actually agree. Take $f(x)=\tfrac12 x^2$ on $\mathbb R$ (so
$f^*(y)=\tfrac12 y^2$), $A=1$, and $g(z)=|z-3|$. For the conjugate of $g$, substitute $w=z-3$:
$g^*(u)=\sup_z(uz-|z-3|)=3u+\sup_w(uw-|w|)$, and $\sup_w(uw-|w|)$ is $0$ when $|u|\le 1$ and $+\infty$
otherwise. So $g^*(u)=3u$ on $[-1,1]$ and $+\infty$ outside — a fact I should not just assert, so I
evaluate $\sup_z(uz-|z-3|)$ on a fine grid: at $u=-1,-0.5,0,0.5,1$ it returns exactly $3u$
($-3,-1.5,0,1.5,3$), and at $u=1.0001$ it already exceeds $3u$ ($3.0050$ vs $3.0003$) and at $u=1.5$
it returns $28$ rather than $4.5$. The box $|u|\le 1$ is genuinely the domain of $g^*$, not something
I imposed.

Now the primal. Minimizing $\tfrac12 x^2+|x-3|$: for $x<3$ the derivative is $x-1$, vanishing at
$x=1$, giving $p^\star=\tfrac12+2=2.5$; a bounded numerical minimizer confirms $x^\star=1.0000$ with
value $2.5$. The dual is $\sup_{|u|\le 1}(-\tfrac12 u^2-3u)$, since $-A^Tu=-u$ and
$f^*(-u)=\tfrac12 u^2$. The unconstrained maximizer of $-\tfrac12 u^2-3u$ is $u=-3$, outside the box,
so the constrained maximum sits at the boundary $u^\star=-1$, where the value is
$-\tfrac12-(-3)=2.5$. A grid search over $[-1,1]$ returns the same: $u^\star=-1$, $d^\star=2.5$.
Primal and dual both equal $2.5$, and the dual optimum lands on the boundary of the conjugate's
domain — so the box constraint was binding, not decorative. This is the first real evidence that the
construction is not merely a lower bound but can be exact.

Weak duality, $d^\star\le p^\star$, needs no example: the dual value was assembled by adding two
inequalities that hold pointwise for every primal $x$, so it can never exceed the primal value. The
more interesting question is what the gap $p^\star-d^\star$ measures. For any pair $(x,u)$ the slacks
in the two Fenchel-Young inequalities are

$$
f(x)+f^*(-A^T u)-\langle -A^T u,x\rangle \ge 0
$$

and

$$
g(Ax)+g^*(u)-\langle u,Ax\rangle \ge 0.
$$

Adding these two and using $\langle -A^Tu,x\rangle+\langle u,Ax\rangle=0$ again, the cross terms
cancel and what is left is $f(x)+g(Ax)+f^*(-A^Tu)+g^*(u)$ — i.e. the primal objective at $x$ minus
the dual objective at $u$. So the gap for a given pair is exactly the sum of two nonnegative
conjugate slacks. I check this identity numerically rather than trust the cancellation: on six random
pairs $(x,u)$ with $|u|\le 1$, the quantity $f(x)+g(Ax)+f^*(-u)+g^*(u)$ matches
$\text{slack}_f+\text{slack}_g$ to machine precision every time (e.g. $x=2.370,u=-0.460$ gives
$2.162642$ both ways), and both slacks are individually $\ge 0$. The gap is not an abstract
discrepancy between two unrelated problems; it is the amount by which the two chosen supports fail to
touch their convex pieces.

That immediately tells me when the gap is zero: both slacks must vanish, and a Fenchel-Young slack
vanishes exactly when the slope is a subgradient, $y\in\partial f(x)\iff f(x)+f^*(y)=\langle
y,x\rangle$. So a zero-gap pair must satisfy

$$
-A^T u\in\partial f(x), \qquad u\in\partial g(Ax).
$$

I can verify these on the worked example. At $x^\star=1$, $\partial f=\{x\}=\{1\}$ and
$-A^Tu^\star=-(-1)=1$, so $-A^Tu^\star\in\partial f(x^\star)$ holds. At $Ax^\star=1$, $g(z)=|z-3|$ is
differentiable with derivative $-1$ (since $1-3<0$), and $u^\star=-1$, so $u^\star\in\partial
g(Ax^\star)$ holds. Both slacks computed at $(1,-1)$ come out as exactly $0$, and $p^\star-d^\star=0$.
The minimizer and the best certificate meet at the same value, with slopes matched through $A$ so the
affine terms cancel.

The one thing the example cannot establish is that the gap is always zero. It happened to close here,
but the construction only forces $d^\star\le p^\star$; closing the gap needs more than convexity. The
slacks can vanish only if there genuinely exist compatible subgradients $-A^Tu\in\partial f(x)$ and
$u\in\partial g(Ax)$, and a separating-hyperplane pathology — where the supports of $f$ and $g$ never
align after the map $A$ — would leave a strictly positive gap. So strong duality has to be a
conditional statement. The standard sufficient condition in the Fenchel-Rockafellar theorem is a
qualification preventing exactly that pathology: in finite dimensions, that the relative interiors of
the relevant domains overlap after the linear map, or more simply that one term is continuous at a
point where the other is finite. Under such a condition the supremum over certificates attains the
primal infimum and the dual optimizer exists; I would want to confirm on a degenerate example (where
the qualification fails) that the gap can indeed be positive, but the example above is enough to show
the mechanism works when it holds.

Stepping back, the whole development is a reversal of the question. Direct optimization asks where the
lowest point is; this construction asks instead for the highest constant floor that convexity permits
beneath the entire objective. The conjugate is the catalog of legal floors indexed by slope, the dual
searches that catalog for the strongest floor whose linear parts cancel through $A$, and the
Fenchel-Young inequality is the single mechanism that both keeps every proposed floor below the
function and accounts, slack by slack, for the entire primal-dual gap.
