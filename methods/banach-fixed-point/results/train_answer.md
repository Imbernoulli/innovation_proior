When an equation is written in the self-referential form x = T(x) and the unknown is a function, sequence, or element of an abstract normed space, direct algebraic solution is usually impossible. Topological fixed-point theorems can certify that at least one solution exists, but they say nothing about how to compute it, give no uniqueness guarantee, and do not ensure that repeatedly applying T will converge. Picard iteration, x_{n+1} = T(x_n), is the natural analytic replacement, yet without a uniform quantitative shrinkage condition the orbit can oscillate, drift, or approach a point outside the space. Non-expansive maps, which merely satisfy d(Tx, Ty) <= d(x, y), preserve too much distance: equal or slowly decreasing gaps can last forever. What is missing is a single global rate at which every pair of points is pulled together, combined with completeness so that the resulting Cauchy behavior actually lands inside the space.

The method that supplies this is the Banach fixed-point theorem. In a nonempty complete metric space (X, d), suppose a self-map T: X -> X satisfies the contraction condition d(Tx, Ty) <= q d(x, y) for all x, y in X, where 0 <= q < 1. Then T has exactly one fixed point x*, and for every starting point x_0 the Picard iterates x_{n+1} = T(x_n) converge to x*. The proof is driven by the same inequality. Because each jump shrinks by at least a factor of q, we have d(x_{n+1}, x_n) <= q^n d(x_1, x_0). Summing a geometric tail via the triangle inequality shows that for m > n, d(x_m, x_n) <= q^n d(x_1, x_0) / (1 - q), so the orbit is Cauchy. Completeness turns this into an actual limit x*. The contraction inequality is strong enough to pass the limit through T: d(Tx*, x*) <= q d(x*, x_n) + d(x_{n+1}, x*) -> 0, hence Tx* = x*. Uniqueness follows immediately, because if u and v are both fixed then d(u, v) = d(Tu, Tv) <= q d(u, v), and q < 1 forces d(u, v) = 0. The theorem therefore turns a quantitative contractive hypothesis into existence, uniqueness, convergence from any start, and explicit error control.

The error estimates are as useful as the existence statement. The a priori bound d(x*, x_n) <= q^n d(x_1, x_0) / (1 - q) tells us before we begin how many iterations are needed for a given accuracy. The a posteriori bound d(x*, x_{n+1}) <= q d(x_{n+1}, x_n) / (1 - q) lets us stop as soon as the most recent observed jump is small. These two bounds make the theorem practical: it is not merely an existence certificate but an approximation scheme with a geometric convergence rate governed by q.

The theorem also explains exactly where the hypotheses bite. Completeness is needed only once, to turn the Cauchy orbit into a limit; without it, the iterates can march toward a missing boundary point. The strict inequality q < 1 is needed for the geometric summation and for the uniqueness argument. When q = 1 the proof collapses: non-expansive maps can have no fixed points, many fixed points, or fixed points that iteration never reaches. The setting is naturally metric rather than linear, since the argument uses only distances, the triangle inequality, and Cauchy limits. This is why the same theorem applies equally well to solutions of integral equations in function spaces and to simple scalar equations on the real line.

```python
def banach_fixed_point(T, x0, q, tol=1e-10, max_iter=10000):
    """
    Approximate the unique fixed point of a contraction T.

    Parameters
    ----------
    T : callable
        A contraction mapping.  Should satisfy d(T(x), T(y)) <= q * d(x, y).
    x0 : object supporting subtraction and a norm
        Initial guess.
    q : float
        Contraction constant with 0 <= q < 1.
    tol : float
        Desired accuracy (uses the a posteriori error estimate).
    max_iter : int
        Safety bound on the number of iterations.

    Returns
    -------
    x : object
        Approximate fixed point.
    info : dict
        Contains the number of iterations, the final jump, and the
        estimated error bound q * jump / (1 - q).
    """
    if not 0 <= q < 1:
        raise ValueError("Contraction constant q must satisfy 0 <= q < 1")

    x = x0
    for n in range(max_iter):
        x_next = T(x)
        jump = abs(x_next - x)
        error_bound = q * jump / (1 - q) if q > 0 else 0.0
        if error_bound < tol:
            return x_next, {
                "iterations": n + 1,
                "jump": jump,
                "error_bound": error_bound,
            }
        x = x_next

    raise RuntimeError(f"Failed to converge within {max_iter} iterations")


# Example: solve x = cos(x) on [0, 1].  The derivative of cos has modulus
# at most sin(1) < 1 on this interval, so cos is a contraction there.
import math
q = math.sin(1.0)
x_star, info = banach_fixed_point(math.cos, x0=0.5, q=q, tol=1e-12)
print(f"fixed point: {x_star:.12f}")
print(f"cos(fixed point): {math.cos(x_star):.12f}")
print(f"iterations: {info['iterations']}, error bound: {info['error_bound']:.2e}")
```
