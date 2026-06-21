I would propose the canonical method name Stone-Weierstrass theorem for this result. The classical Weierstrass approximation theorem says that every continuous real-valued function on a closed interval can be uniformly approximated by polynomials, but it leaves the impression that the proof must be tied to the interval itself, to monomials, Bernstein bases, or the order structure of the real line. The Stone-Weierstrass theorem is the abstraction that explains why the interval result is true at a deeper level: the only thing a polynomial algebra is really using is that it is an algebra of continuous functions, it contains constants, and it can tell points apart. Once those three properties hold on any compact Hausdorff space, the family is already uniformly dense in the space of all continuous real-valued functions.

Let me state the theorem carefully. Suppose X is a compact Hausdorff space and C(X, R) is the Banach algebra of continuous real-valued functions on X equipped with the supremum norm. Let A be a subset of C(X, R) that is a real subalgebra under pointwise addition, scalar multiplication, and multiplication. Assume A contains the constant functions and separates points, meaning that for any two distinct points x and y in X there is some function h in A with h(x) different from h(y). Then A is dense in C(X, R). In other words, for every continuous target function f and every positive tolerance epsilon, there is an element g of A such that the absolute difference between f and g is smaller than epsilon at every point of X simultaneously. There is also a complex version: a complex subalgebra of C(X, C) is dense if it contains constants, separates points, and is closed under complex conjugation. The extra conjugation condition is essential, because without it complex algebras can carry hidden analytic rigidity, as seen in the algebra of functions continuous on the closed disk and holomorphic in its interior, which separates points but is proper and closed.

The proof strategy is not to build explicit approximating formulas for f. Instead, it looks at the uniform closure B of A inside C(X, R). Because uniform limits preserve the algebraic operations on bounded functions, B is itself a closed real algebra containing constants and separating points. The goal becomes showing that any closed algebra with these properties is already all of C(X, R). The key move is to show that B is closed under taking absolute values. Pick any u in B. Since X is compact, the image u(X) lies inside some interval [-M, M]. The classical Weierstrass theorem, applied now in one real variable on that interval, gives real polynomials p_n that converge uniformly to the absolute value function t goes to absolute value of t. Because B is an algebra containing constants, each composition p_n of u lies in B, and these compositions converge uniformly to absolute value of u. Since B is closed, absolute value of u is in B. This is the only place where the interval theorem is used, and it is used as a functional-calculus device rather than as a direct approximation tool.

Once B contains absolute values, it automatically becomes a lattice. The pointwise maximum and minimum of any two functions u and v in B can be written using only addition and absolute value: max(u, v) equals (u + v + absolute value of (u - v)) divided by 2, and min(u, v) equals (u + v - absolute value of (u - v)) divided by 2. So B is closed under max and min. That is the nonlinear operation that makes a coordinate-free proof possible: linear combinations tilt and scale functions, but max and min let us paste together upper and lower envelopes.

Next comes two-point interpolation. Fix a target function f in C(X, R) and two points x and y in X. If x equals y, the constant function f(x) matches f at that point and lies in B because B contains constants. If x and y are different, point separation gives a function h in B with h(x) not equal to h(y). The affine expression g_{xy}(z) = f(x) + (f(y) - f(x)) * (h(z) - h(x)) / (h(y) - h(x)) then lies in B and satisfies g_{xy}(x) = f(x) and g_{xy}(y) = f(y). So B can match the target at any pair of points. The proof now assembles these local witnesses using the lattice structure.

Fix a point x. For each other point y, the function g_{xy} agrees with f at y, so by continuity there is a neighborhood V_y of y on which g_{xy} stays at least f minus a small slack. These neighborhoods cover X, and compactness gives a finite subcover V_{y_1} through V_{y_m}. Taking the pointwise maximum G_x = max(g_{xy_1}, ..., g_{xy_m}) yields a function in B that is everywhere at least f minus the slack and that satisfies G_x(x) = f(x). Continuity then gives a neighborhood U_x of x on which G_x is at most f plus the slack. The neighborhoods U_x cover X, so compactness gives a finite subcover U_{x_1} through U_{x_k}. Taking the pointwise minimum G = min(G_{x_1}, ..., G_{x_k}) produces a function in B that is everywhere at least f minus the slack and, at every point z, at most f(z) plus the slack because z lies in some U_{x_i}. Therefore the uniform distance between G and f is smaller than epsilon. Since epsilon was arbitrary and B is closed, f lies in B. The target was arbitrary, so B equals C(X, R), and A is dense.

The theorem explains why point separation is not merely a necessary condition but the only obstruction in the real compact setting. Any family that cannot separate two points also cannot approximate a continuous function that takes different values there, so separation is necessary. The Stone-Weierstrass theorem shows that once separation is satisfied and the family is an algebra containing constants, no further obstruction remains. This perspective also clarifies the complex case: without conjugation, separation is not enough because complex algebras can be analytically rigid. When conjugation is included, the real and imaginary parts of every function stay in the algebra, the real theorem applies to the self-adjoint real-valued part, and real and imaginary parts of any continuous complex function can be approximated separately and recombined.

The following Python script gives a small concrete illustration rather than a formal proof. It approximates the continuous but non-polynomial function f(x) = absolute value of (x - 0.5) on the unit interval by polynomials, showing that the uniform error shrinks as the polynomial degree grows. It also demonstrates that an algebra that fails to separate points, here the algebra of even polynomials around x = 0.5, cannot approximate the simple linear function g(x) = x no matter how high the degree, because every even polynomial assigns the same value to 0.5 - d and 0.5 + d.

```python
import numpy as np


def chebyshev_nodes(n, a=0.0, b=1.0):
    """Return n Chebyshev nodes in the interval [a, b]."""
    k = np.arange(n)
    t = np.cos((2 * k + 1) * np.pi / (2 * n))
    return 0.5 * (b - a) * t + 0.5 * (a + b)


def lagrange_interpolant(xs, ys):
    """Barycentric Lagrange interpolant for nodes xs and values ys."""
    n = len(xs)
    w = np.ones(n)
    w[0] = 0.5
    w[-1] = 0.5
    w *= (-1.0) ** np.arange(n)

    def p(t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        out = np.zeros_like(t)
        for i, ti in enumerate(t):
            exact = np.isclose(xs, ti)
            if np.any(exact):
                out[i] = ys[np.argmax(exact)]
                continue
            num = np.sum(w * ys / (ti - xs))
            den = np.sum(w / (ti - xs))
            out[i] = num / den
        return out

    return p


def uniform_error(f, g, a=0.0, b=1.0, n=2001):
    xs = np.linspace(a, b, n)
    return np.max(np.abs(f(xs) - g(xs)))


if __name__ == "__main__":
    a, b = 0.0, 1.0
    f = lambda x: np.abs(x - 0.5)
    print("Stone-Weierstrass illustration: polynomial approximation of |x-0.5| on [0,1]")
    for n_nodes in [5, 9, 17, 33, 65]:
        xs = chebyshev_nodes(n_nodes, a, b)
        ys = f(xs)
        p = lagrange_interpolant(xs, ys)
        err = uniform_error(f, p, a, b)
        print(f"  degree {n_nodes - 1}: uniform error = {err:.2e}")

    print("\nNon-separating algebra: even polynomials around 0.5 cannot approximate x")
    g = lambda x: x
    for n_nodes in [5, 9, 17, 33, 65]:
        xs = chebyshev_nodes(n_nodes, a, b)
        t = (xs - 0.5) ** 2
        deg = (n_nodes - 1) // 2
        coeffs = np.polyfit(t, g(xs), deg)
        q = np.poly1d(coeffs)
        xs_test = np.linspace(a, b, 2001)
        err = np.max(np.abs(g(xs_test) - q((xs_test - 0.5) ** 2)))
        print(f"  even-poly degree {2*deg}: uniform error = {err:.4f}")
```

The Stone-Weierstrass theorem is valuable because it turns approximation from a hunt for special formulas into a structural question about closed algebras. In machine learning it helps us understand when a parametric family of functions is rich enough to approximate arbitrary continuous functions on a compact domain: if the family is closed under the algebraic operations, contains constants, and can distinguish inputs, then uniform approximation is guaranteed. The result is not constructive in the sense of giving an explicit approximant for every target, but it is powerfully diagnostic. It tells us that density follows once the right algebraic and separation invariants are in place, and it identifies point separation as the only remaining obstacle in the real compact setting.
