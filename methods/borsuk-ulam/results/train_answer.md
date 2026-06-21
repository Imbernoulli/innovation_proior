The problem is to decide whether every continuous map from an n-dimensional sphere into n-dimensional Euclidean space must identify some antipodal pair. For a continuous g : S^n -> R^n, we want to know if there is always a point x with g(x) = g(-x). In one dimension this is easy: the difference h(x) = g(x) - g(-x) changes sign under the antipodal move, so the intermediate value theorem forces a zero. That local sign argument fails in higher dimensions. A vector in R^n can rotate continuously without ever passing through zero, and a chart-based count of equations and unknowns misses the fact that x and -x are on opposite sides of the sphere rather than in the same neighborhood.

The right way to think about the question is global. If no antipodal pair had the same image, then h(x) = g(x) - g(-x) would be a continuous, nowhere-zero, odd vector field on the sphere. Normalizing it would give an odd map F : S^n -> S^{n-1}, meaning F(-x) = -F(x). So the theorem is really the statement that such a symmetry-respecting collapse of S^n onto a lower-dimensional sphere cannot exist. The proof finds the contradiction not by solving equations in coordinates, but by measuring what the hypothetical map does to the top homology of an equator.

The method is the Borsuk-Ulam theorem. It says that for every continuous map g : S^n -> R^n there exists x in S^n such that g(x) = g(-x), or equivalently that there is no continuous odd map from S^n to S^{n-1}. The proof has two main pieces. First, one shows that any odd continuous self-map of a sphere has odd degree. The argument uses the double covering S^m -> RP^m and the transfer exact sequence in homology with Z/2 coefficients. Because the map is odd, it descends to the quotient projective space, and naturality carries a nonzero map in dimension zero inductively up to the top homology of the sphere. On top homology the induced map is multiplication by the degree modulo 2, so the degree must be odd.

Second, one assumes the theorem is false and derives a contradiction. If g separates every antipodal pair, then h(x) = g(x) - g(-x) is nonzero and odd. Dividing by its norm produces an odd map F : S^n -> S^{n-1}. Restrict F to an equatorial S^{n-1} inside S^n. The restriction u : S^{n-1} -> S^{n-1} is still odd, so by the first step its degree is odd. But the equator bounds a hemisphere D^n, and F itself provides an extension of u across that disk. Any map from S^{n-1} to itself that extends over D^n must have degree zero, because the induced map on top homology factors through H_{n-1}(D^n) = 0. The same map cannot have both odd degree and degree zero, so the original assumption was impossible. Therefore some antipodal pair must be mapped to the same point.

The theorem is sharp in the target dimension. Mapping S^n into R^{n+1} by the inclusion x |-> x separates every antipodal pair, so the obstruction is exactly the one-dimensional drop from the ambient space to the sphere's own dimension. The key insight is not a clever fixed-point trick but the choice of an invariant, degree modulo 2, that can hear the antipodal symmetry.

```python
import numpy as np
from scipy.optimize import minimize_scalar, minimize

# Verify Borsuk-Ulam numerically for low dimensions.

def random_map_s1(t):
    """A continuous map S^1 -> R as a trigonometric polynomial."""
    return (0.7 * np.sin(t) + 0.3 * np.cos(2 * t) +
            0.2 * np.sin(3 * t) + 0.4)

def find_coincidence_s1(g, n_samples=10_000):
    """Find t where g(t) - g(t + pi) ~ 0 using bisection on sign changes."""
    ts = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    diffs = g(ts) - g((ts + np.pi) % (2 * np.pi))
    # Look for a sign change over consecutive samples.
    for i in range(len(ts) - 1):
        if diffs[i] == 0:
            return ts[i]
        if diffs[i] * diffs[i + 1] < 0:
            a, b = ts[i], ts[i + 1]
            fa, fb = diffs[i], diffs[i + 1]
            for _ in range(50):  # bisection
                m = 0.5 * (a + b)
                fm = g(m) - g((m + np.pi) % (2 * np.pi))
                if fa * fm <= 0:
                    b, fb = m, fm
                else:
                    a, fa = m, fm
            return 0.5 * (a + b)
    return None

def random_map_s2(x):
    """A continuous map S^2 -> R^2 using spherical harmonics."""
    # x is a unit vector in R^3.
    y1 = (x[0] * x[1] + 0.5 * x[2]**2 +
          0.3 * x[0] + 0.2 * x[1])
    y2 = (x[1] * x[2] - 0.4 * x[0] * x[2] +
          0.6 * x[2] - 0.1 * x[0])
    return np.array([y1, y2])

def find_coincidence_s2(g, trials=20):
    """Minimize |g(x) - g(-x)|^2 over S^2."""
    def from_spherical(theta_phi):
        t, p = theta_phi
        return np.array([np.sin(t) * np.cos(p),
                         np.sin(t) * np.sin(p),
                         np.cos(t)])
    def obj(theta_phi):
        x = from_spherical(theta_phi)
        d = g(x) - g(-x)
        return float(d @ d)

    best_val, best_x = np.inf, None
    rng = np.random.default_rng(0)
    for _ in range(trials):
        init = rng.uniform([0, 0], [np.pi, 2 * np.pi])
        res = minimize(obj, init, method="Powell",
                       options={"maxiter": 1000})
        if res.fun < best_val:
            best_val = res.fun
            best_x = from_spherical(res.x)
    return best_x, best_val

if __name__ == "__main__":
    # n = 1
    t0 = find_coincidence_s1(random_map_s1)
    print("S^1 coincidence point t:", t0)
    if t0 is not None:
        print("  g(t) =", random_map_s1(t0),
              "g(-t) =", random_map_s1((t0 + np.pi) % (2 * np.pi)))

    # n = 2
    x0, err = find_coincidence_s2(random_map_s2)
    print("\nS^2 coincidence point x:", x0)
    print("  residual |g(x)-g(-x)|^2:", err)
    print("  g(x)  =", random_map_s2(x0))
    print("  g(-x) =", random_map_s2(-x0))
```
