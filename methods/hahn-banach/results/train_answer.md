The problem is how to extend a linear functional defined on a small subspace to the entire vector space without losing the bound that controls it. A basis extension gives values everywhere, but the arbitrary choices destroy continuity and domination. Pure boundedness estimates describe only where the functional is already defined, and finite-dimensional separation arguments collapse once the space is infinite-dimensional and has no manageable enumeration. What is missing is a way to assign each new value so that the domination inequality remains true for every future vector.

The Hahn-Banach theorem is that extension principle. In its analytic form it says: if E is a real vector space, p:E->R is sublinear, M is a subspace, and f:M->R is linear with f(x) <= p(x) on M, then there exists a linear functional F:E->R extending f such that F(x) <= p(x) on all of E. The proof does not build a global basis first. It grows the domain one direction at a time, using sublinearity to guarantee that each new value can be chosen compatibly with all old constraints, and then uses a maximality argument to show that the growth cannot stop before reaching E.

The one-dimensional step is the core of the proof. Take a vector z outside the current domain M. Any linear extension to M+Rz is determined by a single scalar c = F(z), because F(x+tz) must equal f(x)+tc. The domination condition F <= p becomes two families of inequalities. For positive t, dividing by t gives c <= p(y+z)-f(y) for every y in M. For negative t, it gives c >= f(x)-p(x-z) for every x in M. So c must lie in the interval between the supremum of the lower bounds and the infimum of the upper bounds. Sublinearity makes this interval nonempty: for any x,y in M,

f(x)+f(y) = f(x+y) <= p(x+y) = p((x-z)+(y+z)) <= p(x-z)+p(y+z),

which rearranges to f(x)-p(x-z) <= p(y+z)-f(y). Every lower bound is below every upper bound, so a suitable c exists. This is the only place where the geometry of the problem is used.

One direction is not enough when the space is large. The proof therefore collects every possible dominated extension into a partially ordered set, ordered by extension of domain. A chain of such extensions has a natural upper bound: take the union of the domains and define the functional by the unique member of the chain that contains each point. This union is still linear and still dominated by p. Zorn's lemma gives a maximal element. If that maximal domain were not all of E, we could pick a missing vector and apply the one-dimensional step, contradicting maximality. Hence the maximal extension is defined on all of E.

When p is a seminorm and the original bound is |f(x)| <= p(x), the same theorem applied to -x gives -F(x) = F(-x) <= p(-x) = p(x), so |F(x)| <= p(x). In a normed space this means a bounded linear functional on a subspace extends to the whole space with exactly the same operator norm. The theorem also has a geometric face: it produces separating hyperplanes. If M is a closed subspace and z is a point outside it, define f(m+tz) = t dist(z,M) on M+Rz. This is bounded by the norm, and its Hahn-Banach extension is a continuous linear functional that vanishes on M and takes the value dist(z,M) at z, whose kernel is a closed hyperplane separating z from M.

```python
import numpy as np

def one_step_extension(values, basis, p, z, samples=21):
    """Choose c = F(z) so that F(x+tz) <= p(x+tz) holds on a sampled grid."""
    if len(basis) == 0:
        return 0.5 * (-p(-z) + p(z))
    grid = np.linspace(-2, 2, samples)
    mesh = np.array(np.meshgrid(*[grid] * len(basis), indexing='ij'))
    coeffs = mesh.T.reshape(-1, len(basis))
    lowers, uppers = [], []
    key = lambda v: tuple(np.round(v, 12))
    for a in coeffs:
        x = sum(c * b for c, b in zip(a, basis))
        fx = sum(c * values[key(b)] for c, b in zip(a, basis))
        lowers.append(fx - p(x - z))
        uppers.append(p(x + z) - fx)
    lower, upper = max(lowers), min(uppers)
    if lower > upper + 1e-6:
        raise ValueError("domination interval empty")
    return 0.5 * (lower + upper)

def hahn_banach(M_basis, full_basis, f_on_M, p):
    """Extend a linear functional from span(M_basis) to span(full_basis)."""
    M_basis = [np.array(v, dtype=float) for v in M_basis]
    full_basis = [np.array(v, dtype=float) for v in full_basis]
    key = lambda v: tuple(np.round(v, 12))

    current = list(M_basis)
    values = {key(v): float(f_on_M(v)) for v in M_basis}

    missing = [v for v in full_basis
               if not any(np.allclose(v, m) for m in M_basis)]
    current.extend(missing)

    for i in range(len(M_basis), len(current)):
        z = current[i]
        c = one_step_extension(values, current[:i], p, z)
        values[key(z)] = c

    def F(v):
        v = np.array(v, dtype=float)
        coeffs = np.linalg.lstsq(np.column_stack(current), v, rcond=None)[0]
        return sum(c * values[key(b)] for c, b in zip(coeffs, current))

    return F

# Example: extend f(x,0) = 0.5 x on the x-axis to R^2, dominated by the l1 norm.
p = lambda v: np.sum(np.abs(v))
M_basis = [(1.0, 0.0)]
full_basis = [(1.0, 0.0), (0.0, 1.0)]
f_on_M = lambda v: 0.5 * v[0]

F = hahn_banach(M_basis, full_basis, f_on_M, p)

np.random.seed(0)
for _ in range(100):
    v = np.random.randn(2)
    assert F(v) <= p(v) + 1e-6, "domination violated"
print("Hahn-Banach extension verified on random samples.")
```
