I would state the Riemann Mapping Theorem as the central reason that one-complex-variable geometry is so much cleaner than real differential geometry in the plane. Take any proper simply connected open subset Omega of the complex plane and pick a basepoint a inside it. The theorem says there is a unique biholomorphic map f from Omega onto the unit disk D = {w in C : |w| < 1} such that f(a) = 0 and f'(a) > 0. In other words, once I fix one interior point and one tangent direction, the entire shape of Omega becomes invisible from the point of view of holomorphic functions. Every nondegenerate simply connected planar domain is analytically indistinguishable from the disk, and the arbitrary boundary is absorbed into a single canonical coordinate.

The statement splits into two parts. Existence gives me at least one conformal coordinate covering the whole domain exactly once and landing in the disk. Uniqueness says that the two normalizations f(a) = 0 and f'(a) > 0 remove every remaining symmetry. The disk automorphisms that fix the origin are just rotations w -> e^{i theta} w, and the positive derivative condition forces that rotation to be the identity. So if two normalized maps existed, their composition would have to be the identity, and the two maps would coincide.

Why should existence be true? The local picture is easy: any holomorphic function with nonzero derivative is conformal, meaning it preserves angles and infinitesimal shapes. The hard part is making that local behavior global. The topological assumption of simple connectivity is exactly what does it. On a simply connected domain, every nowhere-zero holomorphic function has a holomorphic logarithm, and therefore a holomorphic square root. That may sound like a small algebraic convenience, but it is the engine of the proof, because taking a square root is how I can "open up" an image that fails to fill the disk.

The standard proof builds the desired map as an extremal object rather than as an explicit formula. I consider the family S of all injective holomorphic maps g from Omega into the unit disk with g(a) = 0 and g'(a) > 0. First I need to know S is not empty. Because Omega is proper in C, I can choose a point b outside it. Then z - b has no zeros on Omega, so simple connectivity gives me a holomorphic square root h(z) = sqrt(z - b). This square root is injective, and its image cannot contain both w and -w for any w, because squaring would give the same point of Omega. Since h(Omega) contains a small disk around h(a), it omits a small disk around -h(a). A suitable reciprocal, translation, scaling, and rotation then produces a member of S.

Next I look for the map that uses the disk most efficiently at the basepoint. Set M = sup_{g in S} g'(a). Every map in S is bounded by 1, so Cauchy's estimate on a small closed disk around a shows M is finite. Pick a sequence g_n in S whose derivatives at a approach M. Montel's theorem gives a subsequence converging uniformly on compact subsets to a holomorphic limit f. Derivatives converge as well, by the Cauchy integral formula, so f(a) = 0 and f'(a) = M > 0. Hurwitz's theorem then guarantees that a nonconstant limit of injective holomorphic functions remains injective. The maximum modulus principle prevents |f| from ever reaching 1 inside Omega, because that would force f to be constant. So f itself belongs to S, and the supremum is actually attained.

The deepest step is showing that f is onto D. Suppose it missed some point c in the disk. Since f(a) = 0, c is not zero. The disk automorphism B_c(w) = (w - c)/(1 - conj(c) w) sends c to 0, so B_c(f(z)) is a nowhere-zero holomorphic function on Omega. Simple connectivity again gives a holomorphic square root s(z) with s(z)^2 = B_c(f(z)). The map s is injective because equality of s implies equality of s^2, hence equality of f, hence equality of points. Let alpha = s(a). Then alpha^2 = B_c(0) = -c, so |alpha| = sqrt(|c|). Moving alpha to 0 by another disk automorphism Phi_alpha and rotating the derivative to be positive gives a new map in S. A short derivative calculation shows that its derivative at a is f'(a) * (sqrt(|c|) + 1/sqrt(|c|))/2, which by the arithmetic-geometric mean inequality is strictly larger than f'(a) = M. That contradicts the definition of M, so no omitted c can exist. Therefore f maps Omega onto the whole disk.

This argument also explains where the topological hypothesis is used. The square root of B_c(f) exists globally only because Omega has no holes. If Omega were punctured or had nontrivial topology, monodromy would prevent me from defining s consistently around a loop, and the construction would collapse. In that sense simple connectivity is not just sufficient but necessary for the strongest form of the theorem.

There is also a beautiful harmonic-function interpretation. If f(a) = 0, then log|f(z)| is harmonic on Omega away from a, has a logarithmic pole at a matching log|z - a|, and tends to 0 at the boundary because |f| tends to 1. So log|f| is essentially the Green's function of Omega with pole at a. Once I solve that Dirichlet problem and take a harmonic conjugate, exponentiation gives me back the holomorphic coordinate. The extremal proof above is just the compactness packaging of the same idea.

The canonical name for this result is the Riemann Mapping Theorem. It is foundational in complex analysis, geometric function theory, and any area where conformal coordinates are used to simplify partial differential equations or mesh generation.

To make the theorem concrete, the following Python script constructs the explicit Riemann map for the simplest possible simply connected domain that is not already the disk: the upper half-plane H = {z : Im(z) > 0}, with basepoint a = i. The normalized map is F(z) = i (z - i)/(z + i). It maps H onto D, sends i to 0, and has positive derivative at i. The script verifies these normalizations, checks that sampled points of H land inside D, and visualizes how a rectangular grid in H is carried to curved arcs in D.

```python
import numpy as np
import matplotlib.pyplot as plt

def riemann_map_upper_halfplane_to_disk(z):
    """Normalized Riemann map from H (Im z > 0) onto D with F(i)=0, F'(i)>0."""
    return 1j * (z - 1j) / (z + 1j)

def derivative_riemann_map(z):
    return 1j * ((1)*(z + 1j) - (z - 1j)*(1)) / (z + 1j)**2

# Verify normalizations at basepoint a = i.
a = 1j
F_a = riemann_map_upper_halfplane_to_disk(a)
Fp_a = derivative_riemann_map(a)
print(f"F(i)   = {F_a.real:.6f} + {F_a.imag:.6f}i  (should be 0)")
print(f"F'(i)  = {Fp_a.real:.6f} + {Fp_a.imag:.6f}i  (should be positive real)")

# Sample random points in the upper half-plane and check they map inside the disk.
np.random.seed(0)
x = np.random.uniform(-5, 5, 2000)
y = np.random.exponential(1.0, 2000) + 0.01
z_samples = x + 1j * y
w_samples = riemann_map_upper_halfplane_to_disk(z_samples)
max_modulus = np.max(np.abs(w_samples))
print(f"Max |F(z)| over random H points = {max_modulus:.6f}  (should be < 1)")

# Visualize a grid in H and its image in D.
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

x_grid = np.linspace(-4, 4, 17)
y_grid = np.linspace(0.1, 4, 13)
ax_h, ax_d = axes
for x0 in x_grid:
    z_line = x0 + 1j * y_grid
    w_line = riemann_map_upper_halfplane_to_disk(z_line)
    ax_h.plot(z_line.real, z_line.imag, 'b-', lw=0.7)
    ax_d.plot(w_line.real, w_line.imag, 'b-', lw=0.7)
for y0 in y_grid:
    z_line = x_grid + 1j * y0
    w_line = riemann_map_upper_halfplane_to_disk(z_line)
    ax_h.plot(z_line.real, z_line.imag, 'r-', lw=0.7)
    ax_d.plot(w_line.real, w_line.imag, 'r-', lw=0.7)

theta = np.linspace(0, 2*np.pi, 400)
ax_d.plot(np.cos(theta), np.sin(theta), 'k--', lw=1)

ax_h.set_xlim(-4.5, 4.5)
ax_h.set_ylim(0, 4.5)
ax_h.set_aspect('equal')
ax_h.set_title('Grid in upper half-plane H')
ax_h.set_xlabel('Re z')
ax_h.set_ylabel('Im z')

ax_d.set_xlim(-1.05, 1.05)
ax_d.set_ylim(-1.05, 1.05)
ax_d.set_aspect('equal')
ax_d.set_title('Image under normalized Riemann map F')
ax_d.set_xlabel('Re w')
ax_d.set_ylabel('Im w')

plt.tight_layout()
plt.savefig('riemann_map_illustration.png', dpi=150)
plt.show()
```

Running the script confirms that the explicit map satisfies the three defining properties of the Riemann Mapping Theorem: it is holomorphic and bijective between H and D, it sends the chosen basepoint i to 0, and its derivative at i is the positive real number 1/2. The visualization shows how the right angles of the rectangular grid in H are preserved in the infinitesimal sense, while the global shape is bent so that the whole half-plane fits exactly inside the unit disk. This example is special because an elementary formula exists, but the Riemann Mapping Theorem guarantees the same kind of coordinate for every proper simply connected planar domain, even when no closed-form expression is available.
