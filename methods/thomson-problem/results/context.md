# Context: minimizing Coulomb/Riesz energy of points on a sphere

## Research question

Fix a dimension n and an integer N. Among all ways of placing N points
x_1, ..., x_N on the unit sphere S^{n-1}, which arrangement minimizes the total
pairwise potential energy

    E(x_1, ..., x_N) = sum_{i != j} f(|x_i - x_j|^2),

where f is a repulsive radial potential written as a function of squared distance?
The case that started everything is identically charged particles confined to a
sphere repelling by Coulomb's law: in R^3 the electrostatic potential is
f(r) = 1/r^{1/2} as a function of squared distance r = |x-y|^2 (i.e. 1/|x-y|), and
in R^n the natural harmonic generalization is f(r) = 1/r^{n/2-1}. More generally one
cares about all inverse power laws f(r) = 1/r^s (Riesz s-energy), Gaussians
e^{-cr}, and other decreasing convex potentials.

The numerical question — find low-energy configurations for a specific N and a
specific f — is comparatively easy to attack and notoriously hard to settle. The
energy landscape has a number of local minima that grows roughly exponentially in N,
the global minimizer for N = 8 or N = 20 is not the corresponding Platonic solid,
and even when a beautiful symmetric configuration is the apparent winner there is no
configuration-space argument that proves it. What matters, and what is hard, is a
*proof*: a lower bound on the energy of every admissible configuration that is met
with equality by a specific arrangement. Two things would make such a proof
especially valuable. First, robustness across potentials: a single argument that
holds for a whole family of f at once, rather than one f at a time. Second, it should
explain *why* the exceptional configurations — the regular simplex and cross
polytope, the icosahedron, the minimal vectors of the E_8 and Leech lattices — keep
appearing as optima across many different energies.

## Background

**Spherical codes and designs.** A configuration on S^{n-1} can be summarized by the
set of inner products t = <x,y> that occur between distinct points (since
|x-y|^2 = 2 - 2t when |x| = |y| = 1, energy is a function of these inner products
alone). Two combinatorial notions organize the special configurations. An
(n, N, t)-*spherical code* is a set of N points on S^{n-1} with no two distinct
points having inner product greater than t; maximizing the minimal angle is the
packing/Tammes problem. A *spherical M-design* is a finite set on which every
polynomial of degree at most M has the same average as over the whole sphere. The
most symmetric configurations are simultaneously excellent codes and high-strength
designs.

**Spherical harmonics and positive-definite kernels.** Under the rotation group
O(n), the space L^2(S^{n-1}) decomposes as an orthogonal direct sum of
finite-dimensional spaces V_0, V_1, V_2, ... where V_l is the spherical harmonics of
degree l. Each V_l has a reproducing kernel, and because O(n) acts
distance-transitively, that kernel depends only on the inner product:
K_l(x,y) = C_l(<x,y>) for a degree-l polynomial C_l. As <ev_{l,x}, ev_{l,y}>, this
kernel is *positive-definite*: for any finite point set,
sum_{x,y} C_l(<x,y>) = |sum_x ev_{l,x}|^2 >= 0. Pinning down C_l by orthogonality,
the C_l are exactly the *ultraspherical (Gegenbauer) polynomials* C_l^{lambda} with
lambda = n/2 - 1, orthogonal with respect to (1 - t^2)^{(n-3)/2} dt on [-1,1],
normalized by C_0 = 1, C_1 = 2*lambda*t, with three-term recurrence
i C_i = 2(i + lambda - 1) t C_{i-1} - (i + 2*lambda - 2) C_{i-2}. Schoenberg's
theorem (Schoenberg 1942) states that every continuous distance-only
positive-definite kernel on the sphere is a nonnegative combination of the C_l. A
companion fact (Delsarte, Goethals, Seidel 1977, Thm 5.5): a set is a
spherical M-design if and only if sum_{x,y} C_i(<x,y>) = 0 for 1 <= i <= M.

**Completely monotonic potentials.** A C^infinity function f on an interval is
*completely monotonic* if (-1)^k f^{(k)} >= 0 for all k (so f >= 0, f' <= 0,
f'' >= 0, ...): nonnegative, decreasing, convex, and so on for all derivatives. All
inverse power laws 1/r^s (s > 0) and e^{-cr} are completely monotonic; the harmonic
Coulomb law is a member. By the Bernstein–Widder theorem, completely monotonic
functions are exactly Laplace transforms of nonnegative measures; on a compact
interval they are uniform limits of nonnegative combinations of (4 - r)^k. The
companion notion on inner products is *absolute monotonicity*: a(t) = f(2 - 2t) has
all derivatives nonnegative (a^{(k)} >= 0) exactly when f is completely monotonic.
This class is the natural setting because it is broad (covers Coulomb, all Riesz
powers, Gaussians) yet retains strong structure — nonnegative high derivatives.

**The exceptional configurations.** Empirically and through earlier rigorous work, a
short list of arrangements minimizes specific energies: the regular simplex (N <= n+1
points), the cross polytope (2n points), the regular icosahedron (n = 3, N = 12), the
minimal vectors of the E_8 root lattice (n = 8, N = 240) and of the Leech lattice
(n = 24, N = 196560), and several kissing configurations derived from them. Each is a
high-strength spherical design with only a few distinct inner products. The vertices
of the regular 600-cell (n = 4, N = 120) are an 11-design with eight inner products and
behave almost as well. These same objects recur across number theory, lattice theory,
and analysis, and the recurring question is whether their optimality is one
phenomenon or many.

## Baselines

**Numerical relaxation (steepest descent, simulated annealing).** Treat E as a
function on (S^{n-1})^N and minimize it directly: from a random start, move points
downhill along the sphere-tangential gradient (Claxton–Benson 1966, Erber–Hockney
1991), optionally with simulated annealing or basin hopping to escape local minima
(Altschuler et al. 1994). This finds excellent candidate configurations and the best
known energies for hundreds of points. Its limitation is fundamental: it returns a
*configuration*, never a *proof*. With exponentially many local minima it can never
certify that the reported configuration is the global optimum, and it must be redone
for each N and each f separately. It is a way to guess the answer, not to establish
it.

**Yudin's spherical-harmonic bound.** Yudin (1992) gave a rigorous lower bound for
the harmonic energy of n+1 or 2n points on S^{n-1}, proving optimality of the simplex
and the cross polytope, by expanding the potential in spherical harmonics and using
their positivity. Kolushov and Yudin (1997) extended this to show the E_8 minimal
vectors are the unique minimum for (n,N) = (8,240), and Andreev (1996, 1997) handled
the Leech vectors (24, 196560) and the icosahedron (3, 12). The limitation each
instance shares: each treats essentially one harmonic potential and one configuration,
with an ad hoc auxiliary polynomial chosen by hand for that case; the existing
arguments stop short of any uniform recipe that would cover an arbitrary completely
monotonic f and an arbitrary special configuration at once.

**Linear programming bounds for codes.** Delsarte, Goethals and Seidel (1977) and,
independently, Kabatiansky and Levenshtein (1978) bound the size of a spherical code
of given minimal angle by exhibiting a polynomial with nonnegative Gegenbauer
coefficients that lies below the "forbidden" indicator; positive-definiteness turns it
into a feasibility/size bound (this is the same engine that solves the kissing-number
problem in dimensions 8 and 24). Levenshtein (1992) identified the configurations for
which these LP code bounds are *sharp* — and that list coincides with the exceptional
arrangements above. The construction of the optimal code polynomial used the
Christoffel–Darboux formula and the roots of certain Jacobi-type orthogonal
polynomials. The gap: these bounds are about *packing* (one threshold distance), not
about *energy* for a smooth potential, and they certify cardinality, not an energy
value.

## Evaluation settings

The natural test instances are: the sphere dimension n; the number of points N; and
the potential f, ranging over the Coulomb/harmonic law f(r) = 1/r^{n/2-1}, the Riesz
family 1/r^s for s > 0, the Gaussian e^{-cr}, the log potential log(4/r) (whose
energy is minus the log of the product of distances), and the linear-in-distance
2 - r^{1/2} (sum of distances). The configurations to test against are those in the
known list: simplices, cross polytopes, the icosahedron, the E_8 and Leech minimal
vectors, the 600-cell, and the kissing configurations derived from E_8 and the Leech
lattice. The relevant yardsticks are: whether a candidate is an optimal spherical code
(maximal minimal angle), what spherical-design strength it has, and — the quantity of
interest — its total f-energy compared against a certified lower bound. A useful small
diagnostic instance is N = 20 on S^2 with the Coulomb law, where the numerical
minimizer is known not to be the dodecahedron; and N = 5 on S^2, where one can ask
whether any configuration minimizes energy for *all* completely monotonic potentials
at once.

## Code framework

The existing tools are: spherical-harmonic / Gegenbauer polynomial evaluation
(three-term recurrence, available as a special-function routine), a general-purpose
linear-programming solver, and direct energy evaluation for candidate configurations.
The available numerical slot is a finite-dimensional LP over Gegenbauer coefficients of
the same flavor as the Delsarte–Kabatiansky–Levenshtein code bounds, with constraints
sampled at inner products on a grid. How to put that LP to work, and what would let a
grid-sampled result become a certified proof for the special configurations, is left
open.

```python
import numpy as np
from scipy.special import gegenbauer
from scipy.optimize import linprog

def lp_energy_lower_bound(n, N, f_of_squared_dist, degree=12, grid=600):
    # TODO: set up and solve the LP over Gegenbauer coefficients.
    pass

def coulomb(R):
    # TODO: potential as a function of squared distance R.
    pass

def icosahedron():
    # TODO: construct a known candidate configuration on S^2.
    pass

def energy(P, f):
    # Direct energy of a concrete configuration P on the sphere.
    E = 0.0
    for i in range(len(P)):
        for j in range(len(P)):
            if i != j:
                E += f(np.sum((P[i] - P[j]) ** 2))
    return E
```
