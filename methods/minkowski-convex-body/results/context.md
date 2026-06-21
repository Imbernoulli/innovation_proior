# Context: lattice points in symmetric regions

## Research question

Given a region in the plane or in `n`-dimensional space, does it contain a point with whole-number coordinates other than the origin? More precisely: fix the integer lattice `Zⁿ` (all points whose coordinates are integers, or more generally a full lattice `L` spanned by `n` independent vectors). Take a region `C` placed so that the origin is one of its points. **Under what condition on `C` is it forced to contain a lattice point besides the origin?**

This is the basic existence problem of number theory dressed in geometry. Diophantine questions — can a system of linear forms be made simultaneously small at integer arguments? how well can a real number be approximated by rationals? which integers are sums of two or four squares? — all reduce to: *here is a region built out of the data of the problem; produce an integer point inside it.* The only robust handle a region of arbitrary shape offers is its **volume**. So the question sharpens to: *can a volume threshold alone guarantee a nonzero lattice point, and what is the threshold?*

## Background

**Lattices.** A (full) lattice `L ⊂ Rⁿ` is the set of integer combinations `{z₁v₁ + ⋯ + zₙvₙ : zᵢ ∈ Z}` of `n` linearly independent vectors `v₁,…,vₙ`. It is a discrete subgroup of `Rⁿ`: closed under `±`, and with a positive minimum distance between distinct points. The **fundamental parallelepiped** `F = {x₁v₁+⋯+xₙvₙ : 0 ≤ xᵢ < 1}` has volume `d(L) := |det(v₁,…,vₙ)|`, the **determinant** (covolume) of the lattice, and the translates `{u + F : u ∈ L}` are pairwise disjoint and tile all of `Rⁿ`. The determinant is basis-independent: two bases differ by an integer matrix of determinant `±1`. For the integer lattice `Zⁿ`, `d(L) = 1` and `F` is the unit cube. The covolume measures density: `1/d(L)` lattice points per unit volume, asymptotically.

**Convex, centrally symmetric bodies.** A set `C` is **convex** if for any `x, y ∈ C` the whole segment `{tx+(1−t)y : 0 ≤ t ≤ 1}` lies in `C` (no dents). It is **centrally symmetric** (about the origin) if `x ∈ C ⇒ −x ∈ C`. A **convex body** is a closed, bounded, convex set with the origin in its interior. Such a body is exactly the unit ball of a norm (its *gauge*): set `‖x‖_C = min{λ ≥ 0 : x ∈ λC}`; then `‖·‖_C` is positively homogeneous (`‖tx‖ = t‖x‖` for `t ≥ 0`), subadditive (`‖x+y‖ ≤ ‖x‖+‖y‖`, the triangle inequality, which is precisely convexity), and symmetric (`‖−x‖ = ‖x‖`, precisely central symmetry), and `λC = {x : ‖x‖_C ≤ λ}`. Dilation scales volume by the dimension power: `vol(λC) = λⁿ vol(C)`. Linear maps scale volume by `|det|`. Examples: cubes `{max|xᵢ| ≤ 1}` (volume `2ⁿ`), Euclidean balls, octahedra `{Σ|xᵢ| ≤ 1}`, ellipsoids `{Q(x) ≤ 1}` for a positive-definite quadratic form `Q`.

**The quadratic-form precedent (Hermite).** Hermite, in his letters to Jacobi "sur différents objets de la théorie des nombres" (Crelle's Journal vol. 40), proved a foundational fact: for a positive-definite quadratic form `Q` in `n` variables one can always substitute integers, not all zero, so that the value `Q(x)` stays below a bound that depends **only on the determinant of the form** — `min_{x ∈ Zⁿ\{0}} Q(x) ≤ cₙ (det Q)^{1/n}` for a constant `cₙ` depending only on `n`. Hermite used this as a powerful tool for reduction theory and class-number finiteness. Dirichlet's paper in the same volume of Crelle on the reduction of positive ternary quadratic forms put this circle of ideas in geometric terms. A quadratic-form value bound is exactly a statement about the ellipsoid `{Q(x) ≤ c}`: "an ellipsoid of large enough volume must contain a nonzero integer point."

**The pigeonhole principle (Dirichlet).** Dirichlet's *Schubfachprinzip*: if more than `k` objects are placed in `k` boxes, some box holds two. Its quantitative continuous analogue — comparing a total measure against the measure of a single box — underlies Dirichlet's own theorem that every irrational `α` has infinitely many rational approximations `|α − p/q| ≤ q⁻²`. The principle is elementary and shape-free.

## Baselines

These are the tools on the table that a lattice-point existence result would be measured against or built from.

**Hermite's reduction bound for quadratic forms.** *Idea:* by successively reducing a positive-definite form one bounds its minimum at integer points by a determinant-only quantity, `min_{Zⁿ\0} Q ≤ cₙ (det Q)^{1/n}`. *Math:* algebraic manipulation of the Gram matrix; in the geometric reading it asserts that an ellipsoid `{Q ≤ c}` of volume past a determinant-dependent threshold meets `Zⁿ`. The argument is tied to the quadratic shape and to reduction theory, and the argument is algebraic rather than geometric.

**Direct enumeration / explicit construction.** *Idea:* to show a given Diophantine inequality has an integer solution, build one by hand (continued fractions for one-dimensional approximation; explicit identities such as Brahmagupta–Fibonacci for sums of squares). *Math:* problem-specific. Each region or inequality is addressed by a dedicated construction; high-dimensional or skew regions are handled case by case.

**The pigeonhole/box principle for one-dimensional or finite approximation.** *Idea:* Dirichlet's argument that among `q+1` numbers `{kα}` (fractional parts) two are within `1/q`, giving `|α − p/q| < 1/q²`. *Math:* partition `[0,1)` into `q` boxes; two of `q+1` points share a box. The partition trick exploits the *product* structure of a cube — it is run on the unit interval or a product of intervals.

**Volume/measure comparison.** *Idea:* compare the measure of a set against the measure of a region known to fit it, to deduce overlap or covering. *Math:* additivity of volume over a disjoint partition; volume is translation-invariant and scales by `|det|` under linear maps.

## Evaluation settings

The natural yardsticks for such a result are the classical existence problems it ought to settle, and the worked geometric objects on which it is tested:

- **Simultaneous Diophantine approximation.** Given reals `α₁,…,αₙ`, find integers `(p₁,…,pₙ, q)`, `q > 0`, with `|αᵢ − pᵢ/q| ≤ q^{−1−1/n}` — the multidimensional generalization of `|α − p/q| ≤ q⁻²` (Dirichlet, 1842). The region is a box of small height around a line.
- **Systems of linear forms.** Given linear forms `Lᵢ(x) = Σⱼ bᵢⱼ xⱼ` with `det(bᵢⱼ) ≠ 0` and positive bounds `Aᵢ` with `∏ Aᵢ ≥ |det B|`, decide whether some nonzero integer `x` makes `|Lᵢ(x)| ≤ Aᵢ` for all `i`. The region is the parallelepiped `{|Lᵢ| ≤ Aᵢ} = B⁻¹R`.
- **Representation of integers by quadratic forms.** Express a prime `p ≡ 1 (mod 4)` as a sum of two squares; express every positive integer as a sum of four squares. The region is an ellipse/ellipsoid `{(linear)² + ⋯ < R²}` tuned so its lattice points carry the congruence.
- **The minimum of a quadratic form / shortest lattice vector.** For a lattice `L`, bound `λ(L) = min_{v ∈ L\0} ‖v‖₂` in terms of `d(L)^{1/n}` (the Hermite constant `γₙ = sup_L (λ(L)/d(L)^{1/n})²`); equivalently bound `min_{Zⁿ\0} Q` for a positive-definite form.
- **Worked geometric test objects.** Planar centrally symmetric convex sets of prescribed area (disk, square, ellipse, hexagon) relative to the unit lattice; the cube and the ball in `Rⁿ` against `Zⁿ`. The relevant data per object is its volume and the lattice covolume.

The protocol throughout is non-numeric and existential: state the region and its volume, state the lattice and its covolume, and ask whether a nonzero lattice point is guaranteed — then read off the Diophantine consequence.

## Code framework

A small symbolic/numeric scaffold for *checking instances* of a lattice-point criterion, written purely in terms that already exist (lattices, volumes, the fundamental cell, the pigeonhole comparison). The contribution will be the criterion itself and its proof; the scaffold only lets us sanity-check the threshold on concrete bodies.

```python
import numpy as np
from itertools import product

# --- pre-existing primitives ---

def covolume(basis):
    """d(L) = |det(v_1,...,v_n)| for a full lattice with given basis (columns)."""
    return abs(np.linalg.det(basis))

def fundamental_cell_volume(basis):
    """Volume of the fundamental parallelepiped F; equals d(L)."""
    return covolume(basis)

def in_body(C_membership, x):
    """C is given by a membership predicate x -> bool (closed, bounded, contains 0)."""
    return C_membership(x)

def body_volume(C_membership, box_radius, n, samples=200000):
    """Monte-Carlo estimate of vol(C) for a bounded body inside [-R,R]^n."""
    pts = np.random.uniform(-box_radius, box_radius, size=(samples, n))
    frac = np.mean([C_membership(p) for p in pts])
    return frac * (2 * box_radius) ** n

def lattice_points_in(C_membership, basis, coord_range):
    """Brute-force list of lattice points L ∩ C with integer coords in [-coord_range, coord_range]^n."""
    n = basis.shape[1]
    pts = []
    for z in product(range(-coord_range, coord_range + 1), repeat=n):
        x = basis @ np.array(z)
        if C_membership(x):
            pts.append(x)
    return pts

# --- the slot the result will occupy ---

def guarantees_nonzero_lattice_point(C_membership, basis, box_radius, n):
    """
    Decide, from volume data alone, whether C is FORCED to contain a
    nonzero lattice point of L (basis).
    # TODO: the criterion we will derive — the condition on vol(C), d(L), and
    #       the shape of C under which a nonzero lattice point must exist.
    """
    pass

def witness_nonzero_lattice_point(C_membership, basis, box_radius, n):
    """
    When the criterion holds, exhibit a nonzero lattice point inside C.
    # TODO: the construction our proof will provide.
    """
    pass
```
