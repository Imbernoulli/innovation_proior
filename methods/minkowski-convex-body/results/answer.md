# Minkowski's convex body theorem (geometry of numbers)

## The problem it solves

When is a region forced to contain a lattice point other than the origin, judged from its size alone? The lattice points of a curved or skew region admit no formula and cannot be enumerated; the only robust handle is volume. The theorem converts a volume threshold, together with two soft geometric hypotheses (convexity and central symmetry), into the guaranteed existence of a nonzero lattice point — turning a wide family of Diophantine existence problems into one geometric estimate.

## The key idea

Halve the body and fold it onto the lattice. If `vol(C) > 2ⁿ d(L)`, then `½C` has volume exceeding one fundamental cell, so a continuous pigeonhole forces two points of `½C` onto the same residue modulo `L`; their difference is a nonzero lattice vector, and central symmetry plus convexity carry that difference back into `C`. The factor `2ⁿ` is exactly the price of the halving and is best possible (the cube witnesses sharpness).

## Statement

**Theorem.** Let `L ⊂ Rⁿ` be a full lattice with covolume `d(L) = |det(v₁,…,vₙ)|`, and let `C ⊂ Rⁿ` be convex and symmetric about the origin (`x ∈ C ⇒ −x ∈ C`).

1. **(open / strict form)** If `vol(C) > 2ⁿ d(L)`, then `C` contains a point of `L` other than `0`.
2. **(compact / equality form)** If in addition `C` is compact (closed and bounded), then `vol(C) ≥ 2ⁿ d(L)` already forces a nonzero lattice point of `L` in `C`.

For the integer lattice `L = Zⁿ`, `d(L) = 1`, and the threshold is `vol(C) > 2ⁿ` (resp. `≥ 2ⁿ` for compact `C`).

**Sharpness.** The constant `2ⁿ d(L)` cannot be lowered. With `L = Zⁿ`, the open cube `{x : max_i |xᵢ| < 1}` has volume `2ⁿ` and contains no nonzero lattice point; shrinking it to `{max_i |xᵢ| < 1 − ε/2}` gives volume `(2 − ε)ⁿ` with still no nonzero lattice point. So a strict volume inequality is necessary for an *open* body, and equality requires compactness.

## Proof

Two ingredients: a measure-theoretic **fold lemma** that needs neither convexity nor symmetry, and the convex-body theorem as its corollary. A second, sharper **packing** proof gives the quantitative minimum-distance bound and the compact equality case directly.

### Fold lemma (continuous pigeonhole)

**Lemma.** If `S ⊆ Rⁿ` is measurable with `vol(S) > d(L)`, there exist distinct `x, y ∈ S` with `x − y ∈ L`.

*Proof.* Let `F` be a fundamental parallelepiped of `L`; the translates `{u + F : u ∈ L}` are pairwise disjoint and tile `Rⁿ`. Partition `S` into `Sᵤ = S ∩ (u + F)`. These are disjoint with union `S`, so

`Σ_{u ∈ L} vol(Sᵤ) = vol(S) > vol(F) = d(L).`

Translate each piece into the home cell: `Sᵤ − u ⊆ F`, with `vol(Sᵤ − u) = vol(Sᵤ)`, so `Σ_u vol(Sᵤ − u) > vol(F)`. The translated pieces all lie inside `F`, whose volume is `d(L)`, yet their volumes sum to more than `d(L)`; they cannot be pairwise disjoint. Hence there are `u ≠ v` and a point `a ∈ (Sᵤ − u) ∩ (Sᵥ − v)`. Set `x = a + u ∈ Sᵤ ⊆ S` and `y = a + v ∈ Sᵥ ⊆ S`. Then `x ≠ y` and `x − y = u − v ∈ L \ {0}`. ∎

### Part 1 (strict form)

Apply the lemma to `S = ½C = {½x : x ∈ C}`. Then `vol(S) = 2⁻ⁿ vol(C) > d(L)`, so there are distinct `x, y ∈ ½C` with `x − y ∈ L \ {0}`. Now `2x ∈ C`; by central symmetry `2y ∈ C ⇒ −2y ∈ C`; by convexity the midpoint

`x − y = ½(2x) + ½(−2y) ∈ C.`

So `x − y` is a nonzero lattice point lying in `C`. ∎

### Part 2 (compact equality form)

Let `C` be compact with `vol(C) = 2ⁿ d(L)`; suppose it contains no nonzero lattice point. For each integer `m ≥ 1`,

`vol((1 + 1/m)C) = (1 + 1/m)ⁿ · 2ⁿ d(L) > 2ⁿ d(L),`

so by Part 1 there is a nonzero `xₘ ∈ L ∩ (1 + 1/m)C ⊆ 2C`. Since `2C` is bounded and `L` discrete, `(2C) ∩ L` is finite, so some nonzero `x ∈ L` satisfies `x = xₘ` for infinitely many `m`; equivalently `(1 + 1/m)⁻¹ x ∈ C` for infinitely many `m`. Letting `m → ∞` and using that `C` is closed gives `x ∈ C`, contradicting the assumption. ∎

### Sharp packing form (quantitative)

Let `f(x) = min{λ ≥ 0 : x ∈ λC}` be the gauge of `C` (a norm: `f(tx) = t f(x)` for `t ≥ 0`, `f(x+y) ≤ f(x)+f(y)`, `f(−x) = f(x)`), so `{f < r} = rC` and `vol{f < r} = rⁿ vol(C)`. Let `M = min_{l ∈ L \ {0}} f(l)`, attained because the lattice is discrete in `f`. The bodies `Bₐ = {x : f(x − a) < M/2}` for `a ∈ L` are pairwise disjoint: if `b ∈ Bₐ ∩ B_c` with `a ≠ c`, then `f(a − c) ≤ f(a − b) + f(b − c) < M`, contradicting `f(a − c) ≥ M`. Packing the `(Ω + 1)ⁿ` such bodies around the lattice points with coordinates in `{0, ±1, …, ±Ω/2}` (`Ω` even) into a centered cube of edge `Ω + O(1)` and letting `Ω → ∞` yields `(M/2)ⁿ vol(C) ≤ d(L)`, i.e.

`Mⁿ · vol(C) ≤ 2ⁿ d(L),  hence  min_{l ∈ L \ {0}} f(l) ≤ 2 (d(L) / vol(C))^{1/n}.`

Thus `vol(C) ≥ 2ⁿ d(L) ⇒ M ≤ 1` (nonzero lattice point in the closed `C`) and `vol(C) > 2ⁿ d(L) ⇒ M < 1` (one strictly inside), recovering Parts 2 and 1 with the sharp distance bound. ∎

## Why the hypotheses are exactly right

- **Convexity** is used once and is essential: it puts the midpoint `½(2x) + ½(−2y)` inside `C`. A non-convex body of volume `2ⁿ` (a thin star, a needle) can avoid every nonzero lattice point.
- **Central symmetry** is used once and is essential: it supplies `−2y ∈ C` from `2y ∈ C`, the second endpoint of the midpoint. An off-center convex body of large volume can place only the origin on the lattice.
- **The threshold `2ⁿ d(L)`** is the cost of halving: the fold needs `vol(½C) > d(L)`, i.e. `2⁻ⁿ vol(C) > d(L)`. The cube shows it is sharp.
- **Compactness** is exactly what upgrades `>` to `≥`; the limiting argument needs `C` closed and `(2C) ∩ L` finite.

## Consequences

**Linear forms.** For linear forms `Lᵢ(x) = Σⱼ bᵢⱼ xⱼ` with `det(B) ≠ 0` and positive reals `Aᵢ` with `∏ᵢ Aᵢ ≥ |det B|`, there is a nonzero `x ∈ Zⁿ` with `|Lᵢ(x)| ≤ Aᵢ` for all `i`. *Proof:* the region `P = {x : |Lᵢ(x)| ≤ Aᵢ} = B⁻¹R`, where `R = {|yᵢ| ≤ Aᵢ}` has volume `2ⁿ ∏ Aᵢ`, is convex and symmetric with `vol(P) = |det B|⁻¹ · 2ⁿ ∏ Aᵢ ≥ 2ⁿ`; apply the compact form. ∎

**Simultaneous approximation (Dirichlet).** For reals `α₁,…,αₙ`, applying the linear-forms consequence to `Xᵢ − αᵢ X_{n+1}` and `X_{n+1}` (determinant `1`) with bounds `Q^{−1/n}` and `Q` gives integers with `|xᵢ − αᵢ q| ≤ Q^{−1/n}`, `0 < |q| ≤ Q`; hence `|αᵢ − xᵢ/q| ≤ q^{−1−1/n}`, and letting `Q → ∞` gives infinitely many tuples when some `αᵢ` is irrational.

**Minimum of a quadratic form / shortest vector.** Taking `C` the cube `{‖x‖_∞ < ℓ}` gives a nonzero lattice vector with `‖x‖_∞ ≤ d(L)^{1/n}`, hence `‖x‖₂ ≤ √n · d(L)^{1/n}` and the Hermite constant `γₙ ≤ n`; taking `C` a disk in the plane gives the sharper `γ₂ ≤ 4/π`. With `f = √Q` the bound reads `min_{Zⁿ\0} Q ≤ 4 (vol{Q < 1})^{−2/n}`, recovering a determinant-only bound on positive-definite quadratic forms as the ellipsoid special case.

**Sums of squares.** For a prime `p ≡ 1 (mod 4)`, choose `r` with `r² ≡ −1 (mod p)` and apply the theorem to the ellipse `{(pz₁ + rz₂)² + z₂² < 2p}` (area `> 4`): a nonzero `(z₁, z₂) ∈ Z²` gives `0 < (pz₁ + rz₂)² + z₂² < 2p`, and `(pz₁ + rz₂)² + z₂² ≡ (r² + 1)z₂² ≡ 0 (mod p)` forces the value to be exactly `p`, so `p = a² + b²`. The four-dimensional analogue (with `r² + s² + 1 ≡ 0 (mod p)`) gives Lagrange's theorem that every positive integer is a sum of four squares.

## Checking the criterion on instances

The criterion is "compute `vol(C)`, compare with `2ⁿ d(L)`; if it clears, the body is forced to contain a nonzero lattice point, found by the fold." A direct instance-checker (the threshold test plus a brute-force witness search whose *success* the theorem guarantees):

```python
import numpy as np
from itertools import product

def covolume(basis):                      # d(L) = |det(v_1,...,v_n)|
    return abs(np.linalg.det(basis))

def body_volume(C, box_radius, n, samples=400000):   # Monte-Carlo vol(C)
    pts = np.random.uniform(-box_radius, box_radius, size=(samples, n))
    return np.mean([C(p) for p in pts]) * (2 * box_radius) ** n

def guarantees_nonzero_lattice_point(C, basis, box_radius, n):
    # The criterion: vol(C) >= 2^n * d(L) (compact body) forces a nonzero lattice point.
    return body_volume(C, box_radius, n) >= 2 ** n * covolume(basis)

def witness_nonzero_lattice_point(C, basis, coord_range):
    # When the criterion holds the theorem guarantees this search succeeds.
    n = basis.shape[1]
    for z in product(range(-coord_range, coord_range + 1), repeat=n):
        z = np.array(z)
        if np.any(z != 0) and C(basis @ z):
            return basis @ z
    return None

# Instance: a disk against Z^2 (threshold 2^2 * 1 = 4; the area-4 disk is the sharp
# compact case, with (+-1,0) on/inside it -- here we take area 4.2 > 4 to clear it).
I2   = np.eye(2)
disk = lambda x: x[0]**2 + x[1]**2 <= 4.2 / np.pi        # area = pi r^2 = 4.2 > 4
assert guarantees_nonzero_lattice_point(disk, I2, 2.0, 2)
assert witness_nonzero_lattice_point(disk, I2, 3) is not None   # e.g. (1,0)

# Sharpness: the open square of area (2-eps)^2 < 4 has no nonzero lattice point.
sq = lambda x: max(abs(x[0]), abs(x[1])) < 1 - 1e-9
assert witness_nonzero_lattice_point(sq, I2, 3) is None
```
