**Problem.** Construct a single packing of `26` non-overlapping circles inside the unit square
`[0,1]²`, scored by the sum of radii `Σ rᵢ` (higher is better), feasible within an absolute
tolerance. The radii are free and unequal. The frontier for `n = 26` is `≈ 2.636` (AlphaEvolve
`2.63586276`, ShinkaEvolve `2.635983283`, AutoEvolver record `2.635988438568`); the starting rung
is the principled structured layout.

**Key idea.** Use a regular `5 × 5` grid of equal circles plus one interstitial filler. A `k × k`
grid of unit-cell-inscribed circles has radius `r = 1/(2k)` and sum `k/2`, so finer grids give
larger sums — the sum-of-radii objective rewards many small circles. `5 × 5` is the largest grid
that fits within `26` circles: it places `25` circles of radius `0.1` (sum `2.5`), each tangent to
its neighbors and the walls. The `26th` circle drops into an interior interstitial gap, where four
mutually adjacent grid circles leave room for a circle of radius `(√2 − 1)·r ≈ 0.0414`, tangent to
all four. Total `Σ rᵢ = 2.5 + (√2 − 1)·0.1 ≈ 2.5414`.

**Why these choices.** The construction is forced once `k = 5` is fixed: `26` is between the
squares `25` and `36`, and only `25 = 5²` fits without dropping circles (which would only lower
the sum). The interstitial radius `(√2 − 1)·r` is the exact inscribed-gap circle, the unique
largest circle that fits the lattice hole. This is deliberately the *floor*: a rigid,
parameter-free, symmetric layout that is guaranteed feasible and gives a concrete baseline. It is
not expected to approach the frontier — the equal-radius constraint and the lattice waste the
corners and edges, and a strong sum-of-radii packing is irregular (mixed sizes filling space),
which a grid cannot express. The entire distance from `2.5414` to `2.636` is what later, searched
rungs must buy by letting the radii become unequal and the centers move freely.

**Hyperparameters / contract.** None. Output is the fixed `26`-circle grid-plus-interstitial
packing. Deterministic — same packing every call. Feasible: the `25` grid circles are tangent to
neighbors and walls; the interstitial circle is tangent to its four neighbors; all violations are
at the floating-point floor (`maxviol ~ 6×10⁻¹⁷`).

```python
import numpy as np

N = 26

def construct_packing():
    """5x5 grid of equal circles (r=0.1) + one interstitial filler -> Sigma r ~ 2.5414."""
    k = 5
    r = 1.0 / (2 * k)                       # 0.1, inscribed in each 1/k cell
    centers = [[(i + 0.5) / k, (j + 0.5) / k] for i in range(k) for j in range(k)]
    radii = [r] * (k * k)                   # 25 equal grid circles, sum = k/2 = 2.5
    # 26th circle: largest disk in the interstitial gap of a 2x2 block of grid circles.
    # The four neighbours sit at distance sqrt(2)*r from the gap centre, so r_gap = (sqrt2-1)*r.
    centers.append([2.0 / k, 2.0 / k])      # centre of the (1,1)-(2,2) block interstice
    radii.append((np.sqrt(2) - 1) * r)
    return np.array(centers), np.array(radii)


if __name__ == "__main__":
    c, r = construct_packing()
    print("sum_radii =", r.sum())           # 2.5414213562373…
```
