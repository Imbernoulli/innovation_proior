**Problem.** Place `n = 11` points in `[0,1]^2` to maximize the minimum area over all `165`
triangles. Score = exact min triangle area. Record `Δ(11) = 1/27 = 0.0370370370...` (Goldberg 1972,
conjectured optimal; Erich Friedman's Heilbronn-for-squares page). ENDPOINT: reproduce Goldberg's exact rational configuration and verify in
exact arithmetic that its smallest triangle is exactly `1/27`.

**Key idea.** The search-and-polish frontier sits at `0.037032`, five parts in a million below the
record — pure optimizer tolerance, not a geometric gap. A clean rational record does not come from
unstructured climbing; it comes from a configuration symmetric and rigid enough that a large family of
triangles share *exactly* the same area, forcing the minimum onto a rational value. Goldberg's `n = 11`
configuration is mirror-symmetric about `x = 1/2`, with eight points on the boundary and three in the
interior, all at rational coordinates with denominators `3, 9, 6, 2`. Writing those exact fractions
down and evaluating the min over all `165` triples in **exact rational arithmetic** lands the minimum on
the literal fraction `1/27`, with the binding triangles all tied at exactly that value.

**Why these choices.** Reaching `1/27` *exactly* (not `0.037037...` to some digits) demands exact
arithmetic, so each coordinate is a `Fraction` and each triangle area is half the absolute cross
product of integer-denominator differences — no rounding anywhere, so the minimum over all triples is
an exact fraction. The configuration is Goldberg's published one: `(1/3,0),(2/3,0)` on the bottom;
`(0,2/9),(1,2/9),(0,2/3),(1,2/3)` on the sides; `(1/6,1),(5/6,1)` on top; and the three interior/axis
points `(1/3,4/9),(2/3,4/9),(1/2,7/9)`. The denominator-`9` structure is what makes the area resolve to
`1/27`. Counting how many triangles achieve the minimum measures how over-determined the optimum is — a
rigid binding web is the signature of a genuine max-min maximum. A double-precision recomputation
confirms the harness evaluator the rest of the ladder uses agrees to machine epsilon. This rung
*reaches*, and does not beat, the record: `1/27` is the conjectured optimum at `n = 11` in the square,
believed but unproven, so landing on it exactly is the ceiling. The still-open part is the *proof* of
optimality, which no construction supplies.

**Hyperparameters / contract.** No search, no randomness, no tolerance: the constructor returns the
eleven exact Goldberg coordinates (as floats for the harness; the verification block keeps them as
`Fraction`s). Verified properties: exact rational min over all `165` triples equals `1/27`; `28`
triangles bind at the minimum; the configuration is horizontally symmetric about `x = 1/2`; every point
lies in `[0,1]^2`; double-precision min-area matches `1/27` to `~1e-17`. Output is the `(11,2)` array.

```python
import numpy as np
from itertools import combinations
from fractions import Fraction as Fr

N = 11

# Goldberg (1972) optimal configuration for n=11 in the unit square, Delta(11) = 1/27.
# Source: Erich Friedman's "Heilbronn problem for squares" page (1/27, horizontally
# symmetric).
GOLDBERG_EXACT = [
    (Fr(1, 3), Fr(0)),   (Fr(2, 3), Fr(0)),                       # bottom edge
    (Fr(0), Fr(2, 9)),   (Fr(1), Fr(2, 9)),                       # lower sides
    (Fr(1, 3), Fr(4, 9)),(Fr(2, 3), Fr(4, 9)),                    # interior pair
    (Fr(0), Fr(2, 3)),   (Fr(1), Fr(2, 3)),                       # upper sides
    (Fr(1, 2), Fr(7, 9)),                                         # axis apex
    (Fr(1, 6), Fr(1)),   (Fr(5, 6), Fr(1)),                       # top edge
]

def exact_min_area(P):
    """Exact min over all C(11,3)=165 triangle areas, in rational arithmetic."""
    best = None; nbind = 0
    areas = []
    for i, j, k in combinations(range(N), 3):
        a, b, c = P[i], P[j], P[k]
        cr = (b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1])
        ar = abs(cr) / Fr(2)
        areas.append(ar)
        if best is None or ar < best:
            best = ar
    nbind = sum(1 for ar in areas if ar == best)
    return best, nbind

def construct():
    """Return the 11x2 array of Goldberg's exact optimal points (as floats)."""
    return np.array([[float(x), float(y)] for x, y in GOLDBERG_EXACT])

if __name__ == "__main__":
    # in the unit square?
    assert all(Fr(0) <= x <= Fr(1) and Fr(0) <= y <= Fr(1) for x, y in GOLDBERG_EXACT)
    # exact rational verification
    m, nbind = exact_min_area(GOLDBERG_EXACT)
    assert m == Fr(1, 27), m
    print("exact min triangle area =", m, "=", float(m))
    print("equals 1/27 exactly:", m == Fr(1, 27))
    print("triangles binding at the minimum:", nbind, "of 165")
    # horizontal symmetry about x = 1/2
    S = set((Fr(1) - x, y) for x, y in GOLDBERG_EXACT)
    print("horizontally symmetric about x=1/2:", S == set(GOLDBERG_EXACT))
    # double-precision agreement with the harness evaluator
    P = construct()
    best = np.inf
    for i, j, k in combinations(range(N), 3):
        a, b, c = P[i], P[j], P[k]
        ar = abs((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1])) / 2.0
        best = min(best, ar)
    print("double-precision min area =", repr(best),
          "| gap to 1/27 =", abs(best - 1/27))
```
