The problem is to place $26$ circles inside the unit square $[0,1]^2$, pairwise non-overlapping,
so as to maximize the sum of their radii $\sum_i r_i$. What makes this different from the classical
equal-circle packing problem is that the radii here are free and unequal: I am not maximizing one
common radius, I am maximizing a sum, and that immediately changes the character of the optimum.
The feasible region is governed by the wall constraints $r_i \le x_i \le 1-r_i$,
$r_i \le y_i \le 1-r_i$ and the pairwise constraints $\sqrt{(x_i-x_j)^2+(y_i-y_j)^2} \ge r_i + r_j$,
the latter being nonconvex, so the full problem is a nonconvex QCQP. Before reaching for any
optimizer I want a baseline that is principled, guaranteed feasible, and parameter-free — something
that puts a concrete number on the board so every later searched method has a floor to beat. The
published frontier for $n=26$ sits near $2.636$, but that is the product of large automated search;
what I want first is the honest floor a closed-form construction reaches.

I propose a structured grid baseline: a regular $5\times 5$ grid of equal circles plus one
interstitial filler. The reasoning is driven by a single observation about the objective. If I lay
down a $k\times k$ grid of equal circles, one per cell, the cell width is $1/k$ and the largest
equal radius that keeps each circle inside its cell and tangent to its neighbors is the inscribed
radius $r = 1/(2k)$. The grid then holds $k^2$ circles, and the sum of radii is
$k^2 \cdot \frac{1}{2k} = k/2$. This formula is the load-bearing insight: the sum *grows* with $k$,
because halving the radius while quadrupling the count multiplies the total by two. So for the
sum-of-radii objective, many small circles is not a weakness — it is the whole game. I must hit
exactly $26$ circles, and $26$ lies between the squares $25=5^2$ and $36=6^2$. A $6\times 6$ grid
has too many circles and I cannot simply drop ten, since removing circles only lowers the sum; a
$5\times 5$ grid fits with one circle to spare. The $25$ grid circles have $r = 1/10 = 0.1$ and
contribute $25 \times 0.1 = 2.5$.

The remaining circle goes into an interstitial gap. In a square grid of equal circles of radius
$r$, four mutually adjacent circles centered on the corners of a little grid square of side
$1/k = 2r$ leave a hole at the center of that square. The center of the hole is equidistant from
the four surrounding centers at distance $\sqrt{2}\,r$ (half the diagonal of a $2r\times 2r$
square), so the largest circle that fits the hole without overlapping any of the four has radius
$\sqrt{2}\,r - r = (\sqrt{2}-1)\,r$. With $r = 0.1$ that is $(\sqrt{2}-1)\cdot 0.1 \approx 0.0414$,
and the total becomes
$$\sum_i r_i = 2.5 + (\sqrt{2}-1)\cdot 0.1 \approx 2.5414.$$
The arrangement is feasible at the floating-point floor: the $25$ grid circles sit exactly at the
feasibility boundary, tangent to neighbors (centers $0.2$ apart, radii summing to $0.2$) and tangent
to the walls (centers at $0.1, 0.3, \dots, 0.9$, radius $0.1$), and the interstitial circle is
tangent to its four neighbors by construction. The construction is forced once $k=5$ is fixed: there
is no free parameter and no search. Its limitation is exactly what makes it a baseline — the
equal-radius constraint and the lattice waste the corners and edges where a larger circle could sit,
and a strong sum-of-radii packing is deliberately irregular, mixing a few large circles with many
unequal gap-fillers, none of which a grid can express. That gap of roughly $0.094$ to the frontier
is precisely what later methods must buy by letting the radii become unequal and the centers move
freely.

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
