**Problem.** Place `n = 11` points in the unit square `[0,1]^2` to maximize the minimum area over
all `C(11,3) = 165` triangles. Score = exact min triangle area (higher better). Record at `n = 11`
is `1/27 = 0.037037` (Goldberg 1972, conjectured optimal). This rung is a parameter-free
structured baseline — the floor every searched rung must beat.

**Key idea.** The score is the *minimum* triangle, so any three collinear points sink it to zero.
A regular grid and an equally-spaced boundary ring are both built out of straight lines (grid rows,
square edges), so they contain collinear triples and score exactly `0`. The one structured object
with *no three points ever collinear* is points on a common circle: a line meets a circle in at
most two points, so every triangle has strictly positive area. Take the largest circle that fits —
the inscribed circle of radius `1/2` centered at `(0.5, 0.5)` — and place the `11` points equally
spaced on it (a regular `11`-gon). All points are legal, no triple is degenerate, and the smallest
triangles are the near-adjacent rim triples, a modest positive value.

**Why these choices.** The grid and boundary configurations are reported alongside to make the
collinearity wall explicit: both are "even" in the lattice sense and both score `0`, which is why
evenness here must mean "no straight lines," not "lattice-regular." The circle is the unique simple
closed form that guarantees positivity. The inscribed (radius `1/2`) circle is forced: it is the
largest circle fitting in the square, so it spreads the points maximally; any smaller circle scales
every triangle down, any larger one leaves the square. There is no parameter to tune — equal
spacing on the largest legal circle. It is deliberately the floor: a regular `11`-gon spends its
symmetry on rotational regularity, not on fattening the worst triangle, so it sits well below the
record and leaves the whole gap to `1/27` to be bought by search.

**Hyperparameters / contract.** None. Output is the fixed `11 × 2` array of points on the inscribed
circle. Every coordinate is in `[0,1]` by construction (`0.5 ± 0.5·cos/sin`). Deterministic — same
points every call.

```python
import numpy as np

N = 11

def construct():
    # 11 points equally spaced on the inscribed circle (radius 1/2, center (0.5,0.5)).
    # No three points on a circle are ever collinear, so every triangle has positive area;
    # the grid and equally-spaced-boundary alternatives contain collinear triples and score 0.
    ang = np.linspace(0, 2*np.pi, N, endpoint=False)
    return 0.5 + 0.5 * np.column_stack([np.cos(ang), np.sin(ang)])
```
