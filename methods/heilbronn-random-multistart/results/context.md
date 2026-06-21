# Context: Heilbronn triangle problem at n=11

## Research question

Place `n = 11` points in `[0,1]^2` to maximize the **minimum** triangle area over all
`C(11,3) = 165` triples. The constructor emits 11 points; the exact minimum triangle area is the
score (higher better). A single near-collinear trio sets a near-zero minimum, so the score has a
min-of-many structure.

No closed form reproduces the irregular record configuration at this `n`. The question is how to
produce an 11-point configuration with a large minimum triangle area.

## How the score is defined

```
score(P) = min over all 165 triples of  |(b-a) x (c-a)| / 2,   all points in [0,1]^2.
```

## Record and yardstick

Unit-square record at `n = 11`: `Δ(11) = 1/27 = 0.037037` (Goldberg 1972, conjectured optimal). The
structured baseline (inscribed regular 11-gon) scores `0.021456` (≈0.58 of record); the two
lattice-like baselines (grid, equally-spaced boundary) score 0 by collinearity. A configuration is a
point in 22-dimensional space.

## The editable interface

One editable function, `construct() -> (11,2) array in [0,1]^2`; evaluator and `n = 11` fixed. The
inner scoring is vectorized over the 165 precomputed index triples so that many configurations can
be scored cheaply.

```python
import numpy as np
from itertools import combinations
N = 11
TRIPLES = np.array(list(combinations(range(N), 3)))   # (165,3)
```
