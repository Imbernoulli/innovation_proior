# Context: Heilbronn triangle problem at n=11, random multi-start search

## Research question

Place `n = 11` points in `[0,1]^2` to maximize the **minimum** triangle area over all
`C(11,3) = 165` triples. The constructor emits 11 points; the exact minimum triangle area is the
score (higher better). A single near-collinear trio sets a near-zero minimum, so the difficulty is
the min-of-many structure.

This rung asks: with no closed form able to reproduce the irregular record configuration, what does
the *simplest possible search* — sample many random configurations and keep the best — achieve?
It is the zero-knowledge search baseline, before any local improvement, memory, or schedule.

## How the score is defined

```
score(P) = min over all 165 triples of  |(b-a) x (c-a)| / 2,   all points in [0,1]^2.
```

## Record and yardstick

Unit-square record at `n = 11`: `Δ(11) = 1/27 = 0.037037` (Goldberg 1972, conjectured optimal). The
prior structured baseline (inscribed regular 11-gon) scores `0.021456` (≈0.58 of record); the two
lattice-like baselines (grid, equally-spaced boundary) score 0 by collinearity. Random sampling is
expected to be weak because good min-area configurations occupy a vanishing fraction of the
22-dimensional configuration space.

## The editable interface

One editable function, `construct() -> (11,2) array in [0,1]^2`; evaluator and `n = 11` fixed. To
afford millions of trials the inner scoring is vectorized over the 165 precomputed index triples.

```python
import numpy as np
from itertools import combinations
N = 11
TRIPLES = np.array(list(combinations(range(N), 3)))   # (165,3)
```
