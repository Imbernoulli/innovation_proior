# Context: Heilbronn triangle problem at n=11, local search on positions

## Research question

Place `n = 11` points in `[0,1]^2` to maximize the **minimum** triangle area over all
`C(11,3) = 165` triples. The constructor emits 11 points; the exact minimum triangle area is the
score (higher better).

This rung asks: starting from a configuration and nudging one point at a time, how far does local
search on point positions take us on this objective?

## How the score is defined

```
score(P) = min over all 165 triples of  |(b-a) x (c-a)| / 2,   all points in [0,1]^2.
```

## Record and yardstick

Unit-square record at `n = 11`: `Δ(11) = 1/27 = 0.037037` (Goldberg 1972, conjectured optimal).
Prior rungs: inscribed 11-gon `0.021456` (0.579 of record); random multi-start `0.010872` (0.294).

## The editable interface

One editable function, `construct() -> (11,2) array in [0,1]^2`; evaluator and `n = 11` fixed. The
key efficiency: moving one point changes only the `C(10,2) = 45` triples that involve it, so the
inner loop keeps all 165 areas and recomputes only those 45 per step.

```python
import numpy as np
from itertools import combinations
N = 11
TRIPLES = np.array(list(combinations(range(N), 3)))
INVOLVES = [[t for t, tr in enumerate(TRIPLES) if p in tr] for p in range(N)]  # 45 triples each
```
</content>
