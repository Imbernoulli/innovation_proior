# Context: Heilbronn triangle problem at n=11, SA + soft-min gradient polish (endpoint)

## Research question

Place `n = 11` points in `[0,1]^2` to maximize the **minimum** triangle area over all
`C(11,3) = 165` triples. The constructor emits 11 points; the exact minimum triangle area is the
score (higher better).

This is the endpoint rung. Simulated annealing reaches `0.962` of the record but plateaus, because
near the optimum several triangles are simultaneously near-tight and a random single-point move
cannot grow them all at once. The question: can a *smooth* objective whose gradient inflates all the
near-tight triangles together polish an annealed configuration the rest of the way onto the exact
optimum?

## How the score is defined

```
score(P) = min over all 165 triples of  |(b-a) x (c-a)| / 2,   all points in [0,1]^2.
```

## Record and yardstick

Unit-square record at `n = 11`: `Δ(11) = 1/27 = 0.037037` (Goldberg 1972, conjectured optimal).
Prior rungs: 11-gon `0.021456` (0.579); random `0.010872` (0.294); simulated annealing `0.035639`
(0.962). The endpoint targets the record itself: at `n = 11` the value `1/27` is believed optimal,
so matching it to floating-point precision is the ceiling of a single-machine search-plus-polish.

DeepMind's AlphaEvolve (arXiv:2506.13131) improved Heilbronn records for the *sibling* containers —
unit-area triangle (`n=11`: `0.036 → >0.0365`) and unit-area convex region (`n=13,14`) — not the
unit square; it confirms search-plus-refine (evolutionary global search + local polish) as the right
method, which this endpoint mirrors at small scale.

## The editable interface

One editable function, `construct() -> (11,2) array in [0,1]^2`; evaluator and `n = 11` fixed. The
endpoint uses the prior rungs' SA engine to seed a soft-min (log-sum-exp) gradient polish via
scipy L-BFGS-B with an analytic gradient, annealing the sharpness `β` upward.

```python
import numpy as np
from itertools import combinations
from scipy.optimize import minimize
N = 11
TRIPLES = np.array(list(combinations(range(N), 3)))
# soft-min surrogate: -(1/beta) log sum exp(-beta * area_t)  ->  true min as beta -> inf
```
