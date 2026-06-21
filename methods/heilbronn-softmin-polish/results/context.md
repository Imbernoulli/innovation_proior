# Context: Heilbronn triangle problem at n=11 (endpoint)

## Research question

Place `n = 11` points in `[0,1]^2` to maximize the **minimum** triangle area over all
`C(11,3) = 165` triples. The constructor emits 11 points; the exact minimum triangle area is the
score (higher better).

This is the endpoint rung. The prior rungs reach a configuration with score `0.962` of the record.
The question: starting from such a configuration, how does one drive the minimum triangle area the
rest of the way onto the exact optimum?

## How the score is defined

```
score(P) = min over all 165 triples of  |(b-a) x (c-a)| / 2,   all points in [0,1]^2.
```

## Record and yardstick

Unit-square record at `n = 11`: `Δ(11) = 1/27 = 0.037037` (Goldberg 1972, conjectured optimal).
Prior rungs: 11-gon `0.021456` (0.579); random `0.010872` (0.294); simulated annealing `0.035639`
(0.962). The endpoint targets the record itself: at `n = 11` the value `1/27` is believed optimal,
so the goal is to match it to floating-point precision.

DeepMind's AlphaEvolve (arXiv:2506.13131) improved Heilbronn records for the *sibling* containers —
unit-area triangle (`n=11`: `0.036 → >0.0365`) and unit-area convex region (`n=13,14`) — not the
unit square, via a search-plus-refine recipe (evolutionary global search plus local polish).

## The editable interface

One editable function, `construct() -> (11,2) array in [0,1]^2`; evaluator and `n = 11` fixed. The
prior rungs supply a simulated-annealing engine over the configuration, and the endpoint may seed
from the previous rung's best annealed configuration.

```python
import numpy as np
from itertools import combinations
from scipy.optimize import minimize
N = 11
TRIPLES = np.array(list(combinations(range(N), 3)))
```
