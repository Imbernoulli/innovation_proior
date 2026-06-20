**Problem.** Place `n = 11` points in `[0,1]^2` to maximize the minimum area over all `165`
triangles. Score = exact min triangle area. Record `1/27 = 0.037037` (Goldberg). This rung is the
simplest search: random multi-start — sample many uniform configurations, keep the best min-area.

**Key idea.** Replace "name one clever configuration" with "try a great many and let the evaluator
pick." Draw `11` points uniformly in the square, score the exact minimum over all triples, and keep
the running maximum across millions of independent draws. No state, no schedule, no way to get stuck
— just the max of many independent samples. To afford millions of trials, vectorize: precompute the
`165` index triples once and evaluate a whole batch of configurations with array cross-products,
taking the per-config min and the batch max.

**Why these choices.** Random multi-start is the honest zero-knowledge baseline for the search era
of the ladder — it needs only a sample count and is trivially correct. It is expected to be weak: a
configuration is a point in `22` dimensions, good min-area configurations are a vanishing fraction
of that space, and a random draw almost always has *some* near-collinear trio that makes a thin
sliver and sets a low minimum. So the best of millions of draws beats the typical draw by a wide
margin but stalls well short of the record, because the method has no way to *improve* a promising
configuration — every good draw is discarded the moment the next is scored. That wasted information
is exactly what the next rung fixes.

**Hyperparameters / contract.** `TRIALS = 4,000,000` uniform configurations, processed in batches
of `50,000`; fixed seed `0`. Output is the single best configuration found (exact min-area
verified). All coordinates in `[0,1]` by construction.

```python
import numpy as np
from itertools import combinations

N = 11
TRIPLES = np.array(list(combinations(range(N), 3)))      # (165, 3)
I, J, K = TRIPLES[:, 0], TRIPLES[:, 1], TRIPLES[:, 2]

def batch_min_area(P):                                    # P:(B,N,2) -> (B,)
    a, b, c = P[:, I, :], P[:, J, :], P[:, K, :]
    cross = (b[..., 0]-a[..., 0])*(c[..., 1]-a[..., 1]) \
          - (c[..., 0]-a[..., 0])*(b[..., 1]-a[..., 1])
    return 0.5 * np.abs(cross).min(axis=1)

def construct():
    rng = np.random.default_rng(0)
    best, bestP = -1.0, None
    TRIALS, BATCH, done = 4_000_000, 50_000, 0
    while done < TRIALS:
        P = rng.random((BATCH, N, 2))
        areas = batch_min_area(P)
        idx = int(areas.argmax())
        if areas[idx] > best:
            best, bestP = float(areas[idx]), P[idx].copy()
        done += BATCH
    return bestP
```
