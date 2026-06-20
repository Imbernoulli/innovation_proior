# Randomized greedy multi-start cap set, distilled

Randomized greedy multi-start attacks the weakness of the lexicographic floor — its fixed,
geometry-blind order — by running the same greedy admission rule over many *uniformly random*
offer orders and returning the largest cap found. Each restart admits a point iff it does not
complete a line with an already-admitted point (blocking the completing point of every line it
forms), so every restart produces a valid cap; only the order is randomized. Best-of-many beats a
single arbitrary order because it samples the favorable right tail of the cap-size distribution.

## Problem it solves

Construct a large cap set in `F_3^n` (no three distinct points summing to `0 mod 3`), returning a
verified valid cap scored by its size. This method buys size over the trivial floor at zero risk
to validity, by trading compute for a best-of-sample over random orders.

## Key idea

Lexicographic order is pathological — it grabs a dense low-index cluster whose lines funnel the
greedy fill into the rigid `2^n` pattern. A uniformly random order scatters the early points, so
their lines scatter too and the fill packs more loosely-correlated points. Any single random order
is still arbitrary; taking the *maximum* over many independent restarts is what reliably clears the
floor.

## Why these choices

The maximum over orders, not the typical order, is the operative idea: best-of-`k` reaches the
right tail. The method fixes the *bias* of a fixed order but not its *blindness* — every order is
uniform noise with no preference for structured or low-blocking points — so its best-of-sample lags
the optimum by a margin that *grows* with `n` (the tail thins fast, so extra restarts buy little).
That growing gap is the motivation for biasing the order with a structured, then evolved, priority.

## Hyperparameters / contract

`starts` (number of restarts; more is better with diminishing returns) and `seed` (fixes the RNG so
the returned best cap is reproducible). Output is a verified valid cap; cost `O(starts · |cap|^2 ·
n)`.

## Measured results

Best cap over random orders (verified): `n=4 (5000 starts): 20` (= proven optimum), `n=5 (5000):
39`, `n=6 (3000): 77`, `n=7 (1000): 147`. Clears the lexicographic floor everywhere; reaches the
optimum only at `n = 4`; gap to the optimum grows with `n`.

```python
import itertools
import numpy as np

def _greedy_random_order(n, rng):
    av = np.array(list(itertools.product((0, 1, 2), repeat=n)), dtype=np.int32)
    av = av[rng.permutation(len(av))]                 # random offer order
    powers = np.array([3 ** j for j in range(n - 1, -1, -1)], dtype=np.int64)
    raveled = av.astype(np.int64) @ powers
    is_blocked = np.full(3 ** n, False, dtype=bool)
    cap, capset = [], np.empty((0, n), dtype=np.int32)
    for v, idx in zip(av, raveled):
        if is_blocked[idx]:
            continue
        if len(cap) >= 1:
            blocking = ((-capset - v[None, :]) % 3).astype(np.int64) @ powers
            is_blocked[blocking] = True
        is_blocked[idx] = True
        cap.append(v)
        capset = np.concatenate([capset, v[None, :]], axis=0)
    return np.array(cap, dtype=np.int32)

def construct(n, starts=5000, seed=0):
    """Randomized greedy multi-start: keep the largest cap over `starts` random orders."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(starts):
        c = _greedy_random_order(n, rng)
        if best is None or len(c) > len(best):
            best = c
    return best
```
