**Problem.** Construct a large cap set in `F_3^n`. The constructor returns the set; it is scored by
`|cap|` only if the set is a verified valid cap. This rung attacks the single weakness of the floor
— the fixed, geometry-blind order — by trying many random orders and keeping the largest cap.

**Key idea.** Run the same greedy admission rule (add a point iff it does not close a line with an
already-admitted point, blocking the closing point of every line it forms) but offer the points in
a **uniformly random order**, and repeat over many independent restarts, returning the **best** cap
found. A random order destroys the accidental alignment between lexicographic enumeration and the
line-blocking structure, so on average it clears the `2^n` floor; taking the maximum over many
restarts reaches into the right tail of the cap-size distribution.

**Why these choices.** Every restart still produces a valid cap by construction — only the order
changes — so multi-start trades compute for size at zero risk to validity. The maximum over orders,
not the typical order, is what beats lexicographic: one random order is as arbitrary as the counting
order, but best-of-`k` samples the favorable tail. The method fixes the *bias* of a fixed order but
not its *blindness*: every order tried is uniform noise with no preference for structured or
low-blocking points, so the best-of-sample lags the optimum by a margin that grows with `n` — which
is exactly the limitation the next rung (a structured priority) must address.

**Hyperparameters / contract.** `starts` = number of random restarts (more is better, diminishing
returns); `seed` fixes the RNG so the returned best cap is reproducible. Output is a verified valid
cap. Cost is `O(starts · |cap|^2 · n)`.

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
