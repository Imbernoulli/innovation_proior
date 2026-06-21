Greedy lexicographic gave exactly $2^n$ at every dimension — $16, 32, 64, 128$ at $n = 4\ldots7$ — confirming the rigidity I expected: it is correct by construction but lands on the trivial power-of-two baseline and nothing more, with the gap to the optimum widening monotonically ($-1, -4, -13, -48, -108$). The diagnosis was sharp and it points straight at the fix. Validity was never the problem; the *order* was. Lexicographic is pathological in one specific way — it grabs a dense cluster of low-index points first, and that cluster's lines block a structured swath of the space, funnelling greedy into the power-of-two pattern. The cheapest way to attack a bad order is to stop trusting any single order at all.

The method I propose is **randomized greedy multi-start**. The admission rule is unchanged from the floor — add a point if and only if it does not close a line with an already-admitted point, blocking the closing point $r = (-q - v) \bmod 3$ of every line it forms — so every run still produces a valid cap by construction; I change only the sequence in which points are *offered*. Instead of base-3 counting order I draw a uniformly random permutation of the $3^n$ vectors and run greedy on it, and I do this over many independent restarts, keeping the **largest** cap I ever see:
$$\textsf{cap} \;=\; \arg\max_{\,k = 1 \ldots \texttt{starts}} \; \bigl|\,\textsf{greedy}(\pi_k)\,\bigr|, \qquad \pi_k \sim \text{Uniform}(S_{3^n}).$$
Why a random order should beat lexicographic, and why the *maximum* is the operative word, are two separate points and both matter. A uniformly random order has no low-index bias: the early points it grabs are scattered across $F_3^n$, so the lines they generate are scattered too, and the greedy fill tends to pack more loosely-correlated points before it runs out of room. That breaks the accidental alignment between enumeration and blocking structure, so *on average* a random order clears the $2^n$ floor. But any single random order is just as arbitrary as lexicographic — one shot is one shot. What makes the method work is taking the maximum over many independent orders: best-of-$k$ samples the favorable right tail of the cap-size distribution rather than its typical value, and the more starts I run the further into that tail I reach. The two controls are exactly this: `starts`, the number of restarts (more is better, with diminishing returns), and `seed`, fixing the RNG so the returned best cap is reproducible and its size can be checked rather than taken on faith.

I want to be honest about the regime, because the way this method succeeds *and* the way it ceilings are both the point. At small $n$ the space is tiny and the number of distinct greedy outcomes is limited, so a few thousand restarts can plausibly hit the optimum outright; at $n=4$ the optimum $20$ sits only modestly above the $16$ floor and $20$-caps are not rare among random orders, so I expect multi-start to reach $20$ and match the proven optimum — the first rung to touch a known-optimal value, bought purely by sampling. As $n$ grows the optimum pulls away: the cap-size distribution under random greedy concentrates well below it and the right tail thins fast, so even many restarts lag by a margin that *grows* with $n$ (I expect the high $30$s at $n=5$ against $45$, the mid $70$s at $n=6$ against $112$, the $140$s at $n=7$ against $236$). That growing gap is the real lesson I am setting up. Multi-start fixes the *bias* of a fixed order but does nothing about its *blindness*: every order it tries is still uniform noise, with no preference for points sitting in symmetric or structured positions, no preference for points that block few future candidates over points that block many. It is a lottery over orders, and a lottery cannot learn the geometry. Closing the gap will require *biasing* the order with a structured priority that reflects the symmetry of $F_3^n$ — the move to priority-function greedy — and that is the next rung.

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
