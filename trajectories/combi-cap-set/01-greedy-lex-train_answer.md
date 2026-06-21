I want a large cap set in $F_3^n$ — a subset of $\{0,1,2\}^n$ with no three distinct points $a,b,c$ summing to $0 \pmod 3$ — and before being clever I need a floor I can trust: the simplest construction that is *guaranteed* to emit a valid cap, so every later rung has something concrete to beat. The constraint that makes a cap fail is local in a way that hands me exactly such a construction. Over $F_3$ a line through two distinct points $a,b$ is $\{a, b, -a-b\}$, so any pair of points determines a *unique* third point $r = (-a-b) \bmod 3$ that would close a forbidden triple. That single fact means an incremental rule can never produce an invalid set: walk the points of $F_3^n$ in any fixed order and admit a point only if it is not the closing point of a line through two already-admitted points. Whatever order I use, the result is a cap by construction, because each admitted point was checked against all its predecessors at the moment it entered. Validity is therefore free; the only thing in play is *which order* I walk, and how good the cap that order produces turns out to be.

The method I propose for this rung is **greedy lexicographic** construction: list the $3^n$ vectors as if counting in base 3 — $00\ldots0,\ 00\ldots1,\ 00\ldots2,\ 00\ldots10,\ \ldots$ — and greedily take each vector if and only if it keeps the running set cap-valid. It is the right floor precisely because it is the most naive order imaginable: deterministic, parameter-free, no seed, no tuning. The mechanism is the universal greedy admission rule run on this fixed order. I keep a boolean array `is_blocked` indexed by $\mathrm{idx}(v) = \sum_j v_j\, 3^{\,n-1-j} \in [0, 3^n)$, the base-3 value of each vector. When I reach a vector $v$, if its index is already blocked I skip it — adding it would complete a line. Otherwise I admit it, and for every point $q$ already in the cap I mark the closing point $r = (-q - v) \bmod 3$ of the line $\{q, v, r\}$ as blocked, then mark $v$ itself blocked so it is never revisited. This is the same $O(|cap|^2 \cdot n)$ incremental block check the harness uses to verify caps, run forward as a *constructor*: because every admitted point survives a check against all predecessors, the output is correct no matter the order — there is nothing to verify after the fact.

What this design buys, and what it deliberately gives up, is worth being precise about, because the giving-up is the whole lesson of the rung. The lexicographic walk commits early and locally: it grabs a dense cluster of low-index points first, and the lines those points induce block out a structured swath of the remaining space, funnelling the greedy fill into a rigid pattern. Reasoning about it coordinate by coordinate predicts where it lands. On the last coordinate it takes $0\ldots00$ and $0\ldots01$, then rejects $0\ldots02$ because the three sum to $0$ — so at $n=1$ the cap is $\{0,1\}$, size $2$, which is the optimum there. That two-symbols-per-coordinate pattern cascades upward, and the cap comes out as essentially the vectors that avoid the third symbol once two are pinned — a set whose size is governed by powers of two, $2, 4, 8, 16, \ldots$, not by the richer algebraic structure the strong caps exploit. The counting order is an artifact of how I happen to enumerate tuples, not anything aligned with the geometry of $F_3^n$, and a fixed walk has no mechanism to do better: it cannot look ahead, cannot reconsider, cannot prefer a point that blocks few future candidates over one that blocks many. So I expect $2^n$ for $n = 1\ldots7$ — matching the optimum only at $n = 1, 2$ where $2^n$ *is* the optimum, and falling progressively short everywhere after ($8$ vs $9$, $16$ vs $20$, $32$ vs $45$, $64$ vs $112$, $128$ vs $236$), the gap widening because $2^n$ grows far slower than the true rate near $2.756^n$. The single lever this rung exposes and refuses to pull is the ordering itself; making that gap concrete is exactly what motivates trying *many* orders next.

```python
import itertools
import numpy as np

def construct(n):
    """Greedy lexicographic cap set in F_3^n. Add each vector in counting order iff it
    keeps the set cap-valid. Returns an (m, n) int array of distinct vectors."""
    av = np.array(list(itertools.product((0, 1, 2), repeat=n)), dtype=np.int32)
    powers = np.array([3 ** j for j in range(n - 1, -1, -1)], dtype=np.int64)
    raveled = av.astype(np.int64) @ powers
    is_blocked = np.full(3 ** n, False, dtype=bool)   # third point of each closed line
    cap = []
    capset = np.empty((0, n), dtype=np.int32)
    for v, idx in zip(av, raveled):
        if is_blocked[idx]:
            continue                                  # would complete a forbidden line
        if len(cap) >= 1:
            # block the closing point r = (-q - v) mod 3 of every line {q, v, r}
            blocking = ((-capset - v[None, :]) % 3).astype(np.int64) @ powers
            is_blocked[blocking] = True
        is_blocked[idx] = True
        cap.append(v)
        capset = np.concatenate([capset, v[None, :]], axis=0)
    return np.array(cap, dtype=np.int32)
```
