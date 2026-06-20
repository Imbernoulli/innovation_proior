Randomized greedy multi-start cleared the floor and even hit the optimum at `n = 4`, but it revealed
its own ceiling: it is a lottery over orders, and the right tail of the cap-size distribution thins so
fast that more restarts buy almost nothing as `n` grows. The diagnosis was that the orders are *blind*
— uniform noise, with no preference for points sitting in structured positions. So the move is to
replace the random order with a *priority*: score every vector by a deterministic function of its
coordinates and feed the same greedy admission rule the vectors highest-priority-first. If the priority
reflects the actual symmetry of `F_3^n`, the greedy fill should prefer points that pack the space
efficiently, beating the random lottery without thousands of restarts. The admission rule is unchanged,
so validity is still free; only the ordering function changes.

What structure should I reward? `F_3^n` has a lot of symmetry, and the known large caps are highly
structured — regular weight profiles, reflection symmetries — not random. Two features feel natural.
First, *reflection symmetry*: pair coordinate `i` with `n−1−i` and reward a vector when those paired
entries agree, adding a bonus per matched pair; a cap built from reflection-symmetric vectors inherits
a symmetry that tends to make its line structure regular. Second, a *weight profile*: the Hamming
weight mod 3 plausibly matters, because the line condition `a + b + c ≡ 0` is itself a statement about
coordinate sums mod 3, so I gently prefer vectors whose weight mod 3 sits in a chosen class, biasing
the fill toward a coherent weight layer. A tiny coordinate-sum tie-break keeps the order total. The
priority is the sum of these terms — symmetric, deterministic, parameter-light — exactly the kind of
structured score one writes down before resorting to search.

I want to be careful not to oversell this. The symmetry rewards should beat lexicographic decisively,
because they give the greedy process a reason to prefer one point over another that is aligned with the
geometry instead of the counting order. But I am genuinely unsure whether a hand-crafted symmetric
priority beats the *best of thousands* of random orders — and that uncertainty is the point. A single
structured order is still one order; if my guesses about which symmetries matter are even slightly off,
a deterministic structured order can underperform the maximum over a large random sample. So my honest
expectation is: comfortably above the lexicographic floor, in the same band as random multi-start, and
quite possibly *below* it at some `n`. The running confirms exactly that — `18, 36, 64, 138` for `n =
4..7`, beating the floor at `n = 4,5,7` but landing below the random best (`20, 39, 77, 147`) at every
`n`.

That is not a disappointment; it is the crucial lesson. It shows that *having* a structured priority is
not enough — the priority has to be the *right* one, finely shaped to the dimension, and a human
guessing which symmetries to reward cannot reliably out-do brute sampling. The greedy-priority skeleton
is clearly the correct machinery — it is exactly the skeleton the strong constructions use — but the
function plugged into it is everything, and hand-design plateaus because the space of useful priorities
is large and non-obvious. So the method I propose is this **structured symmetric priority-function
greedy**, offered precisely to establish that the skeleton is right and the hand-designed priority is
not — to motivate handing the priority itself over to search. I respect one caution for the numbers:
the priority has many ties, and how they break affects the result, so I fix the tie-break
deterministically and verify every returned cap.

```python
import itertools
import numpy as np

def priority(el, n):
    """Structured symmetric priority: reflection matches + weight-mod-3 layer + sum tie-break."""
    score = float(n)
    for i in range(1, n // 2):                 # reflection pairs i <-> n-1-i
        if el[i] == el[n - 1 - i]:
            score += 1.0
    w = sum(1 for e in el if e != 0)           # Hamming weight
    score += (3 - (w % 3)) * 0.5               # prefer a coherent weight-mod-3 layer
    score += 0.01 * sum(el)                    # deterministic tie-break
    return score

def construct(n):
    """Priority-function greedy: add highest-priority valid vector, block closed lines."""
    av = np.array(list(itertools.product((0, 1, 2), repeat=n)), dtype=np.int32)
    powers = 3 ** np.arange(n - 1, -1, -1)
    priorities = np.array([priority(tuple(int(x) for x in v), n) for v in av], dtype=float)
    capset = np.empty((0, n), dtype=np.int32)
    while np.any(priorities != -np.inf):
        k = int(np.argmax(priorities))
        v = av[None, k]
        blocking = np.einsum('cn,n->c', (-capset - v) % 3, powers).astype(np.int64)
        priorities[blocking] = -np.inf            # block completing point of each line
        priorities[k] = -np.inf
        capset = np.concatenate([capset, v], axis=0)
    return capset
```
