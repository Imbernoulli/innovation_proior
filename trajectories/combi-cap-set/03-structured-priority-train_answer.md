Random multi-start cleared the floor at every $n$ and even reached the proven optimum $20$ at $n = 4$ — the first rung to hit a known-optimal value — but the feedback made its ceiling plain: it is a lottery over orders, $39, 77, 147$ against $45, 112, 236$, and the right tail of the cap-size distribution thins so fast that more restarts buy almost nothing as $n$ grows. The diagnosis was that the orders are *blind* — uniform noise, with no preference for points in structured positions. So the move is to stop sampling and start *biasing*: replace the random order with a deterministic priority that scores every vector by its coordinates and feeds the same greedy admission rule the vectors in highest-priority-first order. The admission rule is unchanged, so validity stays free; only the ordering function changes.

The method I propose is a **structured symmetric priority-function greedy**, and the design question is which structure of $F_3^n$ to reward. The known large caps are highly structured objects with regular weight profiles and reflection symmetries, so I encode two features. First, *reflection symmetry*: pair coordinate $i$ with coordinate $n-1-i$ and award a bonus when those paired entries agree, $\mathtt{el}[i] = \mathtt{el}[n-1-i]$, because a cap built from reflection-symmetric vectors inherits a symmetry that tends to regularize its line structure. Second, a *weight-mod-3 layer*: the Hamming weight $w$ (number of nonzero coordinates) taken $\bmod 3$ should matter for how vectors interact under the line condition $a + b + c \equiv 0$, since that condition is itself a statement about coordinate sums mod 3, so I gently prefer vectors whose weight mod 3 sits in a chosen residue class, biasing the fill toward a coherent weight layer rather than a mix. The priority is the additive sum of these terms plus a tiny coordinate-sum tie-break to make the order total:
$$\mathrm{priority}(\mathtt{el}, n) \;=\; n \;+\; \sum_{i=1}^{\lfloor n/2\rfloor - 1} \mathbf{1}\!\left[\mathtt{el}[i] = \mathtt{el}[n-1-i]\right] \;+\; 0.5\,(3 - w \bmod 3) \;+\; 0.01 \sum_j \mathtt{el}[j],$$
with the reflection bonus $1.0$ per matched pair, the weight-layer bonus $0.5\,(3 - w \bmod 3)$, and the sum tie-break $0.01$. The skeleton then runs the same way as before: I score all $3^n$ vectors once, repeatedly take the highest-priority vector still in play, set the priority of the closing point of each line it forms to $-\infty$ (the blocking step), set its own priority to $-\infty$, and add it to the cap — greedy on a geometry-aligned order instead of a blind one.

I want to be careful not to oversell this, because the honest expectation is itself the lesson. The symmetry rewards give greedy a *reason* to prefer one point over another that is aligned with the geometry instead of the counting order, so this should beat lexicographic decisively. But a single structured order is still a single order, and the reflection-and-weight priority is a *reasonable guess, not a derived one*. If my chosen symmetries happen to align with the optimal cap's structure I win; if my guesses about which symmetries matter are even slightly off, a deterministic structured order can easily underperform the *maximum over thousands of random orders*, which is a strong baseline. So I expect to clear the floor everywhere, land in the same general band as random multi-start, and quite possibly come in *below* it at several $n$ — at $n = 4$ I would hope to beat $16$ but would not bet on matching the $20$ that sampling found. If it comes out that way it is not a disappointment but the crucial result of the ladder: *having* a structured priority is not enough; the priority has to be the *right* one, finely shaped to the dimension, and a human guessing at which symmetries to reward cannot reliably out-do brute sampling. The greedy-priority skeleton is clearly the correct machinery — it is exactly what the strong constructions use — but the function plugged into it is everything, and hand-design plateaus because the space of useful priorities is large and non-obvious. What reaches the records is not a cleverer human guess but a priority *discovered by search* over the function space, tuned to the specific $n$, encoding regularities a person would not write down. Establishing that the skeleton is right and the hand-designed priority is not is exactly the job of this rung, and it is what motivates handing the priority itself to search in the final one.

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
        priorities[blocking] = -np.inf            # block closing point of each line
        priorities[k] = -np.inf
        capset = np.concatenate([capset, v], axis=0)
    return capset
```
