**Problem.** Construct a large cap set in `F_3^n`. The constructor returns the set; it is scored by
`|cap|` only if the set is a verified valid cap. This is the endpoint: feed the same greedy skeleton
the **priority function discovered by FunSearch** (Romera-Paredes et al., Nature 2024), which builds
the record `512`-cap in `n = 8`.

**Key idea.** Keep the greedy-priority skeleton of the previous rung — score every vector, add the
highest-priority valid one, block the closing point of each line, repeat — but replace the
hand-designed priority with the one an evolutionary program search (LLM + evaluator, millions of
samples) actually *discovered*. The discovered function is deeply non-linear and dimension-specialized:
it branches on the number of zeros in the vector, gives full-weight vectors a large additive boost,
applies *position- and count-dependent multiplicative* factors to each zero (`n·0.5^{in_el}`), and
stacks reflection-pair matches (`el[1]==el[-1]`, `el[2]==el[-2]`, `el[3]==el[-3]`) as `×1.5`
multiplicative bonuses. None of this is hand-derivable from symmetry; it was *selected* for reaching
`512`.

**Why these choices.** The earlier rungs proved the skeleton is correct and the bottleneck is the
priority. A human-written priority plateaus (my structured one fell below random multi-start); only a
priority discovered by search over the function space encodes the right regularities. The function is
verbatim from `github.com/google-deepmind/funsearch` (`cap_set/cap_set.ipynb`). It is tuned for
`n = 8`, so it is **mediocre at smaller `n`** — back near the `2^n` floor — and its entire value is
concentrated at `n = 8`, where it builds exactly `512` (improving the prior best `496`). This rung
reproduces the record, it does not beat it: going further would require running the evolutionary
search itself, not a single constructor.

**Hyperparameters / contract.** None tunable — the priority is the fixed discovered function. Valid
for `n ≥ 4` (it indexes `el[1..3]`). Deterministic given `n`. Output is a verified valid cap; at
`n = 8` it equals the explicit `512`-cap recorded in the FunSearch repository. Cost is `O(3^n · n)`
to score plus the greedy build.

```python
import itertools
import numpy as np

def priority(el, n):
    """Priority function discovered by FunSearch (Romera-Paredes et al., Nature 2024);
    verbatim from github.com/google-deepmind/funsearch cap_set/cap_set.ipynb.
    Builds a 512-cap in n=8, improving the previous best construction of size 496."""
    score = n
    in_el = 0
    el_count = el.count(0)
    if el_count == 0:
        score += n ** 2
        if el[1] == el[-1]:
            score *= 1.5
        if el[2] == el[-2]:
            score *= 1.5
        if el[3] == el[-3]:
            score *= 1.5
    else:
        if el[1] == el[-1]:
            score *= 0.5
        if el[2] == el[-2]:
            score *= 0.5
    for e in el:
        if e == 0:
            if in_el == 0:
                score *= n * 0.5
            elif in_el == el_count - 1:
                score *= 0.5
            else:
                score *= n * 0.5 ** in_el
            in_el += 1
        else:
            score += 1
    if el[1] == el[-1]:
        score *= 1.5
    if el[2] == el[-2]:
        score *= 1.5
    return score

def construct(n):
    """Greedy with the FunSearch-discovered priority. Builds a 512-cap at n=8."""
    av = np.array(list(itertools.product((0, 1, 2), repeat=n)), dtype=np.int32)
    powers = 3 ** np.arange(n - 1, -1, -1)
    priorities = np.array([priority(tuple(int(x) for x in v), n) for v in av], dtype=float)
    capset = np.empty((0, n), dtype=np.int32)
    while np.any(priorities != -np.inf):
        k = int(np.argmax(priorities))
        v = av[None, k]
        blocking = np.einsum('cn,n->c', (-capset - v) % 3, powers).astype(np.int64)
        priorities[blocking] = -np.inf
        priorities[k] = -np.inf
        capset = np.concatenate([capset, v], axis=0)
    return capset
```
