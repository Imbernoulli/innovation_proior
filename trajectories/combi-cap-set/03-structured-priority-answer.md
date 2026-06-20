**Problem.** Construct a large cap set in `F_3^n`. The constructor returns the set; it is scored by
`|cap|` only if the set is a verified valid cap. This rung replaces the random order with a
deterministic **structured priority** over vectors, fed to the same greedy admission rule.

**Key idea.** Score every vector by a hand-designed priority that rewards the symmetry of `F_3^n` —
a bonus per matched **reflection pair** `el[i] == el[n−1−i]`, a gentle preference for a chosen
**weight-mod-3 layer** (Hamming weight `mod 3`, since the line condition `a+b+c ≡ 0` is itself about
coordinate sums mod 3), and a tiny coordinate-sum tie-break to make the order total — then greedily
add the highest-priority vector that does not close a line. A structured order gives the greedy fill
a geometry-aligned reason to prefer one point over another, rather than the geometry-blind counting
or random order of the earlier rungs.

**Why these choices.** Reflection symmetry and a coherent weight profile are the regularities the
known large caps exhibit, so they are the natural features to reward. The admission rule is
unchanged, so validity is still free. But this is a *guess*, not a derivation: a single structured
order is still one order, and if the chosen symmetries are even slightly misaligned with the optimal
cap it can land in the same band as — or below — best-of-thousands random restarts. That is the
intended lesson: the greedy-priority *skeleton* is correct (it is exactly the skeleton the strong
constructions use), but a *hand-designed* priority plateaus, motivating the endpoint where the
priority function is itself discovered by search.

**Hyperparameters / contract.** The priority weights are fixed (reflection bonus `1.0`, weight-layer
bonus `0.5·(3 − w mod 3)`, sum tie-break `0.01`). Deterministic given `n`. Output is a verified
valid cap. Cost is `O(3^n · n)` to score plus `O(|cap|^2 · n)` to build.

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
