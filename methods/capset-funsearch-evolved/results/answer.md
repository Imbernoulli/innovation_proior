# FunSearch-evolved priority cap set, distilled

This is the endpoint of the cap-set ladder: feed the greedy-priority skeleton the priority function
that an evolutionary program search (LLM + evaluator, millions of samples) actually *discovered*,
rather than a hand-designed one. The discovered function — verbatim from the FunSearch repository
(Romera-Paredes et al., Nature 2024) — builds the record `512`-cap in `F_3^8`, improving the
previous best construction of `496`. The skeleton is unchanged; the discovery is the function.

## Problem it solves

Construct a large cap set in `F_3^n` (no three distinct points summing to `0 mod 3`), returning a
verified valid cap scored by its size. The earlier methods proved the greedy-priority skeleton is
correct and the *priority function* is the bottleneck; this method supplies the function that
search found, reaching the strong size hand-design could not.

## Key idea

Keep the skeleton (score every vector, add the highest-priority valid one, block the completing
point of each line, repeat) but use the evolved priority. It is deeply non-linear and
dimension-specialized: it branches on the number of zeros in the vector, gives full-weight vectors
a large additive boost, applies position- and count-dependent *multiplicative* factors to each zero
(`n·0.5^{in_el}`), and stacks reflection-pair matches (`el[1]==el[-1]`, `el[2]==el[-2]`,
`el[3]==el[-3]`) as `×1.5` multiplicative bonuses. None of this is hand-derivable from symmetry — it
was *selected* for reaching `512`.

## Why these choices

A human-written priority plateaus (the structured one fell below random multi-start); only a
priority discovered by search over the function space encodes the right regularities. The function
is tuned for `n = 8`, so it is mediocre at smaller `n` — back at the `2^n` floor — and its entire
value is at `n = 8`, where it builds exactly `512`. This rung reproduces the record, it does not
beat it: exceeding `512` requires running the evolutionary search itself, not a single constructor.

## Hyperparameters / contract

None tunable — the priority is the fixed discovered function. Valid for `n ≥ 4` (it indexes
`el[1..3]`). Deterministic given `n`. Output is a verified valid cap; at `n = 8` it equals, as a
set, the explicit `512`-cap stored in the FunSearch repository (`cap_set/n8_size512.txt`).

## Measured results

Cap sizes (verified; `n ≤ 6` brute-checked): `n=4:16, 5:32, 6:64, 7:128` (dimension-specialized —
back at the floor), and **`n=8:512`** = the record. Independently confirmed the 512 points coincide
exactly with the repository's stored cap. The companion discovered `n = 9` function reaches the
known-best `1082` by the same skeleton.

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
