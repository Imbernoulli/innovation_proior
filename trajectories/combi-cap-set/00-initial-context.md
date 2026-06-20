## Research question

A **cap set** in `F_3^n` — the vectors of length `n` over the integers mod 3 — is a subset
containing no line: no three *distinct* points `a, b, c` with `a + b + c ≡ 0 (mod 3)`.
Equivalently (since over `F_3` a line through two points `a, b` is `{a, b, −a−b}`) it is a set
with no three-term arithmetic progression and no three collinear points. The question is the
oldest one in the subject: **how large can a cap set in `F_3^n` be?** The single thing being
designed here is a *constructor* — a program that emits one concrete subset of `F_3^n` — and it
is scored by the size of the cap set it returns, higher being better, with the hard requirement
that the returned set actually be a valid cap (verified, never assumed).

The maximal sizes are known exactly only for small `n`, and they grow fast:

| `n` | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| max `|cap|` | 2 | 4 | 9 | 20 | 45 | 112 | 236 | **≥ 512** |

The values through `n = 6` are proven optima; `n = 7` is `236`; and at `n = 8` the best known
lower bound is `512`, the cap set that **FunSearch** (Romera-Paredes et al., *Mathematical
discoveries from program search with large language models*, Nature 2024) discovered, improving
on the previous best construction of size `496`. The asymptotic story is that `|cap| ≤ c^n` with
`c < 3` (the Croot–Lev–Pach / Ellenberg–Gijswijt cap-set theorem gives `c ≈ 2.756`), so caps are
exponentially thin in `3^n`, and squeezing the constant in the exponent at each finite `n` is the
genuinely hard, partly-open part of the problem. This task runs the search live at tractable `n`
(reproducing the known optima where feasible) and runs the strong constructions out to `n = 7, 8`
to report what they reach.

## How the score is defined

The score is just `|cap|`, the number of points returned, and there is no partial credit and no
way to game it: a returned set is first checked to be a *valid* cap, and an invalid set scores
nothing. Validity is the entire subtlety. Checking it naively means looking at all triples,
`O(|cap|^3 · n)`; the harness uses the standard incremental check that runs in `O(|cap|^2 · n)`.
Each point of `F_3^n` is given an integer index `idx(v) = Σ_j v_j · 3^{n−1−j} ∈ [0, 3^n)`. The
set is built up one point at a time, carrying a boolean array `is_blocked` over all `3^n` indices;
a point is *blocked* if adding it would complete a line with two already-present points. When a
new point `p` is admitted, for every earlier point `q` the unique third point `r = (−p − q) mod 3`
that closes the line `{p, q, r}` is marked blocked. A returned set is a valid cap iff, replaying
it in order, no point is ever already blocked when it is added (and no duplicates). The harness
also cross-checks small instances with an independent `O(|cap|^3)` triple scan, so a fabricated or
buggy "cap" cannot slip through.

A few fixed yardsticks anchor every rung. The trivial floor at small `n` is `2^n` — there is an
elementary cap of that size (take all vectors whose coordinates avoid one symbol pattern), and a
naive constructor easily lands there. The proven optima `20, 45, 112, 236` at `n = 4..7` are the
mid-ladder targets. The honest ceiling is **`512` at `n = 8`**, the FunSearch record; reaching it
is the endpoint, and nothing here is expected to *beat* it, because `512` is itself the current
frontier of a large evolutionary program search.

| Reference point | `n` | `|cap|` |
|---|---|---|
| Trivial `2^n` floor | 4 / 5 / 6 / 7 | 16 / 32 / 64 / 128 |
| Proven optimum | 4 / 5 / 6 / 7 | 20 / 45 / 112 / 236 |
| **FunSearch record** (Nature 2024; prev. best 496) | 8 | **512** |

## Prior art before the first rung

- **Affine / product constructions (classical).** Small caps lift to larger ones by taking
  products and unions across coordinates; this is how the lower bounds `45, 112, 236` were first
  built. *Gap here:* these are bespoke algebraic constructions per `n`, not a single program, and
  they stall well below `3^n`; the exact optima at `n ≤ 6` and the `n = 7` value `236` came from
  exhaustive/dedicated computer search, not a closed form.
- **Croot–Lev–Pach (2017) → Ellenberg–Gijswijt cap-set theorem.** The polynomial method proves
  `|cap| ≤ O(2.756^n)`, a breakthrough *upper* bound. *Gap:* it bounds how large a cap can be but
  constructs nothing; the matching lower-bound constructions at finite `n` remain the open,
  search-driven side.
- **Greedy / priority-based search.** The natural constructive heuristic is to order the `3^n`
  vectors by some priority and add them greedily, skipping any that would close a line. The order
  is everything: lexicographic order gives a weak cap, random orders give a spread, and a
  cleverly *structured* priority can exploit the symmetry of `F_3^n`. *Gap:* hand-designed
  priorities plateau; finding an order that reaches the records is exactly what is hard.
- **FunSearch (Romera-Paredes et al., Nature 2024; code at github.com/google-deepmind/funsearch).**
  Pairs a pretrained LLM with an evaluator in an evolutionary loop that *evolves the priority
  function* fed to the greedy skeleton. For cap sets it discovered a priority function that builds
  a `512`-cap in `n = 8` (improving the prior best `496`) and one that reaches the known-best
  `1082` in `n = 9`. *Gap:* the discovered functions are the output of millions of LLM samples
  under evaluation; they are dimension-specialized (the `n = 8` function is tuned for `n = 8`) and
  reproducing them from scratch is a search problem, not a derivation — which is what leaves the
  whole ladder to climb.

## The fixed substrate

The harness is a thin, deterministic evaluator built on one greedy skeleton (the FunSearch
skeleton). It enumerates all `3^n` vectors, asks the constructor for a **priority** of each (or,
for the order-based rungs, an explicit order), then greedily adds the highest-priority vector that
is still unblocked, blocking out the third point of every line it closes, until nothing valid
remains. The returned set is then run through the `O(|cap|^2 n)` validity check (and the
independent `O(|cap|^3)` triple scan at small `n`), and its size is reported only if it is a
genuine cap. The enumeration, the line-blocking rule, and the validity check are frozen; the
verifier is shown for reference.

```python
import itertools
import numpy as np

def all_vectors(n):
    return np.array(list(itertools.product((0, 1, 2), repeat=n)), dtype=np.int32)

def is_cap_set(vectors):
    """Valid cap iff no three distinct points sum to 0 mod 3. O(c^2 n) incremental check."""
    if len(vectors) == 0:
        return True
    vectors = np.asarray(vectors, dtype=np.int32)
    c, n = vectors.shape
    powers = np.array([3 ** j for j in range(n - 1, -1, -1)], dtype=np.int64)
    raveled = vectors.astype(np.int64) @ powers
    if len(set(raveled.tolist())) != c:
        return False                                   # no duplicate points
    is_blocked = np.full(3 ** n, False, dtype=bool)
    for i, (v, idx) in enumerate(zip(vectors, raveled)):
        if is_blocked[idx]:
            return False                               # adding v would close a line
        if i >= 1:
            blocking = ((-vectors[:i, :] - v[None, :]) % 3).astype(np.int64) @ powers
            is_blocked[blocking] = True
        is_blocked[idx] = True
    return True
```

Every valid output is a set of distinct vectors in `{0,1,2}^n` with the cap property. There are no
other constraints — the constructor is free to return any cap, built by greedy ordering, random
restart, a structured priority, or an evolved priority function.

## Evaluation settings

A single deterministic objective per `n`: `|cap|` of the returned valid cap. Because a constructor
may search internally (random restarts), the harness fixes any stochastic run to a stated seed so
the number is reproducible, and reports the cap of the *returned* set. The fixed yardsticks — the
`2^n` floor, the proven optima `20/45/112/236`, and the FunSearch record `512` at `n = 8` — are
the rulers every rung is read against. The size is the whole result; an invalid set scores
nothing, and the verifier is run on every reported cap.
