# Calling the Capacity Cliff from a Reuse-Distance Sample

## Problem

A workload with working-set size `N` (the number of distinct items it
repeatedly touches) runs against an LRU-managed cache with raw capacity `C`
lines and `A`-way associativity. As `N` grows, the miss rate stays low while
the working set fits in the cache and then rises toward 1 once it doesn't —
a **capacity cliff**. In this workload family, an access's *reuse distance*
(how many other distinct items were touched since that item's previous
access) scales with `N` itself — a bigger footprint pushes reuses farther
apart — so whether `N` is "small" or "large" for this cache is governed by
`N` relative to an **effective capacity**

```
K = C * A / (A + 1)
```

(associativity `A` recovers a fraction `A/(A+1)` of the raw capacity from
conflict misses — this formula is exact and public). What is **not** public
is exactly how sharply the miss rate transitions from "fits" to "misses" as
`N` crosses `K` — that steepness is a property of the workload's locality,
and every miss-rate measurement you are given was taken far below `K`,
where the true curve is too flat and noise-dominated to reveal it alone.

You additionally get a **reuse-distance sample**: `M` normalized reuse
distances measured directly from traces, independent of any one `N`. Their
distribution's shape is scale-free — the same regardless of which `N` it
was measured under — and it fully determines the transition's steepness.

Your job: emit a closed-form expression for the miss rate as a function of
`n`, accurate not just near the training range but far past it, where the
cliff has already happened.

**Illustrative FORM only — NOT the hidden law (unrelated shape, do not
pattern-match it):** `12.5 + 3.2*log(n) - 0.4*sqrt(n)`

## Input (stdin)

```
t C A M n_train
d_1 d_2 ... d_M
N_1 missrate_1
N_2 missrate_2
...
N_{n_train} missrate_{n_train}
```

`t` is the test id (informational). `C`, `A` are integers. `d_1..d_M` are the
`M` normalized reuse-distance samples (floats). Then `n_train` rows follow,
each a working-set size and its measured miss rate (a finite-sample rate in
`[0,1]`, so it can read exactly `0`).

## Output (stdout)

One line: a closed-form Python-syntax expression for the miss rate in the
single variable `n`. Allowed: `+ - * / **`, unary `-`, numeric constants, and
the functions `sqrt log exp sig tanh absv`. No other names are accepted.

## Feasibility

The output must parse under the grammar above (only the listed names/
functions, finite numeric constants, at most 60 expression-tree nodes) and
must evaluate to a finite real number at every held-out `n`. Any violation
scores `0`.

## Scoring (deterministic, maximization)

Your expression is evaluated on a **held-out set of working-set sizes**,
regenerated inside the grader, that straddle and cross the capacity cliff —
every one far beyond the largest training `N`. Let `p_i` be your prediction
and `t_i` the true (noisy, finite-sample) miss rate at held-out point `i`:

```
metric   = mean_i min(1, |p_i - t_i|)             # clipped absolute error
O        = metric * (1 + LAMBDA * nodes)          # nodes = expression size
baseline = the same metric for the constant predictor mean(train missrate)
Ratio    = min(1000, 100 * baseline / O) / 1000
```

Lower held-out error raises `Ratio` (capped at `1.0`); reproducing the
constant baseline scores about `0.1`. `LAMBDA` is a small fixed parsimony
weight. Non-finite predictions score `0`.

## Why the visible log is a trap

Every training row sits far below `K`, where the true miss rate is a tiny
power-law tail that finite sampling mostly rounds down to noise near `0` —
the log looks flat no matter how sharp the real cliff is. A curve fit to
that log alone keeps extrapolating near `0` long past where the true curve
has already climbed to nearly `1`. The reuse-distance sample is the only
signal that carries the steepness across scales; combined with the public
`K` formula, it lets you place both the location and shape of the cliff.

## Constraints

Time limit 5 s, memory 512 MB. `M` and `n_train` are small (well under a
thousand rows total). Held-out finite-sample noise leaves irreducible error,
so even the correct law does not reach `Ratio = 1.0`.
