# Riverdelta Levee Network: Budgeted Gauge Recalibration

A river-delta flood authority runs `N` water-level gauges. Each gauge `i` wants to
settle at its own calibration target `c_i` with confidence weight `a_i`. Some gauges
share a sluice channel: a **coupling edge** `(i, j, w)` means `i` and `j` must also
be reconciled against each other with stiffness `w`, since water genuinely flows
between them. The coupling **graph** is heterogeneous — most gauges are
hydraulically isolated, while a minority sit inside one or two tightly-coupled
channel clusters.

Formally, minimize the separable-plus-coupling quadratic

```
f(x) = sum_i a_i * (x_i - c_i)^2  +  sum_{(i,j) in E} w_ij * (x_i - x_j)^2
```

You have a **fixed budget of `K` coordinate-update steps**. Each step names one
gauge index `i`; that gauge is reset to its **exact coordinate optimum** given the
current values of every other gauge:

```
x_i  <-  (a_i * c_i + sum_{j in nbrs(i)} w_ij * x_j) / (a_i + S_i)
```

where `S_i` is the sum of `w` over `i`'s incident edges. This move can only
decrease or hold `f`, never increase it. Your program chooses the **order**
(length exactly `K`, repeats allowed) in which gauges get updated, starting from a
given `x0`. `K` is only slightly larger than `N`, so most gauges get exactly one
visit and there is a small pool of "extra" visits to spend. Spending them on an
already-settled isolated gauge is pure waste; a stiff, poorly-conditioned coupled
cluster generally needs several extra visits before it approaches its optimum.

## Program contract

Standalone program: read ONE JSON object from **stdin**, write ONE JSON object to
**stdout**. Runs isolated; sees only the public instance.

### Input (stdin)

```json
{
  "name": "delta104",
  "n": 20,
  "K": 33,
  "a":  [0.71, 1.83, 0.22, ...],
  "c":  [4.1, -7.2, 0.9, ...],
  "x0": [-3.4, 2.0, 8.8, ...],
  "edges": [[3, 11, 41.2], [11, 15, 0.9], ...]
}
```

- `n`, `K`: positive integers, `K > n`.
- `a`, `c`, `x0`: length-`n` lists of floats; every `a_i > 0`.
- `edges`: list of `[i, j, w]` triples, `0 <= i < j < n`, `w > 0`, undirected. A
  gauge with no incident edges is isolated.

### Output (stdout)

```json
{ "order": [0, 11, 3, 11, 15, 3, ...] }
```

`order` must be a list of **exactly `K`** integers, each in `[0, n)`. Any deviation
(wrong length, non-integer or out-of-range index), a crash, a timeout, or non-JSON
output makes that instance score `0.0`.

## Scoring (deterministic)

For each instance the evaluator computes, itself:

- `f_lb`   — the TRUE unconstrained global minimum of `f` (exact linear solve) — a
  generally-unreachable ideal within a finite budget on the stiff clusters,
- `f_base` — `f` reached by the trivial construction "one full pass over every
  gauge in index order `0..n-1`, then waste the remaining `K-n` surplus steps
  re-updating gauge 0" — every gauge gets its mandatory single visit, but the
  surplus is spent with zero regard for the coupling structure, which isolates
  the score on exactly how the *surplus* is spent,
- `f_cand` — `f` reached by **replaying your own `order`**, step by step from
  `x0`, using the closed-form update above.

```
r = clamp( 0.1 + 0.9 * (f_base - f_cand) / max(1e-9, f_base - f_lb), 0, 1 )
```

Matching the "one pass, then waste the surplus" baseline scores ≈`0.1`; reaching
the true optimum scores `1.0`; doing worse than the baseline scores below `0.1`.
Several instances plant a stiff coupled cluster that cannot be fully settled within
budget `K`, so even an excellent allocator stays comfortably below `1.0` — there is
real headroom above any fixed reference strategy.

The reported **Ratio** is the mean of `r` over 10 seeded instances (varying `n`,
cluster count/size/stiffness — some deliberately hard, some larger held-out);
**Vector** lists the per-instance scores.

## Suggested strategies

1. **One pass, waste the surplus** (trivial): visit every gauge once in index
   order, then burn the rest re-updating an already-settled gauge.
2. **Cyclic sweep** (greedy): visit `0, 1, ..., n-1, 0, 1, ...` until the budget
   runs out — every gauge, isolated or coupled, revisited equally, blind to
   structure.
3. **Diagnose, then spend where it counts** (strong): use the edges and current
   gradients to pick, at every step, whichever not-yet-settled gauge would
   shrink `f` most right now (largest `g_i^2 / (4*(a_i+S_i))`). This gives
   every isolated gauge its one needed visit, then concentrates the whole
   surplus on whichever coupled gauge is currently steepest.
