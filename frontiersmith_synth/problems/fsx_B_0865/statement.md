# Coupling-Matrix Balancer (Format B, isolated)

You are given an `n x n` strictly-positive coupling matrix `M` and target row
sums `r` and column sums `c` (`sum(r) == sum(c)`). Your job is to find
diagonal scalings `u, v > 0` so that the rescaled matrix `diag(u) M diag(v)`
has row sums close to `r` and column sums close to `c`. This is the classic
matrix-balancing (Sinkhorn/RAS) problem, but you cannot compute `u, v`
directly: you must submit a **bounded list of moves**, and the grader
replays those moves itself, starting from `u = v = 1`. The number of moves
you may submit is capped by a per-instance **operation budget** `B` — the
grader's replay is the only thing scored, so this cost is fixed and machine
independent, no matter how you decide on your move list internally.

Some instances hide a **near-reducible block structure**: after a secret
permutation, `M` is close to block-diagonal — index groups that couple
strongly to each other but only very weakly across groups. On those, moves
that only ever touch every row/column uniformly correct the weak cross-group
imbalance at a rate proportional to the (tiny) cross-group coupling strength,
so a *global* sweep needs vastly more moves than the budget allows. Other
instances have no such structure (a genuinely well-mixed matrix) — for those,
a plain sweep already does about as well as anything can.

## Public instance (stdin JSON)
```json
{"n": <int>, "budget": <int>, "M": [[...]], "r": [...], "c": [...]}
```
`M` is `n x n`, every entry `> 0`. `r`, `c` are length-`n`, every entry `> 0`,
`sum(r) == sum(c)`.

## Answer (stdout JSON)
```json
{"ops": [ {"type": "row", "omega": 1.0},
          {"type": "col", "omega": 1.0},
          {"type": "block", "axis": "row", "omega": 1.7, "groups": [[0,3,7],[1,2,4,5,6]]},
          ... ]}
```
`ops` is a list of **at most `budget`** moves. `omega` is a finite number in
`[0, 3]` for every move (SOR relaxation factor; `omega=1` is the exact
projection step, `omega>1` over-relaxes). Move types, applied left to right
to the running scalings `u, v` (both start at `1`):

- `{"type":"row","omega":w}` — for every row `i`, let `s_i = u_i * (M[i,:]·v)`
  be its current sum; set `u_i *= (r_i / s_i) ** w`.
- `{"type":"col","omega":w}` — symmetric on columns using `c`, `v`.
- `{"type":"block","axis":"row","omega":w,"groups":G}` — `G` must be a
  **partition** of `{0,...,n-1}` (every index in exactly one group). For each
  group `g`, let `S_g = sum_{i in g} u_i*(M[i,:]·v)` and `R_g = sum_{i in g} r_i`;
  multiply **every** `u_i, i in g` by the **single shared** factor
  `(R_g / S_g) ** w`. `axis:"col"` is symmetric on `v`, `c`.

Any malformed answer, `len(ops) > budget`, an out-of-range/non-finite
`omega`, a `groups` list that is not an exact partition of `{0,...,n-1}`, or
a replay that produces a non-finite scaling, scores this instance **0**.

## Objective & scoring
After replay, let `s_i, t_j` be the final row/column sums. The **imbalance**
is the mean relative deviation from target:
```
imbalance = 0.5 * ( mean_i |s_i - r_i| / r_i  +  mean_j |t_j - c_j| / c_j )
```
Lower is better (minimize). Per instance,
`score = min(1, 0.1 * B_ref / imbalance)`, where `B_ref` is the imbalance the
grader itself gets from the plain textbook recipe (alternating row/column
sweeps, `omega=1`, using the full budget) — so matching that recipe scores
`~0.1`, and the final reported score is the mean over 10 fixed, seeded
instances of varying size, block structure, and budget.

## Suggested strategies (increasing sophistication)
- **Minimal effort** — one row sweep, one column sweep, stop.
- **Plain alternating Sinkhorn** — spend the whole budget on `row`/`col`
  sweeps with `omega=1`.
- **Over-relaxed sweeps only** — tune `omega` on plain row/column moves,
  without ever inspecting the coupling pattern for structure.
- **Block-aware** — infer the group partition from where `M`'s mass
  concentrates, spend a few plain sweeps settling within-group ratios, then
  spend the rest of the budget on over-relaxed `block` moves that correct
  the slow cross-group imbalance directly — falling back to the plain
  recipe when no real block structure is detected.
