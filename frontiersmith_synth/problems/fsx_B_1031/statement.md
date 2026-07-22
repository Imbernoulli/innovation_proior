# Founding-Instant Apical Dominance

A shoot axis differentiates **H lateral-bud positions**, one per developmental
tick: bud `i` differentiates exactly at tick `i` (`i = 1..H`). You control the
plant's auxin schedule — a sequence `r_1, r_2, ..., r_H`, each **0 or 1**, one
value per tick. `r_t = 1` means auxin is **released** ("low") at tick `t`;
`r_t = 0` means the plant is **actively producing** auxin ("high",
suppressing) at tick `t`. Producing auxin costs **growth budget**: the total
number of suppressed ticks may not exceed a given `BUDGET`.

## How a bud's fate is decided

Each bud `i` has a **commitment cost** `c_i` (given in the input).

- **Founding check.** If auxin is **high** (`r_i = 0`) right at bud `i`'s own
  founding instant, the bud is arrested at formation and **never** becomes an
  active branch — no matter what the schedule does afterward.
- **Commitment.** If auxin is **low** at founding (`r_i = 1`), the bud stays
  in play: it commits (becomes an active lateral branch) at the first tick
  `t >= i` such that the *cumulative* number of released ticks in `[i, t]`
  reaches `c_i`. If that sum never reaches `c_i` by tick `H`, the bud never
  commits.

Committed buds get a deterministic **commit order**: sort by commit tick
ascending, breaking ties (same tick) by bud id descending (the more recently
founded bud reports first).

Note what a *constant* schedule does: `r_t = 1` for all `t` lets every bud
pass its founding check, so the plant grows fully bushy — every bud commits,
roughly in creation order, though a bud with a larger `c_i` than its
neighbors can still commit later than them (or, if too little of its own
life remains, never at all). `r_t = 0` for all `t` fails every bud's
founding check — a single unbranched leader. Neither constant hits any other
target architecture; only a genuinely time-varying schedule can.

## Input (stdin)

```
H BUDGET K
c_1 c_2 ... c_H
t_1 t_2 ... t_K
```
`c_i` is bud `i`'s commitment cost (`c_i >= 2`). `t_1, ..., t_K` are the
**target** bud IDs (distinct, subset of `1..H`), listed in the **required
commit order** (`t_1` must commit before `t_2`, etc.). Every bud not in this
list should never commit.

## Output (stdout)

Exactly `H` integers `r_1 ... r_H`, each `0` or `1` (whitespace-separated,
any line layout). Infeasible if the count is wrong, any token isn't `0`/`1`,
or the number of suppressed ticks exceeds `BUDGET`.

## Scoring

Simulate the schedule to get the committed bud set and commit order. Let
`precision`/`recall` compare the committed set against the target set, and
combine them with an **F-beta score (beta² = 0.15)** — a false branch (an
arrested bud that leaks through) hurts far more than a missed target, since
even a few unwanted branches ruin the architecture. Let `order_score` be the
fraction of concordant pairs, among *correctly* committed target buds,
between the required order and their actual commit order (`1.0` if fewer
than 2 such buds). The objective is:

```
F = set_score(precision, recall) * (0.5 + 0.5 * order_score)
```

The grader also builds its own baseline `B` = the `F` achieved by
`r_t = 1` for all `t` (the "always release" constant schedule). Feasible
submissions score:

```
Ratio = min(1000, 100 * F / B) / 1000
```

so the constant-release baseline scores ≈0.1, and a schedule ~10× better in
`F`-terms caps at 1.0. Any infeasibility scores `Ratio: 0.0`.

## Example

For `H=6 BUDGET=4 K=2`, `c = [2,2,2,2,2,2]`, targets `2 5` (bud 2 must commit
before bud 5; buds 1, 3, 4, 6 must never commit), a legal artifact is:

```
0 1 0 0 1 0
```

meaning auxin is suppressed at ticks 1, 3, 4, 6 (4 suppressed ticks, exactly
at `BUDGET`) and released at ticks 2 and 5. Buds 1, 3, 4, 6 all fail their
founding check (`r_i = 0` at their own tick) and are permanently arrested —
good, they were never wanted. Bud 2 passes its founding check (`r_2 = 1`)
and accumulates released ticks from its own founding onward. Work through
the cumulative sums yourself to see exactly when (or whether) buds 2 and 5
end up committing, and in what order — this artifact is only illustrating
the input/output *shape*, not a claim that it scores well.
