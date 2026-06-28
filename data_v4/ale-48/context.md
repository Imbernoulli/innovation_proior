# Time-Indexed Crew Rostering

## Research question

A facility must staff a weekly schedule. There are `W` workers, `D` days, and `S`
work shifts per day (plus the implicit "day off"). We choose, for every worker `w`
and day `d`, a shift code `a[w][d] ∈ {0, 1, …, S}` — `0` means the worker rests that
day, and `s ≥ 1` means the worker works shift `s`. The chosen codes form a `W × D`
**roster grid**.

Each shift type `s` has a fixed length `HOURS[s]` and a clock start `START[s]`; its
clock end is `END[s] = START[s] + HOURS[s]` (which may exceed 24 when a shift crosses
midnight). Each `(day d, shift s)` slot has a **demand** `DEMAND[d][s]` (how many
workers it needs) and a **value** `VALUE[d][s]` (the reward per covered unit). The
business goal is to **cover as much high-value shift demand as possible while paying
as little overtime as possible**, subject to hard labour rules on each worker's week.

Concretely, maximise

```
OBJ = Σ_{d,s} VALUE[d][s] · min(cov[d][s], DEMAND[d][s])      (coverage value)
      − LAMBDA · (total overtime hours, summed over workers)   (overtime penalty)
```

where `cov[d][s]` is the number of workers assigned shift `s` on day `d`. Coverage is
**capped at demand** — staffing a slot beyond its demand is wasted overstaffing and
earns nothing. Overtime is the hours each worker works beyond a soft weekly cap
`MAXH`. This is a constrained assignment / rostering problem: it is NP-hard, there is
no known efficient exact solution at this size, and the benchmark scores a roster by
*how good* its objective is rather than by matching a unique optimum.

## Input / output contract

- **Input (stdin)**, whitespace-separated (line breaks are cosmetic):
  - `W D S`.
  - `HOURS[1..S]` — the length in hours of each shift.
  - `START[1..S]` — the clock start hour of each shift (`0..23`).
  - `MIN_REST MAXCONS MAXH HARDH LAMBDA` — required rest hours between consecutive
    shifts; max consecutive working days; soft weekly-hour cap; hard weekly-hour cap;
    overtime penalty per hour.
  - For each day `d = 0 … D-1`: a line `DEMAND[d][1..S]` then a line `VALUE[d][1..S]`.
  - For each worker `w = 0 … W-1`: `D` lines, line `d` holding `AVAIL[w][d][1..S]`
    (`1` if worker `w` may work shift `s` on day `d`, else `0`).
- **Output (stdout):** the roster grid as `W` lines, line `w` holding the `D` shift
  codes `a[w][0] … a[w][D-1]` (each in `{0, …, S}`), space-separated.
- **Time limit:** about 2 seconds wall-clock per instance. **Memory:** 256 MB.

A solution is **feasible** iff the output parses as exactly `W × D` integers each in
`[0, S]` **and** every worker's row obeys all four hard rules below. Anything else —
wrong count, an out-of-range code, a non-integer token, a rule violation, a missing
file — is **infeasible**.

The four hard per-worker rules (`s' ` is the next day's shift):

- **(A) Availability.** `a[w][d] = s ≥ 1` requires `AVAIL[w][d][s] = 1`.
- **(B) Minimum rest.** If a worker works shift `s` on day `d` and shift `s'` on day
  `d+1`, the rest gap `24 + START[s'] − END[s]` must be `≥ MIN_REST`. (A night shift
  ending after midnight followed by an early shift the next morning is the canonical
  violation.)
- **(C) Max consecutive days.** No worker may work more than `MAXCONS` days in a row
  without a day off.
- **(D) Hard weekly hours.** A worker's total assigned hours must be `≤ HARDH`.

## Background

The roster is a `W × D` grid of small categorical decisions tied together by
per-worker temporal rules and by shared coverage targets. Several approaches sit on
the table before committing:

- **Greedy fill-by-demand.** Process `(day, shift)` slots in decreasing value and, for
  each, assign the lowest-indexed still-available workers who can legally take it
  given what is already rostered. Always feasible, simple, and a natural operational
  baseline — but it commits early and leaves value on the table because a worker
  grabbed for a medium-value slot is then unavailable (rest/hours/consecutive rules)
  for a higher-value slot considered later. This is exactly the reference the scorer
  normalises against.
- **Per-worker "column" construction.** Each worker's whole-week pattern is a *column*
  (in the column-generation sense): a single feasible assignment of shifts to that
  worker's seven days. If we can, for one worker, compute the feasible pattern of
  maximum *marginal* coverage value (value of the residual, still-uncovered-up-to-
  demand slots it would fill, minus its overtime) given everything already rostered,
  then building the roster one strong column at a time is a much better construction
  than slot-by-slot greedy. Because a single worker's days form a short chain, the
  best feasible column can be found by an exact dynamic program over that worker's
  days whose state carries the rest rule (B), the consecutive-day rule (C) and the
  hour budget (D).
- **Local search / simulated annealing on the grid.** Once a roster exists, improve
  it by re-assigning individual cells. The decisive engineering lever is that a
  single-cell change `a[w][d] : s → s'` is **local in two independent senses**: its
  effect on the objective touches only the coverage buckets `cov[d][s]`, `cov[d][s']`
  and worker `w`'s overtime — an `O(1)` delta — and its effect on feasibility touches
  only worker `w`'s days `d−1, d, d+1`, the consecutive-day run around `d`, and that
  worker's hour total — an `O(1)` re-check confined to one worker's adjacent days. So
  thousands of moves per millisecond are possible without ever rescoring the grid or
  re-validating other workers.

The strong method combines the last two: **column-flavoured greedy construction**
to get a high-quality feasible roster fast, then **simulated annealing with
incremental objective deltas and the confined per-worker rest-rule check** to refine
it within the time budget.

## Evaluation settings

- **Instances.** A generator (`verify/gen.py`, parametrised by an integer `seed`)
  deterministically produces a week (`D = 7`) with the three canonical shifts
  (morning / day / night), `W ∈ [18, 40]` workers, per-day per-shift demands and
  values (night shifts carry a value premium), and heterogeneous worker availability
  (each worker blocks a random 10–35 % of slots). Total demand is tuned to exceed
  the supply one feasible roster can cover, so the assignment is genuinely contended —
  the regime where the column abstraction and local search recover the most value.
- **Scoring rule (deterministic, `verify/score.py`).** Read the instance and the
  submitted grid.
  - **Feasibility floor:** if the grid is malformed or violates any of rules (A)–(D)
    for any worker, the score is **`0`**.
  - Otherwise compute `OBJ` (coverage value minus overtime penalty) for the submitted
    grid, and `OBJ(G)` for the scorer's own deterministic **greedy fill-by-demand**
    roster `G` (recomputed inside the scorer, so the reference is reproducible and
    independent of the solver). The score is

    ```
    score = round(1 000 000 × OBJ / OBJ(G))     (feasible, OBJ(G) > 0)
    score = 0                                    (infeasible)
    ```

    The greedy reference scores exactly `1 000 000`; a roster that covers more
    high-value demand with less overtime scores strictly more. (Scores are clamped to
    be non-negative.)
- **Reported metric.** The mean score over a fixed seed set. The trivial *all-off*
  roster (everyone rests every day) is feasible but covers nothing, scoring `0`; it is
  the floor. A genuine solver should land well above `1 000 000` (≈ 1.15–1.45× the
  greedy reference on these instances).

## Code framework

A single self-contained C++17 program reading the instance from stdin and writing a
feasible roster grid to stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int W, D, S;
    if (scanf("%d %d %d", &W, &D, &S) != 3) return 0;
    vector<int> HRS(S + 1, 0), STc(S + 1, 0);
    for (int s = 1; s <= S; s++) scanf("%d", &HRS[s]);
    for (int s = 1; s <= S; s++) scanf("%d", &STc[s]);
    int MIN_REST, MAXCONS, MAXH, HARDH, LAMBDA;
    scanf("%d %d %d %d %d", &MIN_REST, &MAXCONS, &MAXH, &HARDH, &LAMBDA);
    vector<vector<int>> demand(D, vector<int>(S + 1, 0)), value(D, vector<int>(S + 1, 0));
    for (int d = 0; d < D; d++) {
        for (int s = 1; s <= S; s++) scanf("%d", &demand[d][s]);
        for (int s = 1; s <= S; s++) scanf("%d", &value[d][s]);
    }
    vector<vector<vector<char>>> avail(W, vector<vector<char>>(D, vector<char>(S + 1, 0)));
    for (int w = 0; w < W; w++)
        for (int d = 0; d < D; d++)
            for (int s = 1; s <= S; s++) { int x; scanf("%d", &x); avail[w][d][s] = (char)x; }

    // A feasible answer is ANY grid that obeys rules (A)-(D); the all-off grid
    // (everyone rests) is always feasible, so start there and never lose it.
    vector<vector<int>> A(W, vector<int>(D, 0));

    // TODO heuristic: column-flavoured greedy construction (per-worker DP picking the
    // best feasible weekly pattern by marginal coverage), then simulated annealing
    // that re-assigns one cell at a time with an O(1) objective delta and a
    // feasibility re-check confined to that worker's adjacent days. Keep `A` feasible
    // throughout and print the best feasible grid found.

    string out;
    for (int w = 0; w < W; w++)
        for (int d = 0; d < D; d++) { out += to_string(A[w][d]); out += (d + 1 < D ? ' ' : '\n'); }
    fputs(out.c_str(), stdout);
    return 0;
}
```
