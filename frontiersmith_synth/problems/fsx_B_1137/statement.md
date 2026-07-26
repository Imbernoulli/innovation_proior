# One-Way Conveyor: Warm-Up Windows vs. Tool Fatigue

`n` jobs must be released, one at a time, onto a conveyor that threads through `m`
fixed processing stations in a fixed order. The line has **no buffers that can
re-sequence work and no way to send a job backward**: once you choose the order jobs
enter the line, that exact same order is what **every** station sees, forever. You
choose this one global order; you cannot revisit the decision per station.

Each job has an integer `type` in `0..k-1`. Each station `m` maintains its own
rolling **tooling state**, built purely from the type sequence it has already seen
(which — because the line is one-way — is exactly a contiguous view of your global
order):

- **Warm-up window** (`horizon` `H_m`): for the job now arriving with type `t`, count
  `c` = how many of the **last `H_m` jobs** processed at this station (not counting
  the current one) also had type `t` (`0 <= c <= H_m`). The station is "warm" in
  proportion to `c/H_m`. Its cost for this job is
  `warm_cost[m][t] + (cold_cost[m][t] - warm_cost[m][t]) * (1 - c/H_m)`
  (fully warm when `c = H_m`, fully cold when `c = 0`).
- **Tool fatigue** (only on stations with `fatigue_on = true`): let `run` be the
  length of the **unbroken run** of identical-type jobs ending at the current job, in
  your global order. If `run` exceeds that station's `fatigue_threshold`, add
  `fatigue_cost[m][t] * (run - fatigue_threshold)` to the cost.

A station's total cost is the sum of these two terms over all `n` jobs it processes,
multiplied by its `weight`. The **objective to minimize** is the sum of all stations'
weighted totals. The scoring formula does **not** use each job's `proc_time` — it is
provided purely as job metadata (context, not signal): the line's true cost is driven
entirely by the setup-state cascade described above.

**Why this is a trap for local reasoning**: a long same-type run keeps warm-up-window
stations cheap but can blow past a *different* station's fatigue threshold — and
because one shared order feeds every station, there is no way to fix that after the
fact. A rule that looks at each job in isolation (e.g. sorting by `proc_time`) has no
way to see this cross-station tension at all.

## Public instance (stdin JSON)

```json
{
  "n": 60, "m": 4, "k": 4,
  "jobs": [ {"id": 0, "type": 2, "proc_time": 14}, ... ],
  "stations": [
    {"id": 0, "horizon": 3, "weight": 1.5,
     "cold_cost": [31, 44, 28, 37], "warm_cost": [2, 1, 2, 1],
     "fatigue_on": true, "fatigue_threshold": 3, "fatigue_cost": [5, 7, 4, 6]},
    ...
  ]
}
```
`jobs` has exactly `n` entries with distinct `id`s `0..n-1`. `stations` has exactly
`m` entries; `cold_cost`, `warm_cost`, `fatigue_cost` each have `k` entries (indexed
by job `type`). When `fatigue_on` is `false`, `fatigue_threshold`/`fatigue_cost` are
present but unused.

## Answer (stdout JSON)

```json
{"order": [17, 3, 44, 0, ...]}
```
A permutation of `0..n-1`: the order jobs enter the (single, shared) line.

## Feasibility

`order` must be a list of exactly `n` finite integers that is a permutation of
`0..n-1` (every id appears exactly once). Any violation — wrong length, duplicate id,
out-of-range id, non-integer — scores `0` on that instance.

## Scoring

The evaluator also computes a **baseline** `b`: the cost of the *as-given* order
(`order = [0, 1, ..., n-1]`, i.e. doing nothing). For a feasible answer with objective
`obj`:
```
r = min(1, 0.1 * b / obj)
```
so leaving the order untouched scores exactly `0.1`, and an order `k`× cheaper than
the as-given baseline scores `min(1, 0.1k)`. The reported `Ratio` is the mean of `r`
over 10 deterministic, seeded instances (including larger held-out ones). Infeasible
or malformed answers score `0` on that instance.

Your program reads one public instance JSON from stdin and writes one answer JSON to
stdout. It runs in an **isolated subprocess** and only ever sees the public instance.

## Constraints

`30 <= n <= 75`, `3 <= m <= 5`, `4 <= k <= 6`. Time limit 5s.
