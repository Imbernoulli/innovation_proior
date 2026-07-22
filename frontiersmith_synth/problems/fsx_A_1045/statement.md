# Subsystem-Robust Release Schedule

## Problem
A release plan has `N` jobs `1..N` with positive durations, `M` identical build
machines, and a precedence DAG (job `i` may depend on some jobs `< i`; jobs are given
in a topological order). You must output a **priority list**: a permutation of
`1..N`. It is then *replayed*, not trusted: a deterministic list-scheduling
simulator runs it on `M` machines — whenever a machine is free, it starts the
highest-priority job among those whose predecessors have all finished — and reports
the resulting makespan (completion time of the last job).

The catch: the plan comes with `K` **published perturbation vectors** (`K` may be
`0`, meaning no shocks are published and only the nominal schedule is scored).
Vector `k` names a *subsystem* — an explicit subset of job ids, possibly spanning
several different sub-projects — and an inflation factor `> 1`. If shock `k`
occurs, every job in that subsystem (and only that subsystem) runs `ceil(factor_k
* d_i)` time units instead of `d_i`; nothing else changes. You do not know which
shock (if any) will actually occur, so your list is **replayed once per published
vector**, each time with only that vector's subsystem inflated, and your score is
driven by the *worst* of these `K` recomputed makespans — not the nominal one.

Padding slack along the nominal critical path does not defend against this: a shock
inflates a whole subsystem at once, and a subsystem need not be the nominal critical
path — it can be a chain of jobs that only becomes long, or only becomes badly
queued behind other work, once its own shock hits. A robust list is one where no
long chain of jobs ends up owned by, and stuck behind, a single dangerous
subsystem.

## Input (stdin)
```
N M K
d_1 p_1 pred_1,1 ... pred_1,p_1
...
d_N p_N pred_N,1 ... pred_N,p_N
num_1 den_1 s_1 job_1,1 ... job_1,s_1
...
num_K den_K s_K job_K,1 ... job_K,s_K
```
Job `i`'s line gives its duration `d_i`, its number of predecessors `p_i`, then that
many predecessor ids (each `< i`). Perturbation line `k` gives the inflation factor
as a reduced-or-not fraction `num_k/den_k > 1`, the subsystem size `s_k`, then that
many job ids (a subset of `1..N`).

## Output (stdout)
`N` integers: a permutation of `1..N` (the priority list), separated by whitespace.

## Feasibility
The output must contain exactly `N` integers that are pairwise distinct and each in
`1..N` (i.e. a permutation). Any other output — wrong count, duplicates,
out-of-range or non-integer tokens — scores `Ratio: 0.0`.

## Objective
For a feasible list `order`, let `Cmax(order, k)` be the makespan of the
list-scheduling replay under shock `k`'s inflated durations (every job outside
subsystem `k` at nominal duration). Define
```
F(order) = max_{k=1..K} Cmax(order, k)      (nominal Cmax if K = 0)
```
Minimize `F`.

## Scoring
The checker's own baseline `B` assumes **zero parallelism**: run every job
back-to-back on a single resource (never exploiting the `M` machines at all),
then take the worst of the `K` published scenarios, i.e. `B = max_k(sum of all
job durations under shock k)` (the nominal total if `K = 0`). This is always a
valid (if wasteful) schedule.
With your `F`:
```
sc = min(1000.0, 100.0 * B / max(1e-9, F))
Ratio = sc / 1000.0
```
Any use of the `M` machines beats this baseline; a smaller worst-case makespan
scores higher, capped at `1.0`.

## Constraints
`1 <= N <= 400`, `1 <= M <= N`, `0 <= K <= 120`, `1 <= d_i <= 1000`,
`1 < num_k/den_k <= 5`, `1 <= s_k <= N`. Time limit 5s, memory 512MB.

## Example (worked, illustrative shape only)
`N=3, M=1`, jobs: `1` (d=5, no preds), `2` (d=5, pred 1), `3` (d=1, pred 1). One
perturbation: subsystem `{3}`, factor `2/1`. With `M=1` only one job runs at a time,
so any permutation gives the same makespan chain here: `Cmax = 5+5+1=11` nominally,
or `5+5+2=12` if job 3's shock hits and 3 runs last. `F = 12` for every valid
ordering in this tiny example (machine count leaves no real choice); larger
instances with `M >= 2` genuinely reward *which* jobs you let machines start early.
