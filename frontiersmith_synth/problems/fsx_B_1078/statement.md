# Foundry Line Batching: Track Assignment Under a Hidden Chain Ceiling

## Problem
A foundry has `N` candidate jobs. Job `i` earns profit `p_i` if it is completed. Some
jobs use a semi-finished part produced by another job: a **precedence relation** "job
`u` before job `v`" means, if both jobs run, `u` must finish before `v` can start.
Precedence is given as a set of direct edges; it is transitive (if `u` must precede
`v` and `v` must precede `w`, then `u` must also precede `w`, even with no direct
edge `u`-`w`).

The foundry has exactly `k` production **lines**. Each line processes an ordered
sequence of jobs, one at a time, and can hold at most `H` jobs (its horizon). A
sequence of jobs assigned to one line is valid only if, for every two jobs on that
line, one is required to precede the other **and** they appear in that order — i.e.
each line's job list must be a totally ordered chain of the precedence poset. You do
not have to schedule every job, and you may use fewer than `k` lines.

Choose which jobs to run and how to arrange them on at most `k` lines (each of length
at most `H`) to maximize total completed profit. The catch: the highest individual
profit jobs are not necessarily the ones that fit together — some may be pairwise
**incomparable** (no precedence relation either way), and incomparable jobs can never
share a line. A few tempting, mutually incomparable high-profit jobs each permanently
occupy one whole line, wasting up to `H-1` slots a deep, precedence-linked chain of
decent-profit jobs could have filled instead. The true ceiling on reachable profit is
governed by how the job poset decomposes into chains, not by raw profit ranking.

## Input (stdin)
```
N M k H
p_1 p_2 ... p_N
u_1 v_1
...
u_M v_M
```
Each `u v` line means job `u` must precede job `v`. `1 <= N <= 300`, edges form a DAG.

## Output (stdout)
```
m
L_1 t_1 t_2 ... t_{L_1}
...
L_m t_1 t_2 ... t_{L_m}
```
`m` (`0 <= m <= k`) is the number of lines you use. Each following line lists that
line's job count `L_i` (`1 <= L_i <= H`) then the job ids, in the exact order they run
on that line.

## Feasibility
Any of the following scores `Ratio: 0.0`:
- `m` outside `[0,k]`, or any `L_i` outside `[1,H]`;
- a job id outside `[1,N]`, or the same job id appearing more than once (on any line);
- two jobs `t_a`, `t_b` at positions `a < b` on the same line where `t_b` is not
  reachable from `t_a` via the precedence edges (either they are incomparable, or the
  required order is reversed);
- malformed/non-finite output.

## Objective
Maximize `F`, the total profit of every scheduled job.

## Scoring
The checker builds its own trivial baseline `B`: the `min(k,N)` individually most
profitable jobs, each run alone on its own line (always feasible, since a
single-job line trivially satisfies every chain rule). Then:
```
sc = min(1000.0, 100.0 * F / max(1e-9, B))
Ratio = sc / 1000.0
```
Reproducing the baseline scores `0.1`; ten times its profit caps the score at `1.0`.

## Constraints
- `1 <= N <= 300`, `1 <= k <= 8`, `2 <= H <= 15`, profits are positive integers.
- Time limit 5s, memory 512MB.

## Example
Suppose `N=5, k=2, H=3`, profits `[50, 50, 20, 25, 30]`, and edges `3 4`, `4 5` (job 3
before job 4 before job 5; jobs 1 and 2 are incomparable to everything). The baseline
picks the two highest-profit jobs, `1` and `2`, each alone: `B = 50+50 = 100`,
`Ratio = 0.1`. A better output uses line 1 for the chain `3 4 5` (profit `75`) and
line 2 for job `1` alone (profit `50`): `m=2`, lines `3 3 4 5` and `1 1`, giving
`F = 125`, `Ratio = min(1000, 100*125/100)/1000 = 0.125`. Job `2` is left unscheduled
because both lines are already full — this is illustrative only, not a case from the
real test data.
