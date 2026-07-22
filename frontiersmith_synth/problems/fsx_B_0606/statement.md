# Paint-Line Purge Sequencing

## Problem
An industrial paint line must run `N` color jobs, each **exactly once**, in some order you
choose. Between two consecutive jobs the line pays a **changeover** cost; the machine also
accumulates **residue** that must periodically be flushed with a **full clean**. The clean is
expensive, but it resets the line to a pristine state — and starting a job from a clean line is
cheap. Your job is to schedule the jobs *and* decide where to insert full cleans so the total
cost is minimized.

## Input (stdin)
```
N R Cclean
r[0] r[1] ... r[N-1]          # residue each job deposits (r[i] <= R)
s[0] s[1] ... s[N-1]          # startup cost of job j when the line is clean
w[0][0] ... w[0][N-1]         # N rows: changeover matrix (ASYMMETRIC)
...
w[N-1][0] ... w[N-1][N-1]
```
`w[i][j]` is the cost of running job `j` **directly after** job `i`. It is asymmetric:
painting a light color after a dark one is much more expensive than the reverse. Diagonal
entries are unused.

## Output (stdout)
A whitespace-separated sequence of tokens. Each token is either an integer job id in
`[0, N-1]` or the literal `C` (perform a full clean). Every job id must appear **exactly
once**; `C` tokens may appear anywhere (including none). Example: `3 0 C 2 1`.

## Simulation & Feasibility
The line starts **clean** with residue `0`. Process your tokens left to right:
- On `C`: pay `Cclean`, set residue to `0`, line becomes clean.
- On job `j`: pay `s[j]` if the line is currently clean, otherwise `w[L][j]` where `L` was the
  previous job; then add `r[j]` to the residue. **If residue ever exceeds `R`, the schedule is
  INFEASIBLE** (you should have cleaned first). The line is now dirty with last job `j`.

A schedule is feasible iff every job appears exactly once and the residue never exceeds `R`.
Infeasible output scores `0`.

## Objective (minimize)
`cost = (sum of all changeover/startup costs) + Cclean * (number of C tokens)`.

The residue cap forces the run to split into **epochs** (maximal clean-free stretches). Because
a clean makes the *next* job cost only `s[j]` instead of a changeover, an epoch boundary placed
right before an expensive `w[L][j]` transition **absorbs** that penalty into a reset you were
going to pay for anyway. The reset placement, not the raw tour, drives most of the cost.

## Scoring
The checker builds a reference schedule `ref` (clean before every job, in id order) with cost
`cost_ref = sum(s) + (N-1)*Cclean`, then reports
```
Ratio = min(1.0, cost_ref / (10 * cost_yours))
```
Reproducing the reference scores `0.1`; a schedule 10x cheaper caps at `1.0`. Reported per case,
averaged over the 10 cases.

## Constraints
`9 <= N <= 18`. All costs and residues are non-negative integers. Time limit 5 s, memory 512 MB.

## Worked example
`N=3, R=10, Cclean=5`, `r=[4,4,4]`, `s=[2,3,2]`, and `w[0][1]=4`, `w[1][2]=20`, others large.
Reference cost `= (2+3+2) + 2*5 = 17`.
Schedule `0 1 C 2`: job0 startup `2` (residue 4), job1 changeover `w[0][1]=4` (residue 8),
`C` costs `5` (residue 0), job2 startup `2` (residue 4). Total `= 13`, feasible.
Here the clean — required soon anyway because `4+4+4 > R` — is co-located just before job 2, so
the costly `w[1][2]=20` transition is never paid (job 2 starts from a clean line for `2`).
`Ratio = min(1, 17 / (10*13)) = 0.1308`.
Placing the clean one step earlier (`0 C 1 2`) would pay `w[1][2]=20` and cost `30` — far worse.
