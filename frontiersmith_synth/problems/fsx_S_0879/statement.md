# Foundry Order Book: Deadline-Weighted Jobs with Family Setup Switching

## Story
A single machine processes jobs one at a time. Job `i` has a **processing time**
`p[i]`, a **deadline** `d[i]`, a **weight** `w[i]`, and belongs to a **family**
`fam[i]`. Whenever the machine finishes a job and the *next* job it runs is from a
**different** family, it must first run a **setup** — the time cost depends on the
specific *pair* of families (`setup[a][b]`, not necessarily symmetric, and not
necessarily the same as `setup[b][a]`) and is given only in the instance, not in this
statement. Two consecutive jobs from the **same** family need no setup at all.

A job is only worth anything if it finishes on time: if, after any setup and its own
processing, the machine's clock would exceed `d[i]`, the job is simply **never
started** — no time is charged and it earns nothing. There is no partial credit and no
penalty beyond forfeiting that job's weight. Your goal: choose which jobs to run and in
what order, to **maximize the total weight of on-time jobs**.

## Task
Write a **standalone program**: read ONE JSON instance from `stdin`, write ONE JSON
answer to `stdout`.

### Public instance (stdin)
```json
{ "name":"mixed1", "n":34, "F":5,
  "p":[...n positive ints...], "d":[...n positive ints...], "w":[...n positive numbers...],
  "fam":[...n ints in [0,F)...],
  "setup":[[setup[a][b] for b in range(F)] for a in range(F)] }
```
`setup[a][a] = 0`; `setup[a][b]` for `a != b` is the time cost of switching directly
from a job of family `a` to a job of family `b`.

### Answer (stdout)
```json
{ "order": [i_0, i_1, ...] }
```
A sequence of **distinct** job indices in `[0, n)` — the processing order. It need not
include every job.

### Validity
Any violation — not a list, a duplicate index, an out-of-range index, a non-integer /
NaN / boolean entry, a crash, a timeout, or non-JSON output — scores **0.0** on that
instance.

## Transition (deterministic)
Start with clock `t=0` and no previous family. Process `order` left to right: for job
`idx`, let `su = 0` if this is the first job or `fam[idx]` equals the previous
*processed* job's family, else `su = setup[prev_family][fam[idx]]`. If
`t + su + p[idx] <= d[idx]`, the job is **admitted**: `t` advances by `su + p[idx]`,
`prev_family` becomes `fam[idx]`, and `w[idx]` is added to your score. Otherwise the job
is **skipped**: `t` and `prev_family` are unchanged, and it earns nothing — you may
freely list jobs you are unsure about.

## Objective & scoring (deterministic)
Per instance the evaluator computes a relaxation bound:
```
relax = sum(w[i] for i in range(n) if p[i] <= d[i])
```
This is a valid upper bound: any admitted job's completion time is at least its own
processing time, so `p[i] <= d[i]` is *necessary* for admission under any order or
subset whatsoever. It is also deliberately loose — every instance plants pairs of jobs,
in different families, that are each individually feasible but jointly impossible to
both admit in either order — so `relax` counts weight no real schedule can ever fully
collect. Your score on the instance is `clamp(gained / relax, 0, 1)`, where `gained` is
the total weight your admitted jobs earn. The final score is the mean over **10** fixed
seeded instances.

## Why it is open-ended
Sorting purely by deadline (earliest-deadline-first) ignores family structure
completely: on instances where family assignment interleaves with deadline order, it
pays a setup on almost every job and can even burn its budget chasing several
tight-deadline, low-weight jobs scattered across different families while a
high-weight job's own deadline quietly passes. But clustering an entire family into one
contiguous block is *also* a trap when that family holds one high-weight, tight-deadline
job stranded among many low-value, loose-deadline packmates — visiting the block early
saves the stranded job at the cost of its packmates' better placement, while visiting it
at its natural slot loses the stranded job outright. Genuine strategies trade off setup
amortization, family visiting order, and single-job rescue differently; there is no
easy optimum.

## Isolation
Your program runs in a fresh sandboxed subprocess and only ever sees the public
instance above. The relaxation bound is computed by the evaluator process.
