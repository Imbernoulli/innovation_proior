# DVFS Deadline Scheduling with Ramp-Sensitive Transitions

## Problem
A processor executes `J` deadline-bound jobs over `T` discrete time slots
`t = 0..T-1`. Job `j` is released at `r[j]`, must finish by `d[j]`
(exclusive -- it may only use slots `r[j] .. d[j]-1`), and requires
`w[j]` cycles of work; jobs may be preempted and interleaved freely.

At every slot you pick one of `m` fixed DVFS levels. Level `k` executes at
most `s[k]` cycles that slot and burns `p[k]` energy if used, where `p` is
convex in `s` (going faster costs *super-linearly* more power). Level 0 is
idle (`s=0, p=0`).

Switching levels between two consecutive slots costs two things:
1. A fixed **transition energy** `trans[i][j]` (added once, for switching
   from the previous slot's level `i` to this slot's level `j`).
2. A **ramp penalty**: the slot immediately after a switch can execute at
   most `s[k] - ramp` cycles (floored at 0), regardless of level `k` --
   the hardware needs that slot to actually reach the new level.

## Input (stdin)
```
T m J
s[0] p[0]
...
s[m-1] p[m-1]
ramp
trans[0][0] ... trans[0][m-1]
...
trans[m-1][0] ... trans[m-1][m-1]
r[0] d[0] w[0]
...
r[J-1] d[J-1] w[J-1]
```
All values are non-negative integers; `s` is strictly increasing,
`trans[i][i] = 0`.

## Output (stdout)
Exactly `T` non-negative integers, the chosen level for each slot:
```
level[0] level[1] ... level[T-1]
```

## Feasibility
1. Exactly `T` valid integer tokens, each in `[0, m-1]` (no decimals, no
   `nan`/`inf`) -- anything else scores 0.
2. Build the per-slot capacity: `cap[0] = s[level[0]]`; for `t >= 1`,
   `cap[t] = s[level[t]]`, reduced by `ramp` (floored at 0) if
   `level[t] != level[t-1]`.
3. A preemptive earliest-deadline-first packing of all jobs against
   `cap[]` must complete every job's `w[j]` cycles strictly before
   `d[j]`. (Equivalently: no valid way to route `cap[]` cycles to jobs
   leaves any job short at its deadline -- EDF packing is exactly the
   sufficiency test.)

Any violation scores `Ratio: 0.0`.

## Objective
Minimize total energy: `F = sum_t p[level[t]] + sum_t trans[level[t-1]][level[t]]`
(the transition sum only over `t = 1..T-1`).

## Scoring
The checker builds its own reference plan: the shortest possible prefix
at the top level, then idle for the rest ("race to idle"), which is
always feasible. Call its energy `B`. With your feasible energy `F`:
```
Ratio = min(1, 0.1 * B / F)
```
Lower energy scores higher, among feasible plans only.

## Why racing isn't always right
Power grows convexly with speed, so sustaining a *lower* level for
longer can cost far less energy than a short burst at the top level for
the same work -- unless the deadline genuinely leaves no slack, in which
case near-top speed really is required. Separately, every level switch
taxes you twice (transition energy, and lost capacity on the very next
slot); switching exactly when work resumes lets that ramp loss land on
the slot the deadline needed most, while switching one slot earlier --
during idle time you weren't using anyway -- makes the same loss free.

## Example (worked shape only, illustrative sizes -- not the hidden instance)
`T=4, m=4` with levels 0..3 giving `s=[0,3,6,10]`, `p=[0,27,216,1000]`,
`ramp=2`, `trans[0][2]=trans[2][0]=8`, `trans[3][0]=12`, one job
`r=1,d=3,w=5`. Submitting
levels `[0,2,2,0]` (idle, mid, mid, idle): slot 1 switches level
(idle->mid), so it loses `ramp`: `cap[1]=6-2=4`; slot 2 keeps level 2,
so `cap[2]=6`. The job's window (slots 1-2) gets `4+6=10 >= 5`,
feasible. Energy `F = p[0]+p[2]+p[2]+p[0] + trans[0][2] + trans[2][2]
+ trans[2][0] = 0+216+216+0 + 8+0+8 = 448`. The checker's reference
"race to idle" plan needs the top level for the first 2 slots to clear
the same job (`[3,3,0,0]`, since 1 slot of top speed alone loses too
much to the following ramp-down): `B = 1000+1000+0+0 + trans[3][0] =
2012`. `Ratio = min(1, 0.1*2012/448) = 0.449` -- the sustained mid-level
plan beats racing to the top by roughly 4x here, illustrating that a
slower sustained level can beat racing when the deadline leaves slack.

## Constraints
`10 <= T <= 40`, `m = 4`, `1 <= J <= 6`, all cycle/power/energy values
fit in a 32-bit signed integer. Time limit 5s, memory 512MB.
