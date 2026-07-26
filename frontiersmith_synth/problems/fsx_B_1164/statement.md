# Castes of the Cleanroom

## Problem

A fabrication plant has `L` cleanliness grades, `1` (dirtiest) through `L`
(cleanest). A fleet of `R` service robots must perform `J` jobs; job `j`
occupies a zone of grade `g_j`, becomes available at tick `r_j`, takes `dur_j`
ticks once started, is due by tick `dl_j`, and carries a lateness weight
`w_j`.

Every robot starts perfectly clean: it may enter a zone of *any* grade for
its very first job. After a robot works a job in a zone of grade `g`, it
becomes contaminated to level `g`: from then on it may only enter zones of
grade `<= g` **without decontamination**. Contamination is a one-way ratchet
— working a dirtier zone can only lower a robot's current level, never raise
it.

To work a job whose grade exceeds a robot's current contamination level, the
robot must first pass through the facility's single decontamination airlock.
One decon **cycle** takes `T` ticks and admits up to `C` robots simultaneously
(a "batch"); a robot that joins a cycle becomes fully clean (contamination
level `L`) the instant it ends. The airlock is a shared mutex: only one cycle
may be running at any tick, across the whole facility.

You see every job in advance and must produce a full itinerary: which robot
does which job and when, and how robots are grouped into decon cycles.

## Input (stdin)

```
L R J T C KCOST
```
followed by `J` lines, one per job (job `j` is the `j`-th such line, 1-indexed):
```
g_j r_j dur_j dl_j w_j
```

## Output (stdout)

```
NC
s_1
s_2
...
s_NC
ROBOT 1 m_1
<m_1 event lines>
ROBOT 2 m_2
<m_2 event lines>
...
ROBOT R m_R
<m_R event lines>
```
`NC` is the number of decon cycles you use; `s_i` is the start tick of cycle
`i` (it occupies `[s_i, s_i+T)`). Then, for every robot `1..R` in order, give
its event count `m_r` followed by that many events **in chronological
order**, each one of:
```
J job_id start_time      start job <job_id> at tick <start_time>
D cycle_id                join decon cycle <cycle_id> (1-indexed above)
```

## Feasibility

- Every job `1..J` must appear in exactly one robot's event list, exactly once.
- A robot's events must be non-overlapping in time, in the order given: a
  `J` event's `start_time` must be `>=` the robot's current free tick and
  `>= r_j`; a `D` event may only be joined at or after the robot's current
  free tick, i.e. `start_time_of_cycle >= robot's free tick`.
- A `J` event for grade `g` is only legal if `g <=` the robot's current
  contamination level (raised to `L` immediately by the robot's most recent
  completed `D`, if any; otherwise carried over from its last job, or `L`
  if it has done nothing yet).
- All `NC` decon cycles must be pairwise non-overlapping in time (the shared
  airlock mutex), and each cycle may have at most `C` participating robots.

Any violation scores `0` for that test case.

## Objective & Scoring

Minimize
```
cost = sum over jobs of  w_j * max(0, finish_j - dl_j)   +   KCOST * NC
```
The checker also builds its own naive reference: assign jobs round-robin to
robots (ignoring grade entirely), firing an individual, un-batched decon
cycle the instant one becomes necessary, respecting the airlock mutex. This
gives an internal baseline cost `B`. Your score is
```
Ratio = min(1.0, 0.1 * B / max(1e-9, cost))
```
so matching the naive baseline earns about `0.1`.

## What makes it hard

Contamination only ever gets worse until you pay for a decon — so a robot
that is *always* sent to the same grade never needs the airlock at all,
while a robot bounced between grades pays for it constantly. Assigning jobs
by who happens to be free soonest, oblivious to grade, tends to bounce every
robot across the whole grade range. Even once you avoid that, treating each
robot's decon need as its own isolated event wastes the airlock's batch
capacity `C` — the cost `KCOST` per cycle is paid whether one robot uses it
or all `C` do.

## Example

3 jobs, `L=4, T=6, C=3, KCOST=120`. A solution with `finish` times giving
`late` costs `5, 0, 12` and using `NC=1` decon cycle: `cost = 5+0+12 +
120*1 = 137`. If the checker's naive baseline gets `B = 400`, then
`Ratio = min(1, 0.1*400/137) ~= 0.292`.

## Constraints

`1 <= L <= 6`, `1 <= R <= 30`, `1 <= J <= 300`, `1 <= T,C <= 20`, `1 <= KCOST <= 200`.
Time limit 5s, memory 512MB. Scoring is deterministic.
