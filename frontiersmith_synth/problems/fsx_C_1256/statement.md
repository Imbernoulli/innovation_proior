# Bad Node, Good Node: History-Conditioned Checkpoint Intervals

## Problem

A job must complete `W` units of useful work; each unit takes 1 unit of wall-clock time
while the machine is actively computing. At any moment you may pause and **checkpoint**:
this costs `C` time units and permanently saves all progress made so far.

The machine also fails, according to a fixed, known sequence of `m` **active-compute-time
gaps** `g_1, g_2, ..., g_m`: it fails after `g_1` units of active computation counted from
the start, then after another `g_2` units of active computation counted fresh from the
moment failure 1 struck, and so on (only genuine computing time counts toward these gaps
-- checkpointing and restarting do not; if `m` gaps are exhausted before the job
finishes, no further failures occur). When a failure strikes, everything computed since
the last checkpoint (or since the start, if none yet) is lost, a **restart** costs `R`
time units, and the compute-time-since-last-failure counter resets to 0.

Your job: choose a schedule of checkpoints to minimize the total wall-clock time to reach
`W` units of saved progress. The gap sequence is fixed in the input, but its *local
character* varies -- some stretches of the sequence are a handful of large, calm gaps,
others are a single bad node crashing over and over, i.e. several tiny consecutive gaps.
A schedule that reacts only to the failure rate averaged over the whole sequence pays for
that blindness inside every burst.

## Input (stdin)

```
W C R
m
g_1 g_2 ... g_m
```
All values are positive integers.

## Output (stdout)

```
k
p_1 p_2 ... p_k
```
`k` checkpoint marks, strictly increasing integers with `0 < p_i < W`: "take a checkpoint
the first time saved progress would reach `p_i`". (`k = 0` is allowed -- omit or leave the
second line empty.)

## Feasibility

Rejected (score 0) if: any token is missing, non-integer, or out of declared bounds;
`k` is negative or absurdly large; any mark is not strictly inside `(0, W)`; or the marks
are not strictly increasing.

## Replay and Scoring

The checker deterministically replays your schedule against the fixed gap sequence:
starting from wall-clock 0, progress 0, and gap index 0, it computes until either (a) the
next checkpoint mark is reached -- pay `C`, save progress, continue -- or (b) the current
gap is exhausted first -- lose all unsaved progress, pay `R`, resume from the last save,
move to the next gap. This repeats until progress reaches `W`; the total elapsed
wall-clock time is `F` (minimize this).

Let `B` be the wall-clock time of the checker's own reference schedule: NEVER checkpoint
at all (redo everything from scratch on every failure). The score is

```
Ratio = min(1.0, 0.1 * B / F)
```

Fewer wasted restarts and less checkpoint overhead -> smaller `F` -> higher score.

## Example (illustrative, not a worked score)

`W=100, C=5, R=20`, one gap `g_1=30`. Try a single checkpoint at `p=29`: computing
`0->29` consumes 29 units of the gap and finishes the segment (wall-clock `29`), then
checkpoints (`+5=34`, 29 units now saved). The next segment needs only 1 more unit of
active computation to exhaust the gap (`30-29=1 < 71` remaining to `W`), so the failure
strikes almost immediately: wall-clock `34+1=35`, lose nothing (only 1 unit of unsaved
work), pay `R=20` (wall-clock `55`), gap sequence exhausted. The rest, `29->100`, now runs
uninterrupted: `55+71=126`. Total `F=126`.

Now try a single checkpoint at `p=31` instead: computing `0->31` exhausts the gap after
only 30 units (`30 < 31`), so the failure strikes with the checkpoint never reached --
all 30 units are lost, `R=20` is paid (wall-clock `50`), gap sequence exhausted. The
`0->31` segment is redone uninterrupted (`50+31=81`), checkpoints (`+5=86`), then
`31->100` finishes uninterrupted (`86+69=155`). Total `F=155`. Checkpointing just before
a gap boundary (instead of just after) changed the total time from 155 down to 126.

## Constraints

`1 <= W <= 20000`, `1 <= C <= 60`, `1 <= R <= 400`, `0 <= m <= 90`,
`1 <= g_i <= 4*W`. Time limit 5s, memory 512MB.
