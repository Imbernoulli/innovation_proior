# Per-Lock Protocol Assignment Against Priority Inversion

## Problem

A single CPU runs a fixed set of jobs under **strict fixed-priority
preemptive scheduling** (smaller priority number = more urgent; ties broken
by job index). Some jobs hold one of `L` shared **locks** for part of their
execution (a critical section); while a lock is held, no other job may
enter that section -- it blocks until the lock is released.

Strict priority scheduling is right when jobs don't share resources. But
once a low-priority job holds a lock, a plain priority scheduler lets it be
preempted by unrelated medium-priority jobs while holding it, delaying the
release and starving a much more important job waiting for that same lock
-- **priority inversion**, unbounded in principle.

Two classic remedies exist, and you must pick **one per lock** (not one
global rule):

- **inherit**: while a job holds this lock, its *effective* priority is
  boosted to the best (most urgent) priority among jobs **currently
  actually blocked** waiting for this lock (reverts when nobody is
  waiting). Recomputed every tick.
- **ceiling**: while a job holds this lock, its effective priority is
  boosted to that lock's fixed **ceiling priority** (given in the input),
  regardless of whether anyone is currently waiting.
- **none**: no boost at all (the do-nothing baseline).

`inherit` never boosts more than necessary but only reacts *after* a real
waiter shows up -- a medium-priority job stealing cycles *before* the real
waiter arrives is never fended off. `ceiling` protects the holder from the
instant it acquires the lock, but boosts it even when nobody with real
business ever contends for it, needlessly delaying unrelated jobs. Which is
better depends on each lock's own contention pattern -- no protocol is
always right.

## Input (stdin)

```
L J Tmax
c_1 c_2 ... c_L
pri_1 arr_1 dl_1 w_1 k_1  len_1 lock_1  len_2 lock_2  ...  len_k1 lock_k1
... (J job lines total)
```
`L`=#locks, `J`=#jobs, `Tmax`=a safe simulation horizon, `c_i`=lock `i`'s
static ceiling priority. Each job line: `pri`, `arr` (arrival tick), `dl`
(absolute deadline), `w` (cost per tick late), `k` (segment count), then
`k` pairs `len lock` (segments in order; `lock=0`=ordinary execution,
`lock` in `1..L`=runs while holding that lock). A job holds at most one
lock at a time and never re-enters a segment.

## Output (stdout)

Exactly `L` whitespace-separated integer tokens, one per lock in order
`1..L`, each one of `0` (none), `1` (inherit), `2` (ceiling) -- your chosen
protocol for that lock.

## Simulation (how you are scored)

Time advances in integer ticks. Each tick the scheduler picks, among jobs
that have arrived, are unfinished, and are not blocked on a lock held by
someone else, the one with the best **effective priority** (boosted per the
rules above if it currently holds a lock; ties broken by job index), and
runs it one tick. A job's deadline-miss cost is `w * max(0, finish_tick -
dl)`. Any non-integer token, any code outside `{0,1,2}`, or a token count
other than `L`, makes the whole answer infeasible (`Ratio: 0.0`).

## Scoring

Let `F` = total deadline-miss cost with your chosen protocols, and `B` =
the cost of the same simulation with `none` on every lock (the checker's
own baseline). Minimization:
```
sc    = min(1000, 100 * B / max(1e-9, F))
Ratio = sc / 1000
```
Reproducing the baseline scores `0.1`; a 10x cost reduction caps the ratio
at `1.0`. Fully deterministic integer simulation.

## Constraints

`1 <= L <= 4`, `4 <= J <= 21`, all times/costs are small non-negative
integers, `Tmax` generous enough that every job finishes under any
protocol assignment.

## Example

One lock (`L=1`): a low-priority job holds it for 10 ticks; two
medium-priority jobs arrive *before* a tight high-priority waiter `H` starts
waiting on the lock, two more arrive *after*. Output `0` (`none`)
reproduces the baseline: `F = B`, `Ratio = 0.1`. Output `1` (`inherit`)
protects the holder only once `H` is actually waiting, so the two later
jobs no longer delay it but the two earlier ones still do -- `F` drops
below `B`, not all the way. Output `2` (`ceiling`) protects the holder from
the instant it acquires the lock, so none of the four delay it and `H`
meets its deadline -- the lowest `F`. The best choice is not the same
protocol on every lock.
