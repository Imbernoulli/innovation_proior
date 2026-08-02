# Per-Line Adaptive Coherence: Invalidate vs. Update

## Problem
A multi-core machine has `C` private caches sharing `L` memory lines. You are given a
**memory access trace**: a time-ordered log of `(core, line, op)` events, `op ∈ {R, W}`,
recorded across all cores. For every line you must choose a **coherence policy** deciding
what happens on each write to that line. Two textbook policies exist:

- **INV (invalidate-on-write):** a write invalidates every other cache currently holding
  the line (they must re-fetch on their next read).
- **UPD (update-on-write):** a write broadcasts the new value to every other cache
  currently holding the line (they stay valid, no re-fetch needed).

You may also declare a line **ADAPTIVE**: pick a window `W` and a threshold `T ∈ [0,1]`. At
each write, look at the last `W` accesses to that line occurring *before* this write (fewer
if history is short); let `f` be the fraction that were reads. Use UPD-semantics if `f ≥ T`,
else INV-semantics (no prior history ⇒ INV). The window only looks at the past — a
legitimate reactive rule, not lookahead.

Reads are unaffected by policy: a core reading a line it does not currently hold valid pays
a fixed miss cost (fetch); one that already holds it valid reads for free.

## Input (stdin)
```
T C L
MISS INVC UPDC
c_1 l_1 op_1
...
c_T l_T op_T
```
`T` = number of events, `C` = number of cores, `L` = number of distinct lines (indices
`0..L-1`). `MISS` = cost of a read miss (fetch), `INVC` = cost of one invalidate message,
`UPDC` = cost of one update message (all positive integers). Each of the `T` following
lines is one trace event: core `c_i ∈ [0,C)`, line `l_i ∈ [0,L)`, `op_i ∈ {R,W}`. Events for
a given line appear in the same relative order they occur on that line.

## Output (stdout)
Exactly `L` lines, line `i` (0-indexed) giving the policy for line `i`, all in the SAME
3-token shape: `MODE W T`, with `MODE ∈ {INV, UPD, ADAPT}`. `W` (positive integer) and `T`
(real in `[0,1]`) are the adaptive window/threshold; they are only *used* when
`MODE = ADAPT`, but every line must still supply syntactically valid, finite, in-range
values (e.g. `INV 1 0.0`) — there is no shorthand. Example for `L=2`: `INV 1 0.0` then
`ADAPT 4 0.5`.

## Feasibility
Output must contain exactly `L` well-formed policy lines and nothing else: `MODE` must be
exactly `INV`, `UPD`, or `ADAPT`; `W` a finite integer with `1 ≤ W ≤ 5000`; `T` a finite
real with `0 ≤ T ≤ 1`. Any malformed token, wrong line count, trailing/missing data, or
non-finite value makes the whole answer infeasible (score 0), regardless of `MODE`.

## Objective (what the score rewards)
Replay the trace *line by line, independently* (a line's cost never depends on any other
line's policy): each miss costs `MISS`; each write under INV costs `INVC` per other cache
holding the line (which then goes invalid); each write under UPD costs `UPDC` per other
cache holding the line (which stays valid). Total interconnect cost `F` = sum of these
costs; **lower `F` is better**. UPD never re-pays a miss on lines it keeps valid, but pays a
message on *every* write regardless of whether those cores ever read again; INV pays
nothing on repeated writes once sharers are already invalidated, but re-pays a miss the
next time an invalidated core reads. Which is cheaper for a line depends on that line's own
read/write interleaving and sharer count, not the trace's global read/write ratio.

## Scoring
The checker computes `F` for your policy set and `B` = the cost of the all-`INV` policy
(its own reference construction) on the same trace, then reports
`Ratio = min(1000, 100·B/F) / 1000` (lower `F` ⇒ higher ratio; matching `B` gives ≈0.1).
Your score is the mean ratio across 10 hidden test traces.

## Constraints
`1 ≤ T ≤ 1200`, `2 ≤ C ≤ 8`, `2 ≤ L ≤ 10`, `1 ≤ MISS,INVC,UPDC ≤ 20`. Time limit 5s.

## Example (illustrative only — not a real hidden case)
One line, `C=2`, `MISS=8 INVC=2 UPDC=2`, events `(0,R) (1,R) (0,W) (0,W) (0,W)`.
INV: reads cost `8+8=16`; write 1 invalidates core 1 (`+2`); writes 2,3 find no other valid
sharer (`+0` each) → total `18`. UPD: reads cost `16`; all three writes still find core 1
valid and re-broadcast (`+2` each) → total `22`. INV wins because writes cluster with no
interleaved reads — frequent reads by core 1 after each write would flip which is cheaper.
