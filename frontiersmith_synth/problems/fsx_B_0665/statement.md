# One Cache Policy vs a Gauntlet of Access Rhythms

You are designing an **online cache-eviction policy**. Instead of submitting code
that touches raw traces, you submit a small **scoring rule** (six numbers) that a
frozen evaluator will run for you, causally, over a hidden gauntlet of access
patterns. Your policy must work well **everywhere in the gauntlet at once** — no
fixed textbook rule (LRU, MRU, LFU) survives every rhythm.

## Setup

For each of 10 hidden test instances the evaluator has: a cache of `cache_size`
slots, and a suite of **3 traces**. Each trace probabilistically interleaves three
cyclic components of very different periods — a fast short loop, a medium loop, a
slow long scan — plus light noise, so it's never one perfectly uniform scan (that
symmetry would let any static, signal-free address choice act as an accidental
near-optimal reservoir; a real mixed rhythm needs actual history-tracking). The
traces are **hidden** — you only see `cache_size` — your policy must be *one
general rule*, not fit to a trace you were shown.

**Public instance** (stdin): `{"cache_size": <int>}`

## What you submit

**Answer** (stdout): `{"w0":.., "w1":.., "w2":.., "w3":.., "w4":.., "w5":..}` — six
finite floats, each with `|w_i| <= 1000`. These are the weights of a scoring
function the evaluator applies online while it replays each hidden trace against a
cache of size `cache_size`, one access at a time, left to right:

At time `t`, accessing line `x`: if `x` is resident, it's a hit. Otherwise it's a
miss; if the cache is full, the evaluator computes, for every currently RESIDENT
line `y`, features built ONLY from access history up to and including time `t`
(never from the future):

- `recency(y) = t - last_seen[y]` (steps since `y` was last touched)
- `freq(y)` = number of times `y` has been accessed so far in the whole trace
- `gap_est(y) = last_seen[y] - prev_seen[y]` — `y`'s own most recent observed
  inter-access gap (a per-line, self-estimated period); if `y` has been seen only
  once so far, `gap_est(y) = 1000000` (sentinel: "no periodicity known yet")
- `predicted_wait(y) = gap_est(y) - recency(y)` — how many steps until `y` is
  predicted to be needed again, extrapolating from its own last gap
- `regime(y) = 1.0` if `gap_est(y) > cache_size` else `0.0` — flags lines whose own
  cycle can't possibly fit resident in the cache

and evicts the resident line maximizing
`score(y) = w0 + w1*recency(y) + w2*freq(y) + w3*gap_est(y) + w4*predicted_wait(y) + w5*regime(y)`
(ties broken by the smaller line id). Per-line `last_seen/prev_seen/freq` are kept
for every address ever seen, even while it isn't resident — bookkeeping is cheap;
only the cached data itself is capacity-limited.

## Scoring

For each of the 3 traces in an instance's suite, the evaluator computes your
policy's `hitrate = hits/length`, and also (using the same trace, with full
knowledge of the future — something no causal policy can have) Belady's exact
offline-optimal hit rate `hi_belady`. The ceiling used for normalization adds a
fixed headroom margin: `hi = min(1, hi_belady + 0.15)` — even a perfect per-line
reuse-gap estimate should not saturate the score. The trace's ratio is
`clamp(hitrate / hi, 0, 1)`.

**An instance's score is the MINIMUM ratio over its own suite of 3 traces** — a
policy that nails one rhythm but collapses on another is scored by the rhythm it
fails. Your final `Ratio` is the mean of these 10 worst-trace scores. A malformed,
missing-key, non-finite, or out-of-range answer scores 0 on that instance.

## Why no fixed rule wins

Pure LRU (`w1=1`, rest 0) evicts the least-recently-touched line — on a scan whose
period exceeds `cache_size`, every resident line gets evicted exactly before its
next use ("sequential flooding"), driving hit rate toward 0. Pure MRU (`w1=-1`)
fixes long scans but evicts the *just-touched* line even in tight, fast-cycling
short loops, where that line was about to be reused almost immediately. The
`gap_est`/`predicted_wait` features let a policy estimate, per line, *when it will
next be needed* from its own history — the same principle Belady's optimal
algorithm uses with foresight, applied online with an estimate instead.

## Constraints

Time limit 2s per candidate call, memory 512MB. Objective: **maximize** the mean
worst-trace ratio.
