# Humpyard Radix Resort

## Problem

A humpyard receives `N` inbound freight cars on a single working track (track
`0`). Car `i` (`i = 0 .. N-1`, listed in arrival order) must end up in slot
`slot_i` of outbound train `train_i`. Trains are numbered `0 .. T-1`, each
train has exactly `L` cars (`N = T*L`), so a train is correctly assembled iff
its cars sit, from the top of its own siding down, in slot order `0,1,...,L-1`.

Every track in the yard -- the working track, the `Y` classification tracks,
and the `T` final sidings (one per train) -- is a **LIFO stack**: only the top
car is reachable, and cars stay coupled in relative order when moved. A move
takes the top `k` cars of one track as a single cut and shoves them onto the
top of another track, preserving their internal order (the cut's former
topmost car ends up nearest the new track's top). Every track has a hard
capacity; a final siding's capacity is exactly `L`.

There are **2 switch engines**, but only **1 lead track** -- the only place a
cut can physically be transferred. The lead is a mutex: at any instant at most
one engine's move may be in progress, across BOTH engines. A move starting at
time `t` moving `k` cars occupies the lead for `a + b*k` ticks. On top of that,
re-lining the switches costs an extra `s` ticks whenever an engine's move type
differs from that SAME engine's own previous move: type `D` (destination is
one of the `Y` classification tracks) vs type `F` (destination is the working
track or a final siding). A fresh engine's first move never pays this.

**Track indices:** `0` = working track (starts with all `N` cars, cap `N`);
`1..Y` = classification tracks (cap `cap` each); `Y+1..Y+T` = final sidings,
siding `Y+1+i` belongs to train `i` (cap `L`).

**Minimize the makespan**: the time the last move finishes.

## Input (stdin)

Line 1: `N T L Y a b s cap`.
Next `N` lines: `train_id slot_id` for cars `0..N-1` in arrival order (line 0
= top / immediately reachable car of track `0`).

## Output (stdout)

Line 1: `M`, the number of moves.
Next `M` lines: `engine t src dst k` -- engine (`1` or `2`), start time `t`
(real, `>= 0`), source track, destination track, cut size `k >= 1`.

## Feasibility (any violation scores `Ratio: 0.0`)

- `engine in {1,2}`; `t` finite `>= 0`; `0 <= src,dst <= Y+T`; `src != dst`;
  `k >= 1` integer; a move may never pull cars OUT of a final siding
  (`src <= Y`).
- Source must hold `>= k` cars when the move executes; destination must not
  exceed its capacity.
- **Global lead mutex**: sort moves by `t`; consecutive intervals
  `[t, t+dur)` (`dur = a + b*k`, plus `s` on a same-engine mode switch) must
  not overlap.
- At the end, every final siding `Y+1+i` holds exactly train `i`'s `L` cars,
  top-to-bottom in slot order `0,1,...,L-1`.

## Scoring

The checker replays your schedule to get your makespan `F`, and independently
computes its own always-correct reference makespan `B`: dump every car onto
ONE classification track (ignore the radix structure and the other tracks),
then dig each needed car out one at a time via a scratch-and-restore shuffle,
single engine throughout. Then `Ratio = min(1000, 100*B/F) / 1000`: matching
this naive single-bucket baseline scores `~0.1`; a `10x` lower makespan caps
the ratio at `1.0`.

## Constraints

`16 <= N <= 104`, `Y = 4` classification tracks, `T,L` chosen so `N = T*L`.
Checker runs in `O(total moves)`.

## Hint

Reading `1..Y` as digit buckets on the key `train_id*L+slot` (base `Y`) turns
the working track into a radix sorter: distribute cars one at a time by the
current digit, then collect the buckets back in a fixed order (`Y-1` down to
`0`); `D = ceil(log_Y N)` such passes fully sort the working track. A move
preserves relative order (a rigid cut), so collecting a whole bucket in one
shot would silently corrupt the sort -- collecting must stay one car at a
time, exactly like distributing. But once the working track is fully sorted,
each train's `L` cars already sit contiguously in the order its siding needs,
so ONE cut of size `L` finishes a train -- no scratch-track bounce required.
Dedicating one engine to every distribute-type move and the other to every
collect-type move means neither ever re-lines its switches.
