# Tug Escort Chains Through the Tide

## Problem

A harbor channel is modeled as a single line of positions `0..L`. `T` tugs sit at given
integer starting positions at time 0. `N` ships each need an **escort**: ship `j` must be
towed in a straight line from rendezvous point `a_j` to exit point `b_j`. This tow takes
exactly `d_j = |b_j - a_j|` ticks, because a tug's top speed is 1 position per tick and the
tow uses that speed the whole way (no slack).

Crucially, ship `j` requires **`k_j` tugs simultaneously present** for the *entire* tow: all
`k_j` tugs must be at `a_j` at the tow's start tick and at `b_j` at its end tick — a multi-tug
rendezvous, not a solo job. Ship `j` also has a `release_j` (earliest the tow may start) and a
list of disjoint **tide windows** `[o_1,c_1), [o_2,c_2), ...` (increasing, non-overlapping):
the *whole* tow `[s_j, s_j+d_j]` must fit inside **one** window (channels are impassable
between windows — you cannot start in one window and finish in the next).

Any tug not towing moves freely on the line at speed ≤ 1 (it may also simply wait). A tug can
serve many ships over time, one at a time, as long as its own timeline never asks it to be in
two places at once — this is **job-chaining**: the same tug (or persistent team of tugs) can
be routed straight from one tow's exit point into the next tow's rendezvous.

## Input (stdin)
```
T N L
p_1 p_2 ... p_T                (tug start positions, 0<=p_i<=L)
TRAVEL_COEFF PEN
# then, for j = 1..N:
a_j b_j k_j release_j weight_j numWindows_j
o_1 c_1 o_2 c_2 ... o_numWindows_j c_numWindows_j
```

## Output (stdout)
First, one itinerary per tug `t = 0..T-1`, each a token stream
`m t_0 x_0 t_1 x_1 ... t_{m-1} x_{m-1}` (`m>=1` waypoints, strictly increasing times,
`t_0=0`, `x_0` = that tug's given start position, every consecutive pair satisfying
`|x_{i+1}-x_i| <= t_{i+1}-t_i`). Position at any time between two waypoints is the exact
linear interpolation; after the last waypoint the tug is assumed parked there forever.

Then one line `S` (number of ships you claim to serve), followed by `S` lines
`j s_j tug_1 ... tug_{k_j}` — 1-indexed ship id, integer start tick, and exactly `k_j`
**distinct** tug ids (0-indexed). For each such claim your tugs' itineraries must place them
*exactly* at `a_j` at time `s_j` and *exactly* at `b_j` at time `s_j+d_j` (checked with exact
rational arithmetic), `s_j >= release_j`, `[s_j, s_j+d_j]` must lie inside one of ship `j`'s
tide windows, and no tug may be claimed by two ships whose `[s,s+d]` intervals overlap. Any
violation anywhere makes the whole submission score 0.

## Objective (minimize)
`cost = TRAVEL_COEFF * (total distance travelled by all tugs, summed over every itinerary
segment you output, including empty repositioning and the tow itself)`
`+ sum over SERVED ships of weight_j * (s_j - release_j)`
`+ sum over UNSERVED ships of weight_j * PEN`.

Lower is better. The checker computes your `cost = F`, builds its own simple feasible
reference plan with cost `B` (fixed round-robin team assignment, no chaining insight), and
prints `Ratio: <r>` with `r = min(1000, 100*B/max(1e-9,F)) / 1000` — so matching the
reference scores ≈0.1, and doing much better climbs toward (but stays below) 1.0.

## Feasibility (checked strictly)
Malformed tokens, out-of-range/non-finite numbers, wrong waypoint count, non-monotone times,
speed violations, wrong team size, duplicate ship claims, rendezvous point mismatches, window
violations, or a double-booked tug all make the checker print `Ratio: 0.0`.

## Worked example (illustrative shape only)
2 tugs at positions 0 and 10. One ship needs `k=2` tugs from `a=4` to `b=4+d`. A submission
where both tugs walk to `a`, wait if needed, tow together, and are free for the next ship is
feasible; a submission with only one tug present for part of the tow is not.

## Constraints
`1<=T<=12`, `1<=N<=10`, `0<=L<=1000`, `1<=k_j<=T`, `1<=d_j`, all times/positions bounded well
under 10^4. Time limit 5s, memory 512MB.
