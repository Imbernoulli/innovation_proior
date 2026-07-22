# Flower Patrol: Two Guards, No Long Blind Window

## Problem

A museum floor plan is a **flower graph**: one shared hub room (node `0`)
and `k` wings ("petals"). Petal `p` is a simple corridor **loop** of `L_p`
rooms that starts and ends at the hub (so it has `L_p - 1` private rooms
plus the hub; petals share no room other than the hub).

You control two guards. Each guard `g` walks a **closed, periodic route**:
a sequence of `P_g` room visits (one room per time step) that repeats
forever with period `P_g`. Between consecutive steps (including the wrap
from the last step of a cycle back to the first) the guard must either
**stay** in the same room (a wait move) or move to a room **directly
connected** by a corridor. Both guards start walking at time `0`.

An intruder can appear in ANY room at ANY time step. The checker sweeps
**every (room, appearance-time) pair** over one full combined cycle
(`L = lcm(P1, P2)`, since the whole two-guard pattern repeats with that
period) and measures how long the intruder would have to wait until some
guard reaches that room. Your job is to design both routes to minimize the
**worst** such wait over the whole grid of rooms and appearance times.

**The trap**: an obvious fix for "both guards doing the same lap and never
helping each other" is to send them around the flower in **opposite
directions**. That desynchronizes most rooms -- but reversing a route only
changes *where* along the loop each room sits, and reversing a length-`k`
sequence of petals leaves a middle petal's position relative to the other
guard **unchanged** when petal lengths are symmetric. The two guards can
keep meeting at the hub (and passing through that middle petal) in lockstep
forever, no matter how many cycles go by -- geography was fixed, timing
was not. What actually matters is the **joint phase relationship**
between the two periodic walks: padding a route with a few extra wait
steps (changing its effective period) and shifting when it starts can
interleave the two guards' hub visits even when their routes' shapes are
unchanged.

## Input (stdin)
```
k P
L_1 L_2 ... L_k
```
`k` (2..4) is the number of petals, `P` is the maximum period allowed for
EACH guard's route, and `L_1..L_k` (each `>= 3`) are the petal lengths.
Room `0` is the hub. Petal `p`'s private rooms are numbered consecutively
starting right after the previous petal's; the total room count is
`N = 1 + sum(L_p - 1)`.

## Output (stdout)
Four whitespace-separated groups of tokens, in order:
```
P1
w1_0 w1_1 ... w1_{P1-1}
P2
w2_0 w2_1 ... w2_{P2-1}
```
`P1, P2` are the two guards' periods (`1 <= P_g <= P`); `w_g` lists the
room visited at each time step of one full cycle of guard `g`.

## Feasibility
- `1 <= P1, P2 <= P`; every room id in `[0, N-1]`.
- For each guard, every consecutive pair of steps (cyclically, i.e. also
  `w_{P_g-1} -> w_0`) must be the SAME room or a room pair joined by a
  corridor.
- Every room `0..N-1` must appear in `w1` or `w2` (a room no guard ever
  visits has an infinite blind window).
- No missing/extra/non-integer tokens. Any violation scores `Ratio: 0.0`.

## Objective (minimize)
Let `L = lcm(P1, P2)`. For room `v`, let its combined visit times over one
full cycle `[0, L)` be the union of both guards' visit times to `v`
(extended periodically). `idleness(v)` is the largest cyclic gap between
consecutive combined visits. `F = max_v idleness(v)` -- the worst blind
window anywhere, at any moment.

## Scoring
The checker also builds its own naive reference: both guards patrol EVERY
petal, walking to each petal's tip and back (never using the loop-closing
corridor, so a `L_p`-room petal costs `2*L_p - 2` steps instead of `L_p`),
identically. This gives a baseline distance `B`. Your score is
`min(1.0, 0.1 * B / F)` (printed as `Ratio: <value>`).

## Constraints
`2 <= k <= 4`, `3 <= L_p`, `N <= ~20`, `P <= ~35`. Time limit 5s.

## Example (illustrative form only, not a real test case)
`k=2`, petals of length `L_1=3, L_2=3` (a "bowtie": hub `0`, petal 1 rooms
`1,2`, petal 2 rooms `3,4`, `N=5`). One valid guard-1 route of period 3:
`0 1 2` (hub -> room1 -> room2 -> back to hub, using the closing edge
`2-0`). This illustrates the mechanics only; actual instances use larger
petal sets (some with matched petal lengths, which is exactly where naive
route-reversal keeps the guards phase-locked, and some without), and the
two guards must jointly cover all rooms.
