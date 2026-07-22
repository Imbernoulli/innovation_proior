# One Forge, Many Blades: Interleaved Temper Schedule

## Problem
A blacksmith has `W` workpieces heating on the anvil and exactly **one forge**.
Each workpiece must undergo three operations, in fixed order: a HOT operation,
then a MID operation, then a LOW operation. Every operation requires the
piece's *current temperature* to lie inside that operation's temperature
window at the instant the forge starts it. The forge can work on only one
piece at a time, and an operation, once started, occupies the forge for a
fixed duration.

Whenever a piece is **not** being worked on, it is not idle in the useful
sense -- it keeps cooling toward ambient temperature `Tamb` by Newton's law:
if a piece was last set to temperature `T` at time `t0`, its temperature at
a later time `t` is `Tamb + (T - Tamb) * exp(-k * (t - t0))`, using that
piece's own cooling constant `k`. This passive cooling is free and
simultaneous across all pieces -- while the forge attends to one piece, every
other piece keeps sliding toward its own next window on its own schedule.

If a piece has cooled *below* the window required for its next operation, the
only way to bring it back up is to **reheat** it at the forge: reheating
takes a fixed duration and resets the piece's temperature to `REHEAT_TEMP`,
but multiplies that piece's final quality by a penalty factor `PENALTY < 1`
for every reheat it receives.

You must produce a timed schedule of forge actions (operations and reheats,
one piece at a time, no overlaps) within a fixed time horizon.

## Input (stdin)
```
W
Tamb H_lo H_hi M_lo M_hi L_lo L_hi
REHEAT_TEMP REHEAT_DUR OP_DUR PENALTY HORIZON
T0_1 k_1 v1_1 v2_1 v3_1
...
T0_W k_W v1_W v2_W v3_W
```
`[H_lo,H_hi]`, `[M_lo,M_hi]`, `[L_lo,L_hi]` are the HOT/MID/LOW windows (in
that decreasing order). For piece `i`: `T0_i` is its temperature at time 0,
`k_i` its cooling constant, and `v1_i,v2_i,v3_i` the quality values earned by
its HOT/MID/LOW operations respectively. `REHEAT_DUR` and `OP_DUR` are the
fixed durations (time units) a reheat and an operation each occupy the forge.

## Output (stdout)
```
M
TYPE_1 piece_1 start_1
...
TYPE_M piece_M start_M
```
`M` is the number of forge actions. Each action is `OP` (perform the piece's
next pending operation) or `RH` (reheat the piece), a 1-indexed piece id, and
a start time. Actions are executed in the listed order.

## Feasibility
- Actions occupy the forge exclusively: action `j`'s start time must be `>=`
  the end time (`start + duration`) of action `j-1` (no overlaps).
- Every action's end time must be `<= HORIZON`.
- An `OP` action is only valid for a piece's *current* next pending operation
  (HOT, then MID, then LOW); a piece cannot be operated on twice for the same
  stage, nor operated on after finishing all three.
- At an `OP` action's start time, the piece's temperature (from its cooling
  trajectory since its last reheat, or since time 0 if never reheated) must
  lie inside that stage's window (tolerance `1e-6`).
- Any violation (including non-finite or negative times) scores `Ratio: 0.0`.

## Objective
A piece counts only if all three of its operations are completed. Its value
is `(v1+v2+v3) * PENALTY^(reheats it received)`. Maximize the sum `F` of
values over all completed pieces.

## Scoring
The checker computes its own baseline `B`: the value obtained by reheating
every piece before *every one* of its three operations (always safe, never
exploiting cooling). Then
```
sc    = min(1000, 100 * F / B)
Ratio = sc / 1000
```
Matching that always-reheat baseline scores `0.1`. Since every reheat can only
*shrink* a piece's value (by a further factor of `PENALTY`), the very best any
schedule can do on a given instance is complete every piece with **zero**
reheats, worth `1/PENALTY^3` times the baseline (e.g. `~3.64x` when
`PENALTY=0.65`, i.e. `Ratio ~0.364`) -- reaching that ceiling on every piece is
easy for a single workpiece but requires genuine interleaving once several
pieces share the one forge, and on tightly-timed instances even a good
scheduler may be forced into an occasional reheat, scoring below that ceiling.

## Example (illustrative only, not a real test)
With `W=1`, a piece starting at `T0=850` already inside `[800,900]`: doing
`OP 1 0.0` immediately is valid (no wait, no reheat). It then cools; once its
temperature enters `[400,500]`, `OP 1 t` is valid there, and again once it
enters `[150,250]`. Zero reheats gives the piece its full `v1+v2+v3`, versus
only `PENALTY^3` of that value under the always-reheat baseline (e.g. with
`PENALTY=0.65`, that is `0.65^3 ~= 0.27` of full value per piece, so a
zero-reheat piece is worth about `3.6x` its own baseline contribution). With
several workpieces sharing one forge, hitting that zero-reheat rate on more
than one piece at a time only works if you interleave: attend to whichever
piece's window is about to close, and let the others keep cooling toward
theirs in the meantime.
