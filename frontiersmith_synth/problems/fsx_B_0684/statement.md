# Shoulder-Shadow Turning: Stiffness-Aware Cut Interleaving

## Problem
A cylindrical rod of uniform stock radius `R` is chucked at one end and turned into `n`
axial bands, numbered `1..n` from the chuck (`x = 1`) out to the free end (`x = n`).
Band `i` must finish at exact target radius `target_i` (`1 <= target_i <= R-1`). You control
a lathe with `m` tools; tool `t` can remove any depth `d` with `minDepth_t <= d <= maxDepth_t`
in a single pass on a single band. Your submission is a straight-line **tape**: an ordered
list of passes `(tool, band, depth)`, each cutting `band`'s *current* radius down by `depth`.

Three rules govern a legal tape:

1. **Precedence (shoulder shadow).** For adjacent bands `i, i+1` with `target_i != target_{i+1}`,
   whichever band has the LARGER target is the outer shoulder; its material physically
   shadows the deeper neighbour. The neighbour with the smaller target may receive
   ordinary (non-final) passes at any time, but the pass that brings it EXACTLY to its
   target (its *final* pass) is only legal once every such larger-target neighbour has
   already reached its own target.
2. **Overhang chatter.** Cutting band `x` at depth `d` excites chatter proportional to
   `x * d`. This is only safe if the rod is still stiff enough near the chuck. Support
   stiffness is the thinnest cross-section between the chuck and the cut point:
   `S = min(current_1, current_2, ..., current_x)` (current radii, evaluated *before* the
   pass). A pass is legal only if `x * d <= K * S`. Cutting a proximal band down early
   lowers `S` for every band beyond it — including ones that still need deep cuts.
3. **Tool legality & no undercut.** `minDepth_t <= d <= maxDepth_t` for the tool used, and
   the pass must not remove more than the band's remaining headroom (`current - target`).

## Input (stdin)
```
n m K C R
target_1 target_2 ... target_n
maxDepth_1 minDepth_1
...
maxDepth_m minDepth_m
```

## Output (stdout)
```
P
tool_1 band_1 depth_1
...
tool_P band_P depth_P
```
`P` is the number of passes (`0 <= P <= 200000`); each line is one pass, tools/bands
1-indexed. Every band must end at exactly `target_i`.

## Feasibility
The tape is infeasible (score 0) if: any tool id/band id is out of range; any depth
violates its tool's `[minDepth, maxDepth]`; any pass undercuts (`depth >` remaining
headroom, or the band already reached its target); precedence rule 1 is violated on a
final pass; the chatter rule `x*d <= K*S` is violated; or any band ends short of, or
without ever reaching, its exact target.

## Objective
Minimize the total op-cost `F = P + C * (number of tool-change events)`, where a
tool-change event is counted every time the tool used differs from the tool used by the
immediately preceding pass (the very first pass also counts as one change — loading a
tool). Fewer, deeper passes reduce `P`; batching same-tool work reduces changes; both are
capped by the chatter budget available at the moment each pass runs, which depends on
the whole history of cuts so far, not just the current band.

## Scoring
The checker builds an always-feasible reference (unit-depth passes with one fixed tool,
in a precedence-safe order) achieving cost `B`. Your score is
`Ratio = min(1, B / F)`. Lower `F` (fewer passes, fewer tool-changes) scores higher.

## Constraints
`4 <= n <= 60`, `2 <= m <= 3`, `K = 2n`, `1 <= target_i <= R-1`, `20 <= R <= 200`,
time limit 5s.

## Example (worked score)
`n=3 m=1 K=6 C=6 R=10`, `target = 2 4 2`, tool `(maxDepth=8, minDepth=1)`.
Baseline: `B = (10-2)+(10-4)+(10-2) + 6*1 = 8+6+8+6 = 28`.
A tape `3` / `1 2 6` / `1 1 8` / `1 3 8` (finish band 2 first — it's the outer shoulder for
both neighbours — then bands 1 and 3) has `P=3`, one tool used throughout so 1 tool-change,
`F = 3 + 6*1 = 9`. `Ratio = min(1, 28/9) = 1.0` (capped) — this toy example is small enough
to saturate; real test cases are sized so no submitted tier reaches the cap.
