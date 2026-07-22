# Regatta Wake Fanout

## Problem
A fleet of `B` boats races across an `N x N` grid course (rows and columns
`0..N-1`). Every boat starts at column `0`, at its own given row, and must
finish by reaching column `N-1` after crossing all `G` gates. Gate `g` is a
column `col_g` together with a row band `[rowLo_g, rowHi_g]`: a boat satisfies
gate `g` the instant it occupies any cell `(row, col_g)` with
`rowLo_g <= row <= rowHi_g`. Gates may be crossed in **any order**.

Each tick, a boat either **waits** in its current cell (cost 1 tick, no field
check) or **moves** one step `U/D/L/R` into an adjacent cell (must stay on
the grid). Boats churn the water: every cell remembers the ticks at which
any boat has entered or waited in it. A move is priced at the instant it is
attempted, using its UN-delayed candidate tick `ta = current_tick + 1` (what
the arrival would be if this move cost exactly 1 tick). Let `k` = the number
of entries into that cell (any boat, including this boat's own past visits)
already on the books by `ta` and not yet decayed: entry tick `e` with
`0 <= ta - e <= w`. A rival's own later, wake-slowed arrival is never visible
to a boat deciding at the same instant. The move costs `1 + k` ticks, capped
at `CAP`, and this boat's entry is then recorded at its true (possibly
delayed) arrival for everyone's future queries. Ties resolve by boat index:
of two boats deciding a move into the same cell from the same current tick,
the lower-indexed one's entry counts for the higher-indexed one. A fleet
that spreads out in space and time keeps `k` near 0; one that funnels
through the same cells at the same times drives `k`, and the cost, up fast.

A boat may delay its own departure: it "starts" at a chosen tick
`0 <= start <= Smax` (free of direct penalty other than the delay itself),
before which it deposits no wake and occupies no cell.

## Input (stdin)
```
N B G w CAP Smax maxMoves
start_row_1 ... start_row_B
col_1 rowLo_1 rowHi_1
...
col_G rowLo_G rowHi_G
```
Gates are listed in an arbitrary order (not sorted by column).

## Output (stdout)
`B` lines, one per boat in input order:
```
start_i moves_i
```
`start_i` is boat `i`'s departure tick; `moves_i` is a string over
`{U,D,L,R,S}` (`S` = wait), length at most `maxMoves`.

## Feasibility
For every boat, simulating its moves from tick `start_i` at `(start_row_i, 0)`
under the shared wake field described above must:
- never leave the grid,
- end with the boat at column `N-1`,
- have visited every one of the `G` gates at some point (any order).
Any violation (including a malformed line, non-integer start tick,
illegal move character, or an out-of-range start tick) scores `Ratio: 0.0`.

## Objective
**Minimize** the sum, over all boats, of each boat's finish tick (the tick
at which it completes its last move).

## Scoring
Let `F` = your fleet's total finish time. The checker also builds its own
naive feasible fleet plan `B0`: every boat departs at tick 0 and sails the
identical route -- straight to the CENTER of each gate's band, in the exact
(unsorted) order the input lists the gates -- so the whole fleet converges
onto one lane. Score:
```
sc = min(1000, 100 * B0 / F)
Ratio = sc / 1000
```
Reproducing the naive single-lane fleet scores about `0.1`. Spreading the
fleet across lanes and staggering departures scores progressively higher;
there is no known easy optimum, and the best trade-off between using more
lanes vs. more time depends on `w`, the gate bands' widths, and `B`.

## Constraints
- `6 <= B <= 8`, `2 <= G <= 4`, `17 <= N <= 35`
- `7 <= w <= 8`, `CAP = 8`
- each gate's row band has width `>= 2`; every boat starts on the same row

## Example
2 boats, both starting at row 2 on a 6-wide course, one gate at column 3
spanning rows `[2,3]`, `w = 3`. Naive plan: both sail straight along row 2,
tick 0 -- boat 0 (lower index) finishes at tick 5, but boat 1 enters the
same cells at the same ticks right behind it, so most of its moves cost
`1+1=2`; it finishes at tick 9 (total `5+9=14`). Sending boat 1 one row down
onto row 3 (the gate's other lane) instead: it touches no cell boat 0
touched, pays only the 1-tick detour, and finishes at tick 6 (total
`5+6=11`).
