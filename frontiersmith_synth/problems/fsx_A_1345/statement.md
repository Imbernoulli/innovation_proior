# Claim Lines: Bracket the Unwinnable Density

## Problem
A board has `N` cells, numbered `1..N`. A **winning line** is a subset of
cells of size 2 or 3. Two players alternately claim one currently-empty
cell each turn: **Claimer** moves first and wants to end up owning every
cell of at least one winning line; **Blocker** moves second and wants to
prevent that forever (the game ends when the board is full).

You are given `K` nested families of winning lines `L_1 subset L_2 subset
... subset L_K` (a fixed pool of lines, `L_j` = its first `c_j` entries, so
adding lines only ever helps Claimer). Somewhere in this increasing
sequence the game flips, exactly once, from "Blocker always blocks" to
"Claimer always wins" — the density where the board becomes unwinnable for
Blocker. Simulating random games only gives a noisy number, nothing you can
check. Your job: for **each** level, **prove** Claimer wins (an explicit
strategy) or **prove** Blocker is safe (an explicit weighting), or skip it.

## Input (stdin)
```
N K
P
size_1 cell cell ...      (P lines, this is the pool, size in {2,3})
...
c_1 c_2 ... c_K
```
`L_j` consists of pool lines `1..c_j` (1-indexed into the printed pool
order), with `1 <= c_1 < c_2 < ... < c_K <= P`.

## Output (stdout)
```
K
j TAG ...(certificate)...      (K lines, j = 1..K in order)
```
`TAG` is one of:
- `U` — no claim for this level (nothing else on the line).
- `B` (Blocker-safe): `c_j` fractions `p/q` (positive integers), the weight
  `w_i` of pool-line `i` of `L_j` in printed order. Valid iff for every
  line `i` of size `s_i`, `w_i * 2^{s_i} >= 1`, AND the sum of all `c_j`
  weights is strictly `< 1/2` (checked with exact rational arithmetic;
  Claimer moves first, so the potential bound needs the tighter half, not
  a full unit).
- `M` (Claimer-win witness): an integer `T`, then `T` rows
  `d v_1 ... v_d m`. Here `d` is even (`0,2,4,...`), `(v_1,...,v_d)` is a
  full alternating move history (Claimer, Blocker, Claimer, ...) reaching a
  position where it is **Claimer's** turn, and `m` is Claimer's move there.
  The checker replays this table from the empty board against **every**
  legal Blocker reply at every Blocker turn: whenever it is Claimer's turn
  it looks up the exact history in your table (a missing entry fails the
  whole level) and plays `m`; whenever it is Blocker's turn it branches
  over all currently-empty cells. Valid iff every single branch reaches,
  within `N` total moves, a position where Claimer's cells fully contain
  some line of `L_j`.

## Feasibility
Wrong `K`, levels out of order, an unknown tag, a non-numeric or
out-of-range token, a non-positive `p` or `q`, an out-of-range/reused move
in a table row, or any missing/illegal branch during replay makes that
level worth 0. A corrupted top-level format (bad `K`, unparseable tokens,
trailing garbage) scores the whole output 0.

## Scoring
Let `S` = number of levels with a valid certificate (B or M). Let `l` = the
largest level certified B and `h` = the smallest level certified M. Since
`L_j` only grows, every level below `l` is genuinely Blocker-safe and every
level at/above `h` is genuinely a forced Claimer win — if `l < h`, a
**bracket bonus** `max(0, 4 - (h - l))` rewards how tight that proved gap
is. `F = S + bonus`. The checker's own baseline `B0 = 1` (it can always
certify level 1 alone, by construction the smallest, easiest family).
```
sc = min(1000.0, 100.0 * F / B0)     Ratio = sc / 1000.0
```

## Constraints
`8 <= N <= 17`, `K = 7`, line sizes in `{2, 3}`, `P <= 25`. Time 5s,
memory 512m. The instances are small enough that BOTH proof styles are
cheap to check once you have the right one — finding it is the actual
problem. A handful of simulated self-play games only ever gives you a
win-rate number, never a branch-complete table or a valid weight vector.

## Example (illustrative, not a real test)
`N=5`, one level, `L = {{1,2}, {1,3}, {4,5,1}}` (`c=3`). Claimer plays cell
`1` first (shared by the two size-2 lines). Whatever Blocker plays among
`{2,3,4,5}`, at least one of `{2,3}` stays empty; Claimer takes it next
turn and completes `{1,2}` or `{1,3}`. Table: `0 -> 1`; `1 2 -> 3`;
`1 3 -> 2`; `1 4 -> 2`; `1 5 -> 2`. This certifies `M` for that level.
(Real tests use `K=7` nested levels, not one.)

## Notes
Both certificate kinds are exact — rational weights, integer move
sequences — no tolerances, no randomness, no timing. Same submission, same
score, forever.
