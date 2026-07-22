# Frustrated Adhesion: Sorting Cells Into a Chosen Ring Pattern

## Problem

`L` biological cells sit on a ring (positions `0..L-1`, with `L-1` adjacent
to `0`). Each cell has one of `T = 3` types. You are told the initial
arrangement and a **target topology** to sort the cells into. You do not
move cells directly; instead you design a symmetric `3x3` **adhesion-energy
matrix** `J` (`J[a][b] = J[b][a]`, integer, `|J[a][b]| <= Jmax`), and a fixed,
deterministic **energy-minimizing dynamics** (given below) moves the cells
for you, driven purely by `J`.

**Dynamics.** Starting from the given arrangement, repeatedly sweep over all
position pairs `(i, j)`, `0 <= i < j < L`, in that fixed order. For a pair
with different cell types, compute the total interfacial energy of the (at
most 4) ring edges touching `i` or `j` before and after hypothetically
swapping the two cells; if swapping **strictly decreases** this local energy
sum, keep the swap, otherwise revert it. After a full sweep produces no
accepted swap (a local energy minimum), or after 60 sweeps, stop. This
"any-pair swap" rule models cells physically exchanging places to reduce
the ring's total interfacial energy `E = sum over ring edges (i,i+1) of
J[type(i)][type(i+1)]`.

**Target topologies** (given by `target_type`):
- `0` = **engulfment/blob**: the 3 types should each end up as one
  contiguous arc (minimum possible number of type-boundaries on the ring,
  which is exactly 3 whenever all 3 types are present).
- `1` = **interleaved**: adjacent cells should almost never share a type
  (maximum possible mixing).

**The trap**: the classical intuition for differential adhesion ("cells
that are alike should stick together", i.e. `J[a][a]` very negative/sticky,
`J[a][b]` (`a != b`) very positive/repulsive) reliably produces the
engulfment topology — that IS the textbook Steinberg sorting-out behavior.
But it is the worst possible choice for the interleaved topology: making
same-type contacts cheap and different-type contacts expensive is exactly
the opposite of what mixing needs. Reaching the interleaved topology
requires **frustrated** adhesion — off-diagonal (heterotypic) energies
LOWER than diagonal (homotypic) ones — the reverse of the intuitive rule.

## Input (stdin)
```
L T
Jmax
target_type
n_0 n_1 n_2
c_0 c_1 ... c_{L-1}
```
`L` (total cells), `T` (always 3), `Jmax` (bound on `|J[a][b]|`),
`target_type` (0 or 1), `n_0 n_1 n_2` (per-type counts, `n_0+n_1+n_2=L`),
then the `L` initial types `c_0..c_{L-1}` (each in `0..2`) in ring order.

## Output (stdout)
`T` lines, each with `T` integers: row `a` of the symmetric matrix `J`,
`-Jmax <= J[a][b] <= Jmax`.

## Feasibility
- Exactly `T*T` integer tokens, no missing/extra tokens.
- Every value in `[-Jmax, Jmax]`.
- `J[a][b] == J[b][a]` for all `a, b`.
Any violation scores `Ratio: 0.0`.

## Objective (minimize)
Run the dynamics above with your `J` from the given initial arrangement to
get a final arrangement. Let `H` = number of ring edges `(i, i+1 mod L)`
whose two endpoints have the same type in the final arrangement.
- If `target_type = 0`: gap `G = (L - 3) - H` (0 when perfectly segregated
  into exactly 3 arcs; grows as the arrangement fragments).
- If `target_type = 1`: gap `G = H` (0 when no two adjacent cells share a
  type; grows with clustering).
`F = (G + 0.5) / (L + 0.5)` (minimize; the `+0.5` keeps `F` strictly
positive so ratios stay well-defined even at a perfect match).

## Scoring
The checker also runs the SAME dynamics from the SAME initial arrangement
with its own naive reference matrix `J=0` (no adhesion preference at all,
so nothing moves) to get baseline gap/objective `B` (same formula as `F`).
Your score is `min(1.0, 0.1 * B / F)`, printed as `Ratio: <value>` (a
construction 10x closer to target than doing nothing saturates at 1.0).

## Constraints
`9 <= L <= 21`, `T = 3`, `6 <= Jmax <= 9`, each `n_i >= 3`. Time limit 5s.

## Example (illustrative form only, not a real test case)
`L=6`, counts `(2,2,2)`, arrangement `0 1 0 1 2 2`. With
`J = [[5,-5,-5],[-5,5,-5],[-5,-5,5]]` (frustrated: off-diagonal cheaper
than diagonal), the dynamics push toward alternating types, lowering `H`
(fewer same-type neighbors) — the mechanism only; real tests use larger,
harder-to-mix instances.
