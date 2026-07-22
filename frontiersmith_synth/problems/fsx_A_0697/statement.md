# Loading-Dock Shelf: Reuse Under a Known Departure Schedule

## Problem
A warehouse clerk manages one long shelf, modeled as the non-negative integer line.
Today's manifest lists `N` crates. Crate `i` has an integer width `size_i`, arrives at
integer instant `birth_i` and leaves at integer instant `death_i` (`birth_i < death_i`).
All `2N` arrival/departure instants across the manifest are distinct. The clerk already
knows the **entire** schedule before placing a single crate.

The clerk must assign every crate a shelf offset `addr_i >= 0` (an integer). Crate `i`
then occupies the interval `[addr_i, addr_i + size_i)` for its whole stay. Two crates
whose stays overlap in time (`birth_i < death_j` and `birth_j < death_i`) must occupy
disjoint shelf intervals; crates that never coexist may freely reuse the same offsets.

During the day, forklift workers also **check** some crates while they are present: the
manifest additionally lists `M` checks, each a pair `(t, c)` meaning crate `c` is checked
at instant `t` (guaranteed `birth_c <= t < death_c`). The shelf is marked off in fixed
aisles of `PAGE` consecutive units (aisle `k` covers `[k*PAGE, (k+1)*PAGE)`). A check of
crate `c` touches every aisle that `[addr_c, addr_c + size_c)` overlaps.

The clerk is graded on two things at once: how tall the shelf ends up (its **peak**,
the largest `addr_i + size_i` over all crates), and how many **distinct aisles** get
touched by checks over the whole day (crates checked often should end up clustered in
few aisles, not scattered). A per-instance weight `LAMBDA` fixes the trade-off.

## Input (stdin)
```
N M PAGE LAMBDA
size_1 birth_1 death_1
...
size_N birth_N death_N
t_1 c_1
...
t_M c_M
```
Crates are 1-indexed in input order. Each `c_j` is a crate index in `[1, N]`.

## Output (stdout)
`N` lines, the `i`-th line the integer offset `addr_i` assigned to crate `i`.

## Feasibility
- Exactly `N` integer tokens are printed (parse failure, wrong count, or a non-integer
  token scores `Ratio: 0.0`).
- Every `addr_i >= 0` and `addr_i <= 10^12`.
- For every pair `(i, j)` whose stays overlap in time, `[addr_i, addr_i+size_i)` and
  `[addr_j, addr_j+size_j)` must be disjoint.
Any violation scores `Ratio: 0.0`.

## Objective
Let `P = max_i(addr_i + size_i)` (the peak).
Let `pages` be the set of aisle indices `floor(a / PAGE)` for `a` ranging over
`[addr_c, addr_c+size_c)`, unioned over every check `(t, c)`.
```
F = P + LAMBDA * |pages|
```
Minimize `F`.

## Scoring
The checker builds its own baseline `B`: a **bump allocation** that never reuses shelf
space, `addr_i = size_1 + ... + size_{i-1}` (crate order = input order). `B` is `F`
evaluated on that placement. Then, for your submitted `F`:
```
sc = min(1000.0, 100.0 * B / max(1e-9, F))
Ratio = sc / 1000.0
```
Reproducing the bump baseline scores `Ratio = 0.1`; an `F` ten times smaller than `B`
caps the score at `1.0`.

## Constraints
- `8 <= N <= 260`, `0 <= M <= 10*N`, `PAGE, LAMBDA` small positive integers fixed per test.
- `1 <= size_i <= 9`, `0 <= birth_i < death_i < 2N`, all `2N` endpoints distinct.
- Time limit 5s, memory 512m.

## Example
`N=3, M=2, PAGE=4, LAMBDA=2`. Crates: `(size=3,birth=0,death=2)`, `(size=3,birth=1,death=4)`
(overlaps crate 1, so must not share space), `(size=3,birth=3,death=5)` (disjoint from
crate 1's stay). Checks: `(t=1,c=1)`, `(t=4,c=3)`.
A feasible placement is `addr = [0, 3, 0]` (crate 3 reuses crate 1's freed offset since
their stays never overlap). Peak `P = max(3,6,3) = 6`. Check on crate 1 touches aisle
`0`; check on crate 3 (also at offset `0`) touches aisle `0` too, so `pages = {0}`,
`F = 6 + 2*1 = 8`. The bump baseline gives `addr=[0,3,6]`, peak `9`, checks touch aisles
`{0}` (crate 1) and `{1}` (crate 3, offset 6 -> aisle 1), so `B = 9 + 2*2 = 13`.
`sc = 100*13/8 = 162.5`, `Ratio = 0.1625`. (This illustrative example is small; the
graded instances are larger and reward deliberately choosing *which* freed offset to
reuse, not merely reusing *some* offset.)
