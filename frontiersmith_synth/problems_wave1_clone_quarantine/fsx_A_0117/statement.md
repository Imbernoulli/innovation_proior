# Orbital Phase Grid: Maximizing Resolvable Ground-Station Channels

## Problem
You are laying out the antennas of a satellite constellation's ground segment.
Every antenna is assigned a distinct integer **phase slot** on a shared timing
grid `0, 1, ..., M`. Let `A` be the set of the `n` chosen slots.

Two independent radio functions reuse this same set of slots:

- **Uplink beamforming** combines two antennas by *adding* their phase offsets;
  a command channel is resolvable exactly when its combined phase `a + b`
  (with `a, b in A`, repeats allowed, `a = b` permitted for a single antenna
  driven at double phase) is distinct from every other. The number of resolvable
  uplink channels is therefore `|A + A|`, the size of the **sumset**
  `A + A = { a + b : a, b in A }`.

- **Interferometric ranging** compares two antennas by *subtracting* their phase
  offsets; a baseline is resolvable exactly when its signed phase difference
  `a - b` is distinct. The number of resolvable ranging baselines is `|A - A|`,
  the size of the **difference set** `A - A = { a - b : a, b in A }` (a symmetric
  set that always contains `0`).

Your total resolving power is the number of **distinct** channels usable across
both functions:

```
F = |A + A| + |A - A|
```

Because the timing grid `M` is deliberately tight (roughly `0.6` of the span a
perfect collision-free layout would need), you **cannot** make all sums and all
differences simultaneously distinct. You must decide which collisions to accept.

## Input (stdin)
One line with two integers:
```
n M
```
`n` = number of antennas (slots) to place, `M` = largest phase slot index.
It is guaranteed that `M >= n - 1` (so an arithmetic run always fits).

## Output (stdout)
First line: an integer `k`, the number of slots you place (must equal `n`).
Then `k` integers (whitespace/newline separated), the chosen slots.

```
k
p_1
p_2
...
p_k
```

## Feasibility
- Exactly `n` slots must be output (`k == n`).
- Every slot must be an integer in `[0, M]`.
- All slots must be pairwise distinct.

Any violation scores `Ratio: 0.0`.

## Objective
Maximize `F = |A + A| + |A - A|` (larger is better).

## Scoring
The checker builds an internal baseline `B` from the arithmetic run
`A0 = {0, 1, ..., n-1}`, which always fits. For an arithmetic run both the sumset
and difference set have size `2n - 1`, so `B = |A0 + A0| + |A0 - A0| = 4n - 2`.
The score is
```
sc    = min(1000, 100 * F / B)
Ratio = sc / 1000
```
Reproducing the arithmetic-run baseline scores `0.1`; reaching `10x` the baseline
caps at `1.0`. The objective is graded and admits no easy optimum: within the
tight grid you must pack distinct sums and distinct differences, and the two
demands trade off against each other.

## Constraints
- `6 <= n <= 24`, `M <= 300` across the test ladder.
- Deterministic exact scoring; no randomness or timing enters the score.

## Example
Suppose `n = 4`, `M = 8`. The arithmetic run `A0 = {0,1,2,3}` has
`A0 + A0 = {0..6}` (size 7) and `A0 - A0 = {-3..3}` (size 7), so `F = 14 = B`
and `Ratio = 0.1`. The spread layout `A = {0,1,5,8}` has sums
`{0,1,2,5,6,8,9,10,13,16}` (size 10) and differences
`{-8,-7,-5,-4,-3,-1,0,1,3,4,5,7,8}` (size 13), so `F = 23`, giving
`Ratio = min(1000, 100*23/14)/1000 = 0.164286`.
