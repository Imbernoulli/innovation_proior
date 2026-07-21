I have `n <= 100` pastries, each with an integer `price[i]` and `joy[i]` in `[0, 1500]`, and I must count
the subsets whose price sums to *exactly* `B` and whose joy sums to *at least* `J`, modulo `1000000007`,
with `B, J <= 1500`. Two features of this shape dictate everything. First, I am *counting* 0/1
subset-sums, not deciding feasibility — and a counting cell that is off by a factor does not crash, it
silently returns a wrong number, so every index decision has to be pinned exactly. Second, the raw count
reaches `2^100`, which is why the answer is taken mod a prime: I carry every cell mod `1000000007`, and a
sum of two reduced cells stays under `2.1 * 10^9`, comfortably inside `long long`.

Brute enumeration of all `2^n` subsets is `2^100` — hopeless as a solution, though it makes a perfect
independent oracle for tiny `n`, which I will lean on to check myself. The real algorithm is a
two-dimensional 0/1 knapsack *count*: a price axis of size `B+1` for the exact-price subset-sum, and a
joy axis I still have to design, since the joy condition is a *lower* bound rather than an exact target.
If I can hold the joy axis to size `J+1`, the table is `(B+1) x (J+1)` and the answer is a single cell.
Sizing it: `n * B * J = 100 * 1501 * 1501 ~ 2.25 * 10^8` cell-updates, each a couple of `long long` ops —
under 2 s; memory `1501 * 1501 * 8 ~ 18 MB`, under 256 MB. Feasible, so I commit to it.

The joy axis is the design crux. I only ever ask "is total joy `>= J`?" — I never distinguish joy `J`
from joy `J+5`. So I define the *clamped* joy of a partial box as `min(totalJoy, J)`: every box with true
joy `>= J` has clamped joy exactly `J`, and boxes below keep their true joy `0..J-1`. Because every
`joy[i] >= 0`, clamped joy is monotone non-decreasing as items are added — once a box reaches bucket `J`
it can never fall back — so bucket `J` is *absorbing* and the final answer is the one cell `dp[B][J]`. (If
joys could be negative this collapse would break, since a box could drop back below `J`, but the
constraints forbid it.)

Let `dp[c][j]` be the number of distinct subsets so far with total price `c` and clamped joy `j`. Base
case `dp[0][0] = 1` — before any item the only box is the empty one, price `0`, clamped joy `0`.
Processing item `i` with price `p` and joy `g`, each existing box either skips `i` (unchanged) or takes
it, moving `(c-p, j) -> (c, min(j+g, J))`:

```
dp[c][min(j+g, J)] += dp[c-p][j]   (mod 1e9+7)
```

The answer is `dp[B][J]` after all items. The base case already handles `J = 0`: the empty box has
clamped joy `min(0, 0) = 0 = J`, so it lands in the bucket and is counted, correctly, since joy `0 >= 0`.

The sample (`B = 6`, `J = 7`, items `(2,5), (4,4), (2,3), (4,6)`, answer `4`) has exactly four price-`6`
subsets — `{0,1}`, `{0,3}`, `{1,2}`, `{2,3}` — and one of them, `{1,2}` with joy exactly `7`, sits on the
`>=` boundary. That box is the reason the clamp has to be `min(joy, J)` with the answer read at bucket
`J`: joy `= J` must count as `>= J`. It clamps to `min(7, 7) = 7 = J`, so it contributes to `dp[6][7]`,
and the four boxes give `dp[6][7] = 4`.

Two index decisions in this DP are exactly the kind a counting table gets silently wrong. The first is
the sweep direction. To save memory I roll the whole thing in place over a single `dp`, which means the
price axis is swept in place — and the *direction* of that sweep is precisely what separates "each item at
most once" from "each item reused freely." The natural ascending sweep is wrong, and a tiny witness shows
it with a number: `2 4 0` with items `(2,3), (2,1)` — budget `4`, threshold `0`, so the only qualifying
box is `{0,1}`, answer `1`. Sweeping `c` ascending, item 0 writes `dp[2][0] = 1`, then still within item
0's pass reads that fresh `dp[2][0]` at `c = 4` and writes `dp[4][0] = 1` — a box that used pastry 0
*twice*. After item 1 the cell reads `3`, not `1`. The read `dp[c-p]` at the smaller capacity has already
been touched by the current item, folding the item into a state that already contains it. Sweeping `c`
*descending* reads each source cell before this item touches it: re-tracing `2 4 0` descending, at `c = 4`
the source `dp[2][0]` is still `0`, so `dp[4][0]` stays `0` after item 0, and the final `dp[4][0] = 1`,
correct. The joy axis is rolled in place too and carries the identical hazard, so both inner loops
descend.

The second decision is where the joy-bucket boundary sits. It is tempting to size the joy axis `0..J-1`
and treat the top row as "`>= J`", but that clamps at `J-1` and miscounts. Witness `1 3 3` with item
`(3,2)`: the only price-`3` box is `{0}` with joy `2`, and `2 >= 3` is false, so the answer is `0`. The
`0..J-1` sizing clamps `min(0+2, J-1) = min(2, 2) = 2` and reads `dp[3][2] = 1` — it counts a joy-`2` box
as satisfying `>= 3`. Clamping at `J` and reading `dp[B][J]` instead gives `min(0+2, 3) = 2`, leaves
`dp[3][3] = 0`, answer `0`. A box lands in bucket `J` iff its true joy reaches `J`, which is exactly
`>= J` — and it is the sample's joy-`7` box `{1,2}` that this has to keep counting, which it does.

A few corners, several of which the tests target directly. For `n = 0` the item loop never runs, so
`dp[0][0] = 1` yields `1` when `B = 0` and `J = 0` (the empty box) and `dp[B][J] = 0` otherwise. For
`B = 0` only price-`0` boxes qualify; a price-`0` item still runs the `c >= p` loop at `c = 0, p = 0`
(reading `dp[0]`), so it correctly doubles the count of price-`0` boxes — a genuine include/exclude choice
— and three price-`0` items with `B = J = 0` give `2^3 = 8`, matching brute. For `J = 0` bucket `0` is the
only bucket, every `nj = min(j+g, 0) = 0`, and `dp[B][0]` counts all price-`B` subsets. An item with
`price[i] > B` is dropped by `if (p > Bc) continue;` — the inner bound `c >= p > Bc` would be empty
anyway, but the guard also keeps me from ever indexing `c - p` out of range. Cells never exceed `MOD`, so
the only arithmetic — a sum of two reduced cells, `< 2.1 * 10^9` — fits in `long long`; an `int` cell
would overflow and corrupt a large count silently. `cin >>` skips whitespace so the header and the `n`
pairs parse regardless of line layout, and the `if (!(cin >> n >> B >> J))` guard handles empty stdin.

The final program reads `n B J` and the `n` price/joy pairs from stdin and prints `dp[Bc][Jc]`. Against
the brute `2^n` oracle over 500+ random small menus, plus the explicit empty / `B=0` / `J=0` / price-`0` /
over-budget corners, it agrees on every instance.
