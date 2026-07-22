# Levee Budget Against a Storm Sweep

## Problem

A river valley is modeled as `N` cells in a line, indexed `0..N-1`. Cell `i`
has ground elevation `e[i]` and protects an asset worth `v[i]` (both
non-negative integers). You control a **height-budget**: add a non-negative
integer number of levee-height units `h[i]` to any cells, subject to a total
budget `sum(h[i]) <= Budget`. The raised elevation is `H[i] = e[i] + h[i]`.

Nobody knows which storm the coming season will bring, so your levee must
survive a published **sweep of `K` storms**. Storm `k` is a triple
`(a_k, b_k, V_k)`: it drops a total rain volume `V_k` spread evenly over the
contiguous segment `[a_k, b_k]`.

**Flow physics.** After a storm's rain falls, water finds its equilibrium:
it pools in whatever local basin catches it, and once a basin's water rises
to the height of its lowest bounding ridge, the excess spills over that
ridge into whichever basin lies beyond -- which may itself then fill and
spill further into the *next* basin, and so on. Cells `0` and `N-1` both sit
at an open boundary: water reaching them at their own natural height simply
drains away and never pools above it. This is exact "fill-and-spill"
routing, not independent per-cell capping -- raising one ridge does not make
its water vanish, it just forces the same finite volume to find a different
(and possibly farther) escape route, potentially straight into a basin that
used to be safe.

**Damage of storm k** = the total value-weighted depth of water sitting in
every cell once the storm's rain has fully settled: `sum` over flooded cells
`i` of `v[i] * depth[i]`. **Your objective** = the *worst* (maximum) damage
over all `K` storms in the sweep. Choose `h[]` to minimize this worst case.

## Input (stdin)
```
N Budget K
e[0] e[1] ... e[N-1]
v[0] v[1] ... v[N-1]
a_1 b_1 V_1
...
a_K b_K V_K
```
`0 <= e[i] <= 60`, `1 <= v[i] <= 620`, `0 <= a_k <= b_k <= N-1`,
`V_k >= 1`. `K = 24`. `N` up to a few hundred.

## Output (stdout)
`N` non-negative integers `h[0] ... h[N-1]` (whitespace/newline separated),
with `sum(h[i]) <= Budget`.

## Feasibility
Exactly `N` tokens, each parses as a finite non-negative integer, and their
sum does not exceed `Budget`. Any violation scores `0`.

## Scoring
Let `F` be your worst-case damage as defined above (smaller is better). The
checker also builds its own simple reference allocation `B` (spreading the
budget evenly over all cells) and its damage `F_B`. Your score is
`min(1000, 100 * F_B / max(1e-9, F)) / 1000`, so matching the reference
scores about `0.1`, and beating it scores higher (capped at `1.0`). The
budget is never enough to drive every storm's damage to zero, so there is
always headroom to do better.

## Constraints
Time limit 4s, memory 512MB per test. `K = 24` storms per test case, 10 test
cases of increasing size.

## Example (worked, illustrative only -- smaller than the real tests)

`N=7, Budget=8`, elevations `e = [30, 2, 2, 2, 15, 3, 30]`,
values `v = [1, 1, 1, 1, 1, 400, 1]`, one storm `(1, 3, 45)`: 45 units of
rain fall only on cells `1..3` (a basin boxed in by cell `0`, height 30, on
the left and the ridge at cell `4`, height 15, on the right). Cell `5` (the
one high-value asset) gets no direct rain at all -- it only floods if the
basin's overflow reaches it. With `h=0` the checker reports damage `2439`
(the basin overflows the ridge and floods cell 5 deeply). Spending the
whole budget of 8 directly on cell 5 only cuts damage to about `1801` --
it still sits below the overflow's peak. Spending the *same* 8 units on
the ridge at cell 4 instead raises the basin's containment enough that far
less spills over, cutting damage to `45` -- a chokepoint fix beats
individually patching the flooded cell. On the real, larger instances with
a 24-storm sweep, only some storms are big or wide enough to expose which
chokepoint is the real bottleneck; the rest are red herrings.
