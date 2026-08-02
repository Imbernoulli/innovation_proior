# Edge Thermocouples: Backtracking the Hot Spot

## Problem

A thin plate is discretized as an `N x N` grid of cells `(i,j)`,
`0 <= i,j < N`. Its temperature field `u` obeys a discrete 2D heat equation
with diffusivity `r` (`0 < r <= 0.24`), insulated (zero-flux) edges, and
`K` hidden point heat sources. Source `m` switches on at integer time
`t0_m` and instantaneously adds a fixed amount `A` of heat to one cell
`(x_m, y_m)`; before that instant it contributes nothing.

**Simulation** (this exact recipe is what the checker replays from your
guess — replicate it to search intelligently). Start `u` at all zeros.
For `t = 0, 1, ..., T`:
1. For every source with `t0_m == t`: `u[x_m][y_m] += A`.
2. Record the value of every **boundary cell** (`i==0`, `i==N-1`, `j==0`,
   or `j==N-1`) — these are the thermocouples.
3. If `t < T`: update every cell with the 5-point stencil using
   edge-replicated neighbors for insulation:
   `u[i][j] += r * (up + down + left + right - 4*u[i][j])`, where a
   neighbor that would fall outside the grid is replaced by `u[i][j]`
   itself (zero-flux).

You are **only** ever shown the boundary thermocouple readings, at every
time step `0..T` — never the interior. `K`, `T`, `N`, `r`, `A` are given.

**Why naive backward integration is a trap**: reversing the stencil
(`u[i][j] -= r*(...)`) to walk the boundary readings back to `t=0` is
formally reversible, but that reversal amplifies the highest spatial
frequency by a factor `(1+4r)` **per step**. Over `T` steps this factor is
exponential, so on the higher-`r` / longer-`T` cases the reconstructed
interior is swamped by numerical noise long before it reaches useful
information about where and when the sources fired. Forward simulation
from a *guess*, in contrast, is always numerically stable (it's the same
well-posed diffusion the plate itself runs) — so the reliable way to find
the sources is to **search** the space of forward simulations for one that
reproduces the observed boundary trace, not to integrate backward.

## Input (stdin)
```
N T K r A
u_b(0,0) u_b(0,1) ... u_b(0,M-1)
...
u_b(T,0) u_b(T,1) ... u_b(T,M-1)
```
`M = 4N-4` boundary cells, listed in row-major `(i,j)` order restricted to
`i==0 or i==N-1 or j==0 or j==N-1` (i.e. scan `i=0..N-1`, and within each
row `j=0..N-1`, keeping only boundary cells). Row `t` of the matrix holds
that boundary snapshot at time `t`.

## Output (stdout)
Exactly `K` lines, each `x y t0` (integers): your guessed source cell and
onset time. `1 <= x,y <= N-2` (strictly interior — sources never sit on the
boundary you observe), `0 <= t0 <= T-1`.

## Scoring
The checker forward-simulates *your* `K` sources with the exact recipe
above (using amplitude `A` at your integer `(x,y)`, onset `t0`) to get a
predicted boundary trace, and compares it to the given trace with sum of
squared differences `F`. It also computes `B`, the same mismatch for a
fixed, data-independent reference guess (`K` sources all at `(1,1)`,
`t0=0`) — a stand-in for "no idea, guess a corner". The score is
```
ratio = min(1, 0.1 * B / (F + 0.2*B))
```
(the `0.2*B` term keeps the score from saturating on a lucky exact
integer-lattice fit — you get most of the credit for a near-perfect trace
match, never all of it). Final score = mean ratio over 10 hidden test
cases, each an independent `(N,T,K,r)` instance (larger, higher-`r`
instances later in the set).

## Feasibility
Output must have exactly `3K` finite numeric tokens, each within its
stated integer range (a non-integer, out-of-range, `nan`/`inf`, missing,
or extra token scores `Ratio: 0.0`).

## Example (worked, not one of the 10 hidden cases)
`N=6,T=3,K=1,r=0.05,A=8`, one source. If a guess drove `F` to (near) `0`:
`ratio = min(1, 0.1/0.2) = 0.5` — the cap, not `1.0`. (Instances are not
guaranteed to sit exactly on the integer lattice, so expect `F` to stay
strictly positive even for a very good guess; the cap just guarantees a
lucky fit never reads as `1.0`.) A guess far from any real source gives
`F` close to (or above) `B`, so `ratio` lands near `0.1` or lower.
