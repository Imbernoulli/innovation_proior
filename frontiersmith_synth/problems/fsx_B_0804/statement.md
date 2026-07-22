# Varnish Workshop: Humidity-Coupled Coat Staggering

## Problem
A varnish workshop finishes `P` wooden panels. Panel `p` requires an ordered **chain**
of `k_p` coats: coat `1` must be applied and fully dry before coat `2` can be applied,
and so on. Coat `(p, i)` has a **base drying duration** `b_{p,i}` — the time it would
take to dry completely alone, with nobody else's varnish wet.

The workshop has a single humid room: every coat that has been applied but is not yet
dry is **wet**, and wet coats raise the room's humidity together. At any instant, if
`W` coats are simultaneously wet, the *actual* time every wet coat needs to finish
drying is stretched by a **quadratic congestion factor** `(1 + c*W)^2`, where `c > 0`
is a workshop-specific humidity constant given in the input. Concretely: a coat with
base duration `b` that stays wet throughout an interval during which `W` coats are
wet the whole time needs `b*(1+c*W)^2` real time to finish (in general `W` changes
over time as coats are applied/finish, and drying accumulates piecewise accordingly).
A worker may apply any eligible coat at any real-valued time of their choosing —
application itself is instantaneous.

## Input (stdin)
```
P
c_num c_den
k_1 b_{1,1} b_{1,2} ... b_{1,k_1}
k_2 b_{2,1} ... b_{2,k_2}
...
k_P b_{P,1} ... b_{P,k_P}
```
`c = c_num / c_den` (positive rational). All `b_{p,i}` are positive integers.

## Output (stdout)
The application time of every coat, in the **same order** as the input (panel 1's
`k_1` coats first in chain order, then panel 2's, ...): `N = sum(k_p)` numbers total,
each a non-negative number (plain integer, decimal, or `a/b` fraction), separated by
any whitespace.

## Feasibility
An output is valid iff **all** hold:
- exactly `N` tokens, each parseable as a finite non-negative rational `<= 1e9`;
- for every coat `(p,i)` with `i>1`, its application time is `>=` the (simulated)
  finish time of coat `(p,i-1)` — you cannot varnish over a wet undercoat.
Any violation scores `Ratio: 0.0`.

## Objective
Let `F` be the makespan: the time at which the **last** coat anywhere finishes
drying, under the humidity-coupled dynamics above (simulated exactly from your
submitted application times). **Minimize `F`.**

## Scoring
Let `B` be the makespan of the checker's own baseline: apply every coat strictly
**sequentially** (one wet coat in the whole workshop at a time, workshop-wide, not
just per panel) — always feasible, always `W<=1`. With minimization normalization:
```
sc = min(1000.0, 100.0 * B / max(1e-9, F))
Ratio = sc / 1000.0
```
Reproducing the sequential baseline scores `Ratio = 0.1`; a schedule `10x` faster
caps at `1.0`.

## Constraints
- `1 <= P <= 100`, `1 <= k_p <= 5`, `3 <= b_{p,i} <= 18`.
- `0 < c <= 1` (rational, small denominator).
- Time limit 5s, memory 512m.

## Example (illustrative FORM only)
One panel, chain `k=2`, bases `b=[2,3]`, `c = 1/2` (so the congestion multiplier for
`W=1` is `(1+0.5)^2 = 2.25`). Sequential baseline: apply coat 1 at `t=0`, `W=1`
throughout, it dries at `t = 2*2.25 = 4.5`; apply coat 2 at `t=4.5`, dries at
`t = 4.5 + 3*2.25 = 11.25`. So `B = 11.25`. Any schedule that also finishes everything
at `F = 11.25` scores `0.1`; a smarter schedule that manages a smaller `F` (by trading
off how many coats are wet at once against how long the room stays humid) scores
higher — but naively applying **both** coats of *many* panels at once is not free:
more simultaneous wet coats always raises `W`, and the quadratic congestion factor
means the room's humidity penalty grows faster than the parallelism gained once too
many coats are wet together. The best amount of concurrency to run is an interior
value set by `c`, not "as much as possible."

## Notes
Every `b_{p,i}` and `c` needed to reason about the congestion law is given in the
input — nothing about the *magnitude* of the coefficients is hidden, only the shape
of the tradeoff is stated here; you must read the numbers and act on them.
