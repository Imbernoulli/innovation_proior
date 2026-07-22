# Lock Master's Parity Plan

## Problem
You run the single chamber of a canal lock for one day. `N` boats need to
pass through. Boat `i` has a **direction** `dir_i` (`0` = upbound, `1` =
downbound), an **arrival time** `a_i` (it is not at the gate before this),
a **deadline** `d_i`, and a **lateness weight** `w_i`.

A **lockage** moves a batch of up to `K` boats through the chamber at once,
but every boat in one lockage must share the same direction (the chamber
only flows one way per cycle). A lockage can start no earlier than the
arrival time of every boat riding it, and the chamber needs `L` time units
to reset before the next lockage can start (successive lockage start times
must differ by at least `L`).

**Water cost.** The chamber remembers only one bit of state: the direction
of the *previous* lockage (the day starts with the chamber already primed
in direction `s0`, given in the input). If a lockage runs the **same**
direction as the previous one, the chamber must be fully drained and
refilled against gravity: cost `W_same`. If it runs the **opposite**
direction, the previous fill is reused for free flow-through: cost
`W_diff`. `W_diff` is much smaller than `W_same` — repeating a direction is
the expensive move, not the cheap one, no matter how it looks in terms of
"number of setups."

**Lateness cost.** Every boat that completes (its lockage's start time)
after its deadline pays `w_i * (completion - d_i)`; boats on time pay 0.

## Input (stdin)
```
N K L W_same W_diff s0
dir_1 a_1 d_1 w_1
...
dir_N a_N d_N w_N
```
Boat `i` (1-indexed by input order) has direction `dir_i`, arrival `a_i`,
deadline `d_i`, weight `w_i`. `s0` is 0 or 1.

## Output (stdout)
```
M
t_1 dir_1 k_1 id_{1,1} ... id_{1,k_1}
...
t_M dir_M k_M id_{M,1} ... id_{M,k_M}
```
`M` lockages, each with a start time, a direction, a boat count
`1 <= k_j <= K`, and the list of boat ids (1-indexed) riding it. Every boat
`1..N` must appear in **exactly one** lockage.

## Feasibility
Output is worth `Ratio: 0.0` if: token count/parse fails, any value is
non-finite; `k_j` is outside `[1, K]`; any listed id is out of range,
repeated anywhere in the whole output, or missing; a lockage's declared
direction doesn't match every rider's true `dir_i`; a lockage's start time
is earlier than the arrival time of any of its riders; start times are not
non-decreasing across the printed order; or two consecutive lockage start
times differ by less than `L`. `M` above `3*N` is also rejected.

## Objective
Let lockage `j`'s cost be `W_same` if its direction equals the previous
lockage's direction (or `s0` for `j=1`), else `W_diff`. Minimize
```
F = sum_j (setup cost of lockage j) + sum_i w_i * max(0, completion_i - d_i)
```
where `completion_i` is the start time of the lockage carrying boat `i`.

## Scoring
The checker builds its own reference `B`: batch each direction to capacity
`K` (earliest-deadline-first within a direction, so it is not gratuitously
late), but run **all** of one direction's lockages before ever switching to
the other. This reference is deadline- and capacity-aware, but completely
blind to the water-parity bit — every lockage in a block after the first
repeats the previous direction.
```
sc = min(1000.0, 100.0 * B / max(1e-9, F))
Ratio = sc / 1000.0
```
Matching `B` scores `0.1`; cutting the cost to a tenth of `B` caps at `1.0`.

## Constraints
`1 <= N <= 60`, `2 <= K <= 6`, `1 <= L <= 20`, `W_diff < W_same <= 500`,
`0 <= a_i <= d_i <= 2000`, `1 <= w_i <= 10`. Time limit 5s, memory 512MB.

## Example
`N=3, K=2, L=5, W_same=100, W_diff=10, s0=0`. Boats: `(dir=1,a=0,d=3,w=5)`,
`(dir=0,a=0,d=100,w=1)`, `(dir=1,a=0,d=3,w=5)`. Sending boats 1 and 3
(both down) together at `t=0` (direction `1` != `s0=0`, cost `W_diff=10`,
both on time) then boat 2 alone at `t=5` (direction `0` != previous
direction `1`, cost `W_diff=10`, on time) gives `F=20`. Sending them one at
a time in input order instead would cost `10 + 100 + 100 = 210` in setups
alone — batching same-direction boats together, not spreading every
lockage out, is what saves this example; the general trap is the reverse:
when many boats share one direction, batching them all into consecutive
lockages is what maximizes setup cost.
