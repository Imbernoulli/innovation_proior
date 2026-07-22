# Breakwater Berths: Sculpting the Empty Slips of a Ring Marina

## Problem
A circular quay has `m` berths numbered `0..m-1` (berth `m-1` is followed by berth `0`).
You must moor `n` boats (`n < m`). Boat `i` has a **home berth** `h_i` and an
**arrival lane**: it can only be moored at a berth reachable by walking forward from its
home, within a reach limit `R`. Formally boat `i` may occupy any single berth `s_i` with

```
(s_i - h_i) mod m  in  {0, 1, ..., R}.
```

Each berth holds at most one boat. The `m - n` berths left empty are **open slips**.

The harbor log is queried all day. A **present query** for boat `i` occurs `f_i` times; the
dockhand walks from `h_i` forward to `s_i`, inspecting `(s_i - h_i) mod m + 1` berths.
An **absent query** at position `g` (a boat that is *not* here) occurs `a_g` times; the
dockhand walks forward from `g` until the first **open slip** — the empty berth is the proof
of absence — inspecting `(e_g - g) mod m + 1` berths, where `e_g` is the first empty berth at
or after `g` (circularly). Open slips always exist, so every absent walk terminates.

## Input (stdin)
```
m n R
h_0 f_0
h_1 f_1
...
h_{n-1} f_{n-1}
a_0 a_1 ... a_{m-1}
```
`h_i` in `[0,m)`, `f_i >= 1`, `a_g >= 0`. A feasible mooring always exists.

## Output (stdout)
`n` integers `s_0 s_1 ... s_{n-1}` (whitespace-separated): the berth of each boat, in boat
order. Every `s_i` must satisfy the reach constraint and all `s_i` must be distinct.

## Objective (minimize)
```
COST = sum_i  f_i * ((s_i - h_i) mod m + 1)          # present-query walking
     + sum_g  a_g * ((e_g - g) mod m + 1)            # absent-query walking
```
where `e_g` is the first empty berth at or after `g`. Any infeasible output scores 0.

## Scoring
Let `B` be the cost of a reference **home-order first-fit** mooring the checker builds
itself. Your score is
```
Ratio = min(1000, 100 * B / COST) / 1000
```
so reproducing the reference gives ~0.1 and a 10x-cheaper plan caps at 1.0. Ten independent
harbors are scored; your grade is the mean Ratio.

## Why it is not a one-liner
The two terms pull against each other **through the open slips**. Present cost wants every
boat near its home, which packs berths tight and pushes all open slips into a few long runs.
But an absent query buried inside a long packed run walks a long way to the next open slip —
and the `a_g` weights are deliberately heaviest exactly where the packing makes that walk
longest. Open slips are therefore a **resource to be positioned**, not leftover space. Yet you
cannot move one open slip without moving others: sliding an empty berth backward into a hot
zone shifts an entire block of boats one berth forward (raising their present cost and eating
into their reach limits). The absent weights `a_g` couple every placement globally, so
reasoning boat-by-boat — or minimizing only the present cost — leaves the absent bill huge.

## Constraints
- `1 <= n < m`, `m <= 40000`, `0 <= R < m`.
- `1 <= f_i <= 10^4`, `0 <= a_g <= 10^4`.
- Time limit 5 s, memory 512 MB. Scoring is exact integer arithmetic and fully deterministic.

## Example (worked score, illustrative)
Suppose a small harbor yields `COST = 4200` while the checker's reference first-fit yields
`B = 8400`. Then `Ratio = min(1000, 100 * 8400 / 4200)/1000 = 0.200`. A plan that instead
opened two slips just downstream of the two busiest absent zones might reach `COST = 2100`,
for `Ratio = 0.400`.
