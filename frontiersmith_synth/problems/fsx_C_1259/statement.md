# Kernel Graph Placement: Host vs Near-Memory Offload

## Problem
A pipeline of `n` kernels (numbered `0..n-1`) must run on one of two devices:
a fast **Host** and a weaker **Near-Memory (NM)** compute unit that sits next
to main memory. Kernel `v` may depend on earlier kernels (`u -> v`, always
`u < v`); it cannot start until every dependency has finished and, if needed,
its output has been moved across devices.

Each kernel `i` has a compute cost `flops[i]` and a size `bytes[i]` for data
that lives in main memory and is private to that kernel (its own weights /
operands, not what a predecessor produced). Kernels are grouped into
contiguous **offload-granularity groups** given in the input; every kernel in
a group must be placed on the *same* device — you choose per group, not per
kernel.

**Costs (all integers, exact).** A device processes its assigned kernels
strictly in id order (one kernel at a time). For kernel `v` assigned to
device `d`:
- `ready(v)` = 0 if `v` has no dependency, else the max over dependency edges
  `(u,v,b)` of `finish(u)` **plus** `b * L_edge` if `u` and `v` end up on
  *different* devices (0 if same device) — the cost of moving `u`'s output
  across the interconnect.
- Processing time: `ceil(flops[v] / rate[d])`, where `rate[Host]` and
  `rate[NM]` are given in the input (Host is always faster). If `d = Host`,
  add `bytes[v] * L_fetch` — the cost of pulling `v`'s private data out of
  main memory into the Host (NM pays nothing here: it sits next to memory).
- `start(v) = max(ready(v), device_free_time[d])`;
  `finish(v) = start(v) + processing_time`; the device becomes free at
  `finish(v)`.

The **makespan** is `max` over all kernels of `finish(v)`.

## Input (stdin)
```
n m g H_rate N_rate L_fetch L_edge
group_of[0] group_of[1] ... group_of[n-1]      # n ints in [0,g-1], non-decreasing
flops[0] bytes[0]
...
flops[n-1] bytes[n-1]
u v b        # m lines, one dependency edge each, always u < v
```
`1 <= n <= 60`, `0 <= m`, `1 <= g <= n`, all rates/costs positive integers.

## Output (stdout)
Exactly `g` integers, each `0` (Host) or `1` (Near-Memory) — the device for
group `0, 1, ..., g-1` in order. Any other token count, non-integer token, or
value outside `{0,1}` is infeasible and scores 0.

## Objective
Minimize the makespan.

## Scoring
Let `B` be the makespan of the all-Host assignment (a fixed, always-feasible
construction the checker builds itself). With your makespan `F`:
```
Ratio = min(1, 0.1 * B / F)
```
Matching the all-Host baseline scores `0.1`; a 10x-faster makespan caps the
ratio at `1.0`. Malformed or infeasible output scores `0`.

## Constraints
- `1 <= n <= 60`, time limit 5s, deterministic exact-integer scoring.
- `bytes[i]`, `flops[i]`, `L_fetch`, `L_edge`, `H_rate`, `N_rate` are all
  positive integers given per instance — read them, don't assume fixed
  values.

## Example
Two kernels, one group each (`g=2`), no edges. `H_rate=100, N_rate=10,
L_fetch=1, L_edge=1`. Kernel 0: `flops=8000, bytes=300` (compute-dense
despite its byte count). On Host: `ceil(8000/100) + 300*1 = 380`. On NM:
`ceil(8000/10) = 800`. Host wins even though it "moves more data" than NM —
because at this instance's rates, NM's weak compute costs far more than the
fetch. A rule that offloads purely by byte size would send kernel 0 to NM and
lose; a rule that also weighs `flops[i]/bytes[i]` against the device-rate gap
gets it right. (This worked example uses a different, illustrative shape
purely to show the cost formulas — actual instances contain many kernels and
a real dependency graph.)
