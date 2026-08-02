# Bundle Packing Against the Register File

## Problem
A VLIW machine issues one **bundle** per cycle. A bundle has `W` fixed slots;
slot `s` only ever accepts operations of one fixed type (e.g. `'A'` for an ALU
op, `'M'` for a memory/multiply op) — the type of slot `s` never changes across
cycles. You are given a dependence DAG of `N` operations: operation `i` has a
required slot type, a **latency** (the number of cycles after it issues before
its result is usable), and a list of predecessor operations whose results it
consumes. Your job is to place every operation into a `(cycle, slot)` bundle
slot.

The machine also has a register file holding only `R` values at once. An
operation's result must live in a register from the cycle it issues until the
cycle its **last** consumer issues (an operation nobody consumes dies the
instant it issues). If, at some point in your schedule, more values are
simultaneously alive than `R` allows, the excess values must be *spilled* —
this problem does not ask you to write the spill code yourself; the checker
computes, from your schedule alone, the **provably minimum** number of values
that must be spilled to fit the register file (an exact sweep over the live
ranges your schedule implies — see `counter.py`). Each spilled value costs a
fixed, instance-specific penalty (store+reload traffic).

Packing every bundle as full as possible minimizes cycle count — but the
faster you finish computations, the more of their results pile up alive at
once, waiting for something slow (e.g. a single scarce slot type, or a chain
of dependent consumers) to retire them. Sometimes it is cheaper to leave a
slot idle for a cycle than to open a live range you can't afford to keep in
registers.

## Input (stdin)
```
N W R SPILL_COST
<slot type string of length W, one char per slot>
```
then `N` lines, one per operation `i = 1..N` (1-indexed, DAG order — every
predecessor index is smaller than `i`):
```
type_i latency_i k_i  p_1 p_2 ... p_{k_i}
```
`type_i` is a single character (present somewhere in the slot-type string),
`latency_i >= 1`, and `p_1..p_{k_i}` are `i`'s predecessor indices.

## Output (stdout)
`N` lines, one per operation in order `1..N`:
```
c_i s_i
```
— the cycle (`>= 1`) and slot index (`0 <= s_i < W`) operation `i` issues in.

## Feasibility
All of the following must hold, or the output scores `0`:
- Every token parses as a finite integer; exactly `2N` tokens.
- `slot_types[s_i]` equals `type_i` for every `i`.
- For every dependency `i -> j`: `c_j >= c_i + latency_i`.
- No two operations share the same `(cycle, slot)`.

## Objective
Minimize `total_cycles + spill_count * SPILL_COST`, where `total_cycles =
max(c_i)` and `spill_count` is the minimum number of live ranges that must be
evicted so no cycle has more than `R` values held in registers simultaneously
(computed by the checker from your `(c_i, s_i)` choices).

## Scoring
The checker also builds its own simple feasible schedule (one operation per
cycle, in DAG order, each in the first slot of matching type — no register
awareness at all) as baseline `B`, evaluated by the same formula to give
`F_base`. With your cost `F`:
```
Ratio = min(1, 0.1 * F_base / F)
```
Lower `F` scores higher. The true jointly-optimal schedule+register-allocation
is NP-hard in general and unknown for these instances, so headroom remains
above any reference solution.

## Constraints
- `1 <= N <= 200`, `1 <= W <= 8`, `1 <= R <= 16`, `0 <= SPILL_COST <= 1000`.
- `1 <= latency_i <= 8`.
- Deterministic; no timing dependence.

## Example
Suppose `N=3`, two ALU slots (`W=2`, template `"AA"`), `R=5`, `SPILL_COST=4`.
Op 1 has no predecessors (latency 1); ops 2 and 3 each depend only on op 1
(latency 1 each) and are otherwise independent of each other. The checker's
own baseline issues one operation per cycle in DAG order: op 1 at cycle 1,
op 2 at cycle 2, op 3 at cycle 3 — `total_cycles=3`, nothing is ever alive
long enough to exceed `R`, so `spill_count=0` and `F_base=3`. A schedule that
notices ops 2 and 3 can issue in the *same* cycle (op 1 at cycle 1 slot 0; ops
2 and 3 both at cycle 2, in slots 0 and 1) reaches `total_cycles=2` with
`spill_count=0`, i.e. `F=2`, scoring `min(1, 0.1*3/2) = 0.15`.
