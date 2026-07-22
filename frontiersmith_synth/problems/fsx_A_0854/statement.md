# Shared XOR Circuits for Many Linear Forms

## Problem
You are given `R` target parity (XOR) functions over `C` boolean input variables
`x_1..x_C`. The `i`-th target is specified by a set of variable indices; its value is the
XOR of exactly those variables. Your job is to realize ALL `R` targets simultaneously with
a single **straight-line XOR circuit**: a sequence of two-input XOR gates, where each
gate's inputs are either an original variable or the output of an earlier gate ("wire"),
and every target must be produced EXACTLY by the value of some wire in the circuit.

Each gate costs exactly one XOR operation. Building every target independently, bit by
bit, wastes work whenever two or more targets secretly agree on a shared partial sum — a
combination of variables that recurs across several targets. The circuit should commit to
computing such a shared partial sum ONCE, as an intermediate wire, and then extend or
reuse that wire wherever it recurs, instead of recomputing the same combination from
scratch for every target that needs it. A partial sum only pays off once several targets
are built to depend on it, and committing early to the wrong intermediate forecloses
better sharing opportunities discovered later — so minimizing total gates is a genuine
combinatorial search over WHICH partial sums to materialize and reuse, not something a
pass that finishes one target at a time can plan for.

## Input (stdin)
```
R C
```
then `R` lines, the `i`-th formatted as
```
k v_1 v_2 ... v_k
```
— target `i` is the XOR of variables `v_1 < v_2 < ... < v_k` (1-indexed, strictly
increasing, `2 <= k <= C`).

## Output (stdout)
```
G
```
followed by `G` gate lines `a b` (two integers each), and finally `R` lines, each holding
a single integer `w_i`.

Wires are numbered `1..C+G`: wires `1..C` are the input variables (wire `v` is `x_v`);
for `i = 1..G`, wire `C+i` is defined by gate line `i` as `wire[a] XOR wire[b]`, where `a`
and `b` must each reference an EARLIER wire (`1 <= a,b < C+i`) and `a != b`. The final `R`
integers `w_1..w_R` claim that wire `w_i` (`1 <= w_i <= C+G`) computes target `i`.

## Feasibility
Simulate every wire's value as the XOR (over GF(2)) of the input variables it transitively
depends on. The output is feasible iff, for every `i`, wire `w_i`'s value equals target
`i` **exactly** (the same set of variables). Any parse error, wrong token count,
out-of-range or self-referencing gate, non-finite/non-integer token, or a mismatched
target scores `0`.

## Objective
Minimize `G`, the total number of XOR gates.

## Scoring
Let `B` be the number of gates a fully independent construction needs: build each target
bit-by-bit with zero sharing, costing `k_i - 1` gates; `B = sum(k_i - 1)`. With your gate
count `G`,
```
Ratio = min(1, 0.1 * B / G)
```
The independent baseline scores `0.1`. Halving the gate count roughly doubles the ratio.

## Constraints
- `1 <= R <= 60`, `2 <= C <= 60`; each target has `2 <= k <= C`.
- `0 <= G <= 20000`.
- Deterministic exact GF(2) scoring; no timing.

## Example
`C = 4`, two targets: `{1,2}` and `{1,2,3}` (`B = (2-1) + (3-1) = 3`).

A circuit with `G = 2`:
```
gate 1: 1 2      -> wire 5 = x1 XOR x2      (equals target 1)
gate 2: 5 3      -> wire 6 = wire5 XOR x3   (equals target 2, reusing wire 5)
```
Output: `2` / `1 2` / `5 3` / `5` / `6`. This scores `min(1, 0.1*3/2) = 0.15`, beating the
independent construction (`G=3`, ratio `0.1`) because target 2 reused target 1's wire as a
shared prefix instead of rebuilding `x1 XOR x2` from scratch.
