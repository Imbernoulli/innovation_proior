# Monotone Register Ratchet: Minimal Addition Program for a Target Set

## Problem
You control a bank of integer registers built by a straight-line program.
Register `0` starts with value `1`. Every subsequent register is created by
exactly **one** instruction:

```
ADD i j      ->  new register := reg[i] + reg[j]
```

Registers are created in order `1, 2, 3, ...`; instruction number `k`
creates register `k`, and it may only reference registers that already exist
(`0 <= i, j < k`, and `i` may equal `j`). Registers are never overwritten or
deleted -- the machine is strictly append-only.

Every register also has a **reuse budget** (the "ratchet"): register `0` (the
free constant `1`) may be used as an operand any number of times. A register
created by instruction `t` (`t >= 1`) may be used as an operand (across the
*entire rest* of the program, in any later instructions, `i==j` counts
twice) **at most `B0 + floor(t / K) times`**, where `B0` and `K` are given in
the input. This cap depends only on *when* a register was created, and it
only grows as the program advances -- a register born late is born
generous; a register born early stays stingy forever.

Your goal: after running the program, every target value must appear in
*some* register. Minimize the total number of `ADD` instructions.

## Input (stdin)
```
N K B0
t_1 t_2 ... t_N
```
`N` distinct positive integer targets, `K, B0 >= 1`.

## Output (stdout)
```
T
i_1 j_1
i_2 j_2
...
i_T j_T
```
`T` is your instruction count, followed by `T` operand pairs (whitespace
may be split across lines freely). Instruction `k` (1-indexed) creates
register `k` with value `reg[i_k] + reg[j_k]`, and requires `0 <= i_k, j_k <
k`.

## Feasibility
A program is feasible iff:
- every instruction's operand indices are valid (append-only: no reference
  to a not-yet-created register);
- no register's reuse budget is ever exceeded;
- every target value equals the value of at least one register (0..T) when
  the program ends.

Any violation, or malformed/non-finite/out-of-range output, scores `0`.

## Objective
Minimize `T`, the number of `ADD` instructions.

## Scoring
Let `B` be the number of instructions a naive per-target construction needs:
build each target independently from register `0` via the standard binary
double-and-add method (this never touches any register's budget besides the
unconstrained register `0`, so it is always feasible). With your instruction
count `T`:
```
Ratio = min(1, 0.1 * B / T)
```
The naive per-target baseline scores `0.1`. Note that whenever several
targets share a common factor, that factor can be built once and reused --
but only within the reuse budget its creation time earns it, so *when* you
build a shared value matters as much as *whether* you build it.

## Constraints
- `1 <= N <= 40`, `1 <= K <= 20`, `2 <= B0 <= 10`.
- All targets are distinct positive integers, each `< 2*10^6`.
- `0 <= T <= 20000`; every register value stays `<= 10^15`.
- Deterministic exact-integer scoring; no timing.

## Example
Suppose `N=2 K=4 B0=2`, targets `{5, 3}`. One feasible program:
```
3
0 0     # reg1 = 1+1 = 2      (budget of reg1 = 2 + 1//4 = 2)
1 0     # reg2 = 2+1 = 3      (matches target 3)
2 1     # reg3 = 3+2 = 5      (matches target 3+2=5; uses reg1's last unit)
```
`T=3`. The naive baseline builds `3` (cost 2: double, add-1) and `5` (cost 3:
double, double, add-1) independently for `B=5`, so this program scores
`min(1, 0.1*5/3) = 0.1667`.
