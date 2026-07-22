# Batch Power Precompute on Shared Rails

A fixed-point hardware unit precomputes a table of powers of a secret base `x`.
You must produce the exponents `t_1, ..., t_K` as a **straight-line program of
additions**, starting from the constant `1`. Every instruction costs one pipeline
slot; the fewer additions, the cheaper the fabrication run.

## Problem

Register `r_0` permanently holds the value `1` and is free. One instruction
allocates the next register `r_i` (i = 1, 2, ...) and stores

```
r_i = r_a + r_b        with 0 <= a, b < i
```

i.e. the sum of any two *previously* computed registers (the same register may be
used twice, which doubles a value). A single program that reaches all targets is a
*shared (vectorial) addition chain*: intermediate registers computed once can be
reused by every target that needs them.

Your task: given the `K` distinct target integers, output a program whose
registers contain **every** target value, using as few instructions as possible.
Values not equal to any target are allowed as intermediates, but no register may
exceed `T = max(t_i)` (a value larger than every target can never help).

## Input (stdin)

```
K
t_1 t_2 ... t_K
```

- `1 <= K <= 20`
- `2 <= t_i <= 4000`, all distinct.

## Output (stdout)

```
L
a_1 b_1
a_2 b_2
...
a_L b_L
```

`L` = number of instructions (`0 <= L <= 20000`). Line `i` (1-based) defines
`r_i = r_{a_i} + r_{b_i}` with `0 <= a_i, b_i < i`.

## Feasibility

- Every operand index in range, every register value `<= T = max(t_i)`.
- Each target `t_j` equals some register value.
- Any violation, malformed token, or garbage scores `0`.

## Objective and Scoring

Minimize `F = L`, the instruction count. The checker computes an internal
baseline `B` = the total op count of computing **each target independently** with
the classic binary method (`cost(e) = floor(log2 e) + popcount(e) - 1`, summed
over all targets, no sharing). Then

```
Ratio = min(1, 0.1 * B / F)
```

so the independent-binary baseline scores exactly `0.1`, a program `c` times
shorter scores `0.1 * c`, capped at `1.0`. Sharing intermediates across targets is
the only way to score well above `0.1`.

## Example (illustrative; not an actual test)

Input:

```
3
13 26 27
```

Output:

```
7
0 0
1 0
2 2
3 3
4 0
5 5
6 0
```

Trace: `r_1=2, r_2=3, r_3=6, r_4=12, r_5=13, r_6=26, r_7=27`. All three targets
appear, every operand index `< i`, and every value `<= 27`, so the program is
feasible with `F = 7`.

Baseline: `cost(13)=3+3-1=5`, `cost(26)=4+2-1=5`, `cost(27)=4+3-1=6`, so
`B = 16`. Score: `Ratio = min(1, 0.1 * 16 / 7) = 0.228571`.

Note how `26` and `27` were built from the shared `13` in one instruction each:
computing all three targets independently would have needed 16 instructions.
That reuse is the whole game.

## Constraints

- Time limit 2 s, memory 512 MB. Exact integer arithmetic; scoring is fully
  deterministic.
- The instances reward discovering a **shared build DAG**: the target sets hide
  common rails (multiplicative/additive structure across targets) that make
  per-target textbook chains far more expensive than a joint chain.
