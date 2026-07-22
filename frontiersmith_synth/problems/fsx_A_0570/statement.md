# One Ladder, Many Exponents: A Shared Addition Sequence

## Problem
A cryptographic accelerator must raise a fixed base to a whole **batch** of exponents
`n_1, ..., n_k` (each a 60-bit integer). Every exponentiation is compiled to a straight-line
program of *multiplications*; because the base is fixed, a multiplication of two previously
computed powers is exactly one addition on the exponents. So computing all the powers is the
same as building **one addition chain** that contains all the target exponents.

An **addition chain** is a list of values `value[0], value[1], ...` with `value[0] = 1`
where every later value is the sum of two earlier ones. Formally, step `i` (for `i >= 1`)
picks two indices `a, b` with `0 <= a, b < i` and sets `value[i] = value[a] + value[b]`.
Your chain is **feasible** iff every target `n_j` appears as some `value[i]`.

Your job: emit one addition chain that produces all `k` targets using **as few steps as
possible**. Fewer additions = fewer hardware multiplications for the whole batch.

## Input (stdin)
```
k
n_1
n_2
...
n_k
```
`1 <= k <= 120`, each target `1 <= n_j < 2^60`, targets distinct. The targets in a batch are
*related*: they are not independent random numbers. Read them and look for shared structure.

## Output (stdout)
```
L
a_1 b_1
a_2 b_2
...
a_L b_L
```
`L` is the number of steps. Line `i` gives the two source indices for `value[i] = value[a_i]
+ value[b_i]`, with `0 <= a_i, b_i < i` (index 0 is the base value 1). `1 <= L <= 100000`.

## Feasibility
Every target must equal some produced `value[i]` (index 0 counts, `value[0] = 1`). Any
out-of-range index, missing target, malformed token, or non-integer entry scores `0`.

## Objective (minimize)
The cost is the chain length `L` (the number of additions = the number of multiplications the
accelerator performs for the whole batch).

## Scoring
Let `B` be the cost of the **independent square-and-multiply** method — each target compiled
to its own binary addition chain, sharing nothing:
```
B = sum_j ( (bitlength(n_j) - 1) + (popcount(n_j) - 1) ).
```
With your chain length `L`, the score is
```
Ratio = min(1.0, 0.1 * B / L).
```
Reproducing the per-target binary method gives `L = B` and `Ratio = 0.1`. Halving the ops
doubles the score; a `10x` reduction caps at `1.0`. The optimal addition-sequence length is
not known in closed form, so there is room well beyond any reference solution.

## Constraints
Deterministic integer arithmetic only. Time limit 5 s, memory 512 MB.

## Example
Input `k = 2`, targets `n_1 = 5` (`101`), `n_2 = 7` (`111`). A shared chain:
```
4
0 0      value[1] = 1 + 1 = 2
1 1      value[2] = 2 + 2 = 4
2 0      value[3] = 4 + 1 = 5
1 3      value[4] = 2 + 5 = 7
```
The doubling ladder `1, 2, 4` is built **once** and reused by both targets. Here
`B = (2 + 1) + (2 + 2) = 7` and `L = 4`, so `Ratio = 0.1 * 7 / 4 = 0.175`. When many targets
share their low bits, assembling that common low chunk a single time (a meet-in-the-middle
split of each exponent into a high and a low half) saves far more than the doubling ladder
alone.
