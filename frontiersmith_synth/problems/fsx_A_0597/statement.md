# One Furnace, Many Alloys: Shared Addition-Chain Batch Exponentiation

## Problem
A workshop must forge many alloys from a single ingot `g`. Alloy `i` is the power
`g^{e_i}`. The only operation is **fusing two already-forged pieces**: from pieces
holding `g^x` and `g^y` you may forge a new piece holding `g^{x+y}` (one *fusion*).
You start with the single piece `g^1`. Design a schedule of fusions that, somewhere
among the pieces it forges, produces `g^{e_i}` for **every** exponent `e_i` in the
batch. Minimize the total number of fusions.

Formally: output a straight-line multiplication program over the formal symbol `g`.
Each line multiplies two earlier values (their exponents add). Every target exponent
must appear as the exponent of some value. Fewer lines is better.

## Input (stdin)
```
k
e_1
e_2
...
e_k
```
`k` positive integers (`8 <= k <= 14`), one target exponent per line. Exponents are
large (up to ~40 bits). They are **not** random: the batch was planted with hidden
algebraic structure — a shared quotient/residue skeleton — that a schedule can exploit
to fuse many alloys from one common chain. Discovering that structure is the point.

## Output (stdout)
```
L
a_1 b_1
a_2 b_2
...
a_L b_L
```
Index `0` denotes the initial piece `g^1`. Line `n` (for `n = 1..L`) forges value with
index `n`, whose exponent equals `exponent[a_n] + exponent[b_n]`. You must have
`0 <= a_n, b_n < n` (both operands already forged). `L` is your fusion count.

## Feasibility
`L` within `[0, 100000]`; every operand index in range; every produced exponent within
the size bound; and every target `e_i` equal to the exponent of some produced value.
Any violation scores `Ratio: 0.0`.

## Objective (minimize)
`F = L`, the number of fusion lines.

## Scoring
The checker computes a baseline `B` = the cost of the naive schedule that forges each
exponent **independently** by square-and-multiply, with no sharing between alloys:
`B = sum_i ( bitlength(e_i) - 1 + popcount(e_i) - 1 )`. Your score is
```
Ratio = min(1.0, 0.1 * B / F)
```
so the naive independent schedule scores about `0.1`, and a schedule ten times shorter
saturates at `1.0`. The exponents' hidden modulus/residue skeleton (the "furnace
schedule") is **not** given — the coefficients live in the input and must be recovered
from the numbers themselves.

## Constraints
Deterministic integer arithmetic only. Time limit 5s, memory 512 MB. Each input <= 5 MB.

## Example (worked score)
Suppose `k = 2` with `e = [45, 90]`. Naive independent binary method:
`45 = 101101_2` costs `5 + 3 = 8` fusions; `90 = 1011010_2` costs `6 + 3 = 9`, so
`B = 17`. But `90 = 2 * 45`, so forge `g^{45}` (8 fusions) then one doubling gives
`g^{90}` — `F = 9`, `Ratio = min(1, 0.1 * 17 / 9) = 0.188`. Spotting that `90` reuses
the `g^{45}` chain already beats the naive baseline; the planted batches hide deeper
shared skeletons worth far more.
