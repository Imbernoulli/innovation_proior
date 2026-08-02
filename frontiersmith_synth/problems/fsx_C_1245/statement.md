# Predictable Codec: Bit-Width Blocks Under a Branch Tax

## Problem
You are designing a block codec for an integer stream that is *mostly
small*: `A[1..N]`, each `0 <= A_i < 2^30`. The codec stores the stream as
consecutive **blocks**. Each block records only the *delta from its own
minimum* (frame-of-reference coding): if a block spans `A[s..e]`, its base
is `base = min(A[s..e])`, and every element is stored as `A[i]-base` using a
fixed number of bits, the block's **width** `w`. A width-`w` block can only
hold deltas up to `2^w - 1` (a width-0 block must be perfectly constant).

Encoding a block costs a fixed **header** `H` (bits for its length, base and
width) plus `w` bits per element. Decoding reads blocks left to right; before
unpacking a block the decoder must branch on that block's width to know how
many bits to consume. A real branch predictor guesses "same width as the
previous block" — every time the width actually *changes* between two
consecutive blocks, that guess is wrong and the run pays a fixed
**misprediction tax** `C` (the first block never mispredicts).

Given the stream and constants `H`, `C`, choose a partition into blocks and a
width per block minimizing:
```
cost = sum over blocks (H + len(block) * width(block))
       + C * (number of adjacent block pairs whose width differs)
```

## Input (stdin)
```
N H C
A_1 A_2 ... A_N
```
`1 <= N <= 2000`, `1 <= H <= 200`, `1 <= C <= 5000`, `0 <= A_i < 2^30`.

## Output (stdout) — the artifact
```
M
len_1 width_1
...
len_M width_M
```
`M` blocks in order covering the stream left to right: `len_1+...+len_M = N`,
each `len_k >= 1` an integer, each `width_k` an integer with
`0 <= width_k <= 30`. Block `k` spans the next `len_k` elements after the
previous blocks; let `base_k = min` of its elements and
`d_k = max` of its elements `- base_k`. The block is **feasible** iff
`d_k <= 2^width_k - 1` (and `d_k = 0` when `width_k = 0`) — this is exactly
the condition for the block to decode back to its original elements
bit-for-bit. Any infeasible block, malformed/non-finite token, or token-count
mismatch scores `Ratio: 0.0`.

## Scoring
The checker recomputes `cost` exactly from your `(len_k, width_k)` list
(exact integer arithmetic, no tolerance), and compares it against its own
baseline `B`: the cost of a single block covering the *whole* stream at its
own minimal feasible width (no transitions to pay). For this minimization
objective:
```
Ratio = min(1.0, B / cost)        [reported after the harness's 100/1000 scaling]
```
Reproducing the baseline scores `~0.1`; a cheaper codec scores higher, up to
a cap of `1.0` at 10x cheaper than the baseline.

## Why blocks and widths interact
Splitting into more, tighter blocks shrinks the `width*len` term (each block
only pays for the bits *its own* values need) but adds header overhead and,
critically, tends to make the width sequence noisy — every extra boundary is
one more chance for the width to change from its neighbor and trigger `C`.
A partition that is excellent by raw bit count can be far worse by total
cost if the width bounces between many distinct values. Restricting the
whole stream to a small, *reused* set of widths can lose a little on the bit
term while paying the transition tax only a handful of times.

## Example
`N=12, H=2, C=20`, stream
`1 2 0 3 1000 1001 1023 1010 2 1 3 0` (three runs of 4: small, large, small).
Baseline: one block, width `10` (range `0..1023`), `B = 2 + 12*10 = 122`
(→ `Ratio: 0.100000`).

Per-run **minimal** widths are `2, 5, 2` (ranges `3`, `23`, `3`): blocks
`(4,2) (4,5) (4,2)`, cost `= (2+8)+(2+20)+(2+8) + 20*2 = 10+22+10+40 = 82`
→ `Ratio: 0.148780` (two width changes cost `40`, more than the bits saved).

Using the **same** width `5` for all three blocks (not minimal for the
small runs, but stable): blocks `(4,5) (4,5) (4,5)`, cost
`= 3*(2+4*5) + 20*0 = 66` → `Ratio: 0.184848`. Spending 3 extra bits per
small-run element buys zero transitions and wins overall.

Time limit 5s, memory 512MB.
