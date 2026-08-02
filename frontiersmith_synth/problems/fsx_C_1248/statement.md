# Stage Partition and Forwarding Budget: Minimal Total Execution Time

## Problem

A CPU's per-instruction combinational datapath is a fixed sequence of `N` logic
blocks, in program order, block `i` having delay `c_i` (positive integer,
picoseconds). You design the pipeline: cut this sequence into `S` contiguous,
non-empty **stages** (`1 <= S <= N`). The clock **cycle time** is

```
T = max_over_stages(sum of c_i in that stage) + L
```

where `L` is a fixed per-stage latch/setup overhead. Splitting into more,
finer stages can only shrink `T` (or leave it unchanged) — the finest split
(`S = N`, one block per stage) minimizes `T`. That is "peak clock frequency."

But `T` is only half the story. The workload has `K` **hazard classes** and a
**branch profile**, both given in the input:

* Each hazard class `k` has a producer block `result_block_k`, a consumer
  block `need_block_k` (`need_block_k < result_block_k`), an instruction
  **distance** `dist_k` between the two dependent instructions, and a
  frequency `freq_k` (how many times this hazard occurs in the workload).
  Let `stage_of(b)` be the stage number containing block `b` under your
  partition, and `gap_k = stage_of(result_block_k) - stage_of(need_block_k)`.
  Without a forwarding path, class `k` costs `max(0, gap_k - dist_k)` stall
  cycles **per occurrence**. You may build a forwarding path for class `k`,
  which removes its stalls entirely but costs `gap_k` units of a fixed wiring
  **Budget** (sum of costs of all forwarding paths you build must not exceed
  Budget). Deeper, finer partitions make every `gap_k` — hence every
  forwarding path's cost — larger.
* A branch resolves once its condition reaches block `resolve_block`. Every
  misprediction (there are exactly `Mb` of them, out of `Br` branches, both
  given) flushes every stage before and including the resolve stage, costing
  `stage_of(resolve_block) - 1` cycles **per misprediction**. Deeper
  partitions push `resolve_block` into a later stage number, so this
  penalty also grows with stage count.

Total cycles = `I + (sum over unforwarded classes of freq_k * stalls) +
Mb * (stage_of(resolve_block) - 1)`, where `I` is the instruction count.
**Total execution time = total_cycles * T.** Minimize it.

## Input (stdin)

```
N K L Br Mb resolve_block Budget I
c_1 c_2 ... c_N
need_block_1 result_block_1 dist_1 freq_1
... (K lines)
```
All values are non-negative integers; `1<=N<=13`, `2<=K<=6`,
`1<=resolve_block<=N`, `1<=need_block_k<result_block_k<=N`.

## Output (stdout)

```
S
cut_1 cut_2 ... cut_{S-1}
P
fwd_1 fwd_2 ... fwd_P
```
`S` is the stage count (`1<=S<=N`). The `S-1` cut points are strictly
increasing integers in `[1,N-1]`: blocks `1..cut_1` form stage 1,
`cut_1+1..cut_2` form stage 2, etc. (omit the line, i.e. print nothing, if
`S=1`). `P` is how many forwarding paths you build, followed by `P` distinct
hazard-class indices in `[1,K]`.

## Feasibility

Cut points must be strictly increasing and in range; the forwarding index
set must be distinct, in range, and its total cost (`sum of gap_k` over
chosen `k`, computed under YOUR partition) must not exceed `Budget`. Any
violation, or non-finite/non-integer tokens, scores `Ratio: 0.0`.

## Scoring

Let `F` be your total execution time. The checker computes its own baseline
`B` = the execution time of the no-pipelining construction (`S=1`, no
forwarding needed since every block shares one stage). Score
`Ratio = min(1.0, 0.1 * B / F)` (lower `F` is better; matching `B` scores
`0.1`; 10x better than `B` saturates at `1.0`).

## Example

`N=3, K=1, L=1, Br=1, Mb=1, resolve_block=3, Budget=2, I=6`, delays
`[2,3,1]`, hazard class `need=1 result=3 dist=1 freq=2`. Baseline:
`B = 6 * (2+3+1+1) = 42`. Artifact `S=3` (cuts `1 2`), forward class 1
(`0` skipped, `1` chosen): `T = max(2,3,1)+1 = 4`; `gap=3-1=2<=Budget`;
stalls `=0`; branch penalty `=1*(3-1)=2`; cycles `=6+0+2=8`; `F=32`;
`Ratio = min(1, 0.1*42/32) = 0.13125`.

## Constraints

`N<=13`, `K<=6`, all counts fit in 32-bit ints, time limit 5s.
