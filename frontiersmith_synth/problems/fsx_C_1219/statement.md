# Yield Before You're Told: A Window-Update Program for a Shared Link

## Problem
Your flow shares a bottleneck link with several other flows over `T` discrete
ticks (one tick ~ one RTT round). Each tick, every flow sends a window's worth
of packets into a FIFO queue of capacity `Qmax` in front of a link of capacity
`C` packets/tick; the queue serves up to `C` packets, and if the backlog would
exceed `Qmax` the overflow is dropped, split proportionally across the flows
that sent packets that tick. Instead of hand-controlling your window live, you
write a **fixed-schema window-update PROGRAM** that the judge re-executes once
per tick. The score is NOT your own throughput: it is a shared objective —
aggregate fairness and queueing delay across *all* flows, not just yours.

## Input (stdin)
```
T C Qmax n_comp
base_rtt_ego init_cwnd_ego
ALPHA BETA GAMMA
```
followed by `n_comp` competitor lines, each `TYPE p1 p2 p3 base_rtt demand`:
- `AIMD p1 p2 0`: additive-increase/multiplicative-decrease, init window `p1`,
  increase step `p2`; demand is `-1` (uncapped — its target share is `FAIR`,
  see below).
- `CONST p1 0 0`: sends a fixed `p1` packets/tick regardless of loss (an
  inelastic, delay-sensitive flow); demand `= p1`.
- `ONOFF p1 p2 p3`: sends `p1` packets/tick for `p2` ticks, then `0` for `p3`
  ticks, cycling on `t mod (p2+p3)`; demand = its average rate.

## Output (stdout) — the artifact
A straight-line register program over registers `r0..r19`, executed fresh
each tick with fixed inputs: `r0..r3` = your persistent memory (carried from
the previous tick, `0` initially), `r4` = your window last tick, `r5` = `1` if
YOUR packets were dropped last tick else `0`, `r6` = the queue **backlog
observed before this tick** (your delay signal — more backlog now means more
queueing delay), `r7` = `base_rtt_ego`, `r8` = the tick index. Instructions
(operands are a register `rN` or a bare, optionally-signed integer literal `K`):
```
ADD/SUB/MUL/DIV dst a b     MIN/MAX dst a b     LT dst a b  (1 if a<b else 0)
SEL dst c a b   (a if c!=0 else b)      MOV dst a
RESULT c m0 m1 m2 m3        (must be the single, final line)
```
`c` becomes your window for this tick (clamped to `[0, 100000]`); `m0..m3`
become next tick's `r0..r3` (clamped to `[-1e9, 1e9]`). At most 40
instruction lines before `RESULT`; unknown opcodes, bad registers, out-of-range
immediates, or a missing/misplaced `RESULT` make the artifact infeasible.

## Feasibility
Rejected (score `0`) if: the program is malformed per the grammar above, uses
an opcode/register outside the fixed set, exceeds 40 instructions, or is
missing `RESULT`. All arithmetic is exact integers; `nan`/`inf` cannot arise
from valid tokens and are treated as `0` if a non-finite cost still slipped
through (it never validly can).

## Objective (minimize)
Let `FAIR = floor(C / (n_comp+1))`. Every uncapped flow's (yours, and any
`AIMD` competitor's) target share is `FAIR`; a `CONST`/`ONOFF` competitor's
target is its `demand`. Let `J` be the Jain fairness index of each flow's
*delivered / target* ratio, `Q` the average queue backlog over the run
divided by `C`, and `U = max(0, target_total - delivered_total)/target_total`
the shortfall from everyone's aggregate target. The cost is
```
F = ALPHA*(1 - J) + BETA*Q + GAMMA*U
```
The judge builds an internal baseline `B` = the same simulator run with a
never-adapting window (constant `init_cwnd_ego` forever). Score:
```
Ratio = min(1.0, 0.1 * B / max(1e-9, F))
```
Reproducing the baseline scores `0.1`; cutting the cost to a tenth of it caps
at `1.0`. **Grabbing bandwidth is not free**: chasing your own throughput by
growing until you get dropped inflates the shared queue (`Q`) and starves
competitors below their target (`J`), both of which cost you directly —
reacting only after loss is a trap.

## Constraints
`10 <= T <= 90`, `10 <= C <= 30`, `6 <= Qmax <= 30`, `0 <= n_comp <= 5`.

## Example (worked score, illustrative shape only)
Two flows share `C=10`, so `FAIR=5`. Suppose over the run you deliver `48`
(target `50`) and the competitor delivers `20` (target `50`, i.e. it got
starved): normalized values `0.96` and `0.40`, Jain `J=(1.36)^2/(2*(0.9216+0.16))
approx 0.851`. If the average backlog is `Q=0.30` and nobody undershoots the
aggregate target (`U=0`), with `ALPHA=6, BETA=3, GAMMA=2`: `F = 6*0.149 +
3*0.30 = 0.894+0.90 = 1.794`. A passive baseline that never grows might reach
`B ~ 2.4` (lower delay, lower fairness violation, but very high `U`); then
`Ratio = min(1, 0.1*2.4/1.794) approx 0.134`. Real graded cases use `n_comp`
up to `5` and the exact `B` is whatever the judge computes for that instance.
