# Heat-Exchanger Network Synthesis Under a Pinch

## Problem

A plant has `NH` **hot streams** that must be cooled (stream `i` flows from supply
temperature `THs_i` down to target `THt_i`, `THs_i > THt_i`) and `NC` **cold
streams** that must be heated (stream `j` flows from `TCs_j` up to `TCt_j`,
`TCt_j > TCs_j`). Each stream carries a heat-capacity flow `CP` (energy per
degree), so hot stream `i` has `D_i = CP_i*(THs_i-THt_i)` units of heat to give
up, and cold stream `j` needs `E_j = CP_j*(TCt_j-TCs_j)` units to absorb.

You design a network of heat exchangers (**matches**) that transfer duty
directly between a hot stream and a cold stream, without ever mixing the two
fluids. Any duty a stream does not receive/give through a match is made up by
**utility**: unmatched cold-stream duty is supplied by hot utility, unmatched
hot-stream duty is absorbed by cold utility (bought at fixed per-unit prices).

**Feasibility.** A stream may be split into parallel branches (each branch
still runs the stream's full supply-to-target range, just carrying part of the
flow), so a match between hot `i` and cold `j` is thermodynamically legal only
if BOTH ends respect the minimum approach temperature `DTMIN`:
```
THs_i - TCt_j >= DTMIN      (hot inlet vs. cold outlet)
THt_i - TCs_j >= DTMIN      (hot outlet vs. cold inlet)
```
and no stream's total matched duty may exceed `D_i` (hot) or `E_j` (cold).

**Cost.** Every match has an area cost proportional to its duty and inversely
proportional to its log-mean temperature difference (`LMTD`, the standard
counter-current driving force built from the two approach temperatures above):
a match with a big temperature gap needs little area; a tight one needs a lot.
Total cost = utility cost + area cost:
```
F = CH * (hot utility used) + CC * (cold utility used) + A * sum_over_matches( Q_ij / LMTD_ij )
```
**Beware the obvious move.** Matching the globally hottest hot stream against
the globally coldest cold stream always maximizes that one match's driving
force (cheapest area you can buy) -- but every unit of a stream's duty you
spend is gone. Streams with few compatible partners can be starved of the
partner they actually needed, forcing MORE hot utility and MORE cold utility
network-wide, even though your one match looked great in isolation. Good
designs respect the network's **pinch**: the temperature point that splits it
into an above-pinch region (a heat deficit, needs hot utility) and a
below-pinch region (a heat surplus, needs cold utility). Moving heat across
that boundary is always wasteful -- it never lowers utility.

## Input (stdin)
```
NH NC
DTMIN CH CC A
THs_1 THt_1 CP_1     (NH lines, hot streams)
...
TCs_1 TCt_1 CP_1     (NC lines, cold streams)
...
```
`1 <= NH,NC <= 8`. All values are positive reals.

## Output (stdout)
```
M
i_1 j_1 Q_1
...
i_M j_M Q_M
```
`M` matches; `i` (1..NH), `j` (1..NC) name the streams, `Q_ij >= 0` is the duty
transferred. A pair `(i,j)` may repeat (its duties are summed).

## Feasibility
Rejected (score `0`) if: the token stream is malformed; any `Q` is negative,
`NaN`, or `Inf`; any index is out of range; a used match violates the `DTMIN`
rule above; or any stream's total matched duty exceeds its available/required
duty.

## Objective (minimize) and Scoring
`F` is defined above. The checker also builds the trivial feasible network
(zero matches, everything on utility) as baseline `B = CC*sum(D_i) + CH*sum(E_j)`.
```
sc    = min(1000, 60 * B / F)
Ratio = sc / 1000
```
So building nothing scores `~0.06`; a network that beats the do-nothing cost by
more than ~16.7x saturates at `1.0`. Scoring is deterministic floating-point
arithmetic (fixed `1e-6` tolerance on the feasibility checks).

## Constraints
- `2 <= NH+NC <= 16` (small scale), `DTMIN in [5,15]`, costs and areas fixed
  and positive. Time limit 5s, memory 512MB.

## Example
`NH=1, NC=1`, `DTMIN=10`: hot stream `(200 -> 150, CP=2)` so `D=100`; cold
stream `(50 -> 120, CP=1)` so `E=70`. `CH=8, CC=6, A=20`.
Baseline (no matches): `CU=100, HU=70`, `B = 6*100 + 8*70 = 1160`, `Ratio=0.060`.
The single match `(1,1,Q=70)` satisfies `DTMIN` (both approach temperatures are
`80` and `100`, `LMTD ~= 89.6`): leftover cold utility `CU=30`, no hot utility,
area cost `~= 20*70/89.6 ~= 15.6`, so `F ~= 6*30 + 15.6 = 195.6` and
`Ratio = min(1000, 60*1160/195.6)/1000 ~= 0.356`. (Illustrative only -- larger
instances have several hot and cold streams and admit no simple closed-form
optimum: which streams to pair, how to split duty among them, and whether a
tight match is worth its area all interact.)
