# Leaf Vein Remodeling: A Network That Survives a Cut

## Problem

A leaf's vein network is an `R x C` grid of tissue nodes (node `(r,c)` has id
`r*C+c`, `0`-indexed) connected by veins along the 4-neighbour grid edges (every
horizontal and vertical adjacency, no diagonals). One node `S` is the **midrib**
(nutrient source); `K` other nodes are **sinks** that must be fed. Real veins do not
carry one fixed flow forever — sap demand fluctuates between which patches of tissue
are drawing hardest — and the vein network *remodels itself* in response: veins that
carry more flux thicken, veins that carry little decay.

You control two things: (1) a **loading schedule** — non-negative weights over a
given menu of `M` demand scenarios (how much relative "time" the network spends on
each scenario while it remodels), and (2) a **reinforcement exponent** `gamma` that
sets how aggressively flux gets rewarded. The checker runs the fixed, deterministic
remodeling rule below to a stable network and grades *that* network's ability to
survive the loss of any single vein.

**The menu.** Scenario `0` is the *aggregate* load: every sink draws simultaneously
(source supplies `K` units, each sink draws `1`). Scenarios `1..K` are the
*fluctuating* loads: sink `k` alone draws `1` unit (source supplies `1`). The final
two scenarios are the *left-half* and *right-half* groups of sinks (by input order)
drawing together. Only scenarios with positive weight are "active"; weights are
renormalized to sum to `1`.

**The remodeling rule.** Start every edge at conductance `1`. Repeat `28` times:
solve Kirchhoff's current law (`sum of C_e*(p_i-p_j) over edges at i = demand_i`,
one reference node pinned) for **every active scenario** at the current
conductances; take each edge's signed flux `F_e = C_e*(p_i-p_j)` per scenario;
form the schedule-weighted mean squared flux `Phi_e = sum over active scenarios
w*F_e^2`; set `C_e <- Phi_e^gamma` (floored above `0`); rescale so
`sum(C_e) = #edges` (vascular material is conserved — one vein thickening starves
another). `gamma=0` makes every `Phi_e^0=1`, so the network never leaves the
uniform starting mesh.

**Scoring the stable network.** For each edge `e`, sever it (`C_e=0`) and measure
the dissipation `D = sum_i demand_i*p_i` of the resulting network under the
aggregate scenario and under each individual sink scenario (fresh solve each time
on the severed network). The cost of losing `e` is `D_agg(e) + 0.25*sum_k
D_sink_k(e)`. Your score is driven by the **worst** (largest) cost over all
single-edge losses — one badly-protected vein can wreck the whole network.

## Input
```
R C
S
K
sink_1 ... sink_K
gamma_min gamma_max
M
c_1 v_1 ... v_c1
...
c_M v_1 ... v_cM
```
`R,C`: grid shape. `S`: source id. `K` sink ids follow. `gamma_min,gamma_max`:
your exponent's allowed range. `M` (`=K+3`) demand-group lines follow, each
`count v_1..v_count`: group `0` = aggregate (all sinks), groups `1..K` = the
singles, the last two = the half-groups.

## Output
```
gamma
w_1 w_2 ... w_M
```
One real `gamma` in `[gamma_min, gamma_max]`, then `M` non-negative reals (need not
sum to `1` — the checker normalizes); at least one must be strictly positive.

## Feasibility
`gamma` must be finite and within `[gamma_min,gamma_max]`; all `M` weights must be
present, finite, non-negative, and not all zero. Otherwise: `Ratio: 0.0`.

## Scoring formula
Let `F` be the worst-edge-loss cost of your stabilized network and `B` the same
worst-edge-loss cost of the **untouched uniform mesh** (every `C_e=1` — exactly what
`gamma=0` reproduces, whatever weights you submit). Then
```
Ratio = min(1.0, 0.1 * (B / F) ** 2)
```
Lower `F` (a network that survives its worst single cut more cheaply) scores higher;
matching the uniform mesh scores `0.1`.

## Constraints
`2 <= R,C`, `RC <= 100`, `1 <= K <= 10`, `gamma_min=0.0`, `gamma_max=0.85`. Time
limit 5s, memory 512 MiB.

## Example
Toy `2x2` grid (nodes `0`=TL,`1`=TR,`2`=BL,`3`=BR; edges `0-1,1-3,3-2,2-0`, a
4-cycle), `S=0`, sinks `{1,3}` (`K=2`). On the uniform mesh (`gamma=0`, all
`C_e=1`), severing edge `0-1` is worst: both sinks must now be fed the long way
`0-2-3-1` in series, giving cost `9 + 0.25*(3+2) = 10.25 = B` (the other three
single-edge losses all cost less). A network that instead spreads conductance as
roughly `(0.65, 0.75, 1.32, 1.27)` on edges `(0-1,1-3,3-2,2-0)` lowers the
worst-edge-loss cost to `F ~= 8.60`, giving `Ratio ~= 0.142` — beating the uniform
mesh by distributing capacity so no single vein is a catastrophic point of failure.
Reaching such a network from the remodeling rule (not by guessing conductances
directly) is your actual task.
