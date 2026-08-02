# Canary Rollout Under Delayed Failure Signals

## Problem
A fleet team is pushing an OTA firmware update to `V` device variants
(hardware SKUs). Variant `v` has `fleet_v` devices, all on old firmware.

Updating a device is risky: it independently **bricks** (fails permanently)
with probability `p_v`, or succeeds with probability `1-p_v`. A success is
worth `reward_v`; a brick costs `penalty_v` (usually much larger).

The outcome is **not known immediately**: a device updated at time `t` only
reports status at time `t + D_v` — the variant's **signal latency**. Until
then, that cohort's outcome is *in flight* and unobservable.

Because signals lag, the fleet system enforces a **rollback-window** rule
per variant: at any instant, the summed *expected* brick risk of all
still-in-flight cohorts of that variant may not exceed a budget `W_v`. If a
new cohort of size `c` would push in-flight risk (current in-flight risk
plus `c*p_v`) over `W_v`, governance rejects it before any device is
touched: none of its devices update, it earns nothing, no retry.

You design the **staged rollout**: for each variant, a set of cohorts, each
a `(time, size)` pair, over a shared horizon of `T` discrete steps `0..T-1`.

## Input (stdin)
```
V T
fleet_1 p_1 reward_1 penalty_1 D_1 W_1
...
fleet_V p_V reward_V penalty_V D_V W_V
```
`fleet_v` (int), `p_v` (brick probability, `0<p_v<1`), `reward_v`,
`penalty_v` (decimals), `D_v` (int signal latency, `D_v>=1`), `W_v` (decimal
risk budget, `W_v>0`), one line per variant, in order `v=1..V`.

## Output (stdout)
For `v = 1..V`, in input order, print:
```
k_v
t_1 c_1
...
t_{k_v} c_{k_v}
```
`k_v` cohorts for variant `v`. Each `t_i` is an integer in `[0,T)`; all
`t_i` for the same variant must be **distinct**. Each `c_i >= 1` is an
integer cohort size. The sum of `c_i` over variant `v` must not exceed
`fleet_v`.

## Feasibility
Output scores `Ratio: 0.0` if: any token fails to parse, or is non-finite;
any `t_i` is out of `[0,T)` or repeated within a variant; any `c_i < 1`; a
variant's cohort sizes sum above `fleet_v`; or extra/missing tokens remain.

## Objective
Process each variant's cohorts in increasing `t` order. Before admitting
cohort `(t,c)`, compute `inflight` = the sum of `c_j*p_v` over already
ADMITTED cohorts `j` of that variant with `t_j > t - D_v` (still unsignaled
at time `t`). If `inflight + c*p_v <= W_v`, admit the cohort: it contributes
`c * ((1-p_v)*reward_v - p_v*penalty_v)` to the score and its risk joins the
in-flight pool for later cohorts. Otherwise it is rejected: it contributes
`0` and its risk never joins the pool. Maximize the total `F` summed over
all admitted cohorts of all variants.

## Scoring
The checker also builds its own simple reference: for each variant, ONE
cautious cohort issued at `t=0`, sized `floor(W_v/p_v)` (capped at
`fleet_v`) — the largest single shot that alone never breaches the window.
Let `B` be this reference's total score (summed like `F` above).
```
sc = min(1000.0, 100.0 * max(0,F) / max(1e-9, B))
Ratio = sc / 1000.0
```
Matching the one-shot reference scores `0.1`; `10x` more caps at `1.0`.

## Constraints
`2 <= V <= 4`, `10 <= T <= 24`, `0.02 <= p_v <= 0.25`, `1 <= D_v <= 9`,
fleets in the low thousands. Time limit 5s, memory 512MB.

## Example
One variant: `fleet=1000, p=0.1, reward=1, penalty=4, D=2, W=1`. Per device,
`ev = (1-p)*reward - p*penalty = 0.9 - 0.4 = 0.5`. The one-shot reference
cohort is `floor(W/p)=10` devices at `t=0`, so `B = 10*0.5 = 5.0`.

If the solver bunches two 10-device cohorts at `t=0` and `t=1`: at `t=1` the
`t=0` cohort is still in flight (`0 > 1-2` holds), so `inflight=10*0.1=1.0`
and adding the new cohort's `1.0` would total `2.0 > W=1`, so the second
cohort is **rejected**. `F = 10*0.5 + 0 = 5.0`, giving `sc=100*5/5=100`,
`Ratio=0.100` — no better than doing nothing more than the reference.

If instead the solver paces the cohorts at `t=0` and `t=2` (one full
latency window apart), the `t=0` cohort has already signaled by `t=2`
(`0 > 2-2` is false), so `inflight=0` and the second is **admitted**.
`F = 10*0.5 + 10*0.5 = 10.0`, `Ratio=0.200` — twice the reference, from the
same two cohorts, just spaced against the latency instead of bunched.
(Illustrative numbers only — real instances are not this simple.)
