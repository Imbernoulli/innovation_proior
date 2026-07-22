# The Cyclone's Wake

You manage `D` disaster-relief depots grouped into `G` geographic
clusters (same-cluster depots are adjacent). Over `T` steps a storm may
be active over **at most one cluster at a time**, driven by a hidden
Markov chain over `{calm, storm_0, ..., storm_{G-1}}`: from `calm` it
ignites at cluster `0` (prob `p_start`); once active at `g` it **stays**
(`p_stay`), **advances** to `(g+1) % G` (`p_advance`), or **subsides** to
calm (`1 - p_stay - p_advance`, exactly `0` on some instances — once
ignited it just keeps cycling). This transition structure is given; the
sampled regime path is never revealed.

Every depot in the active cluster gets an elevated demand shock; other
depots see only steady background demand. The shock is **largest on the
first step a cluster becomes active** (the strike); later steps of the
same dwell carry a smaller, instance-dependent sustained demand — on many
instances almost all the damage is the strike itself. You never observe
true demand directly, only a coarse **per-cluster, per-bin sensor
reading**: time is divided into consecutive bins of `bin_width` steps,
and for each elapsed bin `b` you get `cluster_activity[b][g]` — the
fraction of cluster `g`'s depot-steps that read "active" during that bin,
itself noisy (each underlying reading independently flipped with
probability `signal_flip_prob`). The exact step within a bin a strike
occurs is never revealed.

Every depot also gets a routine per-step resupply modestly above its own
`base_demand`, so an undisturbed depot is self-sustaining — its buffer
only has to absorb the EXTRA demand a shock brings.

You may ship stock between depots, or from a **regional relief hub**
(virtual location, index `D`) that is never demand-constrained but costs
more per unit (`hub_cost`) and is throttled by its own per-step capacity
(`hub_capacity`, shared by every hub shipment issued that step). A
shipment issued at step `t` **is usable only from step `t+L`** — never at
`t`. A depot-to-depot shipment cannot exceed the source's stock; each
lane carries at most `lane_capacity` units/step (excess clipped, not
rejected). Storage is capped at `capacity[i]`; excess inflow is lost.
Each step, in order: (1) resupply and scheduled arrivals are added
(capped), (2) shipments issued that step leave their source, (3) each
depot serves true demand from its stock — shortfall is **unmet demand**.

## Why your program runs once PER BIN, not once total
Handing you all readings at once would let you read the reading for
whatever future bin a shipment targets and ship accordingly — solving by
lookup, erasing the trap. So **your program is invoked once per bin, in
order**, each time as a fresh process. On the call for bin `epoch` you
only receive `known_activity` for bins `0..epoch` — later bins aren't in
the payload because they don't exist yet. Decide shipments for that
bin's steps only; the evaluator accumulates every bin's output into the
final plan before replaying it against the true hidden demand.

## Per-epoch instance (stdin)
```json
{"D": 6, "T": 40, "L": 3, "G": 3, "clusters": [0,0,1,1,2,2],
 "base_demand": [4.1, ...], "initial_stock": [...], "capacity": [...],
 "lane_cost": [[0,1.0,2.2,...], ...], "lane_capacity": 22.0,
 "hub_cost": 2.0, "hub_capacity": 26.0, "unmet_penalty": 7.0,
 "shock_mean": 18.0, "shock_std": 3.0, "signal_flip_prob": 0.10,
 "bin_width": 5, "transition": {"p_stay": 0.35, "p_advance": 0.65,
 "p_calm_return": 0.0, "p_start": 0.35},
 "epoch": 3, "epoch_start": 15, "epoch_end": 20,
 "known_activity": [[0.8,0.1,0.0], [0.8,0.3,0.2], [0.1,0.7,0.5], [0.0,0.9,0.1]]}
```
The first block is identical every call. `epoch` is the 0-indexed bin
this call decides; `known_activity` has `epoch + 1` rows (bins
`0..epoch`), each `G` per-cluster fractions.

## Candidate program contract
Standalone program: read one JSON object, write one JSON object:
```json
{"shipments": [[t, i, j, qty], ...]}
```
Each entry ships `qty` units from source `i` (depot `0..D-1`, or hub `D`)
to depot `j` (`0..D-1`, `j != i` when `i < D`), issued at step `t` with
**`epoch_start <= t < epoch_end`** for *this* call — a step owned by a
different bin must be decided by that bin's own (later) call; since each
process is stateless, track continuity by recomputing from
`known_activity`. `qty >= 0` finite. A malformed entry on ANY call (bad
shape/types, `t` outside this bin's range, non-finite/negative quantity,
a crash, a timeout, or non-JSON output) scores the WHOLE instance `0.0`.
An infeasible quantity (more than a depot holds, or exceeding a lane's/
the hub's per-step capacity) is clipped during replay, not rejected.

## Objective and scoring
**Minimize**, per instance, `obj = unmet_penalty * (total unmet demand) +
(total shipping cost)`, cost summing `qty * lane_cost[i][j]`
(depot-to-depot) or `qty * hub_cost` (hub) over every executed (post-clip)
shipment across all bins. The evaluator replays the assembled plan
against the TRUE hidden demand (driven by the hidden regime path) across
a fixed, seeded family of 10 instances — several with fast storm-cycling
relative to the lead time, so reacting only to the CURRENT bin's reading
systematically lands at a depot the storm has already left. Normalization:
```
b  = obj of the empty (do-nothing) plan
LB = 0.12 * b
r  = clamp(0.1 + 0.9 * (b - obj) / max(b - LB, 1e-9), 0, 1)
```
`LB` is a generous, hard-to-reach reference — a well-calibrated policy
still incurs unavoidable shock noise and can't get near it, so headroom
remains above any reference solution. **Ratio** is the mean of `r` over
all 10 instances; **Vector** lists the per-instance scores.

## Suggested strategies
1. Do nothing (baseline).
2. Reactive: ship once a depot's cluster looks active *this bin*,
   treating the reading as ground truth, never anticipating; can't
   arrive for `L` more steps.
3. Track per-bin per-cluster readings to identify the active cluster
   with confidence, use the known cyclic structure to predict which
   cluster is hit **next**, and pre-stage stock toward it once
   confidence crosses a threshold — backstopped by reactive top-ups.
4. The same regime-aware pre-positioning, tuned for fast-cycling,
   tight-capacity instances.
