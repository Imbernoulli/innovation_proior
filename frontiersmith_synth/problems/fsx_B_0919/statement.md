# Rack Grid Thermal Placement (Format B, isolated)

A datacenter rack is a **3x3 grid of N=9 nodes** (4-neighbor grid adjacency:
up/down/left/right, no wraparound). A fixed stream of **M jobs** must be
placed on this grid over a **T-step horizon**. Job `j` has an `arrival`
step (earliest step it may start), a `demand` (number of full-rate steps
of compute it needs), and a `heat_rate` (heat generated per step while it
runs, at full rate).

You see the **entire job stream up front** and must submit, in ONE shot,
a schedule: for each job, either a `(node, start_time)` pair with
`start_time >= arrival` and `start_time + demand <= T`, or skip it. A node
can run **at most one job at a time** — chosen intervals on the same node
must not overlap.

## Thermal model
Every node `i` has a temperature `H_i`, starting at 0. Each step `t`:
1. If node `i` is currently running a job, it generates heat equal to
   `heat_rate * rate_multiplier(i)` that step, where `rate_multiplier` is
   `1.0` normally or `throttle_rate` (given in the input, `<< 1`) if the
   node is currently **throttled**. The job's useful compute that step
   equals the SAME `rate_multiplier` (so a throttled node crawls).
2. Heat updates: `H_i += generated_i - decay*H_i + alpha * sum_over_grid_neighbors(H_k - H_i)`.
   The last term is **diffusion**: heat flows from hotter neighbors into
   cooler ones (and out again), so one node's heat spreads spatially —
   a hot node can push a idle or lightly-loaded neighbor over its own
   limit even without ever running a job itself.
3. **Throttle hysteresis**: a non-throttled node becomes throttled the
   moment `H_i > HI`. Once throttled, it does **not** recover at `HI` —
   it stays throttled until `H_i` cools all the way down to `H_i <= LO`,
   where `LO < HI` is given separately. Since a throttled node still
   generates (reduced) heat and still receives diffused heat from hot
   neighbors, escaping the throttled state can take far longer than it
   took to enter it, especially when neighbors stay hot.

## Objective
A job's contribution is the sum of `rate_multiplier(node)` over every
step of its running interval (so a never-throttled job contributes
exactly `demand`; a job that spends part of its interval throttled
contributes less — that lost compute is gone, not recovered later).
Total score = sum of all jobs' contributions. This is normalized against
the absolute ceiling (every job's `demand`, i.e. if every job ran
untouched at full rate) to produce the graded metric — you are NOT
expected to reach the ceiling; some throttling exposure is generally
unavoidable, but WHERE and HOW MUCH is entirely under your control.

## Input (stdin JSON)
```json
{
  "T": 60, "N": 9, "grid_rows": 3, "grid_cols": 3,
  "decay": <float>, "alpha": <float>, "HI": <float>, "LO": <float>,
  "throttle_rate": <float>,
  "jobs": [ {"id": 0, "arrival": <int>, "demand": <int>, "heat_rate": <float>}, ... ]
}
```
Node ids are `row*grid_cols + col`, `0`-indexed.

## Output (stdout JSON)
```json
{"schedule": [ {"id": <int>, "node": <int in [0,N) or -1 to skip>, "start": <int>}, ... ]}
```
Include one entry per job id you want scheduled (a job id absent from
the list is treated as skipped, contributing 0). Any malformed entry
(bad types, `start < arrival`, `start+demand > T`, `node` out of range,
a duplicate job id, or two jobs overlapping on the same node) fails the
**whole test case** (score 0 for it). Non-finite values are rejected.

## Why this is hard
Packing every incoming job onto whichever node reads coolest *right now*
looks efficient, but that node was just chosen because it's about to
absorb this job's heat — and that heat will diffuse into its grid
neighbors on every subsequent step. Once a local hotspot crosses `HI`,
hysteresis means it (and everything next to it) can stay crippled for a
large fraction of the remaining horizon — a cost far larger than the
compute saved by not delaying the placement in the first place. There are
10 test cases; several are engineered so a "coolest node now, start
immediately" policy triggers exactly this cascade.
