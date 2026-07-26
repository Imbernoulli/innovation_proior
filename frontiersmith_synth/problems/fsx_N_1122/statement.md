# Reinforcing Waves

## Problem

A directed influence network has `N` nodes and `M` weighted edges. Each node `i`
has an activation threshold `theta_i` and an accumulator that starts at 0.
The simulation runs for `T` discrete steps `t = 0, 1, ..., T-1`. You control an
**activation schedule**: a set of *external* activation events, each a pair
`(node, time)`, that you may place anywhere in `0 <= time < T`.

Each simulation step applies, in order:

1. **External events.** Every event scheduled for this step adds a fixed
   amount `EXT_BOOST` to its node's accumulator (only if that node is not
   already active).
2. **Propagation (one-shot).** Every node that became active in the
   *previous* step (not earlier) sends its outgoing edge weight to each
   not-yet-active neighbour's accumulator. A node's influence fires exactly
   once, the step right after it activates -- never again.
3. **Threshold cascade.** Any node whose accumulator now meets its threshold
   becomes permanently active.
4. **Temporal decay.** Every node still inactive after step 3 has its
   accumulator shrink: `acc = floor(acc * decay_num / decay_den)`.

Because propagation is one-shot and unmet partial credit decays sharply,
a node that needs contributions from two different upstream chains can only
be crossed for free if **both** chains finish in the exact same step --
otherwise the first pulse decays away before the second arrives, and that
partial credit is gone forever. Firing every external event at time 0 (a
plain "seed set") ignores this: chains of different length reach a shared
successor at different times, so their contributions never combine.

## Input (stdin)

```
N M
decay_num decay_den EXT_BOOST T
theta_0
...
theta_{N-1}
u_1 v_1 w_1
...
u_M v_M w_M
```
Each edge `u v w` means: if `u` is active, it contributes weight `w` to `v`'s
accumulator during the one step right after `u` activates.

## Output (stdout)

```
K
node_1 time_1
...
node_K time_K
```
`K` external activation events, each `(node, time)` with `0 <= node < N`,
`0 <= time < T`. Duplicate/repeated events on the same node are allowed but
each still counts toward `K`.

## Feasibility

Simulate all `T` steps deterministically with your schedule applied. Every
one of the `N` nodes must be active by the end, or the submission scores 0.
(Any node with no incoming edges can only ever be reached by an external
event.)

## Objective (minimize)

`K`, the number of external activation events used to achieve full
activation.

## Scoring

The checker builds its own trivial feasible schedule -- externally
activating every node once (`B = N` events, always feasible) -- as the
baseline, then scores your feasible `K` as
`score = min(1.0, B / (10 * K))`. Fewer events (found by timing, not brute
force) score higher.

## Constraints

`2 <= N <= 400`, `theta_i, EXT_BOOST` positive integers, `EXT_BOOST` is
always at least every `theta_i`, `1 <= decay_num < decay_den <= 10`,
`T <= 60`.

## Example

Take `N=7`, edges `0->2(w=5), 2->3(5), 3->4(5), 4->6(4), 1->5(5), 5->6(5)`,
thresholds `[3,3,5,5,5,5,9]`, `decay_num/decay_den=1/3`, `EXT_BOOST=50`,
`T=13`. Node 6 needs *both* incoming pulses (4+5=9=theta_6) in the same
step; alone, neither clears it and each decays away unspent.

- Firing node 0 at t=0 and node 1 at t=0 (naive): node 4 activates at t=3,
  node 5 at t=1 -- they arrive two steps apart, node 6 never activates.
  **Infeasible.**
- Firing node 0 at t=0 and node 1 at t=2 (delayed to match the longer chain):
  both feeding nodes activate at t=3, node 6 activates at t=4. Full cascade
  with **K=2** events -> `score = min(1, 7/20) = 0.35`.
- Externally activating all 7 nodes directly: **K=7** -> `score = 0.1`
  (the baseline).
