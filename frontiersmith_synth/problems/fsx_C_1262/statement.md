# Cores Are Cheap, Spikes Are Not: Cluster-Aware Neuron-to-Core Mapping

## Problem
You are mapping `n` spiking neurons onto up to `C_max` identical neuromorphic
cores. Neuron `u` fires synaptic spikes to neuron `v` at a fixed integer
**rate** for `m` given directed edges `(u, v, rate)`. You choose, for every
neuron, a **core** in `0..C_max-1` and a **time slot** in `0..T-1` (each core
multiplexes `T` execution slots; a slot represents a recurring time window in
which the core services whichever neurons are scheduled into it).

Three hardware limits apply to every core/slot, all given in the input:
- **Capacity**: a core may host at most `slot_cap` neurons *per slot* (so at
  most `T * slot_cap` neurons total).
- **Fanout-constraint**: the *sum of out-degrees* (number of outgoing edges,
  not their rate) of the neurons placed on one core may not exceed
  `fanout_budget` — this is the core's local routing-crossbar wiring limit.
- **Time-multiplex-slot budget**: the *sum of in-rates* (total incoming spike
  rate, from anywhere) of the neurons scheduled into one `(core, slot)` pair
  may not exceed `slot_rate_budget` — that time window can only physically
  deliver so many spikes.

**Energy model.** For every directed edge `(u, v, rate)`:
- if `u` and `v` land on **different cores**, it costs `rate * INTER` (spikes
  cross the on-chip network — the dominant, expensive path);
- if they share a core but **different slots**, it costs `rate * LOCAL`
  (cheap local routing between time windows);
- if they share **both core and slot**, it costs **0** (true same-cycle,
  same-unit delivery).

Each *active* core (hosting >=1 neuron) additionally costs a fixed `OVER`.
Minimize the total: `F = OVER * (#active cores) + sum of the per-edge terms
above`. `INTER >> LOCAL`, and `INTER` and `OVER` are comparable in scale — cutting
one core does not obviously beat keeping traffic local; read the numbers.

## Input (stdin)
```
N C_max T slot_cap fanout_budget slot_rate_budget INTER LOCAL OVER
M
M lines: u v rate      (directed synapse u -> v, rate >= 1, u != v)
```

## Output (stdout)
```
N
N lines: core slot      (line i, 0-indexed, is neuron i's core and slot)
```

## Feasibility
Rejected (score 0) if: the echoed `N` is wrong; any `core` is not in
`[0, C_max)` or any `slot` is not in `[0, T)`; any output token is
missing/extra/non-integer; any core's neuron count exceeds `T*slot_cap`; any
core's summed out-degree exceeds `fanout_budget`; any `(core, slot)`'s neuron
count exceeds `slot_cap`; or any `(core, slot)`'s summed in-rate exceeds
`slot_rate_budget`.

## Scoring
The checker builds its own baseline: spread neurons round-robin across every
available core in id order (spilling to the next core only when a limit is
hit), then fill slots within each core by descending in-rate (largest first
into the currently-lightest slot). Call its cost `B`. Your artifact's cost is
`F`. `Ratio = min(1, 0.1 * B / F)`.

## Constraints
`4 <= N <= 200`, `2 <= C_max <= 60`, `T = 4`, `1 <= slot_cap <= 20`,
`1 <= fanout_budget, slot_rate_budget <= 2000`, `1 <= M <= 2000`,
`1 <= rate <= 30`. Time limit 5s, memory 512MB.

## Example
`N=3`, `C_max=2`, `T=2`, `slot_cap=2`, `fanout_budget=5`, `slot_rate_budget=100`,
`INTER=10`, `LOCAL=1`, `OVER=3`. Edges: `(0,1,4)`, `(1,0,4)`, `(2,0,1)`. This
tiny example is for output mechanics only — it has one obvious cluster
`{0,1}` and a weak link to `2`; real instances hide many interacting clusters
behind a shuffled neuron numbering and give budgets that make grouping a
whole cluster onto one core infeasible, forcing real placement decisions.
One valid output: `core[0..2] = 0,0,1`, `slot[0..2] = 0,1,0` (0 and 1 share
core 0 in different slots; 2 gets its own core 1). Cost: 2 active cores
(`2*OVER=6`), edge `(0,1,4)` and `(1,0,4)` are same-core-different-slot
(`2*4*LOCAL=8`), edge `(2,0,1)` is inter-core (`1*INTER=10`); total
`F = 6 + 8 + 10 = 24`.

## Note on scale
`n <= 200` and `m <= 2000` keep every instance small; the challenge is
entirely about which neurons you group, not about brute-force search size.
