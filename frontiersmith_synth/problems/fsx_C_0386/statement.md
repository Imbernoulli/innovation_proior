# Asteroid Rig: Causal Map of Subsystem Telemetry

## Story

An autonomous asteroid-mining rig streams **discrete status codes** from a bank
of subsystems every duty cycle — drill speed, regolith flow, ore grade, dust
density, reactor temperature, coolant pressure, thruster load, comms latency,
power draw, gyro drift, hopper fill, solar flux. Each subsystem reports a small
categorical code: `0 = low`, `1 = nominal`, `2 = high`.

These codes are **not** independent. They influence one another through a
**hidden causal wiring diagram** — a directed acyclic graph (DAG) in which each
subsystem's code is a discretized, noisy function of its *direct-cause*
subsystems. A spike in reactor temperature drives coolant pressure, which drives
thruster load, and so on.

You only get **observational telemetry** — a table of past duty cycles. No one
lets you command the drill or force-heat the reactor to watch what happens
downstream. From those logs alone you must **reconstruct the causal wiring**:
output the directed edges `i -> j` meaning "subsystem `i` is a direct cause of
subsystem `j`".

You are graded across many rigs of different size, density, and data volume,
including larger held-out rigs, so the goal is a discovery rule that
**generalizes** rather than one tuned to a single wiring diagram.

## Isolation (how your program is run)

Your program is executed as an **isolated subprocess**. It reads exactly one JSON
object (the *public* view of one rig) from **stdin** and writes exactly one JSON
value (your answer) to **stdout**. You never see the ground-truth DAG, the
topological order, the structural weights, the latent variables, or the
evaluator's memory.

```python
import sys, json
import numpy as np

inst = json.load(sys.stdin)                 # public inputs ONLY
X = np.asarray(inst["data"], dtype=np.int64)   # shape (n_samples, n_nodes), codes in 0..K-1
d = inst["n_nodes"]
K = inst["n_categories"]
# ... infer the causal DAG ...
print(json.dumps({"edges": edges}))         # the ONLY thing the evaluator reads
```

## Public instance (stdin)

```json
{
  "data":          [[int, ...], ...],   // N x d integer telemetry, each entry in 0..K-1
  "n_samples":     int,                 // number of duty cycles N
  "n_nodes":       int,                 // number of subsystems d
  "n_categories":  int,                 // global category count K (codes are 0..K-1)
  "cardinalities": [int, ...],          // per-column category count (max code + 1)
  "node_names":    [str, ...],          // flavor labels for the d subsystems
  "seed":          int                  // per-instance seed you MAY use for your own RNG
}
```

The subsystems are **randomly relabelled**: the column index carries no
information about causal order, so you cannot cheat by assuming `i -> j` whenever
`i < j`.

## Answer (stdout)

A predicted directed graph, in **any** of these equivalent forms:

```json
[[i, j], ...]                          // directed edges  i -> j
{"edges": [[i, j], ...]}               // same, wrapped
{"adjacency": [[0/1, ...], ...]}       // d x d matrix, A[i][j]=1 means i -> j
```

Every index must be an integer in `0..d-1`, and `i != j`. Any of the following
makes a rig score **0**: an out-of-range or self-loop index, a wrong-shaped
adjacency matrix, a non-0/1 adjacency entry, non-list/object output, a crash, a
timeout, or no output.

## Scoring

For each rig the evaluator computes the **Structural Hamming Distance (SHD)**
between your predicted graph and the hidden ground-truth DAG. Over every unordered
pair of subsystems it counts one edit for each mismatch — a **missing** edge, an
**extra** edge, or a **reversed / ambiguous** orientation — and sums them (lower
is better). The evaluator's trivial baseline is the **empty map**, whose SHD
equals the number of true edges `E`. Each rig is normalized by the minimization
form:

```
r = min( 1.0, 0.1 * E / max(SHD_cand, 1e-9) )
```

so predicting no edges (or doing no better than empty) maps to `~0.1`, and a
perfect reconstruction (`SHD = 0`) maps to `1.0`. Valid rigs are floored to a
small positive value; the final reported score is the **mean** of the per-rig
`r`:

```
Ratio:  <mean of per-rig r, in [0,1]>
Vector: [r_1, r_2, ..., r_10]
```

## Objective

**Maximize `Ratio`.** There is no easy optimum. Marginal mutual information flags
every indirect (transitive) association as a false edge; you must screen those off
with **conditional-independence tests** (condition on the right intermediate
subsystems). Co-parents of a collider are marginally independent, so a
marginal-then-condition skeleton avoids adding spurious moralization links — but
orientation from purely observational, discretized data is inherently
under-determined. One real signal: under equal-magnitude disturbances a source
subsystem's code concentrates in the central `nominal` bin (low marginal entropy)
while a descendant's code spreads across the outer `low`/`high` bins (higher
entropy), giving an ascending-entropy topological hint. A genuinely general
causal-discovery routine (PC-style CI pruning, score-based search, or an
equal-disturbance ordering method) is required to push SHD down across the whole
battery, including the larger, denser, data-poorer held-out rigs.
