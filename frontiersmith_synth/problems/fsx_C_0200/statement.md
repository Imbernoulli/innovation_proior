# Warehouse Fleet Telemetry: Causal Influence-Map Discovery

## Background

A robotic fulfillment warehouse continuously logs `d` process signals from its
control plane -- e.g. `intake_rate`, `conveyor_speed`, `robot_utilization`,
`queue_length`, `pick_latency`, `battery_drain`, `error_rate`, ... These signals
are not independent: they are driven by a hidden **structural causal model**
(SCM) laid out on a ground-truth **directed acyclic influence map** (the DAG).
Each signal is a linear function of its direct causal parents plus an independent
random shock, and all shocks share the **same variance** (the equal-variance
regime).

Your job: from a batch of purely *observational* telemetry (no interventions),
reconstruct the influence map as accurately as possible.

You will be scored across a **battery** of warehouse graphs of increasing size
(5-8 signals) and density, including denser held-out layouts. A single cheap
rule does not win everywhere -- you must recover both the *skeleton* (which pairs
are directly linked) and the *orientation* (which direction the influence flows).

> Note on orientation: the public column order is a **hidden permutation** of the
> causal order. You cannot read edge directions off the column indices; you must
> infer them from the data (for instance, in the equal-variance regime a source
> signal has smaller variance than its descendants).

## Program contract

Your program is a standalone process. Read ONE JSON object (the public instance)
from `stdin`; write ONE JSON object (your answer) to `stdout`.

```python
import sys, json
inst = json.load(sys.stdin)
# ...compute...
print(json.dumps(answer))
```

### Public instance schema

```json
{
  "names": ["intake_rate", "conveyor_speed", "..."],   // length d, labels only
  "d": 6,                                               // number of signals (columns)
  "n": 300,                                             // number of samples (rows)
  "data": [[...d floats...], ...]                       // n rows x d columns, observational telemetry
}
```

Columns are indexed `0..d-1` in the order given by `data` / `names`.

### Answer schema

```json
{ "edges": [[i, j], ...] }   // each [i,j] is a directed edge i -> j (signal i directly influences signal j)
```

Constraints (violation => the instance scores 0):

- every `i`, `j` is an integer with `0 <= i,j < d` and `i != j`;
- no pair may appear twice, and you may not include both `[i,j]` and `[j,i]`;
- the resulting directed graph must be **acyclic** (a valid DAG).

An empty edge list (`{"edges": []}`) is always valid.

## Objective and scoring

For each instance the evaluator computes the **Structural Hamming Distance (SHD)**
between your DAG and the hidden ground-truth DAG, over unordered pairs:

- `+1` for every **missing** true edge,
- `+1` for every **extra** (false) edge,
- `+1` for every **reversed** edge (right skeleton, wrong direction).

Lower SHD is better. Let `b` be the SHD of the empty graph (i.e. the number of
true edges). Your per-instance score is

```
r = min(1.0, 0.1 * b / max(SHD, 1e-12))
```

so the empty map scores `~0.1`, and the score rises as your SHD shrinks (an exact
recovery would require `SHD <= b/10`). The reported `Ratio` is the mean of `r`
over all instances in the battery. Infeasible / malformed / cyclic answers score
`0` for that instance.

## Determinism

All instances are generated from fixed seeds; scoring is fully deterministic and
uses no wall-clock or hardware state. Your program is run in an isolated
subprocess and only ever sees the public instance above.
