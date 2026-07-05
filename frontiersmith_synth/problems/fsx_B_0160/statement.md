# Metro Spare-Parts Safety-Stock Placement

You run the spare-parts logistics for a metro (subway) network. A single critical
spare -- a traction-motor brush kit -- is stocked across a four-level **supply tree**:

```
central depot (root)  ->  regional maintenance yards  ->  line stores  ->  stations (leaves)
```

Every node holds **safety stock** of the part. Only **stations** (the leaves) see
random maintenance demand; an internal node's demand is the aggregate of all the
stations it ultimately serves. Because station demands are independent, variances add,
so a node's demand standard deviation is `sigma_i = sqrt(sum of served-station
variances)` (already computed for you in the instance).

You must place safety stock using the classical **Graves–Willems guaranteed-service
model**. Each node `i` **quotes an outbound service time** `S_i` — the whole number of
days it promises to deliver to its customers. It receives an inbound service time from
its single supplier (its parent):

```
SI_i = S_parent(i)          (for the root, SI = Sext, the external supplier's quote)
```

The node's **net replenishment time** is

```
tau_i = SI_i + T_i - S_i     (T_i = the node's local processing/lead time, in days)
```

and it must carry enough safety stock to cover demand variability over `tau_i` at the
required service level. With safety factor `k` (the z-value for the target cycle
service level; `k` is given), the safety-stock **holding cost** at node `i` is

```
cost_i = h_i * k * sigma_i * sqrt(tau_i)
```

The service-level requirement is baked into `k`, so **any feasible `S` already meets
it**. Feasibility is the guaranteed-service structure:

```
S_i >= 0            (integer, whole days)
tau_i = SI_i + T_i - S_i >= 0
S_i <= smax_i       (stations have a finite quoted-service cap; internal nodes: no cap)
```

**Objective (minimize):** the total holding cost `sum_i cost_i`.

There is a real trade-off. Quoting `S_i = 0` everywhere is the naive **decoupled**
policy — each node holds stock for its own full lead time (this is the scoring
baseline). Pushing service times up **pools risk upstream** (safety stock grows only
as `sqrt` of pooled lead time), but the station caps `smax` and the fact that holding
cost **rises downstream** mean you cannot simply pool everything at the leaves. The
exact optimum is a dynamic program over the tree; simple heuristics leave money on the
table.

## Program contract

Your program is a standalone process. Read **one** JSON object (the public instance)
from stdin and write **one** JSON object (your answer) to stdout.

### Input JSON (public instance)

| field      | type          | meaning |
|------------|---------------|---------|
| `n`        | int           | number of nodes |
| `parent`   | list[int]     | `parent[i]` = supplier of node `i`; `-1` for the root |
| `children` | list[list[int]] | `children[i]` = customers of node `i` (convenience; derivable from `parent`) |
| `level`    | list[int]     | depth of each node (0 = depot ... 3 = station) |
| `T`        | list[int]     | local processing/lead time `T_i` (days) |
| `h`        | list[float]   | holding-cost rate `h_i` |
| `sigma`    | list[float]   | demand std dev `sigma_i` served by node `i` |
| `smax`     | list[int]     | outbound service cap `smax_i` (huge sentinel for internal nodes) |
| `Sext`     | int           | external supplier's service time into the root |
| `k`        | float         | safety factor (target service level) |

### Output JSON (your answer)

```json
{"S": [S_0, S_1, ..., S_{n-1}]}
```

`S` is a length-`n` list of **integer** service times (whole days) in node order.

### Example skeleton

```python
import sys, json
inst = json.load(sys.stdin)
n = inst["n"]
# ... choose service times S_i respecting feasibility ...
print(json.dumps({"S": [0] * n}))
```

## Scoring

For each instance the evaluator computes your total holding cost `obj` (rejecting any
answer that is malformed, non-integer, or violates a feasibility constraint — such
answers score `0`). Let `b` be the cost of the all-zero decoupled baseline. Your
per-instance score is

```
r = min(1, 0.1 * b / obj)
```

so the decoupled baseline scores exactly `0.1`, and a placement `x` times cheaper than
baseline scores `min(1, 0.1*x)`. The reported **Ratio** is the mean of `r` over all 12
instances (a mix of medium networks plus larger held-out ones). Your program runs in
an isolated subprocess and only ever sees the public instance.
