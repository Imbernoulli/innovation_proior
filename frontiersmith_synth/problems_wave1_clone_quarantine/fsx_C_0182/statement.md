# Telescope Array: Causal Wiring of Sensor Channels

## Story

A robotic telescope array logs a bank of environmental and control **channels**
every exposure — dome seeing, ambient and mirror temperature, wind speed,
humidity, mount tracking error, guide-star drift, focus position, CCD
temperature, sky background, and so on. These channels are not independent: they
influence one another through a **hidden causal wiring diagram**, a directed
acyclic graph (DAG) in which each channel equals a linear combination of its
direct causes plus its own independent Gaussian disturbance (a linear-Gaussian
structural equation model).

You only get **observational logs** — a matrix of past exposures. No one lets you
intervene on the dome heater or nudge the mount to see what happens. From those
logs alone you must **reconstruct the causal wiring**: output the directed edges
`i -> j` meaning "channel `i` is a direct cause of channel `j`".

You are graded across many arrays of different size, density, and data volume,
including larger held-out arrays, so the goal is a discovery rule that
**generalizes** rather than one tuned to a single graph.

## Isolation (how your program is run)

Your program is executed as an **isolated subprocess**. It reads exactly one JSON
object (the *public* view of one array) from **stdin** and writes exactly one JSON
value (your answer) to **stdout**. You never see the ground-truth DAG, the
topological order, the edge weights, or the evaluator's memory.

```python
import sys, json
import numpy as np

inst = json.load(sys.stdin)                 # public inputs ONLY
X = np.asarray(inst["data"], dtype=float)   # shape (n_samples, n_nodes)
d = inst["n_nodes"]
# ... infer the causal DAG ...
print(json.dumps({"edges": edges}))         # the ONLY thing the evaluator reads
```

## Public instance (stdin)

```json
{
  "data":       [[float, ...], ...],   // N x d observation matrix
  "n_samples":  int,                   // number of exposures N
  "n_nodes":    int,                   // number of channels d
  "node_names": [str, ...],            // flavor labels for the d channels
  "seed":       int                    // per-instance seed you MAY use for your own RNG
}
```

The channels are **randomly relabelled**: the column index carries no information
about causal order, so you cannot cheat by assuming `i -> j` whenever `i < j`.

## Answer (stdout)

A predicted directed graph, in **any** of these equivalent forms:

```json
[[i, j], ...]                          // directed edges  i -> j
{"edges": [[i, j], ...]}               // same, wrapped
{"adjacency": [[0/1, ...], ...]}       // d x d matrix, A[i][j]=1 means i -> j
```

Every index must be an integer in `0..d-1`, and `i != j`. Any of the following
makes an array score **0**: an out-of-range or self-loop index, a wrong-shaped
adjacency matrix, a non-0/1 adjacency entry, non-list/object output, a crash, a
timeout, or no output.

## Scoring

For each array the evaluator computes the **Structural Hamming Distance (SHD)**
between your predicted graph and the hidden ground-truth DAG. Over every unordered
pair of channels it counts one edit for each mismatch — a **missing** edge, an
**extra** edge, or a **reversed / ambiguous** orientation — and sums them (lower
is better). The evaluator's trivial baseline is the **empty graph**, whose SHD
equals the number of true edges `E`. Each array is normalized by the minimization
form:

```
r = min( 1.0, 0.1 * E / max(SHD_cand, 1e-9) )
```

so predicting no edges (or doing no better than empty) maps to `~0.1`, and a
perfect reconstruction (`SHD = 0`) maps to `1.0`. Valid arrays are floored to a
small positive value; the final reported score is the **mean** of the per-array
`r`:

```
Ratio:  <mean of per-array r, in [0,1]>
Vector: [r_1, r_2, ..., r_10]
```

## Objective

**Maximize `Ratio`.** There is no easy optimum. Marginal correlation flags every
indirect path as a false edge; the disturbances are equal-variance, so a source
channel has smaller marginal variance than its descendants (a real orientation
signal) — but colliders create "moralization" edges between co-parents that a
partial-correlation skeleton cannot remove, and orientation from purely
observational data is inherently under-determined. A genuinely general causal
discovery routine (conditional-independence pruning, score-based search, or an
equal-variance ordering method) is required to push SHD down across the whole
battery, including the larger held-out arrays.
