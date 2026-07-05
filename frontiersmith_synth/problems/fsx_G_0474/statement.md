# Riverside Sensor Mesh: Recovering the Causal Wiring of an Environmental Monitoring Network

## Background

A solar-powered **wireless sensor mesh** is deployed along a river reach for
environmental monitoring. Each mesh node streams one physical channel once per
logging interval — rainfall, incident solar radiation, air temperature, relative
humidity, soil moisture, leaf wetness, canopy temperature, stream flow, turbidity,
water pH, dissolved oxygen, wind speed, and so on.

These channels drive one another through a **hidden causal wiring diagram**: a
directed acyclic graph (DAG) in which each channel is a linear function of its
direct causes plus an **independent Gaussian disturbance whose scale differs from
channel to channel** (a *heteroscedastic* linear-Gaussian structural equation
model). You only receive passive **observational** logs — no interventions were
performed.

A few channels are known **weather forcings** (physical drivers such as rainfall,
solar radiation, or wind) that have *no upstream sensor cause*. Their column
indices are published to you as a partial-orientation constraint: any edge that
touches a forcing channel must point **away** from it.

## Your task

Write a standalone program — a **causal-discovery routine** — that reads ONE JSON
instance from **stdin** and writes ONE JSON answer (a predicted set of directed
edges) to **stdout**. Your program is run as an **isolated subprocess**; it only
ever sees the public instance below and never the ground-truth graph.

### Public instance (stdin)

```json
{
  "data":          [[float, ...], ...],
  "n_samples":     500,
  "n_nodes":       6,
  "node_names":    ["rainfall", "solar_rad", "..."],
  "forcing_nodes": [1, 4],
  "seed":          20260480
}
```

- `data` — an `n_samples × n_nodes` observation matrix (rows = logging intervals,
  columns = channels).
- `n_nodes` (`d`) — number of channels; valid channel indices are `0 … d-1`.
- `forcing_nodes` — channels known to have **no parents** (weather drivers). Every
  ground-truth edge incident to such a channel points away from it.
- `seed` — a per-instance seed you MAY use; scoring is deterministic regardless.

### Answer (stdout) — any ONE of these forms

```text
[[i, j], ...]                      # directed edges  i -> j  (channel i causes channel j)
{"edges": [[i, j], ...]}           # same, wrapped
{"adjacency": [[0/1, ...], ...]}   # d x d adjacency matrix, A[i][j]=1 means i -> j
```

Edge indices must be integers in `0 … d-1` with `i != j`. Any malformed,
out-of-range, non-finite, or wrong-shape answer scores **0** on that instance.

## Objective — MINIMIZE structural Hamming distance

For each mesh the evaluator recomputes the **structural Hamming distance (SHD)**
between your predicted graph and the hidden ground-truth DAG. Over every unordered
channel pair, a **missing** edge, an **extra** edge, or a **reversed / ambiguous**
orientation each costs 1. Lower SHD is better.

Each instance is normalized against the empty-graph baseline (whose SHD equals the
number of true edges `E`):

```
r = min( 1.0, 0.1 * SHD_empty / max(SHD_cand, 1e-9) )
```

so an answer no better than the empty graph maps to ~0.1 and a perfect recovery
(SHD = 0) maps to 1.0. Your final score is the **mean of `r`** over a diverse
battery of 10 meshes (sparse/dense, few/many channels, data-rich/data-poor, plus
held-out larger meshes), rewarding a rule that **generalizes**.

## Why it is open-ended

Only observational data is given, so orientation is generally under-determined.
The heteroscedastic disturbances defeat the naive "a source channel has the
smallest marginal variance" trick, and a precision-matrix skeleton also picks up
**moralization** edges between co-parents. Exploiting the forcing-node anchor,
handling collider structure, and choosing an orientation rule that survives
data-poor larger meshes are all genuine design choices — there is no easy optimum,
and multiple strategies (constraint-based, score-based, ordering-based) are viable.
