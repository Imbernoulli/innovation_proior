# Deep-Sea Cable Network — Reconstructing the Signal-Propagation DAG (Format B, isolated)

A deep-sea fibre-optic cable network links `n` signal stations. When a latent
fault/attenuation disturbance hits one station it **propagates downstream** to the
stations that station directly feeds, and every station also picks up its own local
noise. Physically this is a **linear-Gaussian structural causal model** on a directed
acyclic graph (DAG): for each station `j`,

```
X_j = Σ_{i -> j} w_ij · X_i + eps_j ,     eps_j ~ N(0, sigma^2)
```

with **equal disturbance variance** `sigma^2` at every station (identical optical
repeaters). Under equal variances the true propagation DAG is *identifiable* from purely
observational readings — but you have to work for it.

You are given a batch of **observational readings**: many independent snapshots, each a
vector of all `n` station readings. **Reconstruct the directed propagation graph** — which
station directly feeds which. You never see the true graph; the grader keeps it hidden.

## Public instance (stdin JSON)
```json
{
  "n": <int>,                       // number of stations (node ids 0..n-1)
  "samples": [[x_0, ..., x_{n-1}],  // one row per observational snapshot
              ...]
}
```
`m = len(samples)` snapshots are provided. The number of edges and the true graph are
NOT given.

## Answer (stdout JSON)
```json
{"edges": [[i, j], ...]}   // each pair means a directed edge i -> j (station i feeds j)
```
Requirements (any violation ⇒ score 0 for that instance):
- every `[i, j]` has integer `i, j` with `0 ≤ i, j < n` and `i ≠ j`;
- no duplicated edge and no pair appearing in both directions;
- the edge set must be **acyclic** (a propagation graph is a DAG).

## Objective & scoring
For each instance the grader computes the **Structural Hamming Distance (SHD)** between
your graph and the hidden true graph: over all unordered station pairs, count a mismatch
whenever the presence-or-direction of the edge differs (a missing edge, an extra edge, or
a reversed edge each cost 1). **Lower SHD is better.**

Per-instance score = `min(1, 0.1 · SHD_empty / SHD_yours)`, where `SHD_empty` (the
number of true edges) is the SHD of the empty-graph guess. So predicting no edges scores
`≈ 0.1`; recovering the graph exactly scores `1.0`. The final score is the **mean over 12
fixed, seeded networks** of varying size, edge density and sample count — the later,
denser, sample-starved networks are the hard ones, so a good general rule must hold up
across the whole battery.

## Suggested strategies (increasing sophistication)
- **Empty graph** — predict nothing (the `0.1` calibration baseline).
- **Correlation skeleton + variance ordering** — connect strongly correlated stations and
  orient from lower-variance to higher-variance; simple, but marginal correlation also
  fires on indirect pairs so it adds transitive edges.
- **Equal-variance top-down order recovery** — repeatedly pick the remaining station with
  the smallest *residual* variance when regressed on the already-ordered stations (a
  source collapses to the disturbance variance), then prune parents by regression
  coefficient magnitude. This tracks the true DAG closely.
- **Conditional-independence / score-based search** — refine the skeleton and orientations
  with partial-correlation tests or a likelihood/BIC search over the recovered order.
