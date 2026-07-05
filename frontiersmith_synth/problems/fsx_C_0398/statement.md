# Museum Gallery Tour: Recovering the Visitor-Flow Causal Map

A large museum instruments every gallery with an anonymous foot-traffic beacon.
For each visitor *tour* it records a **binary vector**: for every gallery, whether
that visitor entered it (`1`) or not (`0`). Entering one gallery causally raises the
chance of entering certain *other* galleries downstream — a signpost, a shared
doorway, a "if you liked this, continue to…" placard. These influences form a
**hidden directed acyclic graph (DAG)**: each gallery's entry is a **noisy-OR** of a
small private "leak" rate plus activation contributed by each upstream gallery the
visitor already entered. This is a **discrete / categorical Bayesian network** (the
Cancer / Child / Alarm family), *not* a linear-Gaussian model.

Only **observational** tour logs are available — no interventions, no forced routing.
Your job: design a **causal-discovery routine** that reconstructs the directed
visitor-flow map from the logs, and that **generalizes** across a battery of
different museums (varying size, density, and how tour-rich the logs are).

## Isolation

Your program is run as an **isolated subprocess**. It reads ONE JSON *public
instance* from stdin and writes ONE JSON *answer* to stdout. It never sees the
ground-truth DAG, the topological order, the noisy-OR weights, the leak rates, or
any evaluator memory. Anything other than your printed JSON answer is ignored.

## Public instance (stdin)

```json
{
  "data":         [[0, 1, 0, ...], ...],   // N x d binary tour matrix (rows = tours)
  "n_tours":      500,                       // N, number of tours (rows)
  "n_galleries":  8,                         // d, number of galleries (columns)
  "gallery_names":["Antiquities", "..."],   // d flavor labels (no causal info)
  "seed":         20240398                   // per-instance seed you MAY use
}
```

`data[r][j] == 1` iff the visitor on tour `r` entered gallery `j`. Column indices are
randomly relabelled, so **raw column order carries no orientation information**.

## Answer (stdout) — any one of these forms

```json
[[i, j], ...]                        // directed edges  i -> j  (gallery i drives gallery j)
{"edges": [[i, j], ...]}             // same, wrapped
{"adjacency": [[0,1,...], ...]}      // d x d adjacency matrix, A[i][j]=1 means i -> j
```

Every index must be an integer in `0..d-1`; an edge's two endpoints must differ.
Any malformed / out-of-range answer (or a crash / timeout) scores **0** on that
instance.

## Objective & scoring (deterministic)

Quality is the **Structural Hamming Distance (SHD)** between your predicted graph and
the hidden ground-truth DAG. Over every unordered gallery pair, SHD adds 1 for each:
a **missing** edge, an **extra** edge, or a **reversed / ambiguous** orientation.
Lower SHD is better.

The evaluator's own trivial baseline is the **empty map**, whose SHD equals the number
of true edges `E`. Per instance the normalized score is the minimization form

```
r = min( 1.0, 0.1 * SHD_empty / max(SHD_candidate, 1e-9) )
```

so a map no better than empty maps to ~**0.1**, and a perfect recovery (SHD = 0) maps
to **1.0**. Because only observational discrete data is given, orientation is
under-determined (Markov-equivalence) and finite tours make dependence estimates
noisy — so even a strong routine keeps headroom below 1.0, especially on the larger,
sparser, tour-poorer museums. Your final score is the **mean of `r`** over a diverse
battery of 10 museums (including held-out larger ones).

## Hints on strategy

- **Empty map** → ~0.1 everywhere (the floor).
- **Marginal dependence thresholding** (e.g. pairwise mutual information) recovers real
  adjacencies but also every *transitive* dependence along a flow path `A -> B -> C`
  (a false `A–C` edge), and uninformed orientation is a coin flip.
- **Conditional-independence pruning** (drop an edge `X–Y` when some third gallery `Z`
  screens it off, `MI(X,Y|Z) ≈ 0`) removes the transitive false edges; noisy-OR
  colliders never manufacture a co-parent edge to begin with.
- **Orientation from frequency**: with small leak rates a *source* gallery fires rarely
  while its descendants inherit extra activation, so sorting galleries by ascending
  marginal entry frequency estimates a topological order.
- Stronger still: larger conditioning sets, a scored BN search (BDeu/BIC), or an
  explicit noisy-OR likelihood.
