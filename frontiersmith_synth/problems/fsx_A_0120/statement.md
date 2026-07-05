# Greenhouse Zone Partition

An automated greenhouse is a rectangular grid of `H x W` growing cells. The crop
mix is already decided, so each cell `(i, j)` has a fixed **ideal air
temperature** `ideal[i][j]` (an integer). The climate system cannot heat every
cell on its own; instead it partitions the grid into **zones**. A zone is any set
of cells that share the same integer *label*, and one controller drives every cell
in a zone to a single common temperature.

Your job is to choose the labelling — the partition into zones — that minimizes
the total running cost of the greenhouse. This is a static
AtCoder-heuristic-contest instance: the field is fixed and known, and you output
one labelling.

## Costs

The total cost of a labelling is the sum of two parts.

- **Mismatch cost.** Within a zone the controller always picks the temperature
  that minimizes squared error, i.e. the mean of the ideals of that zone's cells.
  The mismatch cost of a zone is therefore its **sum of squared deviations from
  its own mean**, `sum_{cell in zone} (ideal - mean_of_zone)^2`. The total
  mismatch is the sum over all zones.
- **Wall cost.** Every pair of orthogonally adjacent cells (up/down/left/right)
  that lie in **different** zones needs a thermal divider between them, costing
  `wall_penalty` each.

```
total_cost = mismatch + wall_penalty * (number of adjacent cross-zone cell pairs)
```

Minimize `total_cost`.

The two costs pull in opposite directions. One giant zone pays **zero** wall cost
but the full mismatch of the whole field. Giving every cell its own zone pays
**zero** mismatch but a wall between every neighbour — and `wall_penalty` is
chosen so that full fragmentation is *worse* than a single zone. Good partitions
live in between, and there is no closed-form optimum.

## Public instance (stdin)

Your program reads ONE JSON object from standard input:

```json
{
  "name": "gh101",
  "H": 6,
  "W": 6,
  "wall_penalty": 120,
  "ideal": [[30, 31, ...], ...]
}
```

- `H`, `W` — grid dimensions (`ideal` is `H` rows of `W` integers each).
- `wall_penalty` — non-negative integer cost per adjacent cross-zone pair.
- `ideal[i][j]` — integer ideal temperature of cell `(i, j)`.

## Answer (stdout)

Write ONE JSON object to standard output:

```json
{"labels": [l_0, l_1, ..., l_{H*W-1}]}
```

`labels` is a flat **row-major** list of exactly `H*W` **non-negative integers**;
cell `(i, j)` gets `labels[i*W + j]`. Label values are arbitrary — a zone is just
the set of cells that share a label. You may use as few or as many distinct labels
as you like.

An answer is **valid** iff `labels` is a list of exactly `H*W` non-negative
integers. Any other output — wrong length, negative/float/boolean entries, a
crash, a timeout, or non-JSON — scores 0 on that instance.

## Compute budget

Treat the controller as allowed on the order of `1e7` elementary operations per
instance. The grids are small (up to `10 x 10`), so quantile bucketing, band
segmentation, and agglomerative region merging all fit comfortably. The harness
enforces this only as a safety timeout; **the score never depends on wall-clock
time**, so runs are fully reproducible.

## Scoring

For each instance let `q_cand` be your total cost and `q_base` be the cost of the
one-zone labelling (mismatch of the whole field around its global mean, zero
walls). With an unreachable ideal `q_lb = 0` (zero mismatch and zero walls at
once), your per-instance score is

```
r = clamp( 0.1 + 0.9 * (q_base - q_cand) / q_base , 0, 1 )
```

Reproducing the one-zone reference scores `~0.1`; doing worse scores below that
(clamped at 0). Because `q_lb = 0` can never be reached (any mismatch reduction
buys walls), even strong strategies stay well under `1.0`. The reported `Ratio`
is the mean of `r` over all hidden instances, which include larger, noisier
held-out fields to reward strategies that generalize.

## Strategy hints

- A single label reproduces the weak baseline (`~0.1`).
- Bucketing cells by temperature drops mismatch fast but scatters walls.
- Merging adjacent zones only when the merge lowers total cost balances the two
  terms directly and tends to generalize.
