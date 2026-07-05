# Belt Refinery: Assay-Calibration Cache Ordering

## Setting (asteroid mining)

An automated ore refinery in the asteroid belt processes a stream of **survey
probes**. Before a probe's ore batch can be graded, every mineral-**assay stage**
it requires must be *calibrated* on the refinery's shared spectrometer. Calibration
is expensive, so the refinery keeps a **prefix calibration cache**: the spectrometer
always runs a probe's required stages **in one fixed global order**, and it caches
every *prefix* of stage-calibrations it has ever computed. When a new probe's ordered
stage list shares a leading run with a previously-processed probe, that leading run is
a **cache hit** and is reused for free; only the remaining (novel) suffix must be
calibrated afresh.

Concretely, fix a global ordering `order` of all stages. Each probe needs a subset of
stages; it presents them **sorted by their position in `order`**. Treat every probe's
sorted stage list as a path from the root of a **prefix trie**. The number of
**distinct trie nodes** created equals the number of calibrations the spectrometer
must actually run; any node shared by two probes is a cache hit.

Without any cache, the refinery would run `T = sum over probes of |probe|` calibrations.
With the prefix cache and a chosen `order`, it runs `N(order)` = number of distinct trie
nodes. The **cache hit-rate** is `(T - N(order)) / T`.

**You choose the global stage order to maximize the hit-rate.** The order is *global*:
one permutation applies to every probe. A stage that many probes share is worth placing
early (its calibration is then reused by all of them), but stages that co-occur as a
correlated cluster are worth placing *contiguously* so a whole group shares a long
prefix — and marginal frequency alone does not capture that.

## Input (public instance JSON, via stdin)

```json
{
  "n_stages": 10,
  "probes": [[0,3,7], [3,7], [1,2,3,7], ...]
}
```

- `n_stages` (int): number of assay stages, labelled `0 .. n_stages-1`.
- `probes` (list of lists): each inner list is one probe's required stage subset
  (distinct stage ids, given in ascending id order). Every probe needs at least one
  stage.

## Output (answer JSON, via stdout)

```json
{"order": [7, 3, 0, 1, 2, ...]}
```

- `order`: a permutation of `0 .. n_stages-1` — the global calibration order. It must
  contain every stage id exactly once. Anything else (wrong length, duplicates,
  out-of-range ids, non-integers, non-finite values) is rejected and scores 0 for that
  instance.

## Objective

**Maximize** the cache hit-rate, i.e. **minimize** the number of distinct prefix-trie
nodes `N(order)` produced by grading all probes in your chosen global order.

## Scoring (deterministic)

For each instance the evaluator builds the prefix trie itself in a sealed parent process
and counts nodes `N`. Let `N_id` be the node count under the identity order
`[0,1,...,n_stages-1]` (the reference), and `T` the no-cache total. Your per-instance
score is

```
r = clip( 0.1 + 3.0 * (N_id - N) / T , 0, 1 )
```

so the identity order scores exactly `0.1`, a better (fewer-node) order scores higher,
and a worse order scores below `0.1`. The reported `Ratio` is the mean of `r` over a
fixed, seeded set of 10 instances (small clustered instances up to larger ones where
even the optimal order leaves the score well under 1). The candidate program is executed
in an isolated sandbox and only ever receives the public instance above.

## Strategies (open-ended)

- Identity / arbitrary order (reference, ~0.1).
- Marginal-frequency order: most-frequently-needed stage first.
- Correlation-aware order that keeps co-occurring stage clusters contiguous.
- Local search (swap / insertion moves, multi-restart) over the ordering.
