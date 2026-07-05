# Geothermal Logging Tool-String Ordering — Maximize Calibration-Cache Reuse (Format B, isolated)

A geothermal exploration campaign runs a downhole **logging pass** on a large
batch of boreholes. Each borehole is logged by a **modular tool string** built
from a subset of the `K` available sensor-module types (temperature, gamma-ray,
resistivity, flow, caliper, ...). Before a module can take readings it must be
**calibrated / warmed up at the wellhead in sequence**, and the rig keeps a
**calibration cache**: if the tool string for the next borehole *begins* with
the same ordered sequence of modules that some previously-assembled string
began with, that leading calibration is **reused** (a cache hit) and expensive
rig time is saved.

You control **one decision**: a single **global canonical ordering** of the
`K` module types. Every borehole assembles the modules it needs **in that
global relative order**, and the boreholes are logged in a **fixed field
schedule** (the order they are listed). As logging proceeds, the cache holds
every prefix seen so far; a borehole's reused steps = the length of the longest
prefix of its ordered module sequence that already appears in the cache.

This is a prefix-cache reuse problem: **maximize the total reused calibration
steps** across the whole campaign — equivalently, minimize the number of
distinct prefixes in the trie built from all tool-string sequences. There is no
closed-form optimum: fronting the most common modules helps, but tightly
co-occurring module groups must also be kept contiguous, and the two goals
interact.

## Public instance (stdin JSON)
```json
{
  "K": 30,                          // number of sensor-module types, ids 0..K-1
  "wells": [[m, m, ...], ...],      // each borehole = the SET of module ids it needs
  "baseline_order": [0,1,...,K-1]   // the naive reference ordering (normalisation anchor)
}
```
Each entry of `wells` is a list of distinct module ids in `[0, K-1]`. The field
schedule is the given order of `wells`.

## Answer (stdout JSON)
```json
{"order": [p_0, p_1, ..., p_{K-1}]}   // a PERMUTATION of 0..K-1
```
`order` must be a permutation of `0..K-1` (each module id exactly once). Any
other shape/type, a repeated or out-of-range id, or a wrong length is rejected
and scores **0**.

## What the grader computes
For your `order`, each borehole's required modules are emitted **sorted by their
position in `order`**. The grader inserts every borehole's sequence into a
growing trie (the calibration cache) and counts, per borehole, the longest
prefix already present — the reused calibration steps. Let `hit` be
`total_reused / total_steps` for your order and `base_hit` the same quantity for
`baseline_order`. The grader runs this itself; you only ever see the public
instance above.

## Objective & scoring
Maximize the **normalised hit-rate improvement** over the naive reference:
```
score = clip( 0.1 + 0.8 * (hit - base_hit) / (1 - base_hit),  0, 1 )
```
The naive reference (`baseline_order`) has zero improvement and maps to exactly
`0.1`; a perfect-reuse ordering approaches `0.9`. The final result is the mean
over **10 fixed, seeded** campaigns of varying size, group structure, and noise.

## Suggested strategies (increasing sophistication)
- **Naive reference** — assemble modules in natural id order (baseline ≈ 0.1).
- **Frequency-descending** — put the most-used modules first; ignores
  co-occurrence, so tightly-grouped modules get scattered by independent noise.
- **Co-occurrence aware** — keep modules that always appear together contiguous
  while still fronting the common modules.
- **Local search on reuse** — seed from the frequency order and hill-climb with
  pairwise swaps directly on the trie-size / reuse objective, capturing both
  common-module fronting and group contiguity.
