# Context: online one-dimensional bin packing

## Research question

A stream of items arrives **one at a time**; each has a size and must be placed **immediately and
irrevocably**, before the next item is seen, into one of the currently open bins (each of fixed
capacity `C`) that still has room for it, or into a brand-new bin. Once placed, an item never moves.
The single thing being designed is a **priority function** `priority(item, bins) -> array`: given the
incoming item's size and the array of remaining capacities of the bins that can still fit it, it
scores every such bin, and the item goes into the highest-scoring one. The goal is to use as **few
bins** as possible over the whole stream (lower is better).

Because the online optimum is order-dependent and NP-hard to pin down, the honest yardstick is the
**L1 lower bound** `LB = ceil(Σ items / C)` — the bins needed if every bin were filled to the brim
with zero waste. No policy can beat `LB`, so each heuristic is reported as **mean number of bins** and
as **percent excess over the lower bound**, `100 · (mean_bins − LB) / LB`.

## Background

This is the setting of FunSearch (Romera-Paredes et al., *Mathematical discoveries from program search
with large language models*, Nature 2024; code `google-deepmind/funsearch`, `bin_packing/
bin_packing.ipynb`), which used LLM-driven program search to discover online-bin-packing priority
functions that beat the classical First-Fit and Best-Fit baselines on the Beasley OR-Library and on
Weibull-distributed instances. The published metric is exactly the fraction of bins above the L1
lower bound. Published Table 1 (excess over LB; lower is better):

| Dataset | First Fit | Best Fit | FunSearch |
|---|---|---|---|
| OR3 | 5.74% | 5.37% | 3.11% |
| Weibull 5k | 4.23% | 3.98% | 0.68% |
| Weibull 100k | 4.00% | 3.79% | 0.03% |


## The fixed substrate

The harness is the FunSearch online-bin-packing skeleton, reproduced verbatim. It maintains an array
of bin remaining-capacities (pre-allocated large enough that a fresh bin is always available); for
each arriving item it finds the valid bins (`remaining ≥ item`), calls `priority(item,
valid_remaining)`, places the item in the `argmax` bin, and decrements it. A bin is "used" iff its
remaining capacity ever dropped below `C`.

```python
import numpy as np

def get_valid_bin_indices(item, bins):
    return np.nonzero((bins - item) >= 0)[0]

def online_binpack(items, bins, priority):
    for item in items:
        valid = get_valid_bin_indices(item, bins)
        best = valid[np.argmax(priority(item, bins[valid]))]
        bins[best] -= item
    return bins

def num_bins_used(items, capacity, priority):
    bins = np.array([capacity] * len(items), dtype=float)
    return int((online_binpack(items, bins, priority) != capacity).sum())

def l1_bound(items, capacity):
    return int(np.ceil(np.sum(items) / capacity))
```

## Evaluation settings

Two seeded synthetic stream families, five seeds each (0–4), 5000 items per stream, matching the
FunSearch evaluation: **Weibull(scale = 45, shape = 3) at capacity `C = 100`** (the FunSearch "Weibull
5k" regime, Castiñeiras et al. 2012) and **OR-Library-style uniform integer sizes on `[20, 100]` at
capacity `C = 150`**. Every method is run on the same seeded streams; reported as mean #bins and
percent excess over the mean L1 lower bound. The simulator reproduces the published repo numbers to
the digit (OR3 First/Best/FunSearch = 5.74%/5.37%/3.11%; Weibull 5k = 4.23%/3.98%/0.68%) as a
calibration check.

## The editable interface

Exactly one function is editable: `priority(item, bins) -> array`, returning a real score per valid
bin (higher = more preferred); the item goes to its `argmax`. Everything else — the simulator, the
streams, the capacity — is frozen.
