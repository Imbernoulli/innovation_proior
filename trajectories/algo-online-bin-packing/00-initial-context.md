## Research question

Online one-dimensional bin packing. A stream of items arrives **one at a time**; each item has a
size and must be placed **immediately and irrevocably**, before the next item is seen, into one of
the currently open bins (each of fixed capacity `C`) that still has room for it, or into a brand-new
bin opened for it. Once placed, an item never moves. The single thing being designed is a
**priority function** `priority(item, bins) -> array`: given the incoming item's size and the array
of remaining capacities of all bins, it scores every bin, and the item goes into the highest-scoring
bin among those that can still fit it. The whole online policy is this one function; everything else
— the arrival order, the capacity, the new-bin rule — is fixed.

The objective is to use as **few bins** as possible over the whole stream. Lower is better. Because
the optimum of an online stream is itself NP-hard to pin down and order-dependent, the honest
yardstick is the **L1 lower bound** `LB = ceil(Σ items / C)` — the count we would need if every bin
could be filled to the brim with no wasted space. No online (or even offline) policy can beat `LB`,
so we report each heuristic as **mean number of bins used** and, more tellingly, as **percent excess
over the lower bound**, `100 · (mean_bins − LB) / LB`. A policy at `0%` packs with zero wasted
capacity; First-Fit and Best-Fit sit a few percent above; the question is how far below that a
discovered heuristic can push.

## How the score is defined

For a single instance the score is the integer number of bins that end up non-empty. Across a set of
seeded instances we report the mean. The reference is the mean L1 bound over the same instances, and
the headline metric is the percent excess above it. This is exactly the metric reported in the
FunSearch paper (Romera-Paredes et al., *Nature* 2024), which evaluated discovered online-bin-packing
heuristics by their fraction of bins above the L1 optimum on the OR-Library and on Weibull-distributed
instances. The published frontier we measure against:

| Dataset (FunSearch paper) | First Fit | Best Fit | **FunSearch** |
|---|---|---|---|
| OR1 | 6.42% | 5.81% | 5.30% |
| OR3 | 5.74% | 5.37% | **3.11%** |
| OR4 | 5.23% | 4.94% | 2.47% |
| Weibull 5k | 4.23% | 3.98% | **0.68%** |
| Weibull 10k | 4.20% | 3.90% | 0.32% |
| Weibull 100k | 4.00% | 3.79% | **0.03%** |

(Source: *Mathematical discoveries from program search with large language models*, Romera-Paredes et
al., Nature 2024, Table 1; PMC10794145. Code and the two discovered priority functions live in
`google-deepmind/funsearch`, `bin_packing/bin_packing.ipynb`.) The discovered heuristic beats Best
Fit on every dataset and, strikingly, *widens* the gap as the streams grow — on 100k Weibull items it
is `0.03%` off the lower bound. That published improvement over Best Fit is the target this ladder
climbs toward, and the endpoint reproduces the exact discovered function. We reproduced these repo
numbers as a calibration check before building our own seeded streams (OR3: First Fit `5.74%`, Best
Fit `5.37%`, FunSearch-OR `3.11%`; Weibull 5k: `4.23%` / `3.98%` / `0.68%` — matched to the digit).

## The fixed substrate

The harness is the FunSearch online-bin-packing skeleton, reproduced faithfully. It maintains an
array of bin remaining-capacities (pre-allocated large enough that a new bin is always available),
and for each arriving item: finds the valid bins (those whose remaining capacity is `≥ item`), calls
`priority(item, valid_remaining_capacities)`, places the item in the `argmax`-priority bin,
decrements that bin's remaining capacity, and moves on. A bin is "used" iff its remaining capacity
ever dropped below `C`. The simulator, the placement rule (immediate, irrevocable, highest-priority
valid bin), the capacity, and the seeded streams are all frozen.

Two seeded synthetic stream families are used, matching the FunSearch evaluation:

- **Weibull(scale = 45, shape = 3), capacity `C = 100`, 5000 items.** Item sizes drawn from a
  Weibull(45, 3) distribution, rounded to integers in `[1, 100]`. This is the FunSearch "Weibull 5k"
  regime (Castiñeiras–De Cauwer–O'Sullivan 2012, *Weibull-Based Benchmarks for Bin Packing*), the
  setting on which the discovered heuristic shines.
- **OR-Library-style, capacity `C = 150`, 5000 items.** Integer item sizes uniform on `[20, 100]`,
  the size range of the Beasley OR-Library `u`-class instances, at capacity `150`.

Five seeds (`0–4`) per family; every rung is run on the *same* seeded streams so the numbers are
directly comparable.

## The editable interface

Exactly one function is editable: `priority(item, bins)`, returning a real-valued array the same
length as `bins` (the remaining capacities of the bins that can still fit the item). Higher score =
more preferred. Every rung on the ladder is a different body for this one function. The simulator
below is fixed and shown for reference; it is the FunSearch skeleton verbatim.

```python
import numpy as np

def get_valid_bin_indices(item, bins):
    return np.nonzero((bins - item) >= 0)[0]

def online_binpack(items, bins, priority):
    for item in items:                                  # items arrive one at a time
        valid = get_valid_bin_indices(item, bins)       # bins that can still fit it
        best = valid[np.argmax(priority(item, bins[valid]))]
        bins[best] -= item                              # place, immediately & irrevocably
    return bins

def num_bins_used(items, capacity, priority):
    bins = np.array([capacity] * len(items), dtype=float)   # always a fresh bin available
    bins = online_binpack(items, bins, priority)
    return int((bins != capacity).sum())                    # count non-empty bins

def l1_bound(items, capacity):
    return int(np.ceil(np.sum(items) / capacity))           # ceil(Σ items / C)

# ---- EDITABLE: the priority function. Default = Best-Fit. ----
def priority(item, bins):
    return -(bins - item)        # prefer the bin left with the least slack
```

A new bin is opened automatically: the `bins` array is pre-filled with `capacity`, so an item with no
better option lands in a still-full bin (the loosest valid one is the largest-gap option, which
Best-Fit avoids unless forced). The only lever is the geometry of the score.

## Evaluation settings

Per stream family (Weibull-100 and OR-150), five seeded instances of 5000 items each. We report, per
rung, the **mean number of bins** over the five seeds and the **percent excess over the mean L1
lower bound**. The same five seeds are reused across all rungs, so a rung that scores lower truly
packs tighter on identical streams. The metric is exact (an integer bin count from the frozen
simulator); the published FunSearch percentages above are the frontier context the endpoint
reproduces. There is no held-out trickery and no partial credit — the only way to score lower is to
actually waste less capacity as the stream goes by.
