## Research question

Online one-dimensional bin packing. A stream of items arrives **one at a time**; each item has a
size and must be placed **immediately and irrevocably**, before the next item is seen, into one of
the currently open bins (each of fixed capacity `C`) that still has room for it, or into a brand-new
bin opened for it. Once placed, an item never moves. The single thing being designed is a
**priority function** `priority(item, bins) -> array`: given the incoming item's size and the array
of remaining capacities of all bins, it scores every bin, and the item goes into the highest-scoring
bin among those that can still fit it. The arrival order, the capacity, and the new-bin rule are
fixed.

The objective is to use as **few bins** as possible over the whole stream. Lower is better. Because
the optimum of an online stream is itself NP-hard to pin down and order-dependent, the honest
yardstick is the **L1 lower bound** `LB = ceil(Σ items / C)` — the count we would need if every bin
could be filled to the brim with no wasted space. No online (or even offline) policy can beat `LB`,
so we report each heuristic as **mean number of bins used** and as **percent excess over the lower
bound**, `100 · (mean_bins − LB) / LB`. A policy at `0%` packs with zero wasted capacity; the open
question is how far below the classic greedy excess a better priority function can push.

## Prior art / Background / Baselines

The standard online heuristics are **First Fit** and **Best Fit**: both are fast, deterministic, and
require no lookahead.

- **First Fit** scans the bins in order and places the item in the first bin that can accommodate
  it. It spreads items across many bins and leaves fragmented capacity that a more selective rule
  might consolidate.

- **Best Fit** places the item in the valid bin that minimizes the leftover slack. It quickly fills
  bins to near capacity, but the resulting odd residual capacities are often unusable by later
  items, forcing additional bins to open.

Neither baseline adapts to the size distribution of the stream; both leave several percent of excess
over `LB` on the instances used here.

## Fixed substrate / Code framework

The harness is a one-pass online-bin-packing simulator. It maintains an array of bin
remaining-capacities, pre-allocated large enough that a new bin is always available. For each
arriving item it finds the valid bins, calls `priority(item, bins[valid])`, places the item in the
`argmax`-priority bin, decrements that bin's remaining capacity, and moves on. Placement is immediate
and irrevocable; the capacity, the stream generators, and the seeded instances are frozen.

Two seeded synthetic stream families are used:

- **Weibull(scale = 45, shape = 3), capacity `C = 100`, 5000 items.** Item sizes drawn from a
  Weibull(45, 3) distribution and rounded to integers in `[1, 100]`.
- **OR-Library-style, capacity `C = 150`, 5000 items.** Integer item sizes uniform on `[20, 100]`,
  the size range of the Beasley OR-Library `u`-class instances at capacity `150`.

Five seeds (`0–4`) per family; every priority function is evaluated on the *same* seeded streams so
the numbers are directly comparable.

## Editable interface

Exactly one function is editable: `priority(item, bins)`, returning a real-valued array the same
length as `bins` (the remaining capacities of the bins that can still fit the item). Higher score =
more preferred. The simulator below is fixed and shown for reference.

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

Per stream family, five seeded instances of 5000 items each. For each priority function we report
the **mean number of bins** over the five seeds and the **percent excess over the mean L1 lower
bound**. The same seeds are reused across all functions, so a lower score truly means tighter packing
on identical streams. The metric is exact — an integer bin count from the frozen simulator — and
there is no partial credit; the only way to improve is to waste less capacity as the stream goes by.
