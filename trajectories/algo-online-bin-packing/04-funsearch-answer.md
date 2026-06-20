**Problem.** Online 1-D bin packing: items arrive one at a time, placed immediately and irrevocably
into a fitting open bin (capacity `C`) or a new bin. Minimise #bins; report mean #bins and percent
excess over the L1 lower bound `ceil(Σ items / C)`. Edit one function `priority(item, bins) -> array`;
the item goes to the `argmax` bin.

**Key idea.** The endpoint is the **FunSearch-discovered priority function**, reproduced verbatim from
`google-deepmind/funsearch` (`bin_packing/bin_packing.ipynb`), the heuristic evolved by LLM-driven
program search (Romera-Paredes et al., *Nature* 2024) to beat First-Fit and Best-Fit on the OR-Library
and Weibull instances. It keeps the rung-3 relational core — squared distance below the emptiest valid
bin, item-weighted, `(bins − max_bin_cap)² / item` — and adds two further item-power terms,
`bins² / item²` and `bins² / item³`, then applies the same feasibility sign flip on the fitting bins
and the same neighbour differencing across the array. The extra terms make which relational feature
governs the placement depend smoothly on the item size: `1/item²` and `1/item³` dominate for small
items and fade for large ones — an item-size-dependent blend no fixed hand-form expresses. Search did
not invent a new idea; it found a richer parameterisation of the rung-3 idea, fitted by selection.

**Why these choices.** This is the published frontier: on the FunSearch paper's datasets it beats Best
Fit everywhere (OR3 `5.37% → 3.11%`, Weibull 5k `3.98% → 0.68%`) and *widens* the gap as streams grow,
reaching `0.03%` off the lower bound on 100k Weibull items. The two structural choices it shares with
rung 3 — the emptiest-bin landmark and the neighbour differencing — are what break the single-bin
monotone collapse that pins Best-Fit; the two added item-power terms are the search's tuning that a
human would not guess. It is the *Weibull*-discovered function (the repo also ships a separately-
discovered OR heuristic), so its dominance is sharpest on its home distribution; reported honestly on
an off-distribution OR-style family it need not beat the simpler rung-3 core.

**Hyperparameters / contract.** None. The function is fixed (verbatim FunSearch). Returns an array the
length of the valid-bin slice; the item goes to its `argmax`. Deterministic given the stream. Note the
construction reads `max(bins)` over the valid slice and differences along bin order, so it depends on
the whole array, not one bin.

```python
import numpy as np

def priority(item, bins):
    """FunSearch-discovered online bin-packing heuristic (Weibull dataset), verbatim from
    google-deepmind/funsearch, bin_packing/bin_packing.ipynb.

    bins: remaining capacities of bins that can fit `item` (bins - item >= 0).
    Returns a priority per bin; the item is placed in the argmax bin.
    """
    max_bin_cap = max(bins)
    score = (bins - max_bin_cap) ** 2 / item + bins ** 2 / (item ** 2)
    score += bins ** 2 / item ** 3
    score[bins > item] = -score[bins > item]
    score[1:] -= score[:-1]
    return score
```
