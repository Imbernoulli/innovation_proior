**Problem.** Online 1-D bin packing: items arrive one at a time, placed immediately and irrevocably
into a fitting open bin (capacity `C`) or a new bin. Minimise #bins; report mean #bins and percent
excess over the L1 lower bound `ceil(Σ items / C)`. Edit one function `priority(item, bins) -> array`;
the item goes to the `argmax` bin.

**Key idea.** Best-Fit: place the item in the valid bin where it fits **most snugly** — the bin left
with the least slack `r − s` (remaining capacity minus item size) after placement, among all bins
with `r ≥ s`. Score each bin `−(r − s)` so the tightest fit is the `argmax`. This keeps small items
with nearly-full bins (closing them off near-full, the dense packing the lower bound rewards) and
*preserves* roomy bins for genuinely large future items, instead of squandering them on small items
the way First-Fit does.

**Why these choices.** A small leftover hole is doubly good: a bin filled to a sliver of capacity has
effectively retired near-full, and by sending the item to the bin that barely fits it we leave the
roomy bins intact for big arrivals — the matching that wastes the least capacity. New-bin behaviour is
automatic and correct: a still-full bin has the largest slack, hence the most negative score, so
Best-Fit opens a fresh bin only when no used bin can hold the item at all. The residual waste is the
scatter of tight-fit holes Best-Fit cannot avoid because it judges each bin *in isolation*: it is a
monotone function of a single bin's slack, so any reshaping of the per-bin score gives the same
ordering and the same choices. Beating it therefore requires a score that compares a bin *against the
others* — the lever the next rung pulls.

**Hyperparameters / contract.** None. Returns `−(bins − item)`, the negative leftover slack per valid
bin. Deterministic given the stream; works at any capacity and item sizes.

```python
import numpy as np

def priority(item, bins):
    """Best-Fit: prefer the valid bin left with the least slack after placement.

    `bins` are remaining capacities of bins that can fit the item (so bins - item >= 0).
    Maximising -(bins - item) selects the tightest fit; a still-full bin has the
    largest slack and the most negative score, so a new bin opens only as a last resort.
    """
    return -(bins - item)
```
