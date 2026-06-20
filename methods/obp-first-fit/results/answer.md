**Problem.** Online 1-D bin packing: items arrive one at a time and must be placed immediately and
irrevocably into an open bin (capacity `C`) that still fits, or a new bin. Minimise the number of
bins; report mean #bins and percent excess over the L1 lower bound `ceil(Σ items / C)`. The editable
surface is one function `priority(item, bins) -> array` scoring each currently-fitting bin; the item
goes to the `argmax` bin.

**Key idea.** First-Fit: ignore fit quality entirely and place the item in the **first** open bin
that can hold it, opening a new bin only when none can. In the harness the bins array is in a stable
positional order, so "first bin that fits" is "valid bin of smallest index." A priority that strictly
decreases with the bin's position makes the earliest valid bin the unique maximum, realising
First-Fit exactly. A new bin opens for free: when no used bin fits, the only valid bins are still-full
ones, and the earliest of those is a fresh bin.

**Why these choices.** This is the floor every later rung must beat. Greedy reuse — never opening a
new bin when an old one fits — captures most of the achievable saving and can never do worse than one
bin per item, so it is the honest, predictable baseline. Its known weakness, named here so the next
rung can attack it, is the word "first": among several bins that all fit, First-Fit takes the earliest
by accident of creation order, with no regard for the slack left behind. It will spend a roomy bin's
capacity on a small item and fail to top off a nearly-full bin — wasted capacity that becomes extra
bins. The fix is to rank by fit quality (Best-Fit), which is the next rung.

**Hyperparameters / contract.** None. `priority` returns a strictly decreasing function of bin index,
so `argmax` selects the lowest-index valid bin. Deterministic given the stream. Works at any capacity
`C` and any item sizes.

```python
import numpy as np

def priority(item, bins):
    """First-Fit: prefer the earliest (lowest-index) valid bin.

    `bins` holds the remaining capacities of the bins that can currently fit the
    item, in stable positional order. A strictly decreasing score over position
    makes the earliest valid bin the argmax, i.e. 'first bin that fits'.
    """
    return -np.arange(len(bins), dtype=float)
```
