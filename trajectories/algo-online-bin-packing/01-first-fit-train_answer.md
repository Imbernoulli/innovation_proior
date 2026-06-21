The online setting leaves almost no freedom: a stream of items arrives one at a time, each must be placed for good before I see the next, and I want to finish with as few bins as possible. Offline I could sort the multiset and pack cleverly; here I am blind to the future, so every placement is a bet against items I have not yet seen. The only real decision is *which already-open bin to drop the current item into*, because each time I am forced to open a fresh bin I have admitted that the open bins could not absorb the item — and open bins with leftover room are the one thing standing between me and the trivial policy of one bin per item. Before I can claim any heuristic is good, I need a floor: the most conservative, most obviously-correct rule whose behaviour I can fully predict.

I propose **First-Fit**: keep the open bins in a fixed order, and put each item into the *first* bin in that order that still has room for it; only when none do, open a new bin at the end. Its whole appeal is that it never does anything clever and never does anything stupid — it greedily reuses capacity in a deterministic sweep, so it can never do worse than one bin per item and it reuses a bin whenever reuse is possible at all. That makes it the honest baseline: any heuristic worth proposing has to beat the policy that just says "reuse the earliest bin you can."

What makes First-Fit *realisable* in this harness is a small observation about the interface. The harness does not hand me the open bins in creation order with their fill levels; for the current item it hands me only the sub-array of remaining capacities of the bins that *can* fit it — the valid bins — in a stable positional order (index $0$ is the first bin ever opened, and that order never changes across the run). So "first bin that fits" is exactly "valid bin of smallest index." Since the harness places the item in the $\arg\max$ of my score and breaks ties toward the lowest index, I can realise First-Fit either by returning a constant score and leaning on the tie-break, or — more transparently — by returning a score that *strictly decreases* with position in the valid sub-array, so the earliest valid bin is the unique maximum. I write it the second way, as $\text{score}_i = -i$ over the valid bins, so the intent does not lean on any tie-break convention: the lowest-index valid bin always wins.

The new-bin behaviour then falls out for free. The harness pre-allocates a large bins array all sitting at full capacity, so "open a new bin" is not a special branch. When no already-used bin fits the item, the only valid bins left are still-full ones, and First-Fit takes the earliest of those — which *is* opening a fresh bin. I never have to distinguish "reuse" from "open"; the rule is genuinely just "earliest valid bin."

I want to be explicit about why this will not be very good, because that diagnosis is what the next rung must attack. The weakness is the word "first." When several open bins could all fit the item, First-Fit takes the earliest one purely by accident of creation order, with no regard for the slack it leaves behind. Picture a bin with $90$ units of slack and a bin with $12$, and an item of size $10$: if the roomy bin happens to come first, First-Fit pours the $10$ into it, leaving $80$ — spending a big, valuable pocket of capacity on a small item that the tight bin could have swallowed almost perfectly. The tight bins are exactly the ones I most want to finish off; a bin filled to $99/100$ has effectively retired, while a bin left at $80$ is a half-used resource I now have to hope future items fit into. First-Fit systematically spreads items across the early bins and leaves a long tail of partially-filled bins that never get topped off. That wasted capacity is extra bins, and so I expect it to land a few percent above the L1 lower bound $\lceil \sum \text{items}/C \rceil$. The fix — rank by fit *tightness* rather than position — is the next rung.

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
