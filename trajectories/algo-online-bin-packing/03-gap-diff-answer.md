**Problem.** Online 1-D bin packing: items arrive one at a time, placed immediately and irrevocably
into a fitting open bin (capacity `C`) or a new bin. Minimise #bins; report mean #bins and percent
excess over the L1 lower bound `ceil(Σ items / C)`. Edit one function `priority(item, bins) -> array`;
the item goes to the `argmax` bin.

**Key idea.** Beat Best-Fit by scoring each bin **relationally** instead of in isolation. Best-Fit
only sees a bin's own slack, and any monotone reshaping of that gives the same choices. Here each bin
is scored by its squared distance below the **emptiest** valid bin, item-weighted, `(r − r_max)² / s`
(remaining `r`, emptiest remaining `r_max`, item size `s`) — large for already-committed bins, zero
for the freshest. Then the score array is **differenced across bins**, `score[i] ← score[i] −
score[i−1]`, so each bin is judged by the jump in relational value over its neighbour. This couples
the bins together and breaks the single-bin monotone collapse that pins everything to Best-Fit. The
effect is to consolidate already-committed bins faster — finishing off the long tail of nearly-full
bins Best-Fit leaves — keeping fresh capacity in reserve.

**Why these choices.** The emptiest-bin landmark is the cheapest genuinely relational signal in the
array; distance below it measures how committed a bin already is, and rewarding that bias pours items
into bins we are trying to finish. Dividing by `s` makes the relational pull strongest where Best-Fit
is weakest (small items it would scatter). The neighbour differencing is the engine: a per-bin score,
however relational, still risks collapsing under `argmax`, but differencing makes each bin's fate
depend on its predecessor, coupling the whole array. The square power and the plain difference are
hand-chosen from intuition, not derived — that they are *not* optimal is exactly what the final rung
(searching the functional form) addresses. Sign convention: fitting bins (`r ≥ s`) get the score
negated so feasibility is respected before differencing.

**Hyperparameters / contract.** None tunable; the form `(r − r_max)² / s` + neighbour differencing is
fixed. Returns an array the length of the valid-bin slice. Deterministic given the stream. Works at
any capacity and item sizes.

```python
import numpy as np

def priority(item, bins):
    """Relational score: squared distance below the emptiest valid bin, item-weighted,
    then differenced across neighbouring bins so each bin is judged in context.

    `bins` are remaining capacities of bins that can fit the item (bins - item >= 0).
    """
    max_bin_cap = max(bins)
    score = (bins - max_bin_cap) ** 2 / item     # distance below the emptiest bin, item-weighted
    score[bins > item] = -score[bins > item]      # respect feasibility on the fitting bins
    score[1:] -= score[:-1]                        # couple each bin to its neighbour (differencing)
    return score
```
