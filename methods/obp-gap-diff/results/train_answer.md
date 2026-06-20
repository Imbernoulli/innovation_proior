The problem is online one-dimensional bin packing — items arriving one at a time, each placed
immediately and irrevocably into a fitting open bin of capacity $C$ or a new bin, minimizing the bin
count, measured as percent excess over the L1 lower bound $\lceil \sum_i s_i / C \rceil$. Best-Fit,
which sends each item to the valid bin left with the least slack $r - s$, is the strong classical
baseline, and it leaves behind a sharp constraint that I want to take seriously because it tells me
what *cannot* work. Best-Fit scores each bin as a function of that bin's own leftover slack, and the
brutal consequence is that any monotone reshaping of a single bin's slack makes the identical choices:
squaring the slack, exponentiating it, adding a tiny tie-break for how full the bin already is — none
of these move the bin count by a single unit, because they all induce the same $\arg\max$ ordering. So
I will not beat Best-Fit by being cleverer about one hole. Whatever I do must make a bin's score depend
on the *other* bins — on where this bin sits in the landscape of all the open bins — so that the same
leftover slack can be ranked differently depending on context. That is the only door left open.

I propose a relational scoring rule that I will call gap-difference. The cheapest, most natural piece
of relational information Best-Fit throws away is the most salient landmark in the array of valid bins:
the *emptiest* one, the bin with the largest remaining capacity $r_{\max} = \max_j r_j$ (a freshly
opened bin is the extreme case, sitting at capacity). Every other bin's remaining capacity can then be
read as a distance below that emptiest bin, $r - r_{\max}$ — a non-positive number that is $0$ for the
emptiest bin and grows more negative the more a bin has already been filled. This is genuinely
relational: it depends on the whole array, not just one entry, and it has a clean meaning, namely how
much *more committed* this bin already is than the freshest one. I want to *reward* bins that are far
below the emptiest, because those are the bins I am trying to finish — biasing placements into the
already-committed bins consolidates them faster and keeps fresh capacity in reserve, which is exactly
where Best-Fit's residual waste (a long tail of nearly-full-but-not-quite bins) comes from. I write the
reward as the square of that distance, $(r - r_{\max})^2$, which is $0$ at the emptiest bin and grows
quickly for the committed ones, sharply separating "fresh" from "committed."

Raw distance is not the whole story, because the item size has to enter. The same $r - r_{\max}$ means
something very different for a tiny item than for a large one — a large item poured into a committed
bin makes a big difference to closing it, a tiny item barely advances it — so I weight the distance by
how consequential the item is relative to the bin, dividing by the item size to get $(r - r_{\max})^2 /
s$. Now small items, the ones Best-Fit happily scatters, get a gentler relational pull, and the
preference kicks in most strongly where it should. I keep feasibility by negating the score on the
genuinely-fitting bins, so that among bins that actually hold the item a tighter fit is still
preferred. And here is the piece that turns out to be the whole engine. A per-bin score, even a
relational one, is still fed to $\arg\max$ one entry at a time, and I worried it would collapse back
into a Best-Fit-like ordering. What rescues it is making each bin's score depend on its *neighbour* in
the array: I difference the score along its order, $\text{score}[i] \leftarrow \text{score}[i] -
\text{score}[i-1]$, so a bin wins only if it is a local improvement over its predecessor. This
operation looks strange and I reached for it partly empirically, but its meaning is defensible — it
scores a bin by the *jump* in relational value from its predecessor rather than by an absolute value,
which couples the bins together so that no bin's score is decided in isolation. That coupling is
exactly the property I argued was necessary, and it breaks the monotone collapse that pinned every
single-bin rule to Best-Fit. Its precise effect on the selected bin depends on the array's local
structure in a way I cannot fully derive by hand and must measure — and indeed an ablation that drops
the differencing and keeps only the per-bin relational term explodes the excess to roughly $80\%$,
confirming the differencing is doing all the work.

So the rule, in full, is: among valid bins score each one $(r - r_{\max})^2 / s$, flip the sign on the
fitting bins so feasibility is respected, then difference the array so each bin is judged against its
neighbour. Unlike Best-Fit it reads the whole array — the emptiest-bin landmark, the item-relative
weighting, and the neighbour differencing all make a bin's fate depend on its context — and it pulls
items into the already-committed bins, finishing them off faster and cutting into Best-Fit's long tail
of nearly-full bins. I am honest that the specific square power, the single division by the item, and
the plain neighbour difference were all chosen by intuition rather than derived, and I have no reason
to believe these particular powers and this coupling are optimal; searching over that functional form,
rather than guessing it, is what a discovered heuristic does next.

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
