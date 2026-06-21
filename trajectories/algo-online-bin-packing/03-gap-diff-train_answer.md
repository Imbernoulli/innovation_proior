Best-Fit pulled the excess down to $4.146\%$ (Weibull) and $4.411\%$ (OR-style), beating First-Fit on every seed but only modestly — and it left a brutal constraint I had to take seriously. A side-experiment confirmed it literally: replacing $-(r-s)$ with $-(r-s)^2$, $\exp(-(r-s))$, or any monotone reshaping of a single bin's slack leaves the bin count *identical*, because the $\arg\max$ ordering is unchanged. So I cannot beat Best-Fit by being cleverer about one hole. Whatever I do has to make a bin's score depend on the *other* bins — on where this bin sits in the landscape of all the open bins — so that the same leftover slack can be ranked differently depending on context. That is the only door left open.

I propose a **hand-designed relational score**: the squared distance below the emptiest valid bin, item-weighted, then differenced across neighbouring bins. The cheapest, most natural piece of relational information Best-Fit throws away is the most salient landmark in the array — the *emptiest* valid bin, the one with the largest remaining capacity $r_{\max}$ (a still-full freshly-opened bin is the extreme case, sitting at capacity). Every other bin's remaining capacity reads as a distance below it, $r - r_{\max}$, a non-positive number that is $0$ for the emptiest bin and grows more negative the more a bin has already been filled. That distance is genuinely relational — it depends on the whole array, not one entry — and it means "how much more committed this bin already is than the freshest one." I want to *prefer* the committed bins, because those are the ones I am trying to finish: Best-Fit's residual waste was the long tail of bins stuck a few units short of full, and biasing the policy toward pouring items into already-committed bins consolidates them faster while keeping the fresh capacity in reserve. The square $(r - r_{\max})^2$ writes this cleanly: zero at the emptiest bin, growing quickly for the committed ones, sharply separating "fresh" from "committed."

Raw distance is not the whole story, because the item size has to enter. The same $r - r_{\max}$ means something different for a tiny item than a large one — a large item poured into a committed bin makes a big difference to closing it, a tiny item barely advances it. So I weight the distance by how consequential the item is relative to the bin, dividing by the item size: $(r - r_{\max})^2 / s$. Now small items — the ones Best-Fit is happy to scatter — get a gentler relational pull, and the relational preference kicks in most strongly exactly where Best-Fit is weakest. I still respect feasibility by negating the score on the genuinely-fitting bins, so that after the relational term a tighter fitting bin is still preferred before the next operation runs.

The piece that turns out to be the whole engine is the last one. A per-bin score, even a relational one like $(r - r_{\max})^2 / s$, still gets fed to $\arg\max$ one entry at a time, and on its own it risks collapsing back toward a Best-Fit-like ordering. What rescues it is making each bin's score depend on its *neighbour* in the array: I difference the score along its order,

$$\text{score}[i] \;\leftarrow\; \text{score}[i] - \text{score}[i-1],$$

so a bin is judged not by its absolute relational value but by the *jump* in relational value from its predecessor — it only wins if it is a local improvement over its neighbour. This couples the bins together: no bin's score is decided in isolation anymore, which is exactly the property I argued was necessary, and it breaks the monotone collapse that pinned every single-bin rule to Best-Fit. I reached for it partly empirically — the differencing makes the selected bin depend on the array's local structure in a way I cannot fully predict by hand, only measure. But the measurement is decisive: on the Weibull streams this rule collapses nearly all of Best-Fit's residual waste, and an ablation that drops the differencing and keeps only the per-bin relational term explodes to roughly $80\%$ excess — catastrophic — confirming the neighbour coupling is doing all the work.

What I deliberately leave on the table is the *form* of the relational score. I chose $(r - r_{\max})^2 / s$ and a plain neighbour difference by hand, from intuition; I have no reason to believe these specific powers and this specific coupling are optimal, and the asymmetry I expect between the two families — near-optimal on Weibull, only partway on OR-style — would be the fingerprint of a form implicitly tuned to one regime. Searching over that functional form, rather than guessing it, is what the final rung is for.

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
