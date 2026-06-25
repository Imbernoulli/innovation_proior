My hand-built relational rule reached $0.718\%$ excess on the Weibull streams — nearly optimal — but only $3.271\%$ on the OR-style ones, and I am fairly sure I know why: I *guessed* the functional form. I picked the squared distance below the emptiest bin, divided by the item once, and a single plain neighbour difference, all from intuition. Each of those choices — the power on the distance, the power on the item, the number and shape of the extra terms — is a free parameter I set by hand, and there is no reason my guess sits at the optimum. The asymmetry between the two families is the fingerprint of a form tuned, implicitly, to one regime. The honest move is to stop guessing the form and let a *search* over the space of priority functions find it.

I propose the **FunSearch-discovered priority function**, reproduced verbatim. This is exactly the regime program search was built for: an LLM proposes candidate `priority(item, bins)` bodies, each is scored by running the same online simulator and counting bins, the best survivors seed the next round of proposals, and the loop evolves the code. The fitness signal is the one I have been reading all along — fewer bins on the held instances — so whatever it converges to was selected purely for packing tightly, with no human prior on its shape. I do not re-run the search; I take the Weibull heuristic the search already converged to, transcribe it exactly, and read it as the search's verdict on the form I was guessing at.

What the search returns keeps the very piece I had reached for — a relational term built from the distance below the emptiest bin, item-weighted — but it does not stop there. It writes the score as a sum of three terms:

$$\text{score} \;=\; \frac{(\text{bins} - \text{max\_bin\_cap})^2}{\text{item}} \;+\; \frac{\text{bins}^2}{\text{item}^2} \;+\; \frac{\text{bins}^2}{\text{item}^3},$$

then applies the same feasibility sign flip on the genuinely-fitting bins and the same neighbour differencing across the array. The first term is essentially my rung-3 core. So the search *confirmed* the two structural choices I had made by intuition — the emptiest-bin landmark and the neighbour differencing — and then added two more item-power terms I would never have guessed, terms that bring the bin's own remaining capacity in at successively higher inverse powers of the item.

Understanding *why* those extra terms help is what separates copying from knowing. Both added terms, $\text{bins}^2/\text{item}^2$ and $\text{bins}^2/\text{item}^3$, scale with the square of the bin's remaining capacity but with steeply increasing sensitivity to small items — $1/\text{item}^2$ and $1/\text{item}^3$ blow up as the item shrinks. So for very small incoming items these terms dominate and reshape the ranking sharply, while for large items they fade and the distance-below-emptiest term carries the decision. The search discovered an item-size-dependent *blend*: which relational feature governs the placement depends smoothly on how big the arriving item is. That is a subtlety my single $(\,\cdot\,)^2 / s$ term could not express — it had one fixed item weighting — and it is plausibly why my rule was tuned to one regime and faltered on the other. Search did not find a smarter idea so much as a *richer parameterisation* of the same idea, fitted to the training distribution by selection rather than by my taste.

I am scrupulous about what reproducing this shows on my streams, because the reference numbers were measured on the search's own instances. On the Weibull streams — the heuristic's home distribution — I expect it to land essentially at the lower bound, reproducing the reference $\sim\!0.68\%$ excess and edging past my rung-3 rule, since rung 3 is a strict sub-part of it. And it does: $0.668\%$, best of all four rungs, against Best-Fit's $4.146\%$. On the OR-style streams I was genuinely unsure, and the honest reading is the instructive one: the endpoint scores $3.431\%$, *slightly worse* than my hand-built rung-3 core's $3.271\%$, though both still beat Best-Fit's $4.411\%$. This is not a regression in the ladder — it is the expected fingerprint of a searched heuristic. This is the function FunSearch evolved on the *Weibull* distribution (a separately-discovered OR heuristic exists, which scores $3.11\%$ on OR3); its two extra item-power terms are tuned to Weibull item sizes, and off that distribution they do not help and cost a little against the simpler relational core. The discovered heuristic dominates exactly where it was searched — on Weibull, to within $0.03\%$ of the bound at 100k items in the reference numbers — and its dominance is sharpest precisely there. That is the headline the endpoint reproduces: on the distributions it was searched for, the discovered priority function beats both First-Fit and Best-Fit, by a margin no hand-built rule in this ladder, including my own, reaches.

```python
import numpy as np

def priority(item, bins):
    """FunSearch-discovered online bin-packing heuristic (Weibull dataset).

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
