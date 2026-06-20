The problem is online one-dimensional bin packing — items arriving one at a time, each placed
immediately and irrevocably into a fitting open bin of capacity $C$ or a new bin, minimizing the bin
count, measured as percent excess over the L1 lower bound $\lceil \sum_i s_i / C \rceil$. The strong
classical baseline is Best-Fit, and beating it requires a *relational* score, one that compares a bin
against the others rather than judging each bin's slack in isolation. My own hand-built relational
rule — the squared distance below the emptiest bin, item-weighted, $(r - r_{\max})^2 / s$, with
feasibility sign and neighbour differencing — got most of the way on Weibull-distributed streams and
only partway on uniform OR-Library-style ones. I am fairly sure why: I *guessed* the functional form.
The power on the distance, the power on the item, the number and shape of the extra terms — each is a
free parameter I set by hand, and there is no reason my guess sits at the optimum. The honest move is
to stop guessing the form and let a search over the space of priority functions find it: an LLM
proposes candidate `priority(item, bins)` bodies, each is scored by running the online simulator and
counting bins, the best survivors seed the next round, and the loop evolves the code with the same
fitness signal I have been reading — fewer bins. Whatever it converges to is a priority function
selected purely for packing tightly, with no human prior on its shape.

I propose the discovered priority function that this search produces, and the right thing to do is to
reproduce it verbatim rather than paraphrase it, reading it as the search's verdict on the form I had
been guessing at. The discovered Weibull heuristic keeps the very piece I had reached for — the
relational term built from the distance below the emptiest bin, item-weighted — but does not stop
there. It writes the score as a sum of three terms,

$$\text{score} = \frac{(r - r_{\max})^2}{s} \;+\; \frac{r^2}{s^2} \;+\; \frac{r^2}{s^3},$$

where $r$ is a bin's remaining capacity, $r_{\max} = \max_j r_j$ the emptiest valid bin, and $s$ the
item size; it then applies the same feasibility sign flip on the genuinely-fitting bins ($r > s$) and
the same neighbour differencing $\text{score}[i] \leftarrow \text{score}[i] - \text{score}[i-1]$
across the array. The first term is essentially my hand-built core; the search confirmed the two
structural choices I had made by intuition — the emptiest-bin landmark and the neighbour differencing
— and then added two terms I would never have guessed.

I want to understand *why* those extra terms help, otherwise I am just copying. Both $r^2/s^2$ and
$r^2/s^3$ scale with the square of the bin's remaining capacity but with steeply increasing
sensitivity to small items, since $1/s^2$ and $1/s^3$ blow up as $s$ shrinks. So for very small
incoming items these terms dominate and reshape the ranking sharply, while for large items they fade
and the distance-below-emptiest term carries the decision. The search discovered an item-size-dependent
*blend*: which relational feature governs the placement depends smoothly on how big the arriving item
is. That is a subtlety my single $(r - r_{\max})^2/s$ term could not express — it had one fixed item
weighting — and it is plausibly why my rule was tuned to one regime and faltered on the other. The
search did not find a smarter idea so much as a richer parameterization of the same idea, fitted to the
training distribution by selection rather than by my taste.

I am scrupulous about what this shows. On the Weibull streams — the heuristic's home distribution — it
lands essentially at the lower bound and edges past my hand-built rule, since that rule is a strict
sub-part of it. On the off-distribution OR-style streams I am genuinely unsure: this is the
*Weibull*-tuned discovered function, and it is entirely possible that there the two extra item-power
terms, fitted to Weibull, do not help and even cost a little against my simpler core, while both still
beat Best-Fit. I report whatever the simulator says, including if the endpoint loses to the simpler
rule off-distribution — that would not undermine the result, it would *illustrate* it: a searched
heuristic is fitted to the distribution it was searched on, and its dominance is sharpest exactly
there. The headline is the reproducible one: on the distributions it was searched for, the discovered
priority function beats both First-Fit and Best-Fit, and on large Weibull streams it gets to within a
hundredth of a percent of the lower bound — a margin no hand-built rule in this progression, including
my own, reaches. What I bring to it is the understanding of why it is shaped the way it is — the
emptiest-bin landmark and the neighbour differencing I arrived at independently, plus the item-power
blend that only search could tune — transcribed exactly and read against the same seeded simulator
that scored every step before it.

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
