Best-Fit left me with a sharp constraint, and I want to take it seriously because it tells me exactly
what *cannot* work. Best-Fit scores each bin as a function of that bin's own leftover slack `r − s`,
and I checked the brutal consequence directly: any rule that is a monotone reshaping of a single bin's
slack makes the identical choices. Squaring the slack, exponentiating it, adding a tiny tie-break for
how full the bin already is — I tried these and the bin counts did not move one unit, because `argmax`
of a monotone function of slack is `argmax` of slack. So I am not going to beat Best-Fit by being
cleverer about a single hole. Whatever I do has to make a bin's score depend on the *other* bins — on
where this bin sits in the landscape of all the open bins — so that the same leftover slack can be
ranked differently depending on context. That is the only door left open.

Let me think about what context could possibly matter. The harness hands me the remaining capacities
of all the valid bins at once, as an array, in a stable positional order. Best-Fit throws away
everything about that array except each entry's own value. What is the cheapest, most natural piece of
*relational* information I am currently ignoring? The most salient landmark in the array is the
**emptiest** valid bin — the one with the largest remaining capacity. A still-full, freshly-opened bin
is the extreme case of this: it sits at the top, at capacity. Every other bin's remaining capacity can
be read as a *distance below* that emptiest bin: `(r − r_max)`, a non-positive number that is `0` for
the emptiest bin and grows more negative the more a bin has already been filled. That distance is a
genuinely relational quantity — it depends on the whole array, not just one entry — and it has an
intuitive meaning: how much *more committed* this bin already is than the freshest one. A bin far below
the emptiest is a bin I have already invested in.

Why would I want to prefer the bins that are far below the emptiest? Because those are the bins I am
trying to *finish*. Best-Fit's residual waste was the long tail of bins stuck a few units short of
full. If I can bias the policy toward pouring items into the already-committed bins — driving them the
rest of the way to full — rather than touching the freshest ones, I consolidate the committed bins
faster and keep the fresh capacity in reserve. So my relational signal should *reward* a large
distance below the emptiest. The square of that distance, `(r − r_max)²`, is a clean way to write it:
it is `0` at the emptiest bin and grows quickly for the committed ones, so it sharply separates "fresh"
from "committed."

But raw distance is not the whole story, because the item size has to enter. The same `(r − r_max)`
means something very different for a tiny item than for a large one: a large item poured into a
committed bin makes a big difference to closing it, while a tiny item barely advances it. So I want to
weight the distance by how *consequential* the item is relative to the bin, and the simplest
dimensionally-natural way is to divide by the item size: `(r − r_max)² / s`. Now small items, which
are the ones Best-Fit is happy to scatter, get a *gentler* relational pull, and the policy's relational
preference kicks in most strongly exactly where it should. I still have to respect feasibility — among
bins that actually fit the item I do not want to send it somewhere that overflows — so I keep the sign
convention that valid bins (those with `r ≥ s`) are the ones in play and negate the score on them. The
intent is that, after the relational term, a tighter genuinely-fitting bin still comes out ahead. This
is fiddly and I am not at all sure I have the sign bookkeeping right, so before I commit to it I want
to push a concrete state through the rule by hand and watch which bin actually wins.

Take an item of size `s = 3` and three valid bins with remaining `[10, 5, 3]` — all fit, and the last
one is an exact fit. The emptiest is `r_max = 10`. The raw relational term `(r − r_max)² / s` is
`[0, 8.33, 16.33]`. The feasibility flip negates the score on the fitting bins; with the strict mask
`r > s` the first two flip and the exact-fit bin (`r = s = 3`) keeps its positive sign, giving
`[−0, −8.33, 16.33]`. So `argmax` lands on the exact-fit bin — which is the right answer, and it is
exactly what Best-Fit would do here too. Now perturb it so there is *no* exact fit: bins
`[8, 6, 4]`, same item `3`. Every bin satisfies `r > s`, so every score gets negated:
`(8−8)²/3, (6−8)²/3, (4−8)²/3 = [0, 1.33, 5.33]`, negated to `[−0, −1.33, −5.33]`. The `argmax` of an
all-non-positive array is its least-negative entry, here index 0 — the emptiest bin, `r = 8`. But
Best-Fit would pick the tightest, `r = 4`. So with this sign convention the rule actively prefers the
*freshest* bin whenever no exact fit is available, which is the precise opposite of the consolidation I
was reaching for. That is a real problem, not a bookkeeping nuance: with real-valued Weibull sizes,
exact fits essentially never occur, so the rule as it stands would route almost every item into the
emptiest open bin and shred the packing. My consolidation story was wishful — I argued for it but the
numbers I just pushed through say the opposite happens.

So the per-bin score, even this relational one, is not enough on its own; the sign-flipped quadratic
collapses to "prefer the emptiest" exactly where Best-Fit is strong. I need something that re-couples
the bins after the per-bin score, something that makes a bin win only when it is *locally* special
rather than globally least-committed. The cheapest array-level coupling I have not used is the
positional adjacency the harness preserves: replace each bin's score by how much it *exceeds the bin
before it*, `score[i] ← score[i] − score[i−1]`, a plain first difference along the array. Its meaning
is that a bin is judged not by its absolute relational value but by the *jump* in relational value from
its predecessor — a bin wins only if it is a local improvement over its neighbour. Let me run the same
two states through the differenced rule and see whether it rescues the exact-fit case without
re-breaking the no-exact-fit case. For `[10, 5, 3]`, `s = 3`: pre-difference `[−0, −8.33, 16.33]`,
differenced `[−0, −8.33, 24.67]` — `argmax` still the exact-fit bin, good. For `[8, 6, 4]`:
pre-difference `[−0, −1.33, −5.33]`, differenced `[−0, −1.33, −4]` — `argmax` still index 0, still the
emptiest. So the differencing does not, by itself, fix the no-exact-fit pathology I just found; it
preserves the right answer when an exact fit exists and leaves the wrong answer standing when one does
not. What it *does* change is that no bin's score is decided in isolation anymore — the selected bin
now depends on the array's local structure and even on its positional order, which I confirmed by
permuting the same multiset and watching the pick stay locked on the emptiest bin regardless of order.
That coupling is the property I argued was necessary to escape the monotone-collapse to Best-Fit, and
the rule genuinely has it now; whether the coupling helps or hurts the bin count is something I cannot
read off these few states.

That leaves me honest about where this stands. The hand-trace tells me two concrete things: the rule
does the sensible thing on exact fits, and it does the *wrong* thing — prefers the freshest bin — when
no exact fit exists, which is most of the time on continuous sizes. I would expect, then, that run on
the Weibull streams this scores *worse* than Best-Fit, possibly much worse, and I would want to confirm
that on the seeded harness rather than guess at the magnitude. So I am not going to present this as a
finished heuristic that beats Best-Fit; the traced behaviour says it probably does not. What it is, is
a concrete relational scoring *form* — emptiest-bin distance, item-weighted, then neighbour-differenced
— that provably reads the whole array and escapes the single-bin collapse, while clearly having the
wrong sign/coupling somewhere in the no-exact-fit regime. The specific powers and the plain first
difference were chosen by hand from intuition, and the hand-trace has now shown me they are not right
as-is. That is exactly the handle the next rung needs: hold this functional family fixed — distance
below the emptiest bin, an item weighting, an array-coupling difference — and *search* over the powers,
the weighting, and the sign convention, letting measured bin counts on the seeded streams pick the
member of the family that actually consolidates instead of scatters. I have derived the form and found,
by tracing it, where it breaks; what I have not done, and cannot do by hand, is find the member of the
family that wins, which is the job I am handing to the search.

For this gap-difference rule, the risky part is the neighbour first-difference after the item-weighted
squared distance from the emptiest bin; if I cannot keep those signs and exact-fit cases straight
within budget, I will fall back to ordinary Best-Fit, whose trace is simple and correct, and ship that.
A plain correct submission beats an ambitious broken one.
