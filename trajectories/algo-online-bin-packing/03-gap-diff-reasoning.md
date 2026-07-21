Best-Fit left me a sharp constraint, and the rung-2 run confirmed it in the least ambiguous way. The
side-experiment came back exactly as the `argmax` algebra predicted: replacing `−(r − s)` with
`−(r − s)²`, with `exp(−(r − s))`, or with a tiny fill-based tie-break left the bin count *identical*,
the same integer on every seed, because the `argmax` sequence is literally the same on every item. So
any monotone reshaping of a single bin's slack makes the identical choices, and I will not beat
Best-Fit by being cleverer about one hole. Whatever I do has to make a bin's score depend on the
*other* bins. That is the only door left, and the experiment shut every other one. And the numbers
size the prize: Best-Fit sits at `4.146%` (Weibull) and `4.411%` (OR-style), saving a mean `~8` bins
over First-Fit on Weibull and under `5` on OR — modest, and OR gained less, matching the reading that
OR's chunky sizes give tightest-fit fewer snug matches. None of that residual `4%` is reachable by any
single-bin rule, so the whole remaining prize is behind the relational door.

What relational context could matter? The harness hands me the remaining capacities of all valid bins
at once, in stable positional order, and Best-Fit throws away everything but each entry's own value.
The most salient landmark in the array is the **emptiest** valid bin, the one with the largest
remaining capacity. Every other bin's capacity can be read as a *distance below* it, `r − r_max`: `0`
for the emptiest, more negative the more a bin has already been filled. That distance depends on the
whole array, and it means something — how much *more committed* this bin is than the freshest one.

There is a simplification to pin down, because it changes what `r_max` concretely is. The harness
pre-allocates the array full of still-full bins at capacity `C`, and `get_valid_bin_indices` returns
every bin whose remaining is at least the item size — so a fresh bin at `C` is *always* in the valid
slice, since any item fits an empty bin. That means `r_max` is *always* exactly `C`, and
`(r − r_max)² = (r − C)²` is really the square of how much the bin has *already been filled*. So the
relational term rewards already-filled bins and hands a fresh bin exactly zero — cleaner than I first
put it, but it flags something I will have to watch: a fresh bin sitting at score `0` is a very
particular value, and how the rest of the construction treats that zero decides whether the rule ever
*prefers* opening a new bin, which would be fatal.

Why prefer the bins far below the emptiest? Because those are the bins I am trying to *finish*.
Best-Fit's residual was the tail of bins stuck a few units short of full; if I bias toward pouring
items into the already-committed bins — driving them the rest of the way — rather than touching the
freshest ones, I consolidate faster and keep fresh capacity in reserve. The square of the distance,
`(r − r_max)²`, is `0` at the emptiest bin and grows quickly for the committed ones, so it sharply
separates fresh from committed. But item size has to enter: the same `r − r_max` means something very
different for a tiny item than a large one, since a large item poured into a committed bin does much
more to close it. The simplest dimensionally-natural weighting divides by the item, `(r − r_max)² / s`,
so small items — the ones Best-Fit is happy to scatter — get a gentler relational pull. I keep the
sign convention that valid bins (`r ≥ s`) are in play and negate the score on them, so that after the
relational term a tighter genuinely-fitting bin still comes out ahead.

The square and the single division are the two places I am most nakedly guessing. I square the fill
rather than take it linearly because I want the term to *curve*: when I later difference neighbours the
jumps between heavily-committed bins should differ from the jumps between barely-touched ones, carrying
information about where in the fill range a bin sits — a linear fill would make them flat, and the
square is the lowest power that bends. Dividing by the item once keeps `(r − C)² / s` in units of
capacity, but that is taste, not derivation: a different item power, or a second item term, would
weight small and large arrivals very differently and I have no principled reason to prefer one. These
are the knobs I am flagging — a starting point, not an optimum.

This is fiddly and I am not sure I have the sign bookkeeping right, so before I trust it I push a
concrete state through by hand — including a fresh bin at capacity, since I just argued one is always
there. Item `s = 3`, capacity `C = 10`, valid slice `[5, 3, 10]`. The relational term `(r − C)² / s`
is `[25, 49, 0] / 3 = [8.33, 16.33, 0]`. The feasibility flip negates the strictly-fitting bins
`r > s` — here `[T, F, T]` — so the two non-exact bins flip and the exact-fit bin (`r = s = 3`) keeps
its sign: `[−8.33, 16.33, −0]`. The `argmax` lands on the exact-fit bin, index `1` — the right answer,
the same as Best-Fit. Now perturb so there is *no* exact fit: `[7, 5, 10]`, same item. Every bin has
`r > s`, so every score is negated: `[3, 8.33, 0] → [−3, −8.33, −0]`. The `argmax` of this
all-non-positive array is its least-negative entry — index `2`, the *fresh bin* at `10`, because its
fill is zero so its score is zero. So whenever no exact fit exists, the per-bin rule does not just
prefer a roomy partial bin — it prefers to **open a brand-new bin**.

That is a real problem, worse than I first feared. With continuous Weibull sizes exact fits essentially
never occur — I noted at rung 2 that Best-Fit there is always choosing a *near*-fit — so the per-bin
rule as it stands would open a fresh bin on essentially *every* item, degenerating toward one bin per
item, the trivial policy I began the ladder above. My consolidation argument was wishful: the sign flip
turns the fresh bin's zero fill into the winning score. I can already predict that an ablation keeping
only this term and dropping whatever comes next would blow up far past First-Fit, toward
one-bin-per-item territory. (The flip also fires on `r > s` strictly, so the one branch it leaves
positive is the exact fit `r = s` — dead on Weibull, but genuinely live on OR's integers when an item
lands on a bin's gap, one more reason the same code is a different animal on the two families.)

So I need something that re-couples the bins *after* the per-bin score — that makes a bin win only when
it is *locally* special rather than globally least-committed, and in particular stops the fresh bin's
zero from automatically winning. The cheapest coupling I have not used is the positional adjacency the
harness preserves: replace each score by how much it *exceeds the bin before it*, `score[i] ←
score[i] − score[i−1]`, a plain first difference. A bin then wins only if it is a local improvement
over its neighbour. Running the two faithful states through it: `[5, 3, 10]` gives pre-difference
`[−8.33, 16.33, −0]`, differenced `[−8.33, 24.67, −16.33]` — `argmax` still the exact-fit bin, good,
the coupling did not spoil the working case. `[7, 5, 10]` gives `[−3, −8.33, −0]`, differenced
`[−3, −5.33, 8.33]` — `argmax` still index `2`, *still the fresh bin*. On every no-exact-fit slice I
can trace by hand, including a larger `[9, 7, 5, 10, 10]`, the differenced rule opens a new bin too.

But the differencing changes something the small traces hide, and it is exactly the property I argued
was necessary. With the plain per-bin score, the pick on a no-exact-fit slice is the fresh bin no
matter how I order the partials. With differencing, the pick depends on the *order*. Take the multiset
`{5, 7, 10}`, item `3`: in the order `[7, 5, 10]` the rule picks the fresh bin, but in `[5, 7, 10]` it
picks the *partial* bin at `7` — a genuine reuse, not an exact fit and not a new bin. Same three
capacities, two arrangements, two destinations, and in one the rule does exactly the reuse I want. So
the differencing genuinely couples each bin to its neighbour: no score is decided in isolation any
more, even the fresh bin's automatic zero can be beaten, and the positional layout now matters. That
is the escape from the monotone collapse that pinned every single-bin rule to Best-Fit. What I cannot
read off from three- and five-bin toys is what this coupling does on a real array of hundreds of open
bins with continuous sizes, where the fresh tail is long. The differenced ranking is a global,
order-dependent object; the toys tell me the coupling exists and that it can land on reuse *or* a new
bin depending on structure I cannot enumerate. Whether, aggregated over 5000 items, it consolidates or
scatters is genuinely beyond hand computation, and given how badly the toys behave I am not confident.

Before I commit I want the differencing to be the right *kind* of coupling, since the coupling is the
whole novelty. Subtracting the array *mean* shifts every entry by the same constant and leaves every
`argmax` untouched. Coupling by *rank* is just Best-Fit's ordering renamed and collapses the same way.
Dividing each entry by its neighbour detonates near the fresh bins' zeros. The first difference is the
cheapest operation left that is genuinely non-monotone and local: it subtracts a *different* number
from each entry — its own predecessor — so it can reorder the array (unlike the shift) while staying
cheap and not collapsing to a single-bin ordering (unlike the rank). That it can reorder is what I
need; that I cannot predict how is the price I choose to measure.

So the rule I test is: among valid bins, score each `(r − r_max)² / s`, flip the sign on the
strictly-fitting bins for feasibility, then difference the array so each bin is judged against its
neighbour. It reads the whole array — the emptiest-bin landmark, the item weighting, and the
differencing all make a bin's fate depend on context — which is the one thing I proved is necessary to
get past Best-Fit. I am shipping it despite the pathological hand-trace, and clear-eyed about why:
retreating to Best-Fit is pointless, because I have *proven* its ceiling and the entire remaining `4%`
lives behind exactly this relational door, so the honest move is to test a coherent relational form and
let the measurement adjudicate.

Two predictions hold the run to account. First, the per-bin term without the differencing opens a fresh
bin whenever there is no exact fit, which on continuous sizes is nearly every item — so ablating the
differencing should make the bin count *explode* toward one bin per item, far past First-Fit; the
differencing, whatever its aggregate sign, is load-bearing, not a garnish. Second, if the *differenced*
rule actually consolidates, its excess should drop below Best-Fit's `~4%` on at least Weibull, where
fine-grained fillers give the coupling the most to work with; on the chunkier OR-style family I would
not be surprised if it helps less, since these hand-picked powers may sit well in one regime and not
the other. If instead it scatters, the excess climbs toward the ablation's and I learn the coupling I
reached for is wrong.

What this rung deliberately leaves on the table is the *form* of the score: I chose `(r − r_max)² / s`
and a plain neighbour difference by hand, and the trace already shows they are not obviously right.
Holding the functional family fixed — distance below the emptiest bin, an item weighting, a coupling
difference — and *searching* over the powers and the weighting, letting measured bin counts pick the
member that consolidates rather than scatters, is what the final rung is for.
