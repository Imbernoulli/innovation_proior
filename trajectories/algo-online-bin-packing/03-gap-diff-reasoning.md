Best-Fit left me with a sharp constraint, and the rung-2 run confirmed it in the least ambiguous way
possible, so I want to take it seriously because it tells me exactly what *cannot* work. Best-Fit
scores each bin as a function of that bin's own leftover slack `r − s`, and the side-experiment I ran
came back exactly as the `argmax` algebra predicted: replacing `−(r − s)` with `−(r − s)²`, with
`exp(−(r − s))`, or with a version carrying a tiny fill-based tie-break left the bin count *identical*,
not close — the same integer on every seed of both families, because the `argmax` sequence is literally
the same on every item. So any rule that is a monotone reshaping of a single bin's slack makes the
identical choices, and I am not going to beat Best-Fit by being cleverer about a single hole. Whatever
I do has to make a bin's score depend on the *other* bins — on where this bin sits in the landscape of
all the open bins — so that the same leftover slack can be ranked differently depending on context.
That is the only door left open, and the experiment shut every other one.

It is worth reading the rung-2 numbers a moment longer, because they size the prize. Best-Fit landed at
`4.146%` on Weibull and `4.411%` on OR-style, beating First-Fit on every seed — the per-seed Weibull
saving was `[7, 7, 12, 6, 8]` bins, mean `8`, and the OR-style saving `[11, 2, 1, 6, 4]`, mean under
`5`. Two things in that. First, the win was real but modest, exactly the fraction-of-a-percent I
expected, because both rules do greedy reuse and differ only in the disposition of fillers among open
bins. Second, OR-style gained less than Weibull — one seed improved by a single bin — which matches the
rung-2 reading that OR's chunky `[20, 100]` sizes with their floor of `20` give tightest-fit fewer snug
matches to find. And the residual is the whole point: Best-Fit still sits about `4%` above the bound,
per-seed Weibull excess around `[4.17, 3.94, 4.48, 4.20, 3.95]%`, and I have *proven* none of that `4%`
is reachable by any single-bin rule. So the entire remaining prize — a couple of thousand bins' worth of
dead space — is behind the relational door, and nowhere else.

Let me think about what context could possibly matter, because the relational design space is wide and
most of it will be worse than Best-Fit, not better. The harness hands me the remaining capacities of
all the valid bins at once, as an array, in a stable positional order. Best-Fit throws away everything
about that array except each entry's own value. What is the cheapest, most natural piece of
*relational* information I am currently ignoring? The most salient landmark in the array is the
**emptiest** valid bin — the one with the largest remaining capacity. A still-full, freshly-opened bin
is the extreme case of this: it sits at the top, at capacity. Every other bin's remaining capacity can
be read as a *distance below* that emptiest bin, `r − r_max`, a non-positive number that is `0` for the
emptiest bin and grows more negative the more a bin has already been filled. That distance is a
genuinely relational quantity — it depends on the whole array, not just one entry — and it has an
intuitive meaning: how much *more committed* this bin already is than the freshest one. A bin far below
the emptiest is a bin I have already invested in.

There is a simplification here I should pin down before I go further, because it changes what `r_max`
concretely *is*. The harness pre-allocates the bins array full of still-full bins at capacity `C`, and
`get_valid_bin_indices` returns every bin whose remaining is at least the item size — so a fresh bin at
capacity `C` is *always* in the valid slice, since any item fits an empty bin. That means `r_max`, the
largest remaining capacity among the valid bins, is *always* exactly `C`. So `(r − r_max)² = (r − C)²`
is really the square of how much the bin has *already been filled*: "distance below the emptiest valid
bin" and "amount already used" are the same quantity here, and the relational term is, concretely, the
squared fill of a bin divided by the item. That is cleaner than I first put it — the score rewards
already-filled bins and hands a fresh bin exactly zero. It also flags something I will have to watch: a
fresh bin sitting at score `0` is a very particular value, and how the rest of the construction treats
that zero will decide whether the rule ever *prefers* opening a new bin, which would be fatal.

Why would I want to prefer the bins that are far below the emptiest? Because those are the bins I am
trying to *finish*. Best-Fit's residual waste was the long tail of bins stuck a few units short of full.
If I can bias the policy toward pouring items into the already-committed bins — driving them the rest of
the way to full — rather than touching the freshest ones, I consolidate the committed bins faster and
keep the fresh capacity in reserve. So my relational signal should *reward* a large distance below the
emptiest. The square of that distance, `(r − r_max)²`, is a clean way to write it: it is `0` at the
emptiest bin and grows quickly for the committed ones, so it sharply separates "fresh" from "committed."

But raw distance is not the whole story, because the item size has to enter. The same `r − r_max` means
something very different for a tiny item than for a large one: a large item poured into a committed bin
makes a big difference to closing it, while a tiny item barely advances it. So I want to weight the
distance by how *consequential* the item is relative to the bin, and the simplest dimensionally-natural
way is to divide by the item size: `(r − r_max)² / s`. Now small items — the ones Best-Fit is happy to
scatter — get a *gentler* relational pull, and the policy's relational preference kicks in most
strongly exactly where it should. I still have to respect feasibility — among bins that actually fit
the item I do not want to send it somewhere that overflows — so I keep the sign convention that valid
bins (those with `r ≥ s`) are the ones in play and I negate the score on them. The intent is that,
after the relational term, a tighter genuinely-fitting bin still comes out ahead.

The square and the single division by the item are the two places I am most nakedly guessing, and I
should own that rather than dress it up. Why square the fill rather than take it linearly? Because I want
the term to *curve*: I want the jump in score between two heavily-committed bins to differ in size from
the jump between two barely-touched ones, so that when I later difference neighbours the jumps carry
information about *where* in the fill range a bin sits. A linear fill would make those jumps flat and
featureless; the square is the lowest power that bends, so it is my minimal guess at "give the coupling
something to bite on." Why divide by the item exactly once? Dimensionally the fill `(r − C)` is a
capacity and so is the item, so `(r − C)² / s` carries units of capacity, which is at least tidy — but
"tidy" is taste, not derivation. A different power on the item, or a second item term with a different
power, would weight small and large arrivals very differently, and I have no principled reason to prefer
one division over two, or the square over the cube. These are precisely the knobs I am flagging: I chose
the lowest-curvature power and the most dimensionally-innocent item weight and called it a starting
point, not an optimum.

This is fiddly, and I am not at all sure I have the sign bookkeeping right, so before I trust any of it
I want to push a concrete state through the rule by hand and watch which bin actually wins — and I have
to include a fresh bin at capacity in the slice, because I just argued one is always there. Take an item
of size `s = 3`, capacity `C = 10`, and a valid slice `[5, 3, 10]`: two partial bins and a fresh one.
The relational term `(r − C)² / s` is `[25, 49, 0] / 3 = [8.33, 16.33, 0]`. The feasibility flip negates
the score on the strictly-fitting bins `r > s`; here `[5 > 3, 3 > 3, 10 > 3] = [T, F, T]`, so the two
non-exact bins flip and the exact-fit bin (`r = s = 3`) keeps its positive sign, giving
`[−8.33, 16.33, −0]`. The `argmax` lands on the exact-fit bin, index `1` — the right answer, the same one
Best-Fit would pick. Good so far. Now perturb it so there is *no* exact fit: `[7, 5, 10]`, same item `3`.
Every bin satisfies `r > s`, so every score gets negated: `(7−10)²/3, (5−10)²/3, (10−10)²/3 =
[3, 8.33, 0]`, negated to `[−3, −8.33, −0]`. The `argmax` of this all-non-positive array is its
least-negative entry — and that is index `2`, the *fresh bin* at `10`, because the fresh bin's fill is
zero so its score is zero, the least negative of all. So the per-bin rule, whenever no exact fit exists,
does not just prefer a roomy partial bin — it prefers to **open a brand-new bin**, since the freshest
valid bin is always a still-full one at score `0`.

That is a real problem, and a worse one than I first feared, so I need to sit with it because it
undermines the story I told. With continuous Weibull sizes exact fits essentially never occur — I noted
at rung 2 that Best-Fit is really always choosing a *near*-fit, never a clean `r − s = 0` close — so the
per-bin rule as it stands would open a fresh bin on essentially *every* item, degenerating toward one
bin per item, which is the trivial policy I started the whole ladder above. My consolidation argument was
wishful: I reasoned toward preferring committed bins, but the sign flip turns the fresh bin's zero fill
into the winning score, and the numbers I just pushed through say it would scatter about as badly as
anything can. A per-bin score, even this relational one, is not just insufficient — on its own it is
actively catastrophic, and I can already predict that an ablation which keeps only this term and drops
whatever comes next would blow up far past First-Fit, toward one-bin-per-item territory.

There is a subtlety in the feasibility flip worth naming, because it means the two families will
exercise different branches of the very same code. The flip fires on `r > s` strictly, so the *one*
configuration it leaves positive is the exact fit `r = s`. On the Weibull family sizes are effectively
continuous, so `r = s` is a measure-zero coincidence that essentially never happens — the exact-fit
branch is dead, and every decision is governed by the no-exact-fit branch, the one the toys say opens a
fresh bin. On the OR family sizes are integers with a floor of `20`, so `r = s` genuinely does occur
when an item lands exactly on a bin's integer gap, and the exact-fit branch fires a real fraction of the
time, pulling some placements toward clean closes. So before any differencing, the same rule is a
different animal on the two families: near-continuous-and-scattering on Weibull, occasionally-exact on
OR. That is one more reason I should read the two `excess_over_LB` columns as separate experiments and
one more source of the asymmetry I keep predicting — the code's own branch structure is distribution-
sensitive, not just its tuned constants.

So I need something that re-couples the bins *after* the per-bin score, something that makes a bin win
only when it is *locally* special rather than globally least-committed — and in particular something that
stops the fresh bin's zero from automatically winning. The cheapest array-level coupling I have not used
is the positional adjacency the harness preserves: replace each bin's score by how much it *exceeds the
bin before it*, `score[i] ← score[i] − score[i−1]`, a plain first difference along the array. Its
meaning is that a bin is judged not by its absolute relational value but by the *jump* in relational
value from its predecessor — a bin wins only if it is a local improvement over its neighbour. Let me run
the same two faithful states through the differenced rule and see whether it rescues the exact-fit case
without re-breaking the no-exact-fit case. For `[5, 3, 10]`, `s = 3`: the pre-difference vector is
`[−8.33, 16.33, −0]`, and differencing gives `[−8.33, 16.33 − (−8.33), −0 − 16.33] = [−8.33, 24.67,
−16.33]` — `argmax` still the exact-fit bin at index `1`, good, the coupling did not spoil the case that
already worked. For `[7, 5, 10]`: pre-difference `[−3, −8.33, −0]`, differenced `[−3, −8.33 − (−3),
−0 − (−8.33)] = [−3, −5.33, 8.33]` — `argmax` is index `2`, *still the fresh bin*. So on this tiny
faithful array the differencing does *not*, by itself, fix the open-a-new-bin pathology; it preserves
the right answer when an exact fit exists and leaves the fresh bin winning when one does not. I even
pushed a slightly larger no-exact-fit slice `[9, 7, 5, 10, 10]` through it and it again landed on the
fresh bin. On everything I can trace by hand, the differenced rule looks like it opens new bins.

But the differencing changes something the small traces hide, and it is exactly the property I argued
was necessary — and it is enough to stop me writing the rule off. With the plain per-bin score, the pick
on a no-exact-fit slice is the fresh bin no matter how I order the partials. With the differencing, the
pick depends on the *order*. Take the same multiset `{5, 7, 10}` with item `3`: in the order `[7, 5, 10]`
the differenced rule picks the fresh bin at `10`, but in the order `[5, 7, 10]` it picks the *partial*
bin at `7` — a genuine reuse of a slack bin, not an exact fit and not a new bin. Same three capacities,
two arrangements, two different destinations, and in one of them the rule does exactly the reuse I want.
So the differencing genuinely couples each bin to its neighbour: no bin's score is decided in isolation
anymore, even the fresh bin's automatic zero can be beaten, and the positional layout of the array now
matters. That is the escape from the monotone-collapse that pinned every single-bin rule to Best-Fit,
and the rule provably has it. What I cannot do is read off, from three- and five-bin toys, what this
coupling does on a real array of *hundreds* of open bins with continuous sizes, where the fresh tail is
long and the partials are many. The differenced ranking is a global, order-dependent object; the few
states I can trace by hand tell me two things and no more — that the coupling exists, and that it can
land on reuse *or* on a new bin depending on structure I cannot enumerate. Whether, aggregated over
5000 items, it consolidates or scatters is genuinely beyond what I can compute by hand. My honest
expectation is that it could go either way, and given how badly the toys behave I am not confident; the
only way to know is to run it.

Before I commit I want to be sure the differencing is the right *kind* of coupling and not just the
first that came to mind, because the coupling is the whole novelty. A few alternatives are worth ruling
out on paper. I could subtract the array *mean* from every entry — but that shifts all scores by the
same constant, and a constant shift leaves every `argmax` untouched, so it would do literally nothing.
I could couple by *rank*, scoring a bin by its position in the sorted slack order — but that is just
Best-Fit's ordering under a new name and collapses the same way the rung-2 experiment showed. I could
divide each entry by its neighbour rather than subtract — but ratios detonate near the fresh bins'
zeros and are numerically treacherous. The first difference is the cheapest operation left that is
genuinely *non-monotone and local*: it subtracts a *different* number from each entry — that entry's own
predecessor — so it can reorder the array (unlike the constant shift), it looks only one step back so it
stays cheap, and it does not collapse to a single-bin ordering (unlike the rank). That it can reorder is
exactly what I need; that I cannot predict how it reorders on a long array is the price I am choosing to
pay and then measure.

So the rule I will test is: among valid bins, score each one `(r − r_max)² / s`, flip the sign on the
strictly-fitting bins so feasibility is respected, then difference the array so each bin is judged
against its neighbour. It reads the whole array — the emptiest-bin landmark, the item-relative
weighting, and the neighbour differencing all make a bin's fate depend on context — which is the one
thing I proved is necessary to get past Best-Fit. I am shipping it despite the pathological hand-trace,
and I want to be clear-eyed about why: retreating to Best-Fit is pointless, because I have *proven*
Best-Fit's ceiling and the entire remaining `4%` lives behind exactly this relational door, so the
honest move is to test a coherent relational form and let the measurement adjudicate, not to stay on a
rule I know is exhausted.

That gives me two falsifiable predictions to hold the run to. First, the per-bin term without the
differencing opens a fresh bin whenever there is no exact fit — I traced that, and on continuous sizes
that is almost every item — so if I ablate the differencing I predict the bin count *explodes* toward
one bin per item, far past First-Fit, a scatter so bad it dwarfs everything on the ladder; the
differencing, whatever its aggregate sign, is not a garnish but the load-bearing operation, and the
ablation should show that violently. Second, if the *differenced* rule actually consolidates rather than
scatters, its excess should drop below Best-Fit's `~4%` on at least the Weibull family, where
fine-grained fillers give the coupling the most to work with; on the chunkier OR-style family I would
not be surprised if it helps less, since the same hand-picked powers may sit well in one regime and not
the other. If instead the differenced rule scatters too, the excess will climb toward the ablation's and
I will have learned the coupling I reached for is the wrong one. The `excess_over_LB` columns on the two
families will tell me which, and after the hand-traces I have seen I genuinely do not know in advance.

What this rung deliberately leaves on the table is the *form* of the relational score. I chose
`(r − r_max)² / s` and a plain neighbour difference by hand, from intuition, and the hand-trace has
already shown me they are not obviously right — the sign convention prefers the freshest bin off exact
fits, and only the still-mysterious differencing might rescue it. I have no reason to believe these
specific powers and this specific coupling are optimal, and the asymmetry I am predicting between the
two families would be the fingerprint of a form implicitly tuned to one regime. Holding this functional
family fixed — distance below the emptiest bin, an item weighting, an array-coupling difference — and
*searching* over the powers, the weighting, and the sign convention, letting measured bin counts pick
the member of the family that actually consolidates instead of scatters, is what the final rung is for.
I have derived the form and found, by tracing it, where it looks broken; what I cannot do by hand is
find the member of the family that wins, and that is the job I hand to the search.

I want to be explicit about the methodological stance this rung forces on me, because it is the opposite
of how the earlier rungs went. For First-Fit and Best-Fit I could trace a small example and *know* the
rule's behaviour — the toys were faithful miniatures of the full run. Here they are not: the differencing
is a global operation over an array of hundreds of bins, and my three- and five-bin traces are not
scaled-down versions of it but genuinely different objects, which is why they can say "opens a new bin"
while the aggregate might do the reverse. When a derivation reaches a construction whose behaviour does
not survive shrinking to a traceable size, the honest response is to defer to measurement, not to
intuition — especially here, where my intuition already misfired once, arguing confidently for
consolidation right before the hand-trace showed the per-bin term scattering. So I am shipping a rule I
cannot fully predict, with predictions sharp enough that the feedback can prove me wrong, and I am
holding my expectation genuinely open rather than talking myself into optimism I have not earned.
