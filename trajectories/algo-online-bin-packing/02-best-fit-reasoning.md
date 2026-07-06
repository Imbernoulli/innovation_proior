First-Fit left a clear diagnosis on the table, and now I have the numbers to read it against. The rung-1
run put greedy reuse at `4.545%` over the lower bound on the Weibull family and `4.651%` on the OR-style
family — a few percent, exactly the band I expected, and a calibration check I asked for came back clean:
on the reference instances the simulator reproduced First-Fit's known figures (`4.23%` on the Weibull
5k set, `5.74%` on OR3) to the digit, so the harness is faithful and I can trust its readings on my own
seeded streams. Before I change anything I want to turn that `4.545%` into a picture of *what* is being
wasted, because the size of the fix depends on it. The Weibull run averaged `2097.8` bins against a
lower bound of `2006.6`, i.e. `91.2` extra bins, which is `91.2 × 100 = 9{,}120` units of dead space
spread across the packing — about `4.5%` of the `200{,}660` units of capacity actually in play. That
`9{,}120` lines up almost exactly with the structural story I told at rung 1: a couple of hundred
two-item bins each carrying roughly `20` units of un-topped-off leftover would be `~9{,}000` units of
waste, and here it is, measured. So the excess is not exotic; it is precisely the un-closed two-item
bins the filler analysis predicted.

The per-seed spread is worth a look too, because it tells me how much of the number is signal. On
Weibull the five seeds run `[4.51, 4.29, 5.08, 4.50, 4.35]%` — a band a little under a point wide, with
one seed (the `2109`-bin one) sitting high; on OR-style they run `[4.64, 4.69, 4.42, 4.69, 4.81]%`,
tighter still. That is the sampling variability of a 5000-item draw showing through a deterministic
rule, and it is small enough that a fix worth half a point will be visible above it on every seed rather
than drowned in noise. Good — the measurement is precise enough to adjudicate the change I am about to
make.

The two families are close but not identical, and the small gap is informative. First-Fit sits at
`4.545%` on Weibull and `4.651%` on OR-style, so OR-style is marginally the *harder* one for greedy
reuse. In absolute terms the OR run averaged `2092.4` bins against a lower bound of `1999.4`, about `93`
extra bins, and at capacity `150` that is roughly `13{,}950` units of dead space — a larger absolute
waste than Weibull's `9{,}120`, mostly because each OR bin is half again as big. That OR is harder
tracks the family-structure reading from rung 1: OR-style has a floor of `20` and chunky sizes with no
tiny fillers, so a bin that ends up `~30` short is stranded unless an item in the narrow `[20, 30]` band
turns up, whereas Weibull's fine-grained tail almost always has *something* small enough to top a bin
off. Fewer usable fillers means more stranded two-item bins means more excess, and there it is in the
numbers. I will keep expecting the OR family to be the stingier of the two.

One more thing in the per-seed Weibull figures is worth reading, because it tells me what the spread is
made of. The highest-excess seed is the one at `2109` bins, `5.08%`, and its lower bound is `2007` —
squarely mid-range, not the largest. The seed with the *largest* mass, lower bound `2016`, sits at only
`4.51%`. So the seed that packs worst is not the seed with the most items to pack; the excess varies
with how awkwardly a particular stream's sizes happen to interleave, not with its total mass. That is
reassuring, because it means the `~0.8`-point spread is packing difficulty, which a better rule can
attack, rather than an artefact of unequal problem sizes, which it could not. When I compare Best-Fit to
First-Fit on these same five seeds, the mass cancels seed by seed and I read a clean difference of rule.

It is worth stating plainly how much is at stake once I accept the single-bin ceiling. I have shown, by
a fact about `argmax` rather than a hope, that nothing in the single-bin family beats Best-Fit — First-
Fit, Best-Fit, Worst-Fit and every monotone reshaping of slack are the whole family, and Best-Fit is its
best member. So the residual Best-Fit will leave — on the order of `4%` above the bound — is not a little
polish left to do; it is the *entire* remaining prize, and every unit of it is reachable only through a
score that reads the array jointly. That is a large target and a precise instruction at once: whatever I
build next has to make a bin's score depend on the other bins. I do not yet know how to build a good one
— the relational design space is wide open and most points in it will be worse than Best-Fit, not better
— but I know that is the only direction with anything left to win, and I know the exact information Best-
Fit throws away that a relational rule would have to use: everything about the slack array beyond each
entry's own value.

This is also the moment the ladder's shape becomes clear to me. Best-Fit is the classical strong online
baseline — fast, deterministic, no lookahead, the rule any practitioner would reach for — and the honest
milestone is not to tweak it but to *clear* it. Having proven its single-bin ceiling with an `argmax`
argument rather than a hope, I know that clearing it is not a matter of a better hole-shaping trick; it
is a categorical change in what the score reads, from one bin to the whole array. Every rung from here
is therefore committed to relational scoring, and the only open questions are which relational signal to
read and how to couple the bins — not whether to go relational at all. That the single-bin family is
closed is the most useful thing the last two rungs bought me: it turns a vague "do better than Best-Fit"
into a precise search direction. A negative result — "no reshaping of one bin's slack can help" — is
doing more work here than a positive one would, because it rules out an entire wing of the design space
in one stroke and stops me wasting rungs polishing a rule I have proven is at its ceiling.

The fix names itself, and it is the one I flagged at rung 1. Instead of "earliest valid bin," pick the
valid bin where the item fits **most snugly**, the one that will be left with the *least* leftover slack
after I drop the item in. If a bin has remaining capacity `r` and the item has size `s`, the slack left
behind is `r − s`, and among all bins that fit (all bins with `r ≥ s`, so `r − s ≥ 0`) I want the
smallest `r − s`. That is Best-Fit, and I can hand it to the harness as the score `−(r − s)`: the bin
left with the least slack gets the least-negative, hence largest, score, and `argmax` selects it.

Let me reason about why this should be better and not just different, because "tightest fit" is an
intuition and I want the mechanism. The quantity `r − s` is the hole I am leaving in that bin, and a
small hole is good for two reinforcing reasons. First, a bin filled to within a sliver of capacity is
essentially finished — I will rarely see a future item small enough to fit a tiny hole, so that bin has
done its job and effectively retires near-full, which is exactly the dense packing the lower bound
rewards. Second, and more subtly, by sending the item to the bin that *barely* fits it I am
**preserving** the roomy bins. A bin that had `90` units of slack stays at `90`, ready to absorb a
genuinely large future item that the tight bins could never have taken. First-Fit's sin was the
opposite: it would pour a small item into that roomy bin and shrink it to `80`, half-spending a pocket
of capacity I might desperately need later for a big arrival. This is the filler lens from rung 1 turned
into a rule: I complained that First-Fit mis-routes the small filler items into roomy old bins and
strands the near-full bins that the fillers were meant to close; Best-Fit sends each filler to the
tightest fitting bin, which is *by definition* the bin the filler comes closest to completing. Best-Fit
keeps small items with tight bins and large items with roomy bins, which is the matching that wastes
the least. On the very stream that exposed First-Fit — capacity `10`, `[2, 9, 1, 8]` — I can watch it
work: the `2` opens bin 0 (`rem 8`), the `9` opens bin 1 (`rem 1`), and now the `1` arrives with both
bins valid; Best-Fit scores their post-placement slacks, `8 − 1 = 7` for bin 0 and `1 − 1 = 0` for bin
1, and takes the exact fit, bin 1, closing it to `0` and leaving bin 0 at `8`. Then the `8` fits bin 0
exactly, and the whole stream packs into two bins — the lower bound — where First-Fit needed three. The
one decision First-Fit got wrong is the one Best-Fit gets right, and it is worth one whole bin here.

I can put a number on how much the roomy-bin preservation is worth on the Weibull family, which tells me
whether the mechanism is a real channel or a rounding error. A large item can only go into a bin with at
least that much room free, so the items that genuinely *need* a preserved roomy bin are the big ones.
Weibull(45,3) puts a size above `80` with probability `exp(−(80/45)³) ≈ 0.0036`, about `18` items per
5000-item stream, and above `70` with probability `≈ 0.023`, about `116` items. So on the order of a
hundred items per stream arrive needing `30`-plus of free space, and every time First-Fit has carelessly
shrunk a roomy bin below their size, one of them is forced to open a fresh bin that a preserved roomy bin
would have absorbed. It is not a huge population — a hundred-odd items out of 5000 — but it is exactly
the population that turns into extra bins, and Best-Fit's habit of leaving roomy bins untouched until a
genuinely large item needs them is aimed squarely at it. That is the second half of the gain,
complementing the first half of closing tight bins with fillers; both are about routing the right size of
item to the right bin.

I should look at the other fit-aware rules before I commit, so I know Best-Fit is the right one and not
just the first that came to mind. The single number the harness hands me per bin is its slack `r − s`,
and a rule that reads only that number is choosing a monotone direction along it. Best-Fit takes the
*minimum* slack. The opposite choice is Worst-Fit: take the *maximum* slack, `+(r − s)`, deliberately
sending the item into the emptiest fitting bin. It is easy to see this is wrong for my objective, and
worth saying why rather than just dismissing it. Worst-Fit's whole tendency is to keep every bin
half-open and balanced — it pours items into the roomiest bin, which spreads the load and drives *no*
bin toward full. That maximises the very leftover slack I measured as the source of the `9{,}120` units
of waste; it is the anti-consolidation rule. I would expect it to score *worse* than First-Fit, because
First-Fit at least does not actively seek the emptiest bin, whereas Worst-Fit does. The same
`[2, 9, 1, 8]` stream makes the point concrete: when the `1` arrives with bin 0 at `rem 8` and bin 1 at
`rem 1`, Worst-Fit takes the *maximum* slack, bin 0 — the exact wrong move First-Fit made — leaving bin
1 stranded at `1` and forcing a third bin when the `8` comes. Worst-Fit lands on three bins here, tying
First-Fit's failure, while Best-Fit's two. So Worst-Fit is not a different idea to weigh against
Best-Fit; it is First-Fit's mistake made deliberate. There is an "almost-worst-fit" variant that leaves
the second-emptiest, tuned to keep a large bin in reserve, but it inherits the same defect for a
bin-minimisation objective. So the sensible fit-aware direction is unambiguously the tight one, and
Best-Fit is its cleanest expression.

Now I have to be honest about a limit that constrains not just this rung but the whole family of
single-bin rules, and it changes what I can expect from here. Best-Fit scores each bin purely as a
function of *that bin's own* leftover slack `r − s`. It looks at one number per bin and takes the
minimum. But notice: any *other* rule that is a monotone-decreasing function of that same single
number induces the identical ordering. If I scored `−(r − s)²`, or `exp(−(r − s))`, or `−(r − s)³`, the
`argmax` would still land on the smallest `r − s`, because a monotone function preserves the order of
its argument and `argmax` cares only about order. This is not a guess; it is a one-line fact about
composing a monotone map with `argmax`, and it means every "look at one bin's fit" heuristic collapses
onto exactly one of two orderings — Best-Fit's (minimum slack) or Worst-Fit's (maximum slack) — with
nothing in between and no reshaping able to change the choices. I can watch the collapse on a single
decision. Take valid bins with remaining `[12, 3, 7]` and an item of size `3`, so the slacks are
`[9, 0, 4]`. Best-Fit scores `−(r − s) = [−9, 0, −4]`, argmax at index 1. Square it: `−(r − s)² =
[−81, 0, −16]`, argmax still index 1. Exponentiate: `exp(−(r − s)) = [0.0001, 1, 0.018]`, argmax still
index 1. Three very different-looking score shapes, the same choice, because all three are monotone in
the slack and the exact-fit bin has the least slack. So if I wanted to confirm it across whole streams I
would run `−(r − s)²`, `exp(−(r − s))`, and a version with a tiny fill-based tie-break, and I predict
the bin counts would come back *identical* to Best-Fit, not close — identical, because the argmax
sequence is literally the same on every item of every stream. The feedback can check that prediction,
but I do not need it checked to know where the ceiling is: reshaping the per-bin score is a dead end.

That is the single most useful thing this rung teaches, because it tells the *next* rung exactly what it
must do. If simply reshaping the per-bin fit score cannot change the decision, then beating Best-Fit
requires a score that depends on the bins *jointly* — how a candidate bin compares to the others, the
shape of the whole open-bin landscape — not just its own hole. Best-Fit throws away everything about the
array of slacks except each entry's own value; the gain has to come from the information it discards.

Translating Best-Fit itself to the harness is clean, and the new-bin behaviour again takes care of
itself, which I want to verify rather than assume. A still-full, freshly-available bin has the *largest*
possible `r` (the full capacity `C`), hence the *largest* slack `r − s`, hence the *most negative* score
`−(r − s)`; so `argmax` reaches for a fresh bin only when no partially-used bin can hold the item at all,
which is exactly the "open a new bin strictly as a last resort" policy I want. Best-Fit and the correct
new-bin rule are the same single expression `−(r − s)`, with nothing special to code — the same pleasant
property First-Fit had, that the still-full tail supplies new bins without a separate branch. I can see
it in one score vector: an item of size `40` facing a used bin at `rem 45` and a fresh bin at `rem 100`
scores `−(45 − 40) = −5` and `−(100 − 40) = −60`; the `argmax` is the used bin at `−5`, and the fresh
`−60` is chosen only if it is the sole valid option. The fresh bin, having the most slack, always carries
the most negative score, so it can never win a contest against a bin that actually fits — exactly the
last-resort behaviour I want, falling straight out of the arithmetic.

I should also name Best-Fit's own failure mode before I trust it, because every greedy rule has one and
the residual it leaves is what the next rung inherits. By always filling the tightest bin, Best-Fit
closes bins off at slightly *less* than full — it leaves a scatter of small leftover holes, one per bin,
of whatever size `r − s` happened to be the minimum available at that step. If those holes are
systematically the wrong size — a hair too small for any future item — Best-Fit leaves a long tail of
bins stuck at, say, `97/100`, three dead units each, and that is real waste. It is *less* waste than
First-Fit's, because First-Fit leaves holes of essentially arbitrary and often much larger size while
Best-Fit drives each hole to the smallest it could be at the moment of placement; but it is waste all
the same, and Best-Fit cannot avoid it, because it is myopic about *which* hole sizes will turn out to
be reusable. It minimises the hole it can see now, not the hole that will still be fillable later.
Judging the *quality* of the leftover hole against the rest of the open bins — not just its size — is a
relational question, and it is exactly what a rule that reads the whole array could ask and Best-Fit
cannot.

There is a nuance here that sharpens what "tightest fit" actually does on these streams. On the Weibull
family the sizes are effectively continuous, so an *exact* fit — `r − s = 0` — essentially never occurs;
Best-Fit is really choosing the smallest *positive* slack, a near-fit, and the hole it leaves is a small
real number, not zero. That is precisely the regime that breeds the scatter of un-fillable slivers: a
bin closed at `2.7` units short is not closed at all, it is a bin waiting for an item it will almost
never see. On the OR family, integer sizes with a floor of `20` do allow genuine exact fits, but only
when an item lands exactly on a bin's remaining gap, and with gaps concentrated below `30` and items
starting at `20` those coincidences are not common either. So on both families Best-Fit spends most of
its decisions choosing a near-fit rather than a clean close, and the residual `~4%` is the accumulated
cost of near-fits that never became closes. This is the same limitation from a second angle: Best-Fit
cannot ask whether a near-fit is *worth* taking relative to the other bins, only which near-fit is
tightest.

I should also be realistic about *how much* of First-Fit's `9{,}120` units of Weibull waste Best-Fit can
actually recover, so I do not over-promise. Best-Fit only differs from First-Fit on decisions where the
tightest fitting bin is not the earliest one, and of those, only the ones where the tighter choice
eventually *closes a bin that would otherwise have stranded* turn into a saved bin. The rung-1 trace
already showed earliest and tightest do diverge, but not on every item — plenty of decisions have a
single valid used bin, or a fresh bin as the only option, and on those the two rules agree exactly. So
Best-Fit reworks only a slice of the placements, and only a slice of that slice pays off in a bin. That
is why I expect a fraction of a percent rather than a wholesale collapse of the `4%`: the mechanism is
right and points the same direction on every relevant decision, but the relevant decisions are a
minority. And crucially the rule costs nothing extra — `−(bins − item)` is the same per-item `argmax`
over the same valid slice as First-Fit, fully deterministic, no lookahead — so whatever margin it buys
is free, which is the right shape for a baseline improvement.

So my expectation for the run is concrete. Best-Fit should beat First-Fit on every seed of both
families, because closing bins off at the tightest available fit strictly dominates closing them off
wherever happens to be first — but not by a huge margin, because both rules do greedy reuse, which is
most of the battle, and the difference between them is only the disposition of the fillers among the
already-open bins. I would guess a fraction of a percent, not a wholesale collapse of the `4%`, and I
expect it slightly larger on Weibull than on OR-style, since Weibull's fine-grained fillers give
tightest-fit more exact bins to find while OR-style's chunky `[20, 100]` items with their floor of `20`
offer fewer snug matches. In the metric columns that means `mean_bins` a touch below First-Fit's on
both families and `excess_over_LB` down by something like a few tenths of a point, consistent across all
five seeds. The exact size is what the feedback will tell me. What it will *not* tell me is any way past
Best-Fit within the single-bin family — I have shown there is none — so the residual `~4%`, the dead
space in those tightest-fit holes, is the target the next rung has to attack, and it can only do so with
a score that compares a bin against the others rather than reshaping the bin's own slack. That is where
I go next.
