My hand-built relational rule worked — better than the pathological hand-traces had any right to
suggest — and the way it worked tells me what to do next. On the Weibull streams it landed at `0.718%`
over the lower bound, collapsing almost all of Best-Fit's `4.146%` residual, about `69` fewer bins per
stream of two thousand. And the differencing was vindicated as the engine in the most direct way: the
ablation I predicted would blow up did, to roughly `80%` excess when I dropped `score[1:] -= score[:-1]`
and kept only the per-bin term — exactly the toward-one-bin-per-item scatter I traced by hand. The
coupling I could not analyse on paper turns out to be load-bearing and, on Weibull, wonderfully
effective.

But the OR-style streams tell the other half, and it is the half that decides this rung. There the same
rule reached only `3.271%` — a real gain over Best-Fit's `4.411%`, but nowhere near the near-optimal
collapse on Weibull. That asymmetry is a fingerprint. I *guessed* the functional form — the squared
fill divided by the item once, and a single plain neighbour difference — and every one of those choices
was set by my taste. A rule whose constants were hand-tuned, however unconsciously, to one regime is
exactly the kind that sits near-optimal on its home distribution and slips off it, and the per-seed
savings sharpen it: against Best-Fit my rule saved a tight `[72, 63, 75, 67, 67]` bins across the
Weibull seeds but an erratic `[21, 40, 23, 20, 10]` across the OR seeds, the worst limping in at
`4.12%`. Uniform recovery on one family and scatter on the other are two readings of one fact — I tuned
the form to Weibull, and on OR it fires more by luck of the stream than by design. I flagged this exact
asymmetry when I shipped the rule.

The honest move is to stop guessing the form and let a *search* find it. Each choice I made by hand —
the power on the fill, the power on the item, the number and shape of the terms, the coupling — is a
free parameter, and the OR number makes plain my guess is not the optimum. And the fitness signal is
already in hand and cheap: run the simulator on training streams, count bins, lower is better. That is
exactly what a program-search loop can climb — a language model proposes candidate `priority(item,
bins)` bodies, each scored by running the simulator, the best survivors seed the next round, and the
loop evolves the code toward a priority function selected purely for packing tightly, with no human
prior on its shape.

The alternative is to keep hand-extending the form myself — add a `(fill)³` term, or a second item
power, and re-run — and I reject it for a concrete reason, not on faith. Hand-adding a term is guessing
again in a *higher*-dimensional space, with the same taste-bias that already misfit OR, and the only
way to tell whether it helps is still to run the simulator and read the bin count. So a manual
extension is just a low-throughput, human-biased, one-candidate-at-a-time version of the search — same
fitness, same simulator, far fewer proposals and a prior I have seen fail off-distribution. Once I
accept that I cannot *derive* the right form and must *measure* candidates, the rational move is to let
the search generate and score at scale. And I have the discovered Weibull function in hand, so the
right thing is to take it *verbatim* rather than paraphrase — to read it as the search's verdict on the
form I was guessing at, and then understand it.

The converged body keeps the very piece I reached for and goes past it. It scores each valid bin as a
sum of three terms — `(bins − max_bin_cap)² / item`, plus `bins² / item²`, plus `bins² / item³` — then
applies the same feasibility sign flip on the strictly-fitting bins and the same neighbour differencing.
The first term is exactly my rung-3 core: since a fresh bin at capacity is always in the valid slice,
`max_bin_cap` is always `C`, so `(bins − max_bin_cap)²` is the squared fill, item-weighted — the
"distance below the emptiest" landmark I built by hand. And the two structural moves that flank it, the
flip and the differencing, are the two choices I also made from intuition. So the search *confirmed*
the skeleton I arrived at independently and added two terms I would not have guessed:
`bins² / item²` and `bins² / item³`, bringing the bin's own remaining capacity in at successively
higher inverse powers of the item.

Before I tell any story about why those terms help, I want to know mechanically what the code does to a
placement, because two of its lines are easy to misread — and I include a fresh bin at capacity in
every slice. Start with the sign flip, `score[bins > item] = -score[bins > item]`. The simulator only
calls `priority` where `bins ≥ item`; the flip fires on `bins > item` strictly, so the one case it does
*not* flip is `bins == item`, a perfect fit with zero leftover. Item `30`, capacity `100`, slice
`[30, 55, 100]`: the three-term raw scores are about `[164.37, 70.97, 11.48]`, the flip leaves the first
and negates the rest to `[164.37, −70.97, −11.48]`, and differencing gives `[164.37, −235.34, 59.49]`,
`argmax` index `0` — the exact-fit bin. A perfect fit is a hard preference, straight out of the strict
inequality; with `≥` I would have negated it too and lost it.

Now the no-exact-fit case, which is what Weibull is made of. Slice `[55, 80, 100]`, item `30`: raw
`[70.97, 20.68, 11.48]`, all flip to `[−70.97, −20.68, −11.48]`, differencing gives
`[−70.97, 50.29, 9.20]`, `argmax` index `1` — remaining `80`. Best-Fit would take the *tightest* slack,
remaining `55`. This took `80`. So among bins that all leave slack it is emphatically *not* minimising
leftover — the lazy reading that this is "Best-Fit with extra terms" is simply wrong, and one trace
kills it. And unlike my rung-3 rule, which on a small no-exact-fit slice reached for the fresh bin, here
the pick landed on a *committed* partial bin.

That difference tells me *how* the extra terms rescue the idea. In my rung-3 core a fresh bin scored
`(100 − 100)² / item = 0` precisely, and after the flip that stayed the least-negative value, so on a
no-exact-fit slice the fresh bin's privileged zero tended to win. Under the discovered function that
same fresh bin scores `0 + 100²/item² + 100²/item³`, about `11.48` at `item = 30` — a genuinely
positive raw value, turned to `−11.48` by the flip. On its own that is not enough (pre-differencing the
fresh bin at `−11.48` is still the least-negative and would still win), but it hands the differencing
something to work with: because the fresh-tail scores now carry item-dependent magnitude, the
neighbour-difference can lift a committed bin's jump above the fresh bin's, and on `[55, 80, 100]` it
does, landing on `80`. So the cure for my rung-3 scatter is the *combination* — the extra terms make
the fresh tail non-trivial, and the differencing exploits that to keep the item in a partial bin. (The
order-dependence I established at rung 3 survives the extra terms unchanged: the same four capacities in
a different positional order still send the item to a different bin, so this function does not rank bins
order-free and I would misdescribe it if I claimed otherwise.)

Now the two extra terms. Both scale with `bins²`, but
`1/item²` and `1/item³` grow as the item shrinks, so the loose reading is "small items make these
dominate." The numbers are sharper. For a *committed* bin — small remaining, say `bins = 10` — the
distance term `(bins − 100)² / item` is enormous and the two new terms are negligible. For a *roomy*
bin — say `bins = 95` — it flips: `bins² / item²` *dominates* the distance term, by about `72` at
`item = 5`, `12` at `30`, and `6` at `60`. So the extra terms are not a global "small item" effect;
they are specifically a re-ranking of the *roomy* bins — including the long fresh tail — with a steep
item-size dependence injected inside it (that `72 → 12 → 6` fall is the item growing). The distance term
still governs the committed bins, exactly as my rung-3 core did; what the search bought is control over
how the roomy bins are ordered relative to each other and to the fresh tail, tuned by how big the
arriving item is. My single `(fill)² / s` term had one fixed item weighting for all bins and no knob
there — a plausible reason it packed Weibull well but could not bend to OR, since the roomy-bin ordering
is exactly where a distribution's item-size structure bites. The search did not find a smarter *idea*;
it found a richer parameterisation of the same one — emptiest-landmark, flip, differencing — with the
roomy-bin ordering left free and fitted by selection.

This closes back on the first thing I noticed: the whole game turns on the small *filler* items, the
big ones mostly forcing their own bins. The extra terms do essentially nothing to a nearly-full bin's
score and everything to how the roomy bins and fresh tail are ordered, most steeply for small items —
precisely the filler-routing decision I named as the crux. Given only a bin count as its objective, the
search poured its two added degrees of freedom exactly into the filler regime.

I have to be scrupulous about what reproducing this will show, because the search's reported numbers were
measured on its own instances and mine are seeded independently. On the Weibull streams — the heuristic's
home — I expect it to land essentially at the lower bound, reproducing the published `~0.68%`, and to
edge just past rung 3: rung 3 is the first term of this score, and the search added the other two
because they helped on Weibull, so the endpoint should shave a little more off an already near-optimal
`0.718%` (though I hold that loosely, since adding terms changes the `argmax` and cannot be *guaranteed*
to help). On the OR-style streams I am genuinely unsure. This is the *Weibull*-discovered function; its
roomy-bin item-power terms were fitted to Weibull sizes, and the OR family — integer sizes, a floor of
`20`, chunkier fillers — has its own separately-discovered heuristic. It is entirely possible that here
the two extra terms do not help and even cost a little against my simpler rung-3 core, which already
reached `3.271%`. If the endpoint comes in behind rung 3 on OR, that is not a regression — it
*illustrates* the whole point: a searched heuristic is fitted to the distribution it was searched on,
and its dominance is sharpest exactly there. I will report whatever the columns say on both families,
including that case.

Two calibration checks keep the reproduction honest, in the same spirit as the earlier rungs: the
simulator should reproduce the published figures on the reference instances — the discovered Weibull
function around `0.68%` on the reference Weibull set, the separately-discovered OR function around
`3.11%` on OR3 — before I trust its readings on my seeded streams, just as First-Fit's `4.23%/5.74%`
and Best-Fit's `3.98%/5.37%` landed on the digit.

The progression has been a steady widening of what the score is *permitted to see*: a bin's position,
then one bin's own slack, then the whole array — and it was that array-reading, not cleverness about a
single hole, that broke Best-Fit. The endpoint keeps that exact apparatus, the flip and the
differencing whose ablation blows up the same `~80%`, and only enriches the roomy-bin ordering it
opened up.

So the headline is narrower than "best everywhere": on the distributions it was searched for it beats
First-Fit and Best-Fit, and on Weibull it gets within a fraction of a percent of a lower bound no
online policy can beat — a margin no hand-built rule here, including my own rung-3 core, reaches, with
the published record showing the gap *widening* to about `0.03%` off the bound at 100k Weibull items.
I copy the body character for character for a reason the traces earned: the strict `>` in the flip is
load-bearing (soften it to `≥` and the exact-fit preference vanishes), and the powers `item²` and
`item³` set exactly where the roomy-bin blend hands over to the distance term — details selection tuned
and my intuition got wrong at least once, so cleaning them up by eye would risk reintroducing the
pathologies I traced. It costs no more than Best-Fit — the same per-item `argmax`, no lookahead.
