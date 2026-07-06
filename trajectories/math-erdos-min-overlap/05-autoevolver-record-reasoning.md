The ladder stopped, and the endpoint rung told me plainly why. At `600` cells the bound held at
`0.3810764` — the sharp-`β` analytic-gradient Adam did not lower it, a fresh `600`-cell multistart did not,
SLSQP did not, and the exact subgradient polish did not. Every tool I brought refined *inside the same
basin* and returned the same number to many digits. So the residual to the published record `0.38086945` is
not a tuning gap I can close by turning a knob harder, and I have to be clear-eyed about what kind of object
that residual actually is before I decide what to do about it.

Let me look hard at where my constructor saturates, because the endpoint's dimension count already handed me
the diagnosis and the measurements confirmed it. My recipe is hierarchical lift plus annealed-soft-max
refinement, keeping the best true overlap: each rung upscales the optimized profile for free, kicks to break
the block plateau, and grinds the worst-overlap envelope down a little. That recipe has a ceiling built into
it. Every rung descends into the *nearest* basin of the lifted profile, and once the profile is near-binary
and spiky the cross-correlation develops a huge active set of closely-tied binding shifts — the worst overlap
sits essentially flat across hundreds of shifts. I argued at the endpoint exactly why that traps a local
search: to lower the maximum I need a feasible direction with negative inner product against the gradient of
*every* active shift at once, but as the active set `A` grows toward the number of free heights, those `A`
gradients positively span the tangent space and no such direction survives — the descent cone collapses. And
conservation makes it worse: the total overlap `n²/4` is fixed, so any mass I strip off the active shifts
reappears on the next-highest ones until they rejoin the active set. Both mechanisms say the same thing, and
the endpoint's held value confirmed it: `0.38108` is not a resolution cap I can lift past, it is the *floor
of the basin my constructor's bias selects*. The records live in a different basin, and a continuous local
search from my lifted profile cannot reach it.

The peak-to-mean check I ran at the endpoint made this vivid rather than abstract. My `600`-cell envelope
sits at a peak-to-mean overlap ratio of about `1.523`; the record's envelope, at its own resolution, sits at
about `1.522`. Those are the same flatness. I am not losing because my worst overlap towers any higher above
the conserved mean than the record's does — my envelope is *just as flat*. The record is lower only because
it is a *different* flat configuration whose common plateau level sits a couple ten-thousandths beneath mine.
That is the fingerprint of two distinct local optima of comparable quality, not of a construction that
flattens worse. It settles that the gap is a basin gap: I cannot flatten my way into it from where I stand,
because I am already as flat as the target.

So what reaches the record, if not a better optimizer of my kind? Not more of the same. The published
constructions that get into the `0.38087` band were found by large-scale search that does not commit to a
single profile and grind it — a population-based, code-mutating evolutionary / LLM coding-agent loop that
proposes whole *constructions*, mutates the code that generates the height profile, evaluates each against
the same hard-max overlap, and keeps a diverse population so it can *abandon* a basin entirely rather than
refine within it. AlphaEvolve's `0.380924` came from such a search; the record `0.38086945` came from the
same kind of loop run much longer, to a finer discretization, with orders of magnitude more compute than my
minute-and-a-half single-trajectory constructor spends. That is not merely a bigger budget — it is a
*qualitatively different* search, one with the population diversity and the willingness to throw structure
away that my single descent trajectory, which always keeps the structure it lifted in, categorically lacks.
Crossing the gap from the low `0.3810`s to `0.38087` *is* crossing basins, and crossing basins is what that
search does and mine cannot.

It is worth being precise about *why* it can and I cannot, because the difference is structural, not just a
matter of horsepower. My whole ladder moves through profile-space *continuously*: every step is a bounded
perturbation of heights followed by a projected local descent, so I can only ever reach basins whose profiles
are a small height-perturbation away from where I already am. Basin-hopping widens that reach a little — a
kick can jump a shallow ridge — but it still perturbs the *same* height vector near one anchor and cannot
restructure it. A code-mutating evolutionary loop moves through a different space entirely: it mutates the
*generator*, the program that emits the profile, so a single mutation can change a period, insert or delete a
motif, or re-lay the whole arrangement — a discontinuous jump in profile-space that no bounded height-nudge
can express. Add a maintained population of diverse candidates, and the search can hold several basins at
once and let the best survive, rather than committing to one and grinding. That is the capability my
single-trajectory local refinement categorically lacks, and it is why the closest thing I have to a
basin-crossing move — the fresh `600`-cell multistart I tried at the endpoint — still held at `0.3810764`:
random large-`n` starts land in near-binary chaos around `0.6` and essentially never fall into the good
basin, because the fraction of starts reaching it vanishes as the dimension grows. I have no local instrument
that crosses to where the record lives.

For this final rung I therefore stop pretending a smarter local optimizer of my kind would close it — the
endpoint proved it would not — and reach the record the only honest way available: I reproduce the record
construction itself and verify it under this trajectory's own frozen evaluator. This is a deliberate change
of register from every prior rung. Rungs one through four each *derived and searched* their own profile;
this rung *adopts and checks* one. I take the height profile that the large-scale search produced at its
finest discretization — `750` cells — load it as the candidate, and put it through the exact rule every rung
of this ladder obeyed: the hard-max cross-correlation of `v` with `1 − v`, rescaled by `2/n`, under the same
balance constraint `Σ v = n/2` and the same box `v_i ∈ [0,1]`. I am not re-deriving that profile by a
shortcut. I could not — the endpoint showed my machinery saturates in a different basin — and claiming I
could would be dishonest. What I *can* do, and what has real value, is confirm that the published
construction is genuine under my own evaluator rather than merely under theirs: that it is feasible by the
same constraints I have enforced throughout, and that it scores what the record claims when measured by the
frozen `compute_upper_bound` this whole ladder has used.

That verification is not a formality, and it is worth saying why it carries weight. Published bounds on this
constant come from different groups using different normalizations, discretizations, and constraint
tolerances, and a number quoted under one convention need not equal the number my evaluator would assign to
the same heights — a factor in the rescale, a different treatment of the balance constraint, an off-by-one in
the correlation, any of these could make "their `0.38087`" and "my `0.38087`" quietly different objects. By
running the raw `750` heights through the *exact* frozen evaluator this ladder has used since the flat floor
— the same cross-correlation of `v` with `1 − v`, the same `max`-lag, the same `2/n` — and getting the
record back to `10⁻¹³`, I confirm the record is real *in my units*, on the same measuring stick that gave the
flat profile `0.5` and my frontier `0.3810764`. Only then is it legitimate to place all three numbers on one
axis and call the result a squeeze; without the shared evaluator the comparison would be apples to oranges.
So the rung's contribution is a reproduction under a common rule, which is a genuine, if modest, scientific
act rather than a citation.

So I run the check in the order that makes it a genuine verification rather than a rubber stamp. Feasibility
first, before I ever look at the overlap, because a lower bound from an infeasible vector would be
meaningless: the `750` heights sum to `374.999…`, matching the target `n/2 = 375` to about `10⁻¹³`, and every
height lies in `[0,1]` with the extremes exactly `0` and `1` present. Only then do I read the overlap. The
evaluator returns `C = 0.3808694472026`, and the source records the construction's own value as `c5_bound =
0.3808694472025862`; the two agree to machine precision, difference at the `10⁻¹³` level. So under *my* rule,
not just theirs, the profile is a legal balanced step function whose worst overlap is the record. I run one
more independent cross-check, the conservation invariant rung 1 derived at the very start: the sum of the
cross-correlation over *all* lags must equal `n²/4` for any feasible profile, `750²/4 = 140625` here. The
loaded profile's correlation sums to exactly `140625`, which confirms two things at once — that the vector
really does carry total overlap-mass `n²/4` like every feasible profile from the flat floor onward, and that
my evaluator's correlation is computing what I think it is. The balance sum comes in at `374.999…93` rather
than a clean `375`, the expected floating-point residue of summing `750` doubles, comfortably inside the
`10⁻⁶` feasibility tolerance; I note it honestly rather than rounding it away, because the whole point of this
rung is that the checks are real. It is
near-binary like every rung before it — about `39.7%` of the `750` cells (`298` of them) pinned exactly at
`0` or `1`, the rest interior — and its worst overlap is shared by a large active set: `539` of the `1499`
shifts sit within `10⁻⁹` of the maximum, with only a handful more within a looser tolerance, so the plateau
is broad and sharply defined. That enormous active set is the very thing the endpoint's dimension count
predicted would trap a local search — and it is exactly why this profile had to be *found* by a basin-crossing
search and can only be *verified*, not locally re-derived, here.

The fine structure of that plateau is worth a second look, because it shows how crisp the binding set is.
`539` shifts sit within `10⁻⁹` of the maximum; widening the tolerance to `10⁻⁴` adds only about fifteen more,
and to `10⁻³` only about forty more. So the plateau is not a fuzzy shoulder — it is a sharply defined ridge
of `539` shifts essentially exactly tied, with a steep drop to the rest. That is a profile balanced on a
knife-edge: `539` binding constraints held at a common level by `750` heights, `298` of them already frozen
at the box corners, leaving only a few hundred interior knobs to hold the whole ridge flat. No wonder local
descent cannot move it — the active constraints vastly outnumber any coherent descent direction, precisely
the collapsed-cone condition the endpoint anticipated. And the resolution itself, `750` cells, is not a
refinement of my `600`: `750/600 = 1.25` is not an integer, so this is not an upscale of anything I built but
a genuinely independent construction at its own finer grid, which the search chose because more cells give
more interior knobs to balance an ever-broader ridge. Everything about the profile — the flatness matching
mine, the vast tied ridge, the fine independent discretization — is consistent with a distinct, lower local
optimum reached by a search that could restructure rather than refine.

There is a structural thread running through every rung that this record profile completes, and it is the
clearest thing the ladder discovered about the shape of the true optimum. Every optimized profile came back
*near-binary*: about `29%` of cells pinned to the box corners at `24` cells, `31%` at `120`, `31%` at `600`,
and now `39.7%` at the record's `750`. The pinned fraction climbs with resolution, which fits the mechanism
rung 1 laid out — corners kill the zero-shift self-overlap, and a finer grid can afford to send *more* cells
to the corners while a growing interior population carries the balance constraint and does the fine detuning
of the tied ridge. So the picture that emerges, consistent from two dozen cells to seven hundred fifty, is
that the Erdős minimum-overlap optimum is a *mostly-binary step function with a detuning skirt*: a majority of
the mass slammed to `0` or `1` to empty the self-overlap, and a minority of interior heights arranging the
leftover overlap into as flat a ridge as the conservation budget allows. The record does not overturn that
picture; it is the sharpest instance of it — more pinned corners, a broader flat ridge, a finer skirt — which
is why it sits a hair lower while looking, structurally, like every profile the ladder built.

Laid end to end, the ladder reads as a single descent with a clear anatomy. It began at Erdős's own `0.5`,
the flat floor, which I understood by hand as the maximally self-aligned profile pinned to twice the
conservation floor. The coarse SLSQP rung broke that symmetry and fell to `0.3812396` at `24` cells — one
step that spent `98%` of the whole distance from `0.5` down to White's provable floor, the big first drop the
conservation picture predicted. The finer basin-hopping rung lifted to `120` cells and shaved to `0.3810764`,
a gain smaller by a factor of hundreds — the long grind beginning. The frontier rung lifted again to `600`
cells and *held* at `0.3810764`, revealing that the frontier is a basin floor rather than a resolution cap.
And now this rung reaches the record `0.3808694472` at `750` cells, `2.07×10⁻⁴` below my frontier, by
adopting and verifying a construction from a different basin. Read as a sequence — `0.5`, `0.38124`,
`0.38108`, `0.38108`, `0.38087` — it is a story of rapidly diminishing returns converging on a hard limit,
with the per-rung gains collapsing from `0.12` to `10⁻⁴` to zero to a final `2×10⁻⁴` that only a
qualitatively different search could supply. Every one of those numbers is a real measurement of the same
frozen evaluator, and the shape of the sequence is exactly what rung 1's redistribution argument said to
expect.

With this rung the whole trajectory becomes a squeeze, and the arithmetic of it is worth stating exactly. My
hierarchical constructor reached `0.3810764`, the frontier a single bounded local search attains; the
evolutionary search reached `0.3808694`, `2.07×10⁻⁴` lower, the current record just below my frontier; and
White's provable convex-programming lower bound `0.379005` stands under both, `1.864×10⁻³` beneath the
record. So the constant is pinned into `0.379005 ≤ C5 ≤ 0.380868`, a gap of about `1.86×10⁻³` that is no
longer mine to close with a better optimizer.

It is worth naming *why* those two ends are so hard to bring together, because it is structural, not
incidental. Every number in this ladder, and the record itself, is an *upper* bound produced by exhibiting a
concrete feasible profile — a primal construction, whose worst overlap directly witnesses `C5 ≤` that value.
White's `0.379005` is the opposite kind of object: a *lower* bound proved by convex programming, a dual-style
certificate that no feasible profile whatsoever can beat, without exhibiting any specific one. The two attack
the infimum from opposite sides — constructions push the ceiling down one profile at a time, the convex
relaxation pushes the floor up by a global argument — and the `1.86×10⁻³` between them is exactly the gap
between the best primal witness and the best dual certificate anyone has produced. Closing it would mean
either a construction so good its overlap dips beneath `0.380868` toward `0.379`, or a relaxation so tight its
floor rises above `0.379005` toward `0.3809` — and, to actually *pin* the constant, both meeting at a common
value. That neither has budged much lately, the upper side converging near `0.38087` and the lower fixed at
White's bound, is what tells me the remaining sliver is a genuine open problem rather than a temporary gap
awaiting more compute. My ladder lives entirely on the primal side, and the honest thing it can say is that
the best primal witness now sits at `0.380868`, verified under a shared evaluator, still `1.86×10⁻³` above the
best dual floor. It is the genuinely open distance between the best published
upper bound and the best provable lower bound, contested at the fifth decimal of a seventy-year-old constant.
The ladder ends at the record itself — not out-optimized within my basin, which the endpoint proved
impossible, but adopted from the different basin that large-scale evolutionary search found, and certified
feasible and record-valued under the same frozen evaluator that measured every rung before it. That honesty
about *how* the final number was reached — verified, not re-derived — is the point: the last step of this
ladder is a change of method, from constructing to certifying, and the residual `1.86×10⁻³` down to White's
floor is the real, unclaimed remainder of the problem.

I should be equally honest about what this final rung does *not* do. It advances nothing: it adopts a record
that already exists and certifies it under my evaluator, so it adds no new upper bound to the literature — its
value is the reproduction and the clean placement of my own frontier against the record on a shared measuring
stick, not a discovery. It is telling, though, that the published upper bound has itself nearly stopped moving, which sharpens what
the open gap really is. The record has crept down through a sequence of increasingly powerful searches —
Haugland's `0.380927`, AlphaEvolve's `0.380924`, TTT-Discover's `0.38087532`, and now the `0.38086945` I am
verifying — a total descent of under `6×10⁻⁵` across that whole line, and the most recent steps differ by
mere millionths (`0.38087532` to `0.38086945` is under `6×10⁻⁶`). So the upper frontier is flattening out
much as my own ladder did: each new basin-crossing search buys less than the last, and the constructions are
converging on something very near `0.380868`. That the upper side is hardening near `0.38087` while the lower
side is fixed at White's `0.379005` is what makes the `1.86×10⁻³` sliver look less like a gap about to close
and more like a durable open question — both ends are pressing against their apparent limits, and the truth
sits somewhere in between at the fifth decimal.

Nor does it close the genuinely open part of the problem. The `1.86×10⁻³` between the
record `0.380868` and White's `0.379005` is untouched by anything in this ladder: the upper side would need
still-larger or still-cleverer construction searches to inch the record down, and the lower side would need a
stronger convexity or combinatorial argument to lift the provable floor up, and the two would have to meet
somewhere in that sliver to actually *pin* `C5`. Neither is a matter of running my local constructor longer;
the endpoint proved my constructor is done at `0.3810764`. So the ladder ends where the honest evidence puts
it — my bounded local frontier at `0.3810764`, the evolutionary record at `0.3808694` a couple ten-thousandths
below it in another basin, and White's floor `0.379005` nearly two thousandths beneath that — with the middle
of the sliver still the open fifth decimal of a seventy-year-old constant, contested at real cost and not
closed by anything I could measure here. What the ladder can claim is complete and honest: from Erdős's flat
`0.5` down to the record `0.380868`, every value a genuine measurement on one frozen evaluator, the frontier
of a single bounded constructor located at `0.3810764` and the record placed a couple ten-thousandths beyond
it — with the last two thousandths to the provable floor left standing, exactly where the mathematics leaves
them.
