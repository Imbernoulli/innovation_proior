The ladder stopped, and the endpoint told me why. At `600` cells the bound held at `0.3810764` — the
sharp-`β` Adam did not lower it, a fresh `600`-cell multistart did not, SLSQP did not, the exact
subgradient polish did not. Every tool refined inside the same basin and returned the same number to many
digits. So the residual to the record `0.38086945` is not a tuning gap I can close by turning a knob
harder, and I have to be clear about what kind of object it is.

The diagnosis is already in hand: `0.38108` is not a resolution cap but the floor of the basin my
constructor selects. The recipe descends into the *nearest* basin of the lifted profile, and once that
profile is near-binary and spiky the cross-correlation carries a huge active set of closely-tied binding
shifts whose gradients positively span the tangent space — no feasible direction lowers every active shift
at once, and conservation reappears the stripped mass on the next-highest until they rejoin the set. The
peak-to-mean numbers said the same: my envelope sits at ratio `≈ 1.523`, the record's at `≈ 1.522`, equally
flat. The record is simply a *different* flat configuration whose plateau sits a couple ten-thousandths
lower — a basin gap I cannot flatten my way into from where I already stand as flat as the target.

So what reaches the record, if not a better optimizer of my kind? A large-scale search that does not commit
to one profile and grind it — a population-based, code-mutating evolutionary / LLM coding-agent loop that
proposes whole *constructions*, mutates the code generating the profile, and keeps a diverse population so
it can *abandon* a basin rather than refine within it. The difference is structural, not just horsepower.
My ladder moves through profile-space continuously: every step is a bounded height-perturbation followed by
projected local descent, so I only reach basins a small perturbation away; basin-hopping widens that a
little but still nudges the same vector near one anchor. A code-mutating loop moves through generator-space
— one mutation can change a period, insert a motif, or re-lay the whole arrangement, a discontinuous jump
no bounded height-nudge can express — and a maintained population holds several basins at once. That
capability is what my single descent trajectory lacks — even the closest thing to a basin-crossing move I
had, the fresh `600`-cell multistart, still held at `0.3810764`. Crossing from the low `0.3810`s to
`0.38087` *is* crossing basins.

So for this final step I stop pretending a smarter local optimizer of my kind would close it and reach the
record the only honest way: reproduce the record construction and verify it under this trajectory's own
frozen evaluator. This is a deliberate change of register — every earlier step derived and searched a
profile; this one adopts and checks one. I take the `750`-cell height profile the large-scale search
produced at its finest discretization, load it as the candidate, and put it through the exact rule every
step obeyed. I am not re-deriving it by a shortcut — the endpoint showed my machinery saturates in a
different basin — but I can confirm the published construction is genuine under *my* evaluator rather than
only theirs.

That verification carries real weight because published bounds come from different groups using different
normalizations, discretizations, and constraint tolerances, so a number quoted under one convention need
not equal what my evaluator assigns the same heights — a rescale factor, a different balance treatment, an
off-by-one in the correlation could make "their `0.38087`" and "my `0.38087`" quietly different objects. So
I run the check in verification order. Feasibility first, before I look at the overlap: the `750` heights
sum to `374.999…` — matching `n/2 = 375` to about `10⁻¹³`, the expected floating residue of summing `750`
doubles, well inside tolerance — and every height lies in `[0,1]` with exact `0` and `1` present. Then the
overlap: the evaluator returns `C = 0.3808694472026` against the source's `c5_bound =
0.3808694472025862`, agreeing to the `10⁻¹³` level. As an independent cross-check I use the conservation
invariant from the flat analysis — the correlation summed over all lags must equal `n²/4`, here `750²/4 =
140625` — and the loaded profile's correlation sums to exactly that, confirming both that the vector
carries the same total overlap-mass as every feasible profile from the flat floor on, and that my
evaluator's correlation computes what I think.

The profile is near-binary like every one before it — about `39.7%` of the `750` cells (`298`) pinned
exactly at `0`/`1` — with its worst overlap shared by a large active set: `539` of the `1499` shifts
within `10⁻⁹` of the maximum. And the ridge is sharp, not a fuzzy shoulder: widening the tolerance to
`10⁻⁴` adds only about fifteen shifts, to `10⁻³` only about forty. So the profile is balanced on a
knife-edge — `539` binding constraints held at a common level by `750` heights, `298` already frozen at the
corners, leaving a few hundred interior knobs to hold the ridge flat — exactly the collapsed-descent-cone
condition the endpoint anticipated, which is why this profile had to be *found* by a basin-crossing search
and can only be verified here. Its resolution is telling too: `750/600 = 1.25` is not integer, so it is not
an upscale of anything I built but an independent construction at its own finer grid, chosen because more
cells give more interior knobs to balance an ever-broader ridge.

That completes a structural thread every step sharpened. Every optimized profile came back near-binary, and
the pinned fraction climbed with resolution — about `29%` at `24` cells, `31%` at `120` and `600`, `39.7%`
at the record's `750`. That fits the mechanism the flat analysis laid out: corners kill the zero-shift
self-overlap, and a finer grid can send more cells to the corners while a growing interior population
carries the balance and detunes the tied ridge. So the optimum the ladder points at, consistent from two
dozen cells to seven hundred fifty, is a mostly-binary step function with a detuning skirt — a majority of
the mass slammed to `0`/`1`, a minority of interior heights arranging the leftover overlap into as flat a
ridge as conservation allows. The record is the sharpest instance of that picture, not a departure from it.

Read end to end this is one descent: `0.5`, `0.38124`, `0.38108`, `0.38108`, `0.38087` — the flat floor,
the coarse break spending `98%` of the whole distance to White's provable bound, the fine-resolution shave,
the held basin floor, and the record adopted from another basin, `2.07×10⁻⁴` below my frontier. The
per-step gains collapse from `0.12` to `10⁻⁴` to zero to a final `2×10⁻⁴` that only a qualitatively
different search could supply — exactly the shape the redistribution argument predicted.

The upper and lower ends are hard to bring together for a structural reason worth naming. Every number in
this descent, and the record, is an *upper* bound produced by exhibiting a concrete feasible profile — a primal
witness of `C5 ≤` that value. White's `0.379005` is the opposite: a *lower* bound proved by convex
programming, a dual certificate that no profile can beat, exhibiting none. Constructions push the ceiling
down one profile at a time; the relaxation pushes the floor up by a global argument; the `1.86×10⁻³`
between them is the gap between the best primal witness and the best dual certificate anyone has produced.
And the upper side has nearly stopped moving — Haugland `0.380927`, AlphaEvolve `0.380924`, TTT-Discover
`0.38087532`, now `0.38086945`, a total descent under `6×10⁻⁵` with the latest steps differing by
millionths — so it is hardening near `0.380868` while the lower bound stays fixed, which makes the sliver
look like a durable open question rather than a gap about to close. This final step advances none of that:
it adopts a record that already exists and certifies it under my evaluator, adding no new upper bound. Its
value is the reproduction and the clean placement of my own frontier against the record on one measuring
stick, the last `1.86×10⁻³` to the provable floor left standing exactly where the mathematics leaves it.
