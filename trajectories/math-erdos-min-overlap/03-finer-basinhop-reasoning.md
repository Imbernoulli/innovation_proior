The coarse rung landed at `0.381240` with two dozen cells, and it left me a clean falsifiable claim to
test: the cap there is *resolution*, not a missed basin. The whole descent from the flat floor to White's
provable lower bound is only `0.5 − 0.379005 = 0.120995`, and the coarse rung already spent `98%` of it in
one step; what remains is a thin slice of ten-thousandths, and I argued that two dozen cells simply cannot
represent the fine detuning needed to shave the worst-overlap envelope any thinner — too few interior knobs
for too many competing shifts. If that reading is right, adding *pieces* should lower the bound a little
while adding more *coarse restarts* would not. So the natural next move is to lift the optimized profile to
more pieces and refine it there. But I have to be careful *how* I lift: the coarse profile already carries
the gross structure that bought `98%` of the descent, and re-randomizing at higher `n` would throw that
away — a fresh random `120`-cell start lands back in near-binary chaos, where I measured random balanced
profiles averaging `0.628`, worse than the flat floor. The lift has to *preserve* structure, not discard
it.

The clean way to add resolution for free is to upscale: replace each cell by `r` identical finer cells of
the same height. This is the same step function `h` on `[0,2]`, just expressed on a finer grid — same
integral overlap, same bound — so the upscaled point *starts* at exactly the coarse `C` and now carries `r`
times as many degrees of freedom. I want to be sure this is genuinely free and not approximately free, so I
check the mechanism. Repeating each cell `r` times sends the discrete cross-correlation's peak to lag
`r·k*` (where `k*` was the coarse worst lag) with value exactly `r` times the coarse peak, and the rescale
`2/(rn) = (1/r)·(2/n)` cancels the factor `r` precisely — so `C` is invariant to machine precision, which I
confirmed on a test vector at `×2`, `×5`, `×10` (identical to ten digits). Crucially the upscaled peak is a
*unique* maximum at the aligned lag, and no intermediate fine shift exceeds it, because `h` is unchanged and
the finer grid only samples the same function more densely. So upscaling is a true no-op on the value that
hands me a richer neighborhood: the fine grid contains every coarse configuration plus many more, and the
optimizer can now break the repeated blocks apart and carve structure the coarse grid could not represent.

There is a wrinkle in that richer neighborhood, and it is the flip side of the free lift. The upscaled point
is a *degenerate plateau*: the `r` sub-cells of a former coarse cell are all equal, and perturbing two of
them in opposite directions (`+δ`, `−δ`) preserves both the sum and — to first order — the overlap, because
the two sub-cells enter every `c_k` almost symmetrically, their contributions cancelling except at the
block's boundary. So the objective is flat in a large subspace of directions right at the upscaled point,
and a gradient or SLSQP step launched from the exact plateau finds vanishing slope and stalls. I can even
count the flat subspace: each former coarse cell becomes a block of `r` equal sub-cells, and within a block
of `r` there are `r − 1` independent zero-sum internal directions, so across the `n_coarse` blocks the
plateau is flat in about `n_coarse·(r − 1)` directions — for a `×2` upscale of `60` cells that is `60`
degenerate directions. Those are *precisely* the new degrees of freedom the lift added: the upscale hands me
`60` fresh knobs and simultaneously parks me at a point where all `60` of them have zero gradient. That is why
the kick is not a cosmetic nicety but the thing that activates the lift at all — without it the refinement
would report the lifted value and the extra freedom would sit unused. The remedy is
a small kick: perturb the upscaled vector slightly and re-project to feasibility. The kick barely moves `C`
— it is well inside the range where the worst shift stays put and the score changes only quadratically, as I
saw at the flat point — but it breaks the block symmetry and gives the optimizer a non-degenerate gradient to
descend. Without the kick the refinement would sit frozen on the plateau reporting the lifted value; with it,
the extra degrees of freedom actually get spent.

Now the real question: what optimizer refines at this larger `n`? The coarse rung's plain multi-start —
fresh random feasible starts, keep the best — is the wrong tool now, for two reasons that reinforce each
other. First, its own arithmetic turns against me: I argued that `12` random starts suffice at `n = 24` only
if a non-trivial fraction `p` of starts reach the good basin, and that `p` shrinks as `n` grows because more
cells means more and narrower basins. At `120` cells a fixed handful of blind starts is unlikely to land
well. Second, and more decisively, fresh random starts *discard* the very structure upscaling just handed me
for free — they begin from chaos near `0.6`, not from the `0.381` profile I want to refine. Both problems
point the same way: I should perturb the *best-so-far* vector and re-solve, rather than start over. That is
basin-hopping — solve the annealed ladder to a local optimum, perturb the current best by Gaussian noise,
re-project, re-solve, and accept only improvements in the *true* overlap, for a budget of hops. Each hop is a
constrained restart *near* the current best: far enough to jump into a neighboring basin, near enough to keep
the good gross structure the lift preserved. I shrink the perturbation over the hops (scale `∝ 0.9^h + 0.1`),
so early hops explore across basins and late hops refine within the best one. This is the "perturbation
search plus basin-hopping" recipe the agentic-search record (AutoEvolver) reports on this problem, and it is
exactly right for a non-convex minimax where local descent is cheap but the basins are many and the good
ones are close together in structure space.

I make one deliberate simplification to the classic basin-hopping template: I accept a hop only if it
*improves* the true overlap, rather than accepting worse candidates with some Metropolis probability. The
textbook version uses stochastic acceptance to escape traps, but here the exploration is already supplied by
the perturbation-plus-re-solve — each hop lands in a genuinely different local optimum — and I always keep the
best-so-far as the anchor, so a greedy accept keeps the good structure anchored while still sampling
neighboring basins through the kick. The risk of greedy acceptance is settling early, and the shrinking kick
schedule is the counterweight: the first hops kick hard (scale near `0.03`) and can jump far, so the anchor
gets a real chance to move to a better basin before the kick decays and the late hops become pure local
refinement. At this middle resolution, where I mainly want to *tune* the schedule, greedy-plus-shrinking-kick
is the simplest thing that both explores and converges, and it is cheap enough to watch its behavior across
all `20` hops before I commit the endpoint's budget.

A second change is forced by the finer resolution: the `β` ladder has to be *sharper* than at the coarse
level. The soft-max surrogate sits below the true worst overlap by at most `(2/n)·log(2n−1)/β`, and two
things move at `120` cells — there are now `2n − 1 = 239` shifts instead of `47`, and the near-binary
profile is spikier, so more shifts crowd close to the maximum and the surrogate has more competitors to
smear over. A `β` that was sharp at `24` cells is too soft at `120`: at `β = 300` the gap is about
`3×10⁻⁴`, as large as the entire prize I am chasing, so the optimizer would be minimizing a systematically
wrong objective. I anneal the hop ladder up to `β = 3600`, where the gap falls to about `2.5×10⁻⁵` — small
compared to the shave I expect — so the surrogate genuinely tracks the hard overlap I report by the final
hops. As always I score the true `max_k` overlap of every candidate, not the surrogate, and keep the best
true value ever seen, because surrogate-best and true-best are not the same vector.

The two `β` ladders in this rung are deliberately different, and the difference tracks how cold each start
is. The fresh multi-start at `60` cold-starts from random feasible profiles, so it needs the full soft-to-sharp
walk `(60, 150, 300, 600, 1200, 2400)` — the very soft early levels are what let a chaotic initial profile
reorganize its gross structure before the surrogate kinks up. The `120`-cell basin-hop, by contrast, always
begins each solve from an anchor that is already near-optimal (the best-so-far, freshly kicked), so it does
not need the soft reorganizing levels at all; it can start moderately sharp at `β = 300` and climb to
`3600`. Starting the hop ladder soft would be wasted work — it would let the near-optimal anchor drift back
toward a smeared surrogate optimum before re-sharpening — so I match the ladder's floor to the anchor's
quality: cold starts get a soft floor, warm restarts get a sharp one. That the sharpest level rises from
`2400` at `n = 60` to `3600` at `n = 120` is the same resolution effect: more shifts and closer ties at the
finer grid demand a sharper surrogate to keep the gap below the shave I am chasing.

How far to lift, and by what path? I do not jump straight to the several hundred cells the records use. I go
to a middle resolution — around a hundred-odd cells — for two concrete reasons. First, I want to *confirm*
lifting actually lowers the bound before spending a long run at high `n`; this rung is the test of the
resolution reading, and the metric that settles it is whether more pieces buy a lower upper bound. Second,
this is where I *tune the hop schedule* — the kick scale, the number of hops, the `β` ceiling — cheaply, so
the endpoint rung can spend its budget on resolution rather than on re-discovering these settings. Concretely
the path is `24 → 60` by a fresh multi-start at `60` (still only `60` variables, so a handful of blind
starts with a `β` ladder to `2400` is affordable and reaches a better base than the `24`-cell profile,
because `60` cells give more interior detuning knobs), then a free integer upscale `×2` to `120`, then `20`
basin-hops with the `(300, 800, 1800, 3600)` ladder and an initial kick of `0.03`. The multi-start at `60`
rather than an upscale of the `24`-cell vector is deliberate: `60` is not an integer multiple of `24`, and at
sixty variables I can still afford to re-solve from scratch and let the optimizer find the best `60`-cell
base directly, which then upscales cleanly by two.

The choice to re-search at `60` instead of just upscaling `24 → 120` directly is doing real work, and it is
worth being explicit about the tension it resolves. Upscaling `24 → 120` (a `×5` no-op) would inherit the
`24`-cell arrangement exactly — the coarse profile's specific placement of corners and interior cells,
frozen — and then only let basin-hopping perturb it locally. A *fresh* multi-start at `60` instead lets the
optimizer discover a genuinely finer arrangement from scratch, with `60` interior knobs available from the
first solve rather than inherited from a `24`-cell template. Sixty variables is about the resolution ceiling
where blind restarts are still affordable — the probability argument from the coarse rung has not yet turned
badly against me, and a QP at `60` variables is still milliseconds — so `60` is the last place I can cheaply
buy a *new* base rather than merely refine an old one. Past that, at `120` and beyond, blind restarts stop
paying and I have to switch to structure-preserving refinement, which is exactly why the `60 → 120` step is a
free upscale followed by basin-hopping rather than another fresh search. The path is thus "search freshly as
long as you can afford to, then lift-and-refine once you cannot," and `60` is the crossover.

A detail I want right, since it governs whether the kick actually helps: after each Gaussian kick I
re-project onto feasibility with the same clip-and-redistribute operator as the coarse rung. The kick scale
`0.03` is small relative to the `[0,1]` height range, so most cells stay inside the box and only those near a
corner get clipped and have their mass spread back over the interior cells; the projection therefore barely
distorts the kick's direction while guaranteeing the scored vector satisfies `Σ v = n/2` exactly. If I kicked
much harder the projection would fight the perturbation — clipping away most of it at the saturated cells —
and the hop would land somewhere unrelated to the anchor; if I kicked much softer it would not break the
plateau or clear the current basin. The `0.03`-with-`0.9^h` decay sits in the band where the kick is large
enough to jump and small enough that projection preserves its intent, which is another setting this middle
rung exists to confirm before the endpoint.

What do I expect? Rung 1's conservation picture predicted a big first drop followed by a long grind, and
rung 2 delivered the drop. This rung is squarely in the grind: the coarse profile already captured the gross
structure, so the gain from `24 → 120` cells is *fine shaving*, not another jump. I expect to clear the
coarse `0.381240` but only modestly — landing a little below it, around `0.38108`, closing most of the
remaining distance to the Haugland / AlphaEvolve landmarks near `0.380924`–`0.380927` but not reaching them,
and still a couple of ten-thousandths above the AutoEvolver record `0.38086945`. A modest drop from lifting
— and none obtainable from merely piling more restarts onto `n = 24` — is exactly the signature that would
confirm the resolution reading rather than a basin miss; a large drop would refute it, and I would have to
concede the coarse rung had simply landed in a poor basin. I expect the former.

It is worth writing down where a landing near `0.3810764` leaves me in the two directions that bound the
problem, because it re-frames how thin the remaining air is. Above me, the record `0.38086945` would be only
`2.07×10⁻⁴` below — so the entire distance from a single bounded constructor's fine-resolution result to the
best published step function is about two ten-thousandths. Below me, White's provable floor `0.379005` sits
`2.07×10⁻³` down — an order of magnitude further, and unreachable by any construction. The shave this rung
buys, on the order of `1.6×10⁻⁴`, is itself comparable to the *whole* residual gap to the record: I am now
deep in the regime where one rung's gain and the distance to the frontier are the same size, which is the
quantitative meaning of the "long grind" rung 1 predicted. Every subsequent ten-thousandth costs
disproportionately more effort than the last, because conservation forces me to lower a growing set of tied
shifts together rather than one dominant peak.

The ratio of successive gains makes the grind concrete. Rung 2 dropped the bound by about `0.118` (from `0.5`
to `0.381240`); this rung, if it lands near `0.381076`, drops it by about `1.6×10⁻⁴`. That is a shrink of the
per-rung gain by a factor of roughly seven hundred in a single step — the unmistakable signature of a search
approaching a hard limit, where the easy structure has all been captured and only fine adjustments remain. It
also sets an honest expectation for the endpoint: I should *not* assume that a much larger lift buys a
proportionally larger gain. A `×5` lift to several hundred cells adds many more knobs, but if the profile has
already found its basin, the extra resolution may only let the same envelope be expressed more smoothly
rather than pushed materially lower. I expect the endpoint to confirm the frontier of a single bounded
constructor is near here, not far below it — a prediction the record's small residual `2.07×10⁻⁴` above me
already makes plausible, since that entire residual is what the largest published searches, not just more
cells, were needed to close.

One structural prediction I can check on the returned vector: as resolution rises and the envelope flattens,
the number of shifts sitting essentially tied at the worst overlap should *grow*. At the flat floor exactly
one shift was worst; a good coarse profile ties a handful; and as I push toward the frontier the worst
overlap should be shared by an ever-larger active set of shifts, because that is what a flatter envelope
means under fixed total mass. If I see the active set widening from `24` to `120` cells, that both confirms
the flattening mechanism and warns me about the endpoint — a very large active set is exactly the condition
under which local descent has nowhere to go, since lowering any tied shift raises another. I will read the
active-set size off the `120`-cell result and keep it in mind as the endpoint's likely obstacle. The returned profile should
again be near-binary and asymmetric, with a similar fraction of cells pinned to the box corners as the coarse
one but now with more interior knobs to detune the spikier envelope; I report the true hard-max overlap of
the best `~120`-cell vector, and I expect it to read about `0.3810764`.

I can quantify why the lift should help at all, in the same interior-knob terms the coarse rung used to
explain its cap. There I argued that with roughly a third of `24` cells pinned to the box corners I had only
about a dozen interior heights to detune a `47`-shift envelope — a coarse instrument. At `120` cells, if a
similar fraction pins, I have on the order of eighty interior knobs to work a `239`-shift envelope: the lift
roughly quintuples the detuning degrees of freedom while only quintupling the number of shifts, so each
interior knob now addresses fewer shifts and can separate near-tied ones more finely. That is the concrete
mechanism by which more pieces buy a lower bound — not by changing the gross near-binary structure, which
upscaling preserved, but by giving the refinement enough independent knobs to flatten the ridge one notch
further. It also tells me the gain must be *modest*: quintupling the knobs sharpens the detuning but does not
change the basin, so I expect a fine shave, not a jump — precisely the small drop that distinguishes a
resolution cap from a basin miss.

The limitation this rung will expose is, once more, resolution — but now at the fine end. A hundred-odd cells
resolves the profile better than two dozen, yet the published frontier lives at several hundred cells, where
the worst-overlap envelope can be made to tie across many more shifts at a still-lower common level. There is
also a cost wall I can already see coming from the coarse rung's aside about numeric gradients and QP
subproblems being super-linear in `n`. Each basin-hop here runs a full annealed SLSQP ladder, and SLSQP
finite-differences the surrogate at `O(n)` objective evaluations per gradient with an `O(n²)` correlation
each, plus a QP subproblem that is itself super-linear in `n` — so the per-solve cost grows steeply, roughly
like `n³` or worse. At `120` cells, with `20` hops of `4` levels, this is still seconds; but a naive `×5`
scale-up to `600` cells would inflate each solve by more than two orders of magnitude and the whole hop
budget with it, turning a seconds-long refinement into minutes per solve. I can feel the endpoint's central
problem forming already: I cannot simply carry this exact machinery to record resolution, because the solver
that was the right tool at `24` and `120` cells becomes the bottleneck at `600`. So the endpoint rung has to do two things at
once — lift again to the several-hundred-cell scale the records use, and switch to an optimizer that scales —
and refine there with a longer budget plus a final pass that descends the true worst overlap directly rather
than the smooth surrogate, pushing toward `~0.3809` and reading off, honestly, how close a single bounded
constructor gets to the record. The one number I will watch across that lift is the upper bound itself: a
further small drop as pieces grow would say the frontier is still ahead, while a value that simply holds at
this rung's would say I have reached the floor of the basin my whole recipe selects — and either reading is a
real result, not a disappointment.
