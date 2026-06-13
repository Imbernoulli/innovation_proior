I keep coming back to one fact that the standard schedules quietly ignore: in these nets, what decides
generalization is not how *fast* I reach a minimum but *which* minimum I reach — wide, flat ones generalize,
sharp, narrow ones do not. Every schedule I run is tuned for optimization speed: cosine and linear both
start lowering the rate from the very first step, step decay holds one high plateau whose length is a
hand-picked milestone and then drops. None of them was designed around the geometry. So before I touch the
curve I want to take the wide-minimum story seriously and ask what it *implies* the rate should do, because
if there is accuracy being left on the table, that is where it is.

Start from why the learning rate has anything to do with minimum width at all. The clean way to see it is
through the SGD noise scale. To first order the magnitude of the stochastic-gradient noise per step is
`g ≈ eps * N / (B * (1 - m))` — proportional to the rate `eps`, the dataset size `N`, inversely to the batch
`B` and to `(1 - m)` for momentum `m`. So a higher learning rate means more noise in the descent. And noise
is exactly what lets SGD escape a narrow basin: a sharp minimum is a small target with steep walls, and a
noisy step easily bounces out of it, whereas a wide minimum is a large flat region that the same noise
cannot escape. So a *high* learning rate is a wide-minimum-seeking device — it keeps the search hot enough
to be ejected from narrow basins and to settle only where the basin is wide enough to hold it. The
learning rate is the *temperature* of the search, and "decay the rate" is really "cool the search." Once I
say it that way, the question reorganizes itself: not "what decay shape is fastest" but "when should I cool,
and how long should I stay hot first."

Now I need the missing empirical fact, because the temperature picture alone does not tell me the *budget*
to spend hot. The fact I want is about the relative *density* of wide versus narrow minima in the loss
landscape. Suppose — and this is the hypothesis I will lean on — that wide minima are *rarer* than narrow
ones: the landscape is full of narrow basins and only sparsely dotted with wide ones. If that is true, then
a search that cools quickly is overwhelmingly likely to fall into one of the many narrow basins long before
it stumbles onto a rare wide one, simply because the narrow ones are everywhere and the wide ones are not.
The only way to find a rare wide basin is to *keep searching at high temperature for a long time* — to stay
hot long enough that the rare wide basins get sampled — and only *then* cool down to settle into the wide
basin you have found. Let me sanity-check the hypothesis against something I already believe: large-batch
training generalizes worse. Under the noise scale, large `B` means *less* noise, a colder search, which by
this argument settles into a narrow basin sooner — and indeed large batch is known to find sharper minima
and generalize worse. The pieces line up. And I can imagine measuring the density directly: sample minima
under more-vs-less exploration and look at the spread of widths — more time hot should shift the
distribution toward wider minima. I will treat "wide minima are lower-density, so you must explore long to
find them" as the working hypothesis and design from it.

This immediately indicts the standard schedules. Cosine annealing is at its peak rate for a single instant
— `epoch = 0` — and from there it cools monotonically; under the density hypothesis it barely explores
before it starts settling, so it finds *a* minimum fast but is biased toward the abundant narrow ones.
Linear decay is the same disease in a straighter line: it is the best of the simple monotone decays in the
budgeted-training analyses, which makes it the bar, but it too cools from step one. Step decay holds a high
plateau, which is closer to the right idea, but the plateau length is a milestone I tuned for speed, not a
deliberate explore budget, and the cooling is a discrete shock. So every baseline cools too early. The fix
the hypothesis prescribes is blunt and specific: *hold the rate high — at the peak — for a long, deliberate
explore phase, and only after that cool down*. An explore phase, then an exploit phase. This is a curriculum
in temperature: stay hot to find a rare wide basin, then cool to lock into it.

So the schedule has two phases and I have to design each, plus decide the split. Take the explore phase
first because it is the new thing. During explore I want the rate *constant at the peak* — not a ramp, not
a slow decay, just held at `peak_lr` — because the whole point is to keep the temperature maximal and
constant so the search keeps sampling basins for the full explore budget. Any decay during explore is
premature cooling, the exact thing I am trying to avoid. So the explore phase is flat at `peak_lr`. What is
`peak_lr`? The largest rate the net trains stably at — the same value the well-tuned baselines use as their
starting rate (here, the reference `base_lr`), since that is already chosen to be as high as the net
tolerates. I do not need to push it higher; I need to *hold* it longer.

Now the exploit phase. Once the explore phase has (hopefully) parked the iterate in or near a wide basin, I
want to cool the search to settle into it — drive the rate down so the noise shrinks and SGD descends to
the bottom of the wide basin and stays. What cooling shape? Here I deliberately do *not* reach for a cosine
or anything clever, because the explore phase is carrying the generalization load and the exploit phase
just has to bring the rate to zero monotonically over the remaining budget. The simplest honest choice is a
*linear* decay from `peak_lr` to 0 over the exploit epochs. Linear is the strongest simple monotone decay
in the budgeted-training analyses, it has one obvious slope, and it reaches exactly 0 at the end so the
final steps are vanishingly small and the wide basin is fully settled. A linear ramp down from the peak to
zero, with the slope set so it hits zero precisely at the budget boundary, is `slope = -peak_lr /
decay_epochs` and the rate is `peak_lr + slope * (epochs into exploit)`, clamped at 0 so it never goes
negative. No milestones, no shape parameter.

The split between explore and exploit is the one real hyperparameter, and the density hypothesis tells me
how to think about it but not the exact number. Too little explore and I cool before finding a wide basin —
back to the cosine/linear failure. Too much explore and I leave too few epochs to actually settle, so I end
near a wide basin but never descend into it and the training loss stays high. So there is an interior
optimum, and the hypothesis says it should be *large* — a substantial fraction of the budget spent hot,
because wide basins are rare and need a long search. I would set it by trying a few explore fractions —
say 0, 30%, 60%, 100% of the budget — and watching two things: the final accuracy and, as a direct test of
the mechanism, the *width* of the minimum reached (the largest weight perturbation that keeps the loss in a
band). If the hypothesis is right, more explore epochs should monotonically widen the minimum and raise
accuracy, up to the point where too little exploit budget remains to settle. I expect that to land around
half the budget — enough exploration to find a rare wide basin, enough exploitation to settle it — so my
default is **explore = 50% of the total budget**, exploit = the rest. That single fraction is the method's
one knob.

One more piece composes in: warmup. On the deep nets and especially on the Transformers, the rate at the
sharp initial region can exceed the curvature ceiling, so a short linear ramp from a small value up to
`peak_lr` over the first few steps protects the init — and it is *complementary* to the explore-exploit
structure, not part of it. So the full schedule is three segments: a brief linear *warmup* ramp to
`peak_lr`; a flat *explore* plateau at `peak_lr` for ~50% of the budget; a linear *exploit* decay from
`peak_lr` to 0 over the remainder. The warmup is just init hygiene; the explore plateau is the
generalization engine; the linear exploit is the settling. Let me write the decay-step bookkeeping so the
three segments tile the budget exactly: with `warmup_steps`, `explore_steps`, and total `total_steps`, the
decay length is `decay_steps = total_steps - (warmup_steps + explore_steps)`, and that must be
non-negative.

Let me check the segments against each other at the seams. At the end of warmup the rate is `peak_lr`; the
explore phase opens at `peak_lr` — continuous, no jump. At the end of explore the rate is `peak_lr`; the
exploit phase opens at `peak_lr` (its first decay step has zero progress) — continuous. The exploit phase
ends at 0 at the budget boundary. So the only non-smoothness is a slope change at each seam (flat-to-flat is
fine; the warmup's positive slope into the flat explore, and the flat explore into the negative exploit
slope), and none of them is a *value* discontinuity, so there is no shock to the dynamics or the momentum
velocity. Good.

Now write it as the function the harness calls. Three branches on the step index: warmup linear ramp,
explore constant, exploit linear decay clamped at 0. The concrete code belongs in the answer, while the
reasoning here is just the derivation of those branches.

The causal chain back to the start: generalization in deep nets is set by *which* minimum SGD finds — wide
ones generalize, narrow ones do not — and the learning rate, through the noise scale `g ≈ eps·N/(B(1−m))`,
is the *temperature* that decides this, since a high rate keeps the search hot enough to be ejected from
narrow basins and to settle only in wide ones. The standard schedules (cosine, linear, step) all start
*cooling* from the very first step, so under the hypothesis that wide minima are *lower-density* — rare in a
landscape full of narrow basins, consistent with large-batch/low-noise training generalizing worse — they
cool before sampling a rare wide basin and are biased toward the abundant narrow ones. The fix is an
explore-then-exploit schedule: hold the rate *flat at the peak* for a long, deliberate explore phase to
search for a rare wide minimum, then *linearly decay to 0* over an exploit phase to cool into it; the one
knob is the explore fraction, which the density argument says should be *large*, and a few explore-vs-width
sweeps put it at ~50% of the budget. A short linear warmup composes in front as init hygiene. The three
segments tile the budget with continuous values at the seams, and the whole thing is a per-step function
from training progress to a scalar rate.
