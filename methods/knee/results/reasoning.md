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
cannot escape. So a *high* learning rate keeps the search hot enough to be ejected from narrow basins and to
settle only where the basin is wide enough to hold it. The learning rate is, in this reading, the
*temperature* of the search, and "decay the rate" is really "cool the search." Once I say it that way, the
question reorganizes itself: not "what decay shape is fastest" but "when should I cool, and how long should I
stay hot first."

Before I trust that reframing I want to make sure it is not just a metaphor that happens to sound right. The
cleanest external prediction it makes is about batch size, because `g` depends on `B` the same way it
depends on `eps`. Take a CIFAR-scale setting and put numbers in: `N = 50000`, `m = 0.9`, `eps = 0.1`. At
`B = 128`, `g ≈ 0.1 · 50000 / (128 · 0.1) = 5000 / 12.8 ≈ 390`. Push the batch to `B = 4096` and
`g ≈ 5000 / 409.6 ≈ 12.2`. That is a factor of 32 less noise — exactly the batch ratio, as the formula
demands — for the large-batch run. So the temperature view *predicts* that large batch runs a much colder
search and should therefore settle into narrower minima and generalize worse, holding everything else fixed.
That is precisely the large-batch generalization gap that is reported in practice. The metaphor survives a
quantitative prediction it did not have to, so I will reason with it.

Now the temperature picture alone does not tell me the *budget* to spend hot — it says hot is good, not how
long. To get a budget I need a fact about the landscape itself: the relative *density* of wide versus narrow
minima. Suppose — and this is the hypothesis I will lean on — that wide minima are *rarer* than narrow ones:
the landscape is full of narrow basins and only sparsely dotted with wide ones. Then a search that cools
quickly is overwhelmingly likely to fall into one of the many narrow basins long before it stumbles onto a
rare wide one, simply because the narrow ones are everywhere and the wide ones are not. To find a rare wide
basin under that density you have to keep sampling at high temperature for a long time, and only then cool.
The same batch-size arithmetic above is consistent with this: the colder large-batch search not only settles
faster but settles into the abundant narrow basins it cannot escape, which is what its worse generalization
looks like. I cannot prove the density claim from here — I would want to measure it directly, sampling minima
under more-versus-less exploration and watching whether more time hot shifts the width distribution wider —
but it is the working hypothesis, and I will design from it and keep that measurement in mind as the test
that could falsify the whole thing.

This reshapes how I read the standard schedules. Cosine annealing is at its peak rate for a single instant —
`epoch = 0` — and from there it cools monotonically; under the density hypothesis it barely explores before
it starts settling, so it finds *a* minimum fast but should be biased toward the abundant narrow ones. Linear
decay is the same shape in a straighter line: it is the best of the simple monotone decays in the
budgeted-training analyses, which makes it the bar to beat, but it too cools from step one. Step decay holds
a high plateau, which is closer to the right idea, but the plateau length is a milestone tuned for speed, not
a deliberate explore budget, and the cooling arrives as a discrete shock. The common defect is that all of
them start cooling before the search has had time to sample a rare wide basin. So the hypothesis is pushing
me toward a schedule whose first job is to *not* cool — hold the rate high, at the peak, for a long deliberate
stretch — and only after that stretch begin to cool. Two phases, where the standard schedules effectively
have one cooling curve: a long high-temperature search, then a settling.

So the schedule has two phases and I have to design each, plus decide the split. Take the high-rate phase
first because it is the new thing. During it I want the rate *constant at the peak* — not a ramp, not a slow
decay, just held at `peak_lr` — because the whole point is to keep the temperature maximal and constant so
the search keeps sampling basins for the full budget I allot to it. Any decay during this phase is premature
cooling, the exact thing I am trying to avoid. So this phase is flat at `peak_lr`. What is `peak_lr`? The
largest rate the net trains stably at — the same value the well-tuned baselines use as their starting rate
(here, the reference `base_lr`), since that is already chosen to be as high as the net tolerates. I do not
need to push it higher; I need to *hold* it longer.

Now the second phase. Once the high-rate phase has (hopefully) parked the iterate in or near a wide basin, I
want to cool the search to settle into it — drive the rate down so the noise shrinks and SGD descends to the
bottom of the wide basin and stays. What cooling shape? Here I deliberately do *not* reach for a cosine or
anything clever, because the high-rate phase is carrying the generalization load and this phase just has to
bring the rate to zero monotonically over the remaining budget. The simplest honest choice is a *linear*
decay from `peak_lr` to 0 over the remaining epochs. Linear is the strongest simple monotone decay in the
budgeted-training analyses, it has one obvious slope, and it reaches exactly 0 at the end so the final steps
are vanishingly small and the wide basin is fully settled. A linear ramp down from the peak to zero, with the
slope set so it hits zero precisely at the budget boundary, is `slope = -peak_lr / decay_epochs`, and the
rate is `peak_lr + slope * (epochs into the decay)`, clamped at 0 so it never goes negative. No milestones, no
shape parameter. I will call these two phases *explore* (the flat high-rate search) and *exploit* (the linear
settle), since that is what each is doing.

The split between explore and exploit is the one real hyperparameter, and the density hypothesis tells me how
to think about it but not the exact number. Too little explore and I cool before finding a wide basin — back
to the cosine/linear failure. Too much explore and I leave too few epochs to actually settle, so I end near a
wide basin but never descend into it and the training loss stays high. So there is an interior optimum, and
the hypothesis says it should be *large* — a substantial fraction of the budget spent hot, because wide
basins are rare and need a long search. I would set it by trying a few explore fractions — say 0, 30%, 60%,
100% of the budget — and watching two things: the final accuracy and, as a direct test of the mechanism, the
*width* of the minimum reached (the largest weight perturbation that keeps the loss in a band). If the
hypothesis is right, more explore epochs should widen the minimum and raise accuracy, up to the point where
too little exploit budget remains to settle. I expect the turnover to sit near the middle: enough exploration
to find a rare wide basin, enough exploitation to settle it. So my default is **explore = 50% of the total
budget**, exploit = the rest. That single fraction is the method's one knob, and the explore/width sweep above
is the experiment that would actually pin it down rather than my guess.

One more piece composes in: warmup. On the deep nets and especially on the Transformers, the rate at the
sharp initial region can exceed the curvature ceiling, so a short linear ramp from a small value up to
`peak_lr` over the first few steps protects the init — and it is *complementary* to the explore-exploit
structure, not part of it. So the full schedule is three segments: a brief linear *warmup* ramp to `peak_lr`;
a flat *explore* plateau at `peak_lr` for ~50% of the budget; a linear *exploit* decay from `peak_lr` to 0
over the remainder. The warmup is just init hygiene; the explore plateau is the generalization engine; the
linear exploit is the settling. Let me write the decay-step bookkeeping so the three segments tile the budget
exactly: with `warmup_steps`, `explore_steps`, and total `total_steps`, the decay length is
`decay_steps = total_steps - (warmup_steps + explore_steps)`, and that must be positive or there is no room
to settle at all.

Before I trust that the three segments tile cleanly I should actually trace the rate the harness would emit,
not just assert the seams line up. Take a tiny budget I can step through by hand: `total_steps = 10`,
`warmup_steps = 2`, `explore_steps = 4`, so `decay_steps = 10 - (2 + 4) = 4`, and `peak_lr = 0.1`. The
slope in the exploit phase is `-0.1 / 4 = -0.025`. Walking the step index from 1 (the scheduler starts its
counter at 1) and applying the three branches — warmup `step <= 2`, explore `step <= warmup+explore = 6`,
else exploit:

```
step 1 : warmup   0.1 * 1/2                    = 0.0500
step 2 : warmup   0.1 * 2/2                     = 0.1000
step 3 : explore                               = 0.1000
step 4 : explore                               = 0.1000
step 5 : explore                               = 0.1000
step 6 : explore                               = 0.1000
step 7 : exploit  0.1 - 0.025*(7-6) = 0.1-0.025 = 0.0750
step 8 : exploit  0.1 - 0.025*(8-6)            = 0.0500
step 9 : exploit  0.1 - 0.025*(9-6)            = 0.0250
step 10: exploit  0.1 - 0.025*(10-6)           = 0.0000
```

So warmup hits exactly `peak_lr` at its last step (step 2), explore holds `0.1` flat across steps 3–6, and
exploit walks down to exactly `0.0` at the budget boundary (step 10). That last value matters and it lands:
`0.1 + (-0.025)·(10-6) = 0.1 - 0.1 = 0`, so the budget really does end with the rate fully cooled, no clamp
needed and no leftover rate.

But the trace also caught something my mental model of the seams got slightly wrong. I had been telling
myself the exploit phase "opens at `peak_lr`" because its first decay step has zero progress — but it
doesn't. The last explore step (step 6) returns `peak_lr` via the `<=` branch, and the *very first* exploit
step (step 7) already has `global_step - (warmup+explore) = 1`, not 0, so it returns
`peak_lr - peak_lr/decay_steps = 0.1 - 0.025 = 0.075`. The explore→exploit seam is therefore not exactly
continuous: it drops by one slope-step of size `peak_lr/decay_steps` right at the transition. With
`decay_steps = 4` that drop is `0.025`, a quarter of the peak — visible. But that is an artifact of the tiny
toy budget. At a realistic budget — say 200 epochs with ~half on decay, `decay_steps ≈ 100` — the seam drop
is `peak_lr/100`, one percent of the peak in a single step, which is smaller than the per-step noise the
schedule is deliberately running on and vanishes as the budget grows. So the practical statement is: the
warmup→explore seam is exactly continuous (both equal `peak_lr`), and the explore→exploit seam is continuous
*to one discretization step* that shrinks like `1/decay_steps`. Either way there is no *value* discontinuity
on the scale of a sharp step-decay shock — the largest jump anywhere is `peak_lr/decay_steps`, not a factor
of 5 — so there is no shock to the momentum velocity. That is the property I actually wanted, and now I know
it holds for the real reason rather than the one I first assumed.

Now write it as the function the harness calls. Three branches on the step index: warmup linear ramp, explore
constant, exploit linear decay clamped at 0. The concrete code belongs in the answer, while the reasoning here
is just the derivation of those branches.

Tracing back along the chain I actually walked: generalization in these nets is set by *which* minimum SGD
finds, and the noise scale `g ≈ eps·N/(B(1−m))` ties that to the learning rate as a temperature — a claim I
made concrete by checking that the same formula predicts the measured large-batch generalization gap to the
right factor. From there the *density* of wide minima (rare among many narrow basins, the part I can only
hypothesize and would test with an explore-vs-width sweep) is what forces a *long* hot search before cooling.
The two phases that follows are an `explore` plateau held flat at the peak and a linear `exploit` decay to 0,
with the explore fraction the one knob the density argument pushes to be large and the sweep would pin near
half the budget; a short linear warmup composes in front as init hygiene. The three segments tile the budget,
and stepping the real rate through a tiny budget confirmed they reach 0 exactly at the boundary and never
introduce a value shock larger than one slope-step — the whole thing a per-step function from training
progress to a scalar rate.

## Minimal knee-point stub

The name "knee" is worth taking literally for a moment, because the schedule's shape *is* a knee: a flat
explore plateau that bends down into the linear exploit decay, and the bend — the corner where holding gives
way to cooling — is exactly where a diminishing-returns curve turns. So a small helper that finds the "knee"
of a 2-D curve in its geometric form is the right companion: the point farthest from the chord joining the
curve's first and last points. Let me get the geometry right rather than guess it. For a point `(xi, yi)` and
the chord from `(x0, y0)` to `(x1, y1)` with direction `(dx, dy) = (x1-x0, y1-y0)`, the perpendicular
distance is `|dy·xi - dx·yi + x1·y0 - y1·x0| / sqrt(dx² + dy²)` — the standard point-to-line formula written
with the chord's two endpoints. I should not ship that on trust; let me work it on a concrete concave curve
that has an obvious knee.

Take a diminishing-returns curve, `x = [0,1,2,3,4]`, `y = [0, 0.7, 0.9, 0.97, 1.0]` — a steep early rise that
flattens, the kind of curve where the elbow should sit at the first big gain. The chord runs `(0,0)→(4,1.0)`,
so `dx = 4`, `dy = 1.0`, `norm = sqrt(16 + 1) = sqrt(17) ≈ 4.123`, and the constant `x1·y0 - y1·x0 = 0`.
Plugging each interior point into `|1.0·xi - 4·yi| / 4.123`:

```
i=1 (1,0.70): |1.0 - 2.80| / 4.123 = 1.80 / 4.123 = 0.437   <- largest
i=2 (2,0.90): |2.0 - 3.60| / 4.123 = 1.60 / 4.123 = 0.388
i=3 (3,0.97): |3.0 - 3.88| / 4.123 = 0.88 / 4.123 = 0.213
```

The maximum is at `i = 1`, the point `(1, 0.70)` — which is exactly the elbow of that curve, where the steep
initial gain gives way to the flat tail. So the helper returns index 1 here, and it returns it for the right
reason: that point is genuinely the farthest off the chord, by the same formula the code uses. The endpoints
both sit *on* the chord so their distance is 0 and they can never be selected, which is the behavior I want —
a knee is interior by definition. Good; the helper computes what its name claims.

```python
def find_knee_point(x, y):
    """Return the index of the 'knee' — the point farthest from the line
    joining the first and last points of a 2-D curve."""
    if len(x) != len(y) or len(x) < 3:
        raise ValueError("x and y must have equal length >= 3")

    x0, y0 = x[0], y[0]
    x1, y1 = x[-1], y[-1]
    dx, dy = x1 - x0, y1 - y0
    norm = (dx * dx + dy * dy) ** 0.5
    if norm == 0:
        return 0

    max_dist, knee_idx = -1.0, 0
    for i, (xi, yi) in enumerate(zip(x, y)):
        # perpendicular distance from (xi, yi) to line (x0, y0)-(x1, y1)
        dist = abs(dy * xi - dx * yi + x1 * y0 - y1 * x0) / norm
        if dist > max_dist:
            max_dist, knee_idx = dist, i
    return knee_idx
```
