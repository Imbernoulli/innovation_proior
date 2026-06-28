I start with the cost, because that is the part I cannot explain away. I have a fixed image
classifier, a fixed optimizer, and a menu of transformations that mostly preserve class labels.
The thing I am asked to invent is a policy over that menu. The standard answer is to search:
learn a controller, run a density-matching procedure, evolve a population, or at least grid-search
the remaining knobs. But the search is not a small implementation detail. AutoAugment has a
controller RNN choosing five sub-policies, two operations per sub-policy, and probability and
magnitude for each operation. With 16 operations, 10 magnitude bins, and 11 probability bins, one
sub-policy already has `16 * 10 * 11 = 1760` configurations per operation, so a two-operation
sub-policy has `1760^2 ~= 3.1e6`, and five of them give `(3.1e6)^5 ~= 2.9e32`. That is the space
the controller samples thousands of child trainings from. I want to know whether the search is
actually the active ingredient, or whether it is mostly a costly way to approximate broad
augmentation coverage.

The first thing I check is the proxy assumption. If augmentation strength were a stable property of
a dataset, then searching on a small proxy would be defensible. RandAugment reports the opposite:
the best global distortion magnitude rises with model width and with training-set size. That means
a cheap proxy can point at the wrong strength for the real run. The more I look at that, the less I
trust a learned policy as a transferable object. It may work, but the reason cannot simply be "the
search found the task's true policy," because the task changes when the model and data scale change.

The second observation is more direct. RandAugment compares several ways to set the global
magnitude during training: constant, random, linearly increasing, and random with an increasing
upper bound. The random strategies tie the best number, while the constant strategy is kept because
it leaves only one hyperparameter. That is a useful admission. If random magnitude is already
competitive with a tuned constant, then the constant `M` is not sacred. It is a convenient knob,
and if my goal is to remove knobs, I should not keep it just because it is easy to search.

Now I need a reason that removing the search does not remove the benefit. I use the
augmentation-orbit view. Without augmentation, I minimize
`(1/n) sum_i loss(f_theta(x_i), y_i)`. With transforms `t` sampled from an approximately
label-preserving set `T`, the empirical objective becomes
`(1/n) sum_i E_{t~T}[loss(f_theta(t(x_i)), y_i)]`. For each image, the optimizer sees an average
over transformed versions of that image. If those transforms stay inside the class, this average
reduces variation in the loss and gradient across the orbit. The argument asks for coverage of the
orbit. It does not ask for an elaborate learned distribution over the orbit unless there is evidence
that the fine weighting matters.

For the finite augmentation space, the coverage picture is concrete. If `A` is the operation set
and `M = {0,...,30}` is the strength set, then the simplest fully symmetric distribution is the
uniform mixture `(1/(|A|*|M|)) sum_{a in A} sum_{m in M} (a_m)_# P_hat`. With `|A| = 14` and
`|M| = 31` that is a mixture of `14 * 31 = 434` pushed copies of the dataset, each weighted
`1/434 ~= 0.0023`. That is not a learned policy. It is just the empirical dataset pushed through
every operation-strength pair and averaged uniformly. Writing the weight out also corrects a
tempting oversimplification: I cannot average only over operations and forget strengths. If I
collapsed the inner sum I would weight each operation `1/14` and lose the 31-way spread of strengths
that is the whole point of the orbit. The strength bins are part of the actual transform surface,
so they have to stay in the mixture.

UniformAugment already moves in this direction by replacing search with uniform sampling from an
approximate invariant set. I agree with that instinct, but I do not want to keep the extra parts
unless they are forced. The original UniformAugment algorithm samples an operation, an application
probability, and a continuous magnitude for each selected transform; the practical comparison in the
later automatic-augmentation setup treats the leave-out probability as fixed at `0.5`. Either way,
the drop mechanism is a separate choice. The number of operations is another choice. The continuous
strength range is another choice. If the mechanism is coverage, I should test whether each of those
choices is necessary, and I should be suspicious that any of them is doing hidden tuning work.

I start with composition depth. Chaining operations can create rich distributions, but richness is
not automatically good. The in-class assumption is fragile: a mild shear or color change may be
label-preserving, while a stack of large distortions can move the image outside the class. If I
apply one operation per image, the distribution is easy to understand: a uniform mixture over single
operation-strength pushes of the dataset, exactly the `434`-component mixture above. That is the
cleanest way to cover the available augmentation surface without inventing a composition knob.
RandAugment already treats `N` as a small tunable integer, and at least some of its sweeps show that
more operations are not always needed. So I set `N = 1`, not because one is magical, but because it
is the only value that removes the composition question rather than tuning it.

Once I use exactly one operation, I want to fold the application probability into the same draw. The
idea is that "do not apply an augmentation" is just the identity operation, so if `Identity` is in
`A`, a separate leave-out rule should be redundant. Before I commit to that, I should check what drop
rate it actually implies, because the surrounding methods do not treat the no-op as rare. Under a
uniform draw over 14 operations, the probability of picking `Identity` is `1/14 ~= 0.071`. The
practical UniformAugment comparison leaves an operation out with probability `0.5`. Those are not the
same number; my folding gives a no-op rate about seven times smaller. So folding the no-op into the
operation set is not a faithful re-encoding of a `0.5` drop probability — it is a genuinely different
and far more aggressive augmentation regime, where roughly 93 percent of images get transformed.

That forces me to decide which one the coverage argument actually wants, rather than assuming they
agree. A `0.5` leave-out spends half of every image's training exposure on the untransformed point.
If the benefit is exposing the optimizer to diverse in-class views, half the budget on the bare
image is a large tax, and it is a tax whose size came from no principle I can point to — `0.5` is a
round number, not a derived one. A `1/14` no-op rate spends almost the entire budget on coverage and
still revisits the clean image once every fourteen draws, plus more often through the near-identity
strength bins (small shears, `Solarize` at threshold `255`, the lowest enhancer factors). So the
folding does not reproduce the `0.5` rule; it replaces it with a drop rate that is fixed by the
operation set and that lands on the coverage-heavy side. I keep the folding, but now for the right
reason: it removes a free hyperparameter and the rate it leaves behind is the one the orbit argument
prefers, not an accident I am pretending matches the baseline.

Then I handle the magnitude. A continuous magnitude sounds more general, but the surrounding methods
already discretize strengths, and the RandAugment diagnostics say a small number of distinct
magnitudes can be enough. UniformAugment itself notes the tradeoff: discretizing reduces augmentation
variance by limiting the parameter space. That is not obviously a defect. I want enough diversity to
cover useful orbits, not unlimited jitter. So I keep the inherited 31 bins `{0,...,30}`, and instead
of fixing a single global magnitude, I sample a fresh strength bin for each image. This gives weak
and strong perturbations in the same training distribution, rather than forcing the whole run onto
one shell of the orbit.

At this point the method is sample one operation uniformly from `A`, sample one strength uniformly
from `{0,...,30}`, apply the operation at that strength, and return the image. I want to be careful
that this is not just RandAugment with `N = 1`. RandAugment with `N = 1` still uses one fixed `M`
chosen for the run, so its sampled distribution puts all its mass on a single strength shell per
operation. Here the strength is resampled, so each operation's mass is spread across all 31 bins.
The distinction is in the support of the per-image distribution, not just the parameter count.

I also need to be precise about implementation, because small constants change the actual policy.
The operation set I land on is the 14-operation RandAugment-style set: identity, auto-contrast,
equalize, rotate, solarize, color, posterize, contrast, brightness, sharpness, shear-x, shear-y,
translate-x, and translate-y. For the wide variant in torchvision, shear uses 31 values from `0` to
`0.99`, translation uses `0` to `32` pixels, rotation uses `0` to `135` degrees, solarize uses
threshold `255` down to `0`, and posterize uses retained bits from `8` down to `2`.

I should actually evaluate the posterize schedule rather than trust the endpoints, because the
formula is the kind of thing that is off-by-one in either direction. The expression is
`8 - round(arange(num_bins) / ((num_bins - 1) / 6))`. With `num_bins = 31` the divisor is
`30 / 6 = 5`, so the term is `8 - round(i / 5)` for `i = 0..30`. At `i = 0` that is `8`; at
`i = 5` it is `8 - 1 = 7`; stepping by five it walks `8, 7, 6, 5, 4, 3` and at `i = 30` reaches
`8 - 6 = 2`. So the retained-bit values run `8` down to `2` and hit every integer in between,
matching the table. Good — the lowest bin keeps the image, the highest crushes it to two bits, and
the schedule is monotone with no gap.

The shear representation is the other place I do not want to assume. The natural thing is to put the
shear factor into the off-diagonal of an affine matrix, but torchvision's `F.affine` takes shear in
degrees, so the canonical code passes `degrees(atan(magnitude))`. I check that this round-trips:
for the largest bin, `magnitude = 0.99`, the angle is `atan(0.99) ~= 44.71` degrees, and
`tan(44.71 deg) = 0.99` again. So the off-diagonal the affine actually applies is the tangent of
that angle, which is exactly the sampled shear factor — the angle wrapper recovers the intended
factor rather than distorting it. Translation is cast to integer pixels.

The four enhancers (brightness, color, contrast, sharpness) use magnitudes from `0` to `0.99` and
then apply factor `1 + magnitude`; after the fair sign flip this gives factors from `1 - 0.99 = 0.01`
to `1 + 0.99 = 1.99`, symmetric around the identity factor `1`. The wide-space table rounds the
enhancer range as `0.01` to `2.0`, and the original AutoML one-file code implements that direct
factor range, but the torchvision landing uses the signed `0.99` representation, which is why the
upper end is `1.99` rather than a clean `2.0`. Since the scaffold uses torchvision, I align the
landing code to torchvision and accept the `0.99` endpoints. Scalar operations such as identity,
auto-contrast, and equalize ignore the sampled strength entirely, so for those operations all 31
strength draws collapse to the same image. Those details are not cosmetic; they define the
distribution the training loop sees.

What remains is a parameter-free augmentation rule whose logic follows the evidence in order. Search
is expensive and proxy search is miscalibrated. Random strength already works about as well as a
tuned constant in the relevant diagnostic. The orbit-averaging argument asks for diverse in-class
coverage, not a finely optimized policy distribution. Removing unsupported choices leaves one
operation, no separate drop probability, and one fresh discrete strength per image. The result fills
the augmentation slot with a uniform finite mixture over the standard operation-strength surface —
the `434`-component mixture I started from, now realized as a per-image sampler rather than a learned
policy.
