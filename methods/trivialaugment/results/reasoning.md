I start with the cost, because that is the part I cannot explain away. I have a fixed image
classifier, a fixed optimizer, and a menu of transformations that mostly preserve class labels.
The thing I am asked to invent is a policy over that menu. The standard answer is to search:
learn a controller, run a density-matching procedure, evolve a population, or at least grid-search
the remaining knobs. But the search is not a small implementation detail. AutoAugment has a
controller RNN choosing five sub-policies, two operations per sub-policy, and probability and
magnitude for each operation. The policy space is enormous, and the controller samples thousands
of child trainings. RandAugment is much simpler, but it still searches `N` and a global `M` on the
target task. I want to know whether the search is actually the active ingredient, or whether it is
mostly a costly way to approximate broad augmentation coverage.

The first crack is the proxy assumption. If augmentation strength were a stable property of a
dataset, then searching on a small proxy would be defensible. RandAugment shows the opposite:
the best global distortion magnitude rises with model width and with training-set size. That
means a cheap proxy can point at the wrong strength for the real run. The more I look at that,
the less I trust a learned policy as a transferable object. It may work, but the reason cannot
simply be "the search found the task's true policy," because the task changes when the model and
data scale change.

The second crack is even more direct. RandAugment compares several ways to set the global
magnitude during training: constant, random, linearly increasing, and random with an increasing
upper bound. The random strategies tie the best number, while the constant strategy is kept
because it leaves only one hyperparameter. That is a useful admission. If random magnitude is
already competitive with a tuned constant, then the constant `M` is not sacred. It is a convenient
knob, and if my goal is to remove knobs, I should not keep it just because it is easy to search.

Now I need a reason that removing the search does not remove the benefit. I use the
augmentation-orbit view. Without augmentation, I minimize
`(1/n) sum_i loss(f_theta(x_i), y_i)`. With transforms `t` sampled from an approximately
label-preserving set `T`, the empirical objective becomes
`(1/n) sum_i E_{t~T}[loss(f_theta(t(x_i)), y_i)]`. For each image, the optimizer sees an average
over transformed versions of that image. If those transforms stay inside the class, this average
reduces variation in the loss and gradient across the orbit. The argument asks for coverage of
the orbit. It does not ask for an elaborate learned distribution over the orbit unless there is
evidence that the fine weighting matters.

For the finite augmentation space, the coverage picture is concrete. If `A` is the operation set
and `M = {0,...,30}` is the strength set, then the simplest fully symmetric distribution is the
uniform mixture `(1/(|A|*|M|)) sum_{a in A} sum_{m in M} (a_m)_# P_hat`. That is not a learned
policy. It is just the empirical dataset pushed through every operation-strength pair and averaged
uniformly. This formula also corrects a tempting oversimplification: I cannot average only over
operations and forget strengths. The strength bins are part of the actual transform surface.

UniformAugment already points in this direction by replacing search with uniform sampling from an
approximate invariant set. I agree with that instinct, but I do not want to keep the extra parts
unless they are forced. The original UniformAugment algorithm samples an operation, a probability,
and a continuous magnitude for each selected transform; the practical comparison in the later
automatic-augmentation setup treats the leave-out probability as fixed at `0.5`. Either way, the
drop mechanism is a separate choice. The number of operations is another choice. The continuous
strength range is another choice. If the mechanism is coverage, I should test whether each of
those choices is necessary.

I start with composition depth. Chaining operations can create rich distributions, but richness is
not automatically good. The in-class assumption is fragile: a mild shear or color change may be
label-preserving, while a stack of large distortions can move the image outside the class. If I
apply one operation per image, the distribution is easy to understand: a uniform mixture over
single operation-strength pushes of the dataset. That is the cleanest way to cover the available
augmentation surface without inventing a composition knob. RandAugment already treats `N` as a
small tunable integer, and at least some of its sweeps show that more operations are not always
needed. So I set `N = 1`, not because one is magical, but because it is the only value that
removes the composition question rather than tuning it.

Once I use exactly one operation, the application probability becomes unnecessary. "Do not apply
an augmentation" is just the identity operation. If `Identity` is in `A`, then the no-op event has
probability `1/|A|` under uniform sampling. I do not need a sampled `p`, a fixed `0.5`, or any
other leave-out rule. Folding the no-op into the same uniform draw also keeps the method honest:
the probability of doing nothing is determined by the operation set, not by an extra hidden
hyperparameter.

Then I handle the magnitude. A continuous magnitude sounds more general, but the surrounding
methods already discretize strengths, and the RandAugment diagnostics say a small number of
distinct magnitudes can be enough. UniformAugment itself notes the tradeoff: discretizing reduces
augmentation variance by limiting the parameter space. That is not obviously a defect. I want
enough diversity to cover useful orbits, not unlimited jitter. So I keep the inherited 31 bins
`{0,...,30}`, and instead of fixing a single global magnitude, I sample a fresh strength bin for
each image. This gives weak and strong perturbations in the same training distribution, rather
than forcing the whole run onto one shell of the orbit.

At this point the method is almost embarrassingly small: sample one operation uniformly from `A`,
sample one strength uniformly from `{0,...,30}`, apply the operation at that strength, and return
the image. It is important that this is not just RandAugment with `N = 1`. RandAugment with
`N = 1` still uses one fixed `M` chosen for the run. Here the strength is sampled anew, so the
distribution covers all strength bins instead of one chosen bin.

I also need to be precise about implementation, because small constants change the actual policy.
The operation set I land on is the 14-operation RandAugment-style set: identity, auto-contrast,
equalize, rotate, solarize, color, posterize, contrast, brightness, sharpness, shear-x, shear-y,
translate-x, and translate-y. For the wide variant in torchvision, shear uses 31 values from
`0` to `0.99`, translation uses `0` to `32` pixels, rotation uses `0` to `135` degrees, solarize
uses threshold `255` down to `0`, and posterize uses retained bits from `8` down to `2`. The four
enhancers use magnitudes from `0` to `0.99` and then apply factor `1 + magnitude`; after the fair
sign flip this gives factors from `0.01` to `1.99`. The wide-space table rounds the enhancer
range as `0.01` to `2.0`, and the original AutoML one-file code implements that direct factor
range, but the torchvision landing uses the signed `0.99` representation. Since the scaffold uses
torchvision, I align the landing code to torchvision.

There is another implementation trap in shear. A raw PIL affine matrix can put the shear factor
directly into the off-diagonal entry, but torchvision's `F.affine` expects shear angles in
degrees. The canonical torchvision code therefore passes `degrees(atan(magnitude))`, so the
tangent of the angle is the intended shear factor. Translation is cast to integer pixels.
Posterize uses `8 - round(arange(num_bins) / ((num_bins - 1) / 6))`, which is `8` down to `2` for
31 bins. Scalar operations such as identity, auto-contrast, and equalize ignore the sampled
strength. Those details are not cosmetic; they define the distribution the training loop sees.

What remains is a parameter-free augmentation rule whose logic follows the evidence in order.
Search is expensive and proxy search is miscalibrated. Random strength already works about as
well as a tuned constant in the relevant diagnostic. The orbit-averaging argument asks for
diverse in-class coverage, not a finely optimized policy distribution. Removing unsupported
choices leaves one operation, no separate drop probability, and one fresh discrete strength per
image. The result fills the augmentation slot with a uniform finite mixture over the standard
operation-strength surface.
