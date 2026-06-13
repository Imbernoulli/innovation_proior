Let me start from what actually costs me when I want good data augmentation for an image classifier, because the pain is concrete. I have a fixed model, a fixed optimizer, a fixed schedule, and a set of label-preserving image operations — rotate, shear, translate, color, contrast, brightness, sharpness, solarize, posterize, equalize, auto-contrast, identity — each a function `a(x, m)` that distorts an image `x` by some strength `m` drawn from `{0, ..., 30}`. The thing I'm supposed to produce is a *policy*: a distribution over which of these to apply, with what probability, at what strength, and how many to compose per image. And the whole field's answer to "what policy?" is: search for one. Train a controller with reinforcement learning, or run a grid search, or a Bayesian density match, or an evolutionary population — until you've found the policy that maximizes validation accuracy for this exact model and dataset. That search is the expense. The reinforcement-learning route of Cubuk, Zoph, Mané, Vasudevan and Le — AutoAugment — spends over half a GPU-year on CIFAR-10 before the real training starts: an RNN controller, trained by PPO, emitting policies of five sub-policies, each two operations, each operation tagged with an application probability and a magnitude, a space of order `(16·10·11)^10 ≈ 2.9×10^32`, sampled fifteen thousand times, the reward being a child model's validation accuracy. Even the cheap descendants don't escape it: RandAugment cuts the policy down to two scalars but still grid-searches them, up to eighty trainings per task. The question that won't leave me alone is whether all this search is actually earning its keep, or whether it's elaborate machinery sitting on top of something that doesn't need optimizing at all.

So let me interrogate the search, not assume it. What is the search actually buying? It claims to fit the policy to the model and dataset. But there's a measurement that undercuts that claim. If I take a single global distortion magnitude and sweep it on CIFAR-10 across Wide-ResNet widths and across training-set sizes, the *best* magnitude isn't a fixed property of the task — it climbs monotonically with model size and with dataset size. Larger nets and larger datasets want stronger augmentation. That's damning for the standard cost-saving trick, which searches the policy on a small proxy model and a reduced dataset and then transfers it to the full target: the proxy is systematically *under*-strengthened relative to the target, so the searched magnitude is calibrated to the wrong point. The search isn't finding the right answer for the task I care about; it's finding the right answer for a smaller surrogate and hoping it carries over. So one of the two things the search sells me — task-specific fitness — is partly an illusion when it's run on a proxy, which it usually is for cost reasons.

And here's the observation that really cracks it open. When the magnitude *schedule* over training was studied — hold it constant, ramp it linearly, sample it uniformly at random per step between two bounds, sample it with a rising upper bound — the random one did as well as the best of them, edging out the constant schedule, and only a handful of distinct magnitudes were needed at all; finely resolving the strength bought essentially nothing. Sit with that. A *random* magnitude, no schedule, no tuning, matched the carefully chosen constant. And yet the field's response was to keep the magnitude *fixed* and tune it — because a single constant `M` is one clean hyperparameter you can grid-search, whereas "random" felt like giving something up. The whole apparatus of searching for the right `M` was preserved to protect a knob that the data was already saying you don't need. If random ties tuned, then the reason to fix `M` is bookkeeping, not performance.

Now let me bring in the only argument I have for *why* augmentation helps at all, because if I'm going to throw away the search I need to know I'm not throwing away the benefit. Training minimizes empirical risk `(1/n) Σ_i l(f_θ(x_i), y_i)` as a proxy for the expected risk over the true data distribution. When I augment, I replace each `x_i` by a random transform `t(x_i)` drawn from a set `T` of approximately label-preserving operations, and the objective becomes `(1/n) Σ_i E_{t∼T}[ l(f_θ(t(x_i)), y_i) ]` — for each example, the loss *averaged over the transforms applied to it*. Think of `T` as acting on the data so that the augmented versions of `x_i` trace out an orbit, and if `T` really is label-preserving that whole orbit stays inside the class. What does averaging the loss over the orbit do? It's an average of a noisy quantity, so it cuts variance — the variance of the per-example loss and, more to the point, of the gradient the optimizer follows. If the loss swings wildly as the image moves along its orbit, averaging across the orbit denoises the gradient. That's the mechanism: augmentation tightens the gap between the empirical risk I can compute and the expected risk I actually want, by reducing the variance of my estimate of the gradient.

Stare at what that argument needs and, crucially, what it does *not* need. It needs `T` to be approximately label-preserving, so the orbit stays in-class. It needs me to *cover* the orbit — to actually average over a diverse set of transforms, because a degenerate `T` that always applies the same transform averages nothing and cuts no variance. What it does *not* need is for the sampling distribution over `T` to be finely optimized. The variance reduction comes from averaging over a broad, in-class orbit; it doesn't care whether I weight transform A at 0.31 and transform B at 0.52 the way a learned policy would. So the theory is quietly telling me that *diversity of coverage* is the active ingredient, and *policy fitness* is not load-bearing. That lines right up with the empirical cracks: random magnitude tied tuned magnitude; a proxy-searched policy is mis-fit anyway; a few magnitudes suffice. LingChen and colleagues already pushed on exactly this with UniformAugment — drop the search entirely, sample transforms uniformly over `T`, on the grounds that if `T` is approximately invariant then uniform coverage is about as good as a searched policy. I think that's the right instinct. But I want to take it further, because UniformAugment still carries a pile of design decisions that, when I check them against this same argument, I can't justify.

Let me lay UniformAugment out and audit it knob by knob. It applies `N = 2` operations per image; each operation is drawn uniformly from `A`; each is applied with a probability `p ∼ Uniform(0, 1)`; each at a magnitude `λ ∼ Uniform(0, 1)` over the *continuous* strength range. Four choices there: combine two ops, the per-op application probability, the continuous strength, and the uniform draw. The uniform draw is the one I keep — it's the whole point, it's what the invariance argument endorses. The other three I want to put on trial.

Why two operations? Combining operations is what lets the earlier methods build elaborate composite distributions — chain a shear and a solarize and you get a richer, weirder distribution than either alone. But the variance argument doesn't ask for richness of the *composite*; it asks for coverage of an in-class orbit. And there's a cleaner way to think about coverage when I apply exactly *one* operation per image. If each image gets a single op `a` drawn uniformly from `A`, then the distribution of my augmented dataset is just the mixture `(1/|A|) Σ_a (distribution of a applied to the whole dataset)` — a plain, uniform average of the per-operation distributions. No composition, no stochastic chaining into a complex shape; the simplest possible cover of `A`, the literal arithmetic mean of the single-operation distributions. Picture a two-class dataset, crosses and circles, with a few deterministic augmentations of the cross class scattered around each cross; applying one op uniformly just samples uniformly from all those scattered crosses, fattening the class cloud evenly. That's coverage in its most honest form. And the empirical hints agree: when the global magnitude work looked at the *number* of composed operations, `N = 1` came out best on at least one Wide-ResNet/CIFAR-100 benchmark; and stacking more operations risks over-distorting an image past the label-preserving boundary, which the variance argument warns against because it breaks the in-class assumption that the whole thing rests on. So combining buys complexity and a knob — *how many to chain* — that I have no principled value for. Drop to one operation per image.

The moment I commit to one operation per image, the application probability `p` evaporates on its own. With two or more chained ops you need a way to sometimes *not* apply one of them, so each carries a probability. But with a single op, "don't apply it" is just "apply the identity." And identity is already an operation sitting in `A`. So I don't need a separate probability mechanism at all — sampling the identity *is* the no-op event, folded into the same uniform draw over `A` as everything else. One fewer moving part, and it isn't even a loss: the probability of a no-op is now exactly `1/|A|`, set by the geometry of the augmentation set rather than by a hyperparameter I'd otherwise have to pick. That's strictly cleaner.

Now the continuous strength `λ ∼ Uniform(0, 1)`. Two reasons to discretize instead. First, every prior method — AutoAugment, RandAugment, all of them — works on the discrete strength set `{0, ..., 30}`, and the diagnostic finding was that *a few* distinct magnitudes already capture the benefit; finely resolving strength changed almost nothing. Second, and this is UniformAugment's own observation turned against its own choice: discretizing the strength space *reduces augmentation variance* by limiting the range of possible distortions to a finite set, which is a feature, not a bug — I want enough diversity to cover the orbit, but unbounded continuous jitter just adds variance without adding coverage I can use. So discretize to the same `{0, ..., 30}` everyone else uses. And then, the same way I argued for `M`: don't fix the strength, *sample it per image*, uniformly from `{0, ..., 30}`. This is the random-magnitude finding cashed in — the thing that tied tuned `M` and was discarded only to protect a knob. Sampling the strength per image gives me, for free and across the dataset, a *mixture of weak and strong augmentations*: some images barely touched, some heavily distorted, the full spread. That mixture is exactly the broad orbit coverage the variance argument wants, and it's why fixing a single `M` was always slightly wrong — a fixed `M` covers only one shell of the orbit, while a per-image random `m` covers all of them.

So watch what's left after I've stripped every choice I couldn't defend. For each image: draw one operation `a` uniformly from `A`; draw one strength `m` uniformly from `{0, ..., 30}`; return `a(x, m)`. That's it. No `N` to choose, no per-op probability, no global magnitude to tune, no continuous range to bound, no search of any kind — there is nothing left to tune. Every removed knob was removed for a reason traceable back to the variance-reduction argument (coverage, not policy fitness, is what helps) or to the measured facts about the prior systems (random magnitude ties tuned; a proxy search is mis-calibrated; a few strengths suffice). The simplicity isn't a stylistic preference; it's what remains once I delete everything the benefit didn't depend on. I'll call the thing TrivialAugment because it really is trivial — almost embarrassingly so for something meant to stand next to a GPU-year of reinforcement learning.

I should be careful to state precisely what TrivialAugment is *not*, because there's a tempting misreading. It is not a special case of RandAugment with `N = 1`. RandAugment with `N = 1` still uses a *fixed* magnitude `M` chosen once for the whole run and applied to every image; TrivialAugment samples a *fresh* magnitude per image. The fixed-vs-per-image-random magnitude is the whole difference, and it's the difference between covering one shell of the orbit and covering all of them — the same distinction that made random tie tuned in the schedule study. So even though both reduce to "one op per image," the strength behavior is genuinely different, and it's the per-image randomness that the conclusion hinges on.

Two things still need pinning down before this is real code: the augmentation set `A`, and the details of how an operation consumes its strength. For `A` I take the fourteen operations that the prior methods converged on — identity, auto-contrast, equalize, rotate, solarize, color, posterize, contrast, brightness, sharpness, shear-x, shear-y, translate-x, translate-y. Geometric ones (rotate, shear, translate) and photometric ones (the four enhancers, plus solarize/posterize/equalize/auto-contrast), and identity for the no-op. The geometric ops and the enhancers are *signed*: a rotation can go clockwise or counter-clockwise, a shear or translation either direction, an enhancer can push the attribute above or below the original. The prior ops handle this by flipping the sign with probability one half, which doubles the effective diversity of those operations at no extra cost — and more diversity is exactly what I want, so I keep the coin flip. There's a choice of *range* for each operation's strength, and here a wider range pairs naturally with per-image random sampling. With a fixed `M` a wide range is dangerous — pick `M` too high and every image is overcooked — but with a random per-image `m`, a wide range just means the *spread* of strengths is larger, more genuinely-weak and genuinely-strong images in the mix, without committing every image to the extreme. So I'll use the wide ranges: shear up to `0.99`, translate up to `32` pixels, rotate up to `135°`, posterize down to `2` bits, the enhancers spanning roughly `0.01` to `2.0`, solarize across the full `0` to `255`. The strength `m ∈ {0, ..., 30}` indexes linearly into each operation's range — strength `0` is the gentlest end (often a near-no-op), strength `30` the strongest.

For how an operation reads its strength, I'll mirror the standard image-library implementations so this is faithful and not invented. With `num_bins = 31` magnitude bins indexed by `m`, each operation maps `m` to a value on its range. ShearX/Y: a shear factor linearly from `0` to `0.99`, fed straight into the off-diagonal slot of an affine transform. TranslateX/Y: a pixel offset linearly from `0` to `32`, fed into the affine transform's translation slot. Rotate: degrees linearly from `0` to `135`. The enhancers — Brightness, Color (saturation), Contrast, Sharpness — a factor linearly from `0` to `0.99`, applied as `enhance(1.0 + magnitude)` so that magnitude `0` is the identity and the signed coin flip pushes the factor above or below one. Posterize: the number of retained bits stepping from `8` down to `2` as `m` rises (fewer bits = stronger). Solarize: the threshold sweeping from `255` (no-op) down to `0` (invert everything) as `m` rises. AutoContrast, Equalize, Identity: strength-free, they ignore `m`. The signed operations — shear, translate, rotate, and the four enhancers — flip their magnitude's sign with probability one half.

Let me write the contribution as the strategy that fills the one open slot in the pipeline — a callable that takes one PIL image and returns one augmented PIL image — grounded in how the standard libraries actually do each operation:

```python
import random
from PIL import Image, ImageOps, ImageEnhance


def _apply_op(img, op_name, magnitude):
    # Each primitive op reads a single scalar magnitude; signed ops have already had
    # their sign chosen by the caller. Mirrors the standard PIL-based implementations.
    if op_name == "Identity":
        return img
    if op_name == "ShearX":
        return img.transform(img.size, Image.AFFINE, (1, magnitude, 0, 0, 1, 0))
    if op_name == "ShearY":
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, magnitude, 1, 0))
    if op_name == "TranslateX":
        return img.transform(img.size, Image.AFFINE, (1, 0, magnitude, 0, 1, 0))
    if op_name == "TranslateY":
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude))
    if op_name == "Rotate":
        return img.rotate(magnitude)
    if op_name == "Brightness":
        return ImageEnhance.Brightness(img).enhance(1.0 + magnitude)
    if op_name == "Color":
        return ImageEnhance.Color(img).enhance(1.0 + magnitude)
    if op_name == "Contrast":
        return ImageEnhance.Contrast(img).enhance(1.0 + magnitude)
    if op_name == "Sharpness":
        return ImageEnhance.Sharpness(img).enhance(1.0 + magnitude)
    if op_name == "Posterize":
        return ImageOps.posterize(img, int(magnitude))
    if op_name == "Solarize":
        return ImageOps.solarize(img, magnitude)
    if op_name == "AutoContrast":
        return ImageOps.autocontrast(img)
    if op_name == "Equalize":
        return ImageOps.equalize(img)
    raise ValueError(op_name)


class TrainAugmentation:
    """One operation per image, drawn uniformly from A, at a strength drawn uniformly
    from {0, ..., num_bins-1}; signed ops flip direction with probability 1/2.
    No N, no probability, no global magnitude, no search -- nothing to tune."""

    def __init__(self, num_bins=31):
        self.num_bins = num_bins  # 31 discrete strengths, the {0, ..., 30} of prior work

    def _augmentation_space(self, n):
        # op_name -> (per-bin magnitudes over the WIDE range, is the op signed)
        idx = [b for b in range(n)]
        lin = lambda lo, hi: [lo + (hi - lo) * b / (n - 1) for b in range(n)]
        return {
            "Identity":     ([0.0] * n,                     False),
            "ShearX":       (lin(0.0, 0.99),                True),
            "ShearY":       (lin(0.0, 0.99),                True),
            "TranslateX":   (lin(0.0, 32.0),               True),
            "TranslateY":   (lin(0.0, 32.0),               True),
            "Rotate":       (lin(0.0, 135.0),              True),
            "Brightness":   (lin(0.0, 0.99),                True),
            "Color":        (lin(0.0, 0.99),                True),
            "Contrast":     (lin(0.0, 0.99),                True),
            "Sharpness":    (lin(0.0, 0.99),                True),
            # fewer bits = stronger: 8 bits at m=0 down to 2 bits at m=30
            "Posterize":    ([int(round(8 - b / ((n - 1) / 6.0))) for b in idx], False),
            # threshold 255 (no-op) at m=0 down to 0 (full invert) at m=30
            "Solarize":     (lin(255.0, 0.0),              False),
            "AutoContrast": ([0.0] * n,                     False),
            "Equalize":     ([0.0] * n,                     False),
        }

    def __call__(self, img):
        space = self._augmentation_space(self.num_bins)          # the set A
        op_name = random.choice(list(space.keys()))              # one op, uniform over A
        magnitudes, signed = space[op_name]
        m = random.randint(0, self.num_bins - 1)                 # one strength, uniform
        magnitude = magnitudes[m]
        if signed and random.random() < 0.5:                     # signed ops: coin-flip direction
            magnitude = -magnitude
        return _apply_op(img, op_name, magnitude)
```

And the pipeline this drops into is the fixed one — the designed augmentation at the front, then the standard 32×32 geometric wrappers, then the mandatory tensor-and-normalize tail:

```python
from torchvision import transforms


def build_train_transform(config):
    return transforms.Compose([
        TrainAugmentation(num_bins=31),                       # one random op + random strength
        transforms.RandomCrop(config['img_size'], padding=4), # standard geometric wrapper
        transforms.RandomHorizontalFlip(),                    # standard geometric wrapper
        transforms.ToTensor(),
        transforms.Normalize(config['mean'], config['std']),
    ])
```

Let me trace the causal chain back, because I want to be sure the simplicity is earned and not just stumbled into. I started with the cost: every automatic augmentation method searches the policy space, and the search runs from one extra training to half a GPU-year. Two measured facts undercut the search — the optimal strength tracks model and dataset size, so a proxy-task search is calibrated to the wrong point; and a *random* per-step magnitude tied a tuned constant, the random option discarded only to keep a single tunable knob. The one theory I have for why augmentation works — averaging the loss over each image's label-preserving orbit reduces gradient variance and tightens the empirical-to-expected-risk gap — says the active ingredient is *coverage of a diverse in-class orbit*, not the fine fitness of the sampling distribution, which is precisely why a searched policy isn't earning its cost. UniformAugment already drew the search-free conclusion, but kept `N = 2`, a per-op application probability, and a continuous strength; auditing each against the coverage argument, I dropped to one op per image (coverage is the mean of single-op distributions, the simplest honest cover, and stacking risks leaving the in-class orbit), which folded the application probability into simply having identity in the set, and I discretized the strength to `{0, ..., 30}` (a few strengths suffice; discretizing reduces useless variance) and sampled it *per image* rather than fixing it (the random-magnitude finding, which spreads weak and strong augmentations across the dataset to cover the whole orbit instead of one shell). What's left has no hyperparameters, isn't a special case of RandAugment because the strength is fresh per image, and is one short PIL-based file: sample one operation uniformly, sample one strength uniformly, flip the sign if the op is signed, apply, return.
