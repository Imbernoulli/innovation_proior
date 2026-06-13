# Context: automatic image-augmentation policies (circa 2018-2021)

## Research question

Data augmentation is one of the few regularizers in deep image classification that buys
generalization at zero inference cost: apply label-preserving transformations to each training
image, and the network learns the invariances those transformations encode (a small rotation
should not change the class; a color shift should not either). The catch is that *which*
transformations to apply, *how strongly*, *with what probability*, and *how many to chain*
together is a large design space, and the right answer differs across models and datasets. The
field's response has been to *search* this space — to learn, per model and per dataset, an
augmentation policy that maximizes validation accuracy. That search is the cost center: the
strongest methods spend from one extra training run up to more than half a GPU-year of compute
before the real training even begins, and they bolt a separate, hard-to-reproduce optimization
phase onto an otherwise standard pipeline.

The precise problem: design an augmentation strategy for image classification that (1) matches
the accuracy of these searched policies across architectures and datasets, (2) needs **no search
phase** and (3) **no per-task hyperparameter tuning** — ideally no tunable knobs at all — so that
it transfers to a new model or dataset out of the box, and (4) is trivial to reimplement, since
reproducibility across augmentation papers is poor. The question is whether the search machinery
is actually earning its cost, or whether something far simpler captures the same benefit.

## Background

The augmentation space everyone works in. All the automated methods of this period share one
structure: an augmentation space is a set of prespecified operations `A` (rotate, shear,
translate-x/y, color, contrast, brightness, sharpness, solarize, posterize, equalize,
auto-contrast, identity, and on some spaces invert/cutout/flip) together with a discrete set of
strength settings, conventionally the integers `{0, ..., 30}`. An operation is a function
`a(x, m)` mapping an image `x` and a strength `m` to an augmented image; most operations use `m`
to set how hard they distort (degrees of rotation, shear factor, color-enhance factor), a few
ignore it (equalize, auto-contrast, identity). Several operations are *signed* — rotation, shear,
and translation can go either direction, and the color/contrast/brightness/sharpness enhancers
can push above or below the original — so the direction is itself chosen by a coin flip. A
"policy" is then some distribution over which operations to draw from `A`, with what probability
each is applied, at what strength, and how many to compose per image.

Why augmentation helps, stated as theory available at the time. Training minimizes the empirical
risk `(1/n) Σ_i l(f_θ(x_i), y_i)` as a stand-in for the inaccessible expected risk over the true
data distribution. Augmentation replaces each example by a random transform `t(x_i)` drawn from a
set `T` of approximately label-preserving transforms. Viewed through group theory, `T` acts on
the data so that for an approximately invariant `T` the augmented points `t(x_i)` form an *orbit*
of `x_i` that stays inside the class. The augmented objective becomes
`(1/n) Σ_i ∫ l(f_θ(t(x_i)), y_i) dT(t)` — for each example, the loss is *averaged over its
orbit*. A line of analysis (Chen et al. 2020, on the group-invariance view of augmentation)
shows that this averaging over `T` reduces the variance of the loss and of the gradient estimate,
which tightens the gap between empirical and expected risk; intuitively, if the loss swings a lot
as an image moves along its augmentation orbit, averaging across the orbit denoises the gradient
the optimizer follows. The consequence that matters here: the benefit comes from *covering the
orbit with diversity*, and if `T` is genuinely close to label-preserving, sampling transforms
broadly should help — the analysis does not, by itself, require that the sampling distribution be
finely *optimized*.

Diagnostic findings about searched policies (Cubuk et al. 2020). Two measured observations about
existing automated methods set up the problem. First, the *optimal* augmentation strength is not a
fixed property of a task: sweeping a single global distortion magnitude on CIFAR-10 across
Wide-ResNet widths and across training-set sizes shows the best magnitude rises monotonically
with both model size and dataset size. So a policy whose strength is tuned on a small *proxy*
model-and-dataset (the standard cost-saving trick) is systematically mis-strengthened for the
larger target — the searched magnitude is chasing a moving target. Second, when the schedule of
the global magnitude over training was varied — held constant, ramped linearly, sampled uniformly
at random per step between two bounds, or sampled with a rising upper bound — the *random* schedule
performed as well as the best of the others (it tied the top, slightly ahead of the constant
schedule) on a Wide-ResNet-28-10 / CIFAR-10 sweep. And only a handful of distinct magnitudes were
needed to get good results; finely resolving the magnitude bought little. These are facts about
the prior systems, observable before any new method exists.

Reproducibility pain. Across these papers, published material is uneven: some release policies but
no training code, some no code at all, some only partial. Comparisons mix different training
pipelines, epoch budgets, and augmentation spaces, so reported gaps conflate the method with
setup details. A method that is one short, dependency-light file would sidestep much of this.

## Baselines

These are the prior augmentation methods a new strategy would be measured against and reacts to.
They are ordered, as the field orders them, by search cost.

**AutoAugment (AA) — Cubuk et al., CVPR 2019.** The first automated-augmentation method and the
most expensive. A recurrent-network controller, trained by reinforcement learning (PPO), emits an
augmentation policy; the reward is the validation accuracy of a child model trained from scratch
under that policy. A policy is 5 sub-policies, each a sequence of 2 operations, and each operation
carries two extra parameters — a probability of being applied and a discrete magnitude. With 16
operations, magnitudes discretized to 10 values and probabilities to 11, the search space is on
the order of `(16·10·11)^10 ≈ 2.9×10^32` policies, and roughly 15,000 are sampled. To afford this
the search runs on a *reduced proxy* — a small Wide-ResNet-40-2 on ~4,000 CIFAR-10 images — and
the found policy is transferred to the full model and dataset; the CIFAR run alone costs over half
a GPU-year. The standard CIFAR pipeline AA established is also inherited downstream: standardize,
horizontal flip with probability 0.5, zero-pad and random crop, then the learned policy, then a
final 16×16 Cutout. **Gap:** the search is enormous, and it leans on the assumption that a policy
optimal on a small proxy is optimal on the real task — an assumption later measurements about
strength-vs-size call into question.

**RandAugment (RA) — Cubuk et al., 2020.** A drastic simplification built to delete the separate
search phase by folding the augmentation parameters into the ordinary training hyperparameters.
It throws away the learned per-operation probabilities entirely: for each image it picks `N`
operations uniformly at random from `A` (each with probability `1/|A|`) and applies them in
sequence, all at a *single global magnitude* `M` shared across operations. That leaves exactly two
human-interpretable scalars, `N ∈ {1, 2, 3}` and `M ∈ {0, ..., 30}`. In two lines:

```python
transforms = [Identity, AutoContrast, Equalize, Rotate, Solarize, Color,
              Posterize, Contrast, Brightness, Sharpness, ShearX, ShearY,
              TranslateX, TranslateY]          # K = 14 operations
def randaugment(N, M):
    ops = np.random.choice(transforms, N)
    return [(op, M) for op in ops]
```

Because `N` and `M` are tuned directly on the target task rather than a proxy, RA matches or
beats the costlier learned methods. **Gap:** the two scalars are still found by *grid search* —
sweeping the candidate `(N, M)` settings means up to ~80 full trainings per task — so the search
phase is reduced, not removed; and the magnitude `M` is a *single value fixed for the whole run*,
chosen per task, not varied across images. The method's own diagnostics (see Background) sit in
tension with this last choice: a per-step random magnitude was measured to do as well as a tuned
constant `M`, yet `M` was kept fixed precisely to preserve a single tunable knob.

**UniformAugment (UA) — LingChen et al., 2020.** The first to drop the search phase outright,
justified by the orbit-variance argument above: if `T` is approximately distribution-invariant,
then sampling transforms *uniformly* over `T` covers the orbit and should be about as effective
as any searched policy, so no optimization is needed. Concretely UA fixes `N = 2`, and for each of
the two operations samples a probability of application `p ~ Uniform(0, 1)` and a magnitude
`λ ~ Uniform(0, 1)` over the *continuous* strength range. **Gap:** "search-free" still leaves
several un-argued design choices baked in — why two operations, why a continuous strength range
when its own analysis notes that discretizing reduces augmentation variance, and why an
independent application probability per operation. Each is a lever that was set rather than
derived, and the continuous-`λ`, drop-probability, two-op construction is more elaborate than the
invariance argument demands.

**Cutout — DeVries & Taylor, 2017.** A single masking operation: zero out one contiguous square
region (16×16 on CIFAR-style 32×32 inputs) at a random location, read as a "spatial prior to
dropout in input space" that forces the network to use the whole image rather than a few
discriminative patches. **Gap:** it is one fixed transform, complementary to the augmentation
strategies above rather than a competitor — on CIFAR it is conventionally appended *after* the
chosen policy as the last step of the pipeline.

Also in the landscape, named for completeness: Fast AutoAugment (Bayesian density-matching search,
about one extra training), Population-Based Augmentation (an online evolutionary schedule across
parallel workers), Adversarial AutoAugment (online RL with 8× batch augmentation), and Mixup /
AugMix (convex image mixing for robustness). All but Cutout share the "learn or sample a policy
over `A × strengths`" framing.

## Evaluation settings

The natural yardsticks already in use at the time, all pre-method facts (the datasets, models, and
metric existed before any new strategy):

- **CIFAR-10 and CIFAR-100** (Krizhevsky 2009): 50K training images, 32×32, 10 and 100 classes.
  Standard models are Wide-ResNet-40-2 and Wide-ResNet-28-10, and Shake-Shake-26-2×96d. The
  established CIFAR pipeline (from AA) wraps the augmentation strategy with horizontal flip,
  pad-and-crop, and a final 16×16 Cutout, then normalizes by training-set channel mean and std.
- **SVHN** (street-view house numbers): a 73K-image core set plus 531K extra images; Wide-ResNet-
  28-10. Color-based augmentations dominate; a final 16×16 Cutout is used, without the extra
  geometric augmentations.
- **ImageNet**: 1.2M images, 1000 classes; ResNet-50, with randomly-resized crop, horizontal flip,
  color jitter, and lighting noise as the surrounding pipeline.
- The smaller-scale task family of interest here mirrors these: ResNet-style classifiers on
  CIFAR-10 / CIFAR-100 (and a Fashion/grayscale variant), 32×32 inputs, SGD with momentum,
  cosine-annealed learning rate over a fixed epoch budget, all held fixed so that only the
  training-time augmentation varies.
- Protocol: identical training pipeline across the augmentation methods being compared; many runs
  per setting with confidence intervals, since the accuracy differences are small. The metric is
  best test accuracy (%, higher is better).

## Code framework

The augmentation strategy plugs into a fixed torchvision pipeline: a single
`build_train_transform(config)` returns a `transforms.Compose([...])` of PIL/tensor transforms
that runs on each training image. The surrounding pieces already exist — the standard 32×32
geometric wrappers (`RandomCrop` with 4-pixel padding, `RandomHorizontalFlip`), and the
two mandatory tail steps `ToTensor()` and `Normalize(mean, std)` that hand a normalized tensor to
the model. The model, optimizer (SGD with momentum, weight decay), cosine schedule, and training
loop are all fixed and outside this function. The only open slot is the augmentation transform
that goes at the front of the pipeline: a callable that takes one PIL image and returns one
augmented PIL image.

```python
import random
from torchvision import transforms
from PIL import Image, ImageOps, ImageEnhance


class TrainAugmentation:
    """Per-image training augmentation: maps one PIL image to one augmented PIL image.

    The set of primitive operations on (A x strengths) already exists in the field --
    geometric ops (rotate, shear-x/y, translate-x/y), photometric ops (color, contrast,
    brightness, sharpness, solarize, posterize, equalize, auto-contrast), and identity,
    each a function op(image, strength). The open question is the *strategy* that turns
    that primitive set into a per-image augmentation: which op(s) to apply, at what
    strength, with what probability, how many to compose. That strategy is the design slot.
    """

    def __init__(self):
        # any configuration the strategy turns out to need would be set here
        pass

    def __call__(self, img):
        # TODO: the augmentation strategy we will design.
        #       Given the primitive operation set on (A x strengths), decide how to
        #       produce one augmented image from the input image and return it.
        raise NotImplementedError


def build_train_transform(config):
    """Fixed pipeline: the designed augmentation, then the standard 32x32 wrappers,
    then the mandatory ToTensor + Normalize tail."""
    return transforms.Compose([
        TrainAugmentation(),                                  # the strategy slot
        transforms.RandomCrop(config['img_size'], padding=4), # existing geometric wrapper
        transforms.RandomHorizontalFlip(),                    # existing geometric wrapper
        transforms.ToTensor(),                                # required
        transforms.Normalize(config['mean'], config['std']),  # required
    ])
```

The strategy `__call__` is the single empty slot; the rest of the pipeline is fixed and shared
with the baselines.
