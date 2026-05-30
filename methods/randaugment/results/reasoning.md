Let me start from the thing that's actually slowing everyone down. Learned data augmentation works — AutoAugment-style policies give real accuracy gains and cost nothing at inference. But to get one I have to run a *separate search*: stand up an RNN controller, train it with reinforcement learning (PPO) to propose augmentation policies, train a child network for each proposal, reward the controller by the child's validation accuracy, and do all of this on a small *proxy* task — a reduced dataset and a small model — then transfer the winning policy to the real task. That's a whole second optimization loop bolted onto training, expensive and fiddly. I want the accuracy without that loop.

Before I try to remove it, let me check whether the proxy step is even *sound* for augmentation, because if it's not, that's a much stronger reason to kill it than mere cost. The proxy assumption is: a policy that's good on a small model / small dataset is good on the big target. For architecture search that's plausible. For augmentation? Think about what augmentation *does* — it's regularization, and the right amount of regularization depends on capacity and data. A big model on a big dataset can absorb — and wants — *more* aggressive augmentation than a tiny model on a sliver of data. So the optimal augmentation *strength* should grow with model size and dataset size. If that's true, then a policy whose strength was pinned down on a small proxy is *systematically* mis-set when I move to the large target, and the search gives me no handle to re-tune it. So the proxy isn't just expensive — it's structurally the wrong place to fix the one thing (strength) that most depends on the target. That settles it: I don't want to search on a proxy at all. I want the augmentation settings to be ordinary training hyperparameters, tuned directly on the target task alongside learning rate and weight decay.

The obstacle to that is the search space. AutoAugment's policy is enormous: 5 sub-policies, each two operations in sequence, each operation a (type out of ~16 transforms, probability out of 11 discrete values, magnitude out of 10 discrete values). That's about (16 × 10 × 11)^10 ≈ 3 × 10^32 policies, 30-plus parameters. You can't grid-search that on the target; you *need* a learned controller, which is exactly the loop I'm trying to delete. So to remove the search I first have to make the space small enough that no search is needed. The question becomes: how much of that 10^32 is actually carrying the benefit, and how much can I throw away?

Here's the lever. The reported reason learned policies help is *diversity* — the network sees many different transformed versions of each image. It is not, as far as anyone has shown, the specific learned *probabilities* of each operation; those are just the knobs the controller happened to twiddle to maximize diversity-and-accuracy. So what if I stop learning the probabilities entirely and just *force* diversity by construction? Instead of a learned probability per transform, sample transforms *uniformly at random*. At each application, pick one of the K = 14 transforms with probability 1/K — parameter-free, no probabilities to learn or search. Apply N of them in sequence to each image. That still gives K^N possible operation combinations per image, plenty of diversity, but now there is *nothing to learn* about which to pick. The entire probability dimension of the search space — the bulk of those 10^32 — is gone, replaced by a coin flip.

Now the magnitudes. AutoAugment searches a separate magnitude for each transform, each on an integer scale 0 to 10. Do I need per-transform magnitudes? Two clues say no. First, the population-based work noticed that the optimal magnitudes tend to move *together* over training — they rise on a shared schedule rather than each doing its own thing, which means they're not carrying much independent information. Second, I can test the sensitivity directly: fix every transform's magnitude at a shared value, then vary just *one* transform's magnitude across its whole range and watch the accuracy. If per-transform tuning mattered, that should swing accuracy a lot. It barely moves it — on the order of a tenth of a percent. So tying *all* the magnitudes to a single global value M loses almost nothing. That collapses the entire magnitude dimension to one number.

So I'm left with exactly two knobs: N, how many transforms I apply per image, and M, the single magnitude shared by all of them. Both are monotone in regularization strength — crank either up and the augmentation gets more aggressive — so they're human-interpretable, which is exactly what I wanted for tuning strength to the target model and dataset. The whole algorithm is: sample N transforms uniformly from the list, apply each at magnitude M.

```python
transforms = [
    'Identity', 'AutoContrast', 'Equalize',
    'Rotate', 'Solarize', 'Color', 'Posterize',
    'Contrast', 'Brightness', 'Sharpness',
    'ShearX', 'ShearY', 'TranslateX', 'TranslateY',
]

def randaugment(N, M):
    sampled_ops = np.random.choice(transforms, N)   # uniform, 1/K each
    return [(op, M) for op in sampled_ops]           # all share the one magnitude
```

The search space is now K^N — for K = 14 and N = 2 or 3 that's a few hundred to a few thousand — against AutoAugment's ~10^32. That's a reduction of more than thirty orders of magnitude, and it's the whole point: K^N is small enough that I don't need a controller at all. Plain *grid search* over N and M on the *target* task finds a policy that matches or beats the proxy-searched ones, and N and M just fold into the normal hyperparameter sweep. No separate phase, no proxy, no RL.

One loose end: M is a single number, but should it be *constant* over training, or should it follow a schedule — the population-based result did say magnitudes rise during training, so maybe M should ramp up. Let me lay out the candidates: a constant M; a random M sampled uniformly between two bounds each step; a magnitude that increases linearly over training; and a random M whose upper bound increases over training. I'd run all four (say on CIFAR-10 with a Wide-ResNet) and compare. They come out essentially tied — within noise of each other. Given a tie, take the option with the fewest hyperparameters, which is the constant: one number, M, fixed for the whole run. The ramp and the random-range schemes each add bounds I'd have to set, for no measured gain. So M is just a constant.

Now the magnitudes have to actually mean something per transform, since "M = 9" has to turn into a rotation angle, a shear, a solarize threshold, and so on. Keep AutoAugment's linear scale with PARAMETER_MAX = 10: a level on 0–10 maps to each transform's native parameter by level/10 times that transform's maximum. So rotation is ±(M/10)·30°, shear is (M/10)·0.3, translation is (M/10)·(max pixels), solarize inverts pixels above a threshold int((M/10)·256), posterize keeps int((M/10)·4) bits, and the photometric ops — color, contrast, brightness, sharpness — blend by (M/10)·1.8 + 0.1 (so 0.1 to 1.9× enhancement). AutoContrast, Equalize, and Identity ignore M. Larger M means a larger angle, a harsher shear, a stronger color shift — strength rising monotonically, as promised. (Nothing forces M ≤ 10, either: a very large model on large images can use M of 17 or 28, which is just the same linear scale extrapolated — exactly the strength-scales-with-model-size adjustment the proxy approach couldn't make.)

```python
import numpy as np
import random

# the 14 transforms; magnitude m is on the 0..10 (PARAMETER_MAX) scale
PARAMETER_MAX = 10

def _float_param(level, maxval):
    return float(level) * maxval / PARAMETER_MAX

def _int_param(level, maxval):
    return int(level * maxval / PARAMETER_MAX)

def apply_op(img, name, m):
    if name == 'Identity':     return img
    if name == 'AutoContrast': return auto_contrast(img)          # magnitude-free
    if name == 'Equalize':     return equalize(img)               # magnitude-free
    if name == 'Rotate':       return rotate(img, _int_param(m, 30) * random.choice([-1, 1]))
    if name == 'Solarize':     return solarize(img, 256 - _int_param(m, 256))
    if name == 'Posterize':    return posterize(img, 4 - _int_param(m, 4))
    if name in ('Color', 'Contrast', 'Brightness', 'Sharpness'):
        return enhance(img, name, _float_param(m, 1.8) + 0.1)     # 0.1 .. 1.9
    if name in ('ShearX', 'ShearY'):
        return shear(img, name, _float_param(m, 0.3) * random.choice([-1, 1]))
    if name in ('TranslateX', 'TranslateY'):
        return translate(img, name, _int_param(m, 10) * random.choice([-1, 1]))

transforms = [
    'Identity', 'AutoContrast', 'Equalize',
    'Rotate', 'Solarize', 'Color', 'Posterize',
    'Contrast', 'Brightness', 'Sharpness',
    'ShearX', 'ShearY', 'TranslateX', 'TranslateY',
]

def randaugment(N, M):
    """Two interpretable knobs: N ops per image, single global magnitude M."""
    ops = np.random.choice(transforms, N)            # uniform sampling, no learned probs
    return [(op, M) for op in ops]                    # all share magnitude M

def augment(img, N, M):
    # applied on top of the default flip + crop pipeline
    for op, m in randaugment(N, M):
        img = apply_op(img, op, m)
    return img

# N and M are tuned by grid search on the TARGET task, alongside lr / weight decay;
# everything else (default flip+crop, network, SGD + cosine schedule) is unchanged.
```

The chain: learned augmentation needs a separate proxy search, and the proxy is doubly wrong — costly, and unable to set the augmentation strength, which actually scales with model and dataset size. The benefit of those learned policies is diversity, not the learned probabilities, so replace the probabilities with uniform 1/K sampling of N transforms — diversity by construction, no probabilities to search. The per-transform magnitudes barely matter and move together anyway, so tie them all to one global magnitude M, held constant (all magnitude schedules tie, so pick the one with the fewest knobs). That leaves two interpretable knobs, N and M, both monotone in strength, over a search space of K^N — small enough that grid search on the target task replaces the entire controller, and the augmentation strength can finally be dialed to the model and dataset at hand.
