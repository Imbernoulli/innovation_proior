Let me start from the thing that's actually slowing everyone down. Learned data augmentation works — AutoAugment-style policies give real accuracy gains and cost nothing at inference. But to get one I have to run a *separate search*: stand up an RNN controller, train it with reinforcement learning (PPO) to propose augmentation policies, train a child network for each proposal, reward the controller by the child's validation accuracy, and do all of this on a small *proxy* task — a reduced dataset and a small model — then transfer the winning policy to the real task. That's a whole second optimization loop bolted onto training, expensive and fiddly. I want the accuracy without that loop.

Before I try to remove it, let me check whether the proxy step is even *sound* for augmentation, because if it's not, that's a much stronger reason to kill it than mere cost. The proxy assumption is: a policy that's good on a small model / small dataset is good on the big target. For architecture search that's plausible. For augmentation? Think about what augmentation *does* — it's regularization, and the right amount of regularization depends on capacity and data. A big model on a big dataset can absorb — and wants — *more* aggressive augmentation than a tiny model on a sliver of data. So the optimal augmentation *strength* should grow with model size and dataset size. If that's true, then a policy whose strength was pinned down on a small proxy is *systematically* mis-set when I move to the large target, and the search gives me no handle to re-tune it. So the proxy isn't just expensive — it's structurally the wrong place to fix the one thing (strength) that most depends on the target. That settles it: I don't want to search on a proxy at all. I want the augmentation settings to be ordinary training hyperparameters, tuned directly on the target task alongside learning rate and weight decay.

The obstacle to that is the search space. AutoAugment's policy is enormous: 5 sub-policies, each two operations in sequence, each operation a (type out of ~16 transforms, probability out of 11 discrete values, magnitude out of 10 discrete values). Let me actually size it. One operation is 16 × 10 × 11 = 1760 choices. A policy is effectively 10 such operations (5 sub-policies × 2), so 1760^10. I want the order of magnitude: log10(1760) ≈ 3.246, times 10 is ≈ 32.5, so the space is around 10^32 — call it 2.85 × 10^32 if I carry it out exactly. That's 30-plus parameters to set jointly. You cannot grid-search 10^32; you *need* a learned controller, which is exactly the loop I'm trying to delete. So removing the search isn't a matter of choosing a cheaper search — I first have to make the space small enough that no search is needed at all. The question becomes: how much of that 10^32 is actually carrying the benefit, and how much can I throw away?

So I go after the largest factor first. Of the three things each operation specifies — type, probability, magnitude — the probability is the one with the most discrete values (11) and it appears in every one of the ten operations, so it dominates the count. Is it earning its keep? The reported reason learned policies help is *diversity* — the network sees many different transformed versions of each image. I have not seen evidence that it's the specific learned *probabilities* of each operation that matter; those are just the knobs the controller happened to twiddle to maximize diversity-and-accuracy. If diversity is really the active ingredient, then I don't need to *learn* the probabilities — I can *force* diversity by construction. Replace the learned probability per transform with uniform random selection: at each application, pick one of the K = 14 transforms with probability 1/K. Parameter-free, nothing to learn or search. Apply N of them in sequence to each image, which still yields K^N possible operation combinations per image — plenty of diversity. The entire probability dimension is gone, replaced by a coin flip. That's the factor of 11-per-operation, the bulk of the count, eliminated at one stroke.

Now the magnitudes. AutoAugment searches a separate magnitude for each transform, each on an integer scale 0 to 10. Do I need per-transform magnitudes? Two clues say maybe not. First, the population-based work noticed that the optimal magnitudes tend to move *together* over training — they rise on a shared schedule rather than each doing its own thing, which means they're not carrying much independent information. Second, this is testable directly: fix every transform's magnitude at a shared value, then sweep just *one* transform's magnitude across its whole 0–10 range and watch validation accuracy. If per-transform tuning mattered, that sweep should swing accuracy substantially. The cheap prediction is that it won't — that the curve is nearly flat — because if magnitudes are highly coupled, one transform's magnitude is mostly redundant with the shared level. I'd want to run that sweep on, say, CIFAR-10 with a Wide-ResNet before trusting it; my expectation is a swing on the order of a tenth of a percent, i.e. within run-to-run noise. If that holds, then tying *all* the magnitudes to a single global value M loses essentially nothing, and the entire magnitude dimension collapses to one number.

Granting that, I'm left with exactly two knobs: N, how many transforms I apply per image, and M, the single magnitude shared by all of them. Both are monotone in regularization strength — crank either up and the augmentation gets more aggressive — so they're human-interpretable, which is exactly what I wanted for tuning strength to the target model and dataset. The whole algorithm is: sample N transforms uniformly from the list, apply each at magnitude M.

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

Let me put a number on what that bought, because the whole argument for "no search needed" rests on the new space being small. The space is now K^N. For K = 14: N = 2 gives 14^2 = 196, N = 3 gives 14^3 = 2744 — a few hundred to a few thousand. Compare to AutoAugment's ≈ 2.85 × 10^32. The reduction in orders of magnitude is log10(2.85e32) − log10(196) ≈ 32.45 − 2.29 ≈ 30.2 at N = 2, and ≈ 32.45 − 3.44 ≈ 29.0 at N = 3. So roughly thirty orders of magnitude, which I'd been gesturing at — now I've actually subtracted the logs and it lands where I hoped. And the operative consequence isn't the headline number, it's this: I don't just search a smaller space, I can enumerate it. The two knobs are integers in tiny ranges — try N ∈ {1, 2, 3} and M ∈ {1, …, 10} and that's 30 configurations total. Thirty training runs on the *target* task is an ordinary hyperparameter sweep, the kind I'd do for learning rate anyway. No controller, no proxy, no RL — the controller existed only because 10^32 couldn't be enumerated, and now it can. Whether those 30 runs actually *match or beat* the proxy-searched policies is the empirical question this whole construction stakes itself on; I can't settle it on paper, but the bound makes the experiment cheap enough to just run.

One loose end on M: it's a single number, but should it be *constant* over training, or follow a schedule? The population-based result did say magnitudes rise during training, so a ramp is at least plausible and I shouldn't assume constant is best. Candidates: a constant M; a random M sampled uniformly between two bounds each step; a magnitude that increases linearly over training; and a random M whose upper bound increases over training. I'd run all four (CIFAR-10, Wide-ResNet) and read off the accuracies. I expect them to come out close, because the magnitude-sweep above already suggested the network is fairly insensitive to exactly how M is set — if it barely cares about a static level, it's unlikely to care much about the level's time-profile. If they do land within noise of each other, the tie-break is not accuracy but parameter count: the constant has one number, M; the random and ramped schemes each add bounds I'd have to set. Fewest knobs wins a tie, so M is a constant — but I'm holding that lightly until the four-way comparison actually comes back tied rather than assuming it.

Now the magnitudes have to actually mean something per transform, since "M = 9" has to turn into a rotation angle, a shear, a solarize threshold, and so on. Keep AutoAugment's linear scale with PARAMETER_MAX = 10: a level on 0–10 maps to each transform's native parameter by level/10 times that transform's maximum. So rotation is ±(M/10)·30°, shear is (M/10)·0.3, translation is (M/10)·(max pixels), solarize inverts pixels above a threshold int((M/10)·256), posterize keeps int((M/10)·4) bits, and the photometric ops — color, contrast, brightness, sharpness — blend by (M/10)·1.8 + 0.1 (so 0.1 to 1.9× enhancement). AutoContrast, Equalize, and Identity ignore M.

Let me trace the mapping at M = 9, the strongest in-range value, to confirm the numbers are sane: rotation ±int(9·30/10) = ±27°, shear ±(9·0.3/10) = ±0.27, translate ±int(9·10/10) = ±9 px, solarize threshold 256 − int(9·256/10) = 256 − 230 = 26 (invert almost everything), posterize 4 − int(9·4/10) = 4 − 3 = 1 bit, enhance 9·1.8/10 + 0.1 = 1.72. All in their natural ranges, all stronger than at small M — strength rising monotonically, as promised.

Then there's the claim that M can exceed 10 to give a big model stronger augmentation than the in-range maximum — the strength-scales-with-model-size adjustment the proxy approach couldn't make. I should check that the extrapolation actually behaves, not just assert it, so trace M = 28 (the value I'd want for a large EfficientNet): rotation ±int(28·30/10) = ±84°, shear ±0.84, translate ±28 px, enhance 28·1.8/10 + 0.1 = 5.14. Those are all "more of the same" — a continuous parameter pushed further along the same line, which is exactly the point. But the same trace exposes something I'd have papered over: solarize threshold is 256 − int(28·256/10) = 256 − 716 = −460, and posterize is 4 − int(28·4/10) = 4 − 11 = −7. Negative. Those two ops have *clamped, discrete* parameters — a threshold in [0, 256] and a bit count in [0, 4] — so once M passes about 10 they don't extrapolate at all; they saturate. So "it's just the same linear scale extrapolated" is true for the *continuous* geometric and photometric ops but not literally for the two integer-clamped ones, which simply bottom out (solarize inverts everything, posterize crushes to the fewest bits) and then stop changing. That's fine — saturating at maximum strength is the sensible thing for a clamped op to do — but it means M > 10 buys extra strength only on the continuous ops, which is most of the list. Good to know the failure mode is "harmless saturation" and not "garbage," and I'd only have learned that by actually running the level through the formula.

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
