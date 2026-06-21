Learned data augmentation works: AutoAugment-style policies give real accuracy gains and cost nothing at inference. The price is a *separate search phase*. To obtain a policy I have to stand up an RNN controller, train it with reinforcement learning (PPO) to propose augmentation policies, train a child network for each proposal, reward the controller by the child's validation accuracy — and do all of this on a small *proxy* task, a reduced dataset and a small model, before transferring the winning policy to the real task. That is a whole second optimization loop bolted onto training, expensive and fiddly. Faster variants (Fast AutoAugment via density matching, Population Based Augmentation via an evolutionary schedule) cut the cost of that loop but never remove it. I want the accuracy without the loop at all.

Before trying to delete the proxy step, I should check whether it is even *sound* for augmentation, because if it is not, that is a far stronger reason to kill it than cost. The proxy assumption is that a policy good on a small model and a sliver of data is good on the big target. For architecture search that is plausible. For augmentation it is suspect, because augmentation is regularization, and the right amount of regularization depends on capacity and data: a large model on a large dataset can absorb — and wants — *more* aggressive augmentation than a tiny model on little data. So the optimal augmentation *strength* should grow with model size and dataset size. If that holds, a policy whose strength was pinned on a small proxy is *systematically* mis-set when moved to the large target, and the search gives no handle to re-tune it. The proxy is therefore not merely expensive but structurally the wrong place to fix the one thing — strength — that most depends on the target. That settles it: I do not want to search on a proxy at all. I want the augmentation settings to be ordinary training hyperparameters, tuned directly on the target task alongside learning rate and weight decay.

The obstacle is the size of the search space. AutoAugment's policy is enormous — 5 sub-policies, each two operations applied in sequence, each operation a choice of (type out of $\approx 16$ transforms, probability out of 11 discrete values, magnitude out of 10 discrete values) — so the space is about $$(16 \times 10 \times 11)^{10} \approx 2.9 \times 10^{32}$$ policies, 30-plus parameters. You cannot grid-search that on the target; you *need* a learned controller, which is exactly the loop I am trying to delete. So to remove the search I first have to shrink the space until no search is needed, which means asking how much of that $10^{32}$ actually carries the benefit and how much can be thrown away.

I propose RandAugment. The lever is the reported reason learned policies help: it is *diversity* — the network sees many different transformed versions of each image — not the specific learned per-operation *probabilities*, which are merely the knobs the controller twiddled to maximize diversity-and-accuracy. So I stop learning the probabilities entirely and force diversity by construction. Instead of a learned probability per transform, I sample transforms *uniformly at random*: at each application pick one of the $K = 14$ transforms with probability $1/K$, parameter-free, nothing to learn or search. Applying $N$ of them in sequence to each image still yields $K^N$ possible operation combinations — plenty of diversity — but the entire probability dimension of the search space, the bulk of that $10^{32}$, is gone, replaced by a coin flip.

Next the magnitudes. AutoAugment searches a separate magnitude for each transform on an integer scale $0$ to $10$. Two clues say per-transform magnitudes are not needed. First, the population-based work observed that the optimal magnitudes move *together* over training — rising on a shared schedule rather than each evolving independently — so they carry little independent information. Second, the sensitivity can be probed directly: fix every transform's magnitude at a shared value, then sweep just *one* transform's magnitude across its whole range and watch the accuracy; if per-transform tuning mattered, accuracy should swing, but it barely moves, on the order of a tenth to two-tenths of a percent. Tying *all* magnitudes to one global value $M$ therefore loses almost nothing and collapses the entire magnitude dimension to a single number.

That leaves exactly two knobs: $N$, how many transforms are applied per image, and $M$, the single magnitude shared by all of them. Both are monotone in regularization strength — crank either up and the augmentation grows more aggressive — so both are human-interpretable, which is precisely what is wanted for tuning strength to the target model and dataset. The whole policy is: sample $N$ transforms uniformly from the list, apply each at magnitude $M$. The search space is now $K^N$ — for $K = 14$ and $N = 2$ or $3$, a few hundred to a few thousand — against AutoAugment's $\approx 10^{32}$, a reduction of more than thirty orders of magnitude. That is the whole point: $K^N$ is small enough that plain *grid search* over $N$ and $M$ on the *target* task finds a policy that matches or beats the proxy-searched ones, with $N$ and $M$ folding into the normal hyperparameter sweep. No separate phase, no proxy, no controller, no RL.

One loose end is whether $M$ should be *constant* over training or follow a schedule, since the population-based result reported that magnitudes rise during training. I lay out four candidates — a constant $M$; a random $M$ sampled uniformly between two bounds each step; a magnitude increasing linearly over training; and a random $M$ whose upper bound increases over training — and compare them (say on CIFAR-10 with a Wide-ResNet). They come out essentially tied, within noise of one another. Given a tie, take the option with the fewest hyperparameters, the constant: one number $M$, fixed for the whole run. The ramp and the random-range schemes each add bounds I would have to set for no measured gain.

Finally the magnitudes have to *mean* something per transform, since "$M = 9$" must turn into a rotation angle, a shear, a solarize threshold, and so on. I keep AutoAugment's linear scale with $\text{PARAMETER\_MAX} = 10$: a level on $0$–$10$ maps to each transform's native parameter by $(\text{level}/10)$ times that transform's maximum. So rotation is $\pm(M/10)\cdot 30^\circ$, shear is $(M/10)\cdot 0.3$, translation is $(M/10)\cdot(\text{max pixels})$, solarize inverts pixels above a threshold $\lfloor(M/10)\cdot 256\rfloor$, posterize keeps $\lfloor(M/10)\cdot 4\rfloor$ bits, and the photometric ops — color, contrast, brightness, sharpness — blend by $(M/10)\cdot 1.8 + 0.1$, i.e. a $0.1$ to $1.9\times$ enhancement; AutoContrast, Equalize, and Identity ignore $M$. Larger $M$ means a larger angle, harsher shear, stronger color shift — strength rising monotonically. Nothing forces $M \le 10$ either: a very large model on large images can use $M$ of $17$ or $28$, simply the same linear scale extrapolated, which is exactly the strength-scales-with-model-size adjustment the proxy approach could never make. Empirically the optimal $M$ does rise with model size — ResNet-50 at $9$, EfficientNet-B5 at $17$, B7 at $28$.

RandAugment runs as a transform on top of the default flip/crop pipeline; the network, optimizer, and schedule are unchanged, and magnitudes use the linear $0$–$10$ scale.

```python
import numpy as np
import random

PARAMETER_MAX = 10

def _float_param(level, maxval):
    return float(level) * maxval / PARAMETER_MAX

def _int_param(level, maxval):
    return int(level * maxval / PARAMETER_MAX)

def apply_op(img, name, m):
    if name == 'Identity':     return img
    if name == 'AutoContrast': return auto_contrast(img)
    if name == 'Equalize':     return equalize(img)
    if name == 'Rotate':       return rotate(img, _int_param(m, 30) * random.choice([-1, 1]))
    if name == 'Solarize':     return solarize(img, 256 - _int_param(m, 256))
    if name == 'Posterize':    return posterize(img, 4 - _int_param(m, 4))
    if name in ('Color', 'Contrast', 'Brightness', 'Sharpness'):
        return enhance(img, name, _float_param(m, 1.8) + 0.1)        # 0.1 .. 1.9
    if name in ('ShearX', 'ShearY'):
        return shear(img, name, _float_param(m, 0.3) * random.choice([-1, 1]))
    if name in ('TranslateX', 'TranslateY'):
        return translate(img, name, _int_param(m, 10) * random.choice([-1, 1]))

transforms = [
    'Identity', 'AutoContrast', 'Equalize', 'Rotate', 'Solarize', 'Color',
    'Posterize', 'Contrast', 'Brightness', 'Sharpness',
    'ShearX', 'ShearY', 'TranslateX', 'TranslateY',
]

def randaugment(N, M):
    ops = np.random.choice(transforms, N)               # uniform; no learned probabilities
    return [(op, M) for op in ops]                        # all share the single magnitude M

def augment(img, N, M):                                   # applied after default flip + crop
    for op, m in randaugment(N, M):
        img = apply_op(img, op, m)
    return img
```
