# RandAugment

## Problem

Learned augmentation policies (AutoAugment and its faster variants) improve accuracy but require a *separate search phase*: an RNN controller trained by reinforcement learning on a small proxy task (reduced dataset, small model), whose policy is then transferred to the target. This second optimization loop is expensive and complex, and it is structurally wrong for augmentation: the optimal augmentation *strength* scales with model size and dataset size, so a strength fixed on a small proxy cannot be right for a large target, and the search offers no way to re-adjust it. The goal is automated augmentation with no separate search — augmentation settings that are ordinary, interpretable training hyperparameters tuned directly on the target.

## Key idea

Collapse AutoAugment's ~10^32-policy search space to two interpretable hyperparameters by removing what the diversity benefit does not need:

- **Drop the learned per-transform probabilities**: at each application, sample a transform *uniformly* (probability 1/K) from the K = 14 operations. Apply N of them in sequence per image. This forces diversity by construction (K^N combinations) with nothing to learn.
- **Tie all magnitudes to one global value M**: per-transform magnitudes barely affect accuracy (varying one while fixing the rest moves accuracy ~0.1–0.2%) and tend to move together over training, so a single shared magnitude suffices.

The entire algorithm is two lines: sample N transforms uniformly, apply each at magnitude M.

```python
transforms = [
    'Identity', 'AutoContrast', 'Equalize', 'Rotate', 'Solarize', 'Color',
    'Posterize', 'Contrast', 'Brightness', 'Sharpness',
    'ShearX', 'ShearY', 'TranslateX', 'TranslateY',
]

def randaugment(N, M):
    sampled_ops = np.random.choice(transforms, N)   # uniform, 1/K each
    return [(op, M) for op in sampled_ops]
```

Both N and M are monotone in regularization strength, so they are human-interpretable and can be tailored to the model and dataset.

## Why it works

- **Diversity, not learned probabilities, drives the gain.** Uniform sampling of N ops gives K^N varied policies per image — the diversity the controller was implicitly maximizing — without any learned probability parameters.
- **The search space becomes K^N** (a few hundred to a few thousand) versus AutoAugment's ≈ (16×10×11)^10 ≈ 2.9 × 10^32 — a reduction of roughly 30 orders of magnitude (≈10^32 → ≈10^2). This is small enough that plain *grid search on the target task* matches or beats proxy-searched policies; N and M fold into the normal hyperparameter sweep, and no proxy, controller, or RL is needed.
- **Strength adapts to the target.** Because M is a free hyperparameter tuned on the target (and may exceed 10 on the same linear scale), larger models can use stronger augmentation — exactly the adjustment a proxy search cannot make. Empirically the optimal M rises with model size (ResNet-50: 9; EfficientNet-B5: 17; B7: 28).

## Design choices

- **Uniform op sampling (1/K)** instead of learned probabilities: removes the probability search; diversity is built in.
- **Single global magnitude M** instead of per-transform magnitudes: per-transform tuning is negligible and magnitudes share a schedule.
- **Constant M over training** (vs random / linearly increasing / random-with-rising-bound): all schedules perform equally, so the one with the fewest hyperparameters wins.
- **N transforms applied in sequence**: the second knob; larger N strengthens regularization.
- **AutoAugment's 0–10 magnitude scale** (PARAMETER_MAX = 10): level → native parameter linearly, so M is interpretable and monotone in strength.

## Code

RandAugment runs as a transform on top of the default flip/crop pipeline; the network, optimizer, and schedule are unchanged. Magnitudes use the linear 0–10 scale.

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

Typical settings: CIFAR-10 N=3, M between 4 and 9 depending on the network (WRN-28-2: 4, WRN-28-10: 5, Shake-Shake: 9); SVHN N=3, M≈5–9 with Cutout applied after; ImageNet ResNet-50 N=2, M=9, EfficientNet-B5 N=2, M=17, B7 N=2, M=28; COCO detection N=1, M≈5–6 (geometric ops also transform the boxes). N and M are chosen by grid search on the target task.
