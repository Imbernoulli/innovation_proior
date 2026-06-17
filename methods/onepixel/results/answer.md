# One Pixel Attack

## Problem

Craft an adversarial image that fools a classifier under an **`L0` budget**: modify at most `d`
spatial pixels (the namesake case `d = 1`), with each modified pixel allowed to change by an
arbitrary amount. This inverts the usual `Lp` attack, which bounds perturbation *magnitude* and
spreads it over all pixels. The attack is **score-based / black-box** — it uses only the model's
output probabilities, no gradients — so it also applies to non-differentiable models.

## Key idea

Encode a sparse perturbation as a fixed-length real vector and optimize it with **differential
evolution (DE)**, a gradient-free population metaheuristic.

- **Encoding.** One modified pixel is a 5-tuple `(x, y, R, G, B)`: location plus color. A
  candidate solution is `d` such tuples concatenated — a real vector of length `5d` (for RGB).
  Because only `d` tuples are ever written, the constraint `||e(x)||_0 <= d` holds **by
  construction**: no penalty term, no projection. The discrete "which pixel" choice is carried by
  the continuous coordinates, rounded to indices when applied.
- **Optimizer (DE).** A population of candidates is evolved. Each generation, every parent gets a
  child via DE/rand/1 mutation using three mutually distinct random population indices, then the
  child replaces the parent iff its fitness is at least as good (one-to-one tournament). DE needs
  only to *evaluate* the objective, and its
  difference-scaled mutation self-adapts the search radius (large while the population is spread,
  small as it converges); the one-to-one selection preserves diversity, escaping the local optima
  that sink greedy/gradient sparse attacks.
- **Fitness.** Just the softmax probability of the relevant class — `f_adv(x+e)` to maximize
  (targeted) or `f_t(x+e)` to minimize (non-targeted). No surrogate loss; classifier-agnostic.

## Formal statement

Targeted: `maximize_{e(x)} f_adv(x + e(x))  s.t.  ||e(x)||_0 <= d`.
Non-targeted: `minimize_{e(x)} f_t(x + e(x))  s.t.  ||e(x)||_0 <= d`.

DE/rand/1 mutation (Storn & Price 1997):
`x_i(g+1) = x_{r1}(g) + F * (x_{r2}(g) - x_{r3}(g))`, with `r1`, `r2`, and `r3`
mutually distinct random population indices, distinct from `i`.

## Search settings

In the clean DE/rand/1 formulation: population 400, up to 100 generations, scale factor
`F = 0.5`, and no crossover. Initialize locations `~ U(1, S)` over the spatial size `S` (32
for CIFAR, 227 for ImageNet) and RGB `~ N(mu=128, sigma=127)`. Fitness =
target-class probability (targeted, maximize) or true-class probability (non-targeted,
minimize). Early stop in the clean experimental settings: targeted CIFAR when target prob > 90%,
non-targeted ImageNet when true prob < 5%. Cost = population x generations model evaluations.
Budgets `d = 1, 3, 5`.

## Torchattacks implementation

Faithful to the `torchattacks.OnePixel` implementation: images are in `[0, 1]`, so channel
value bounds are `(0, 1)`; the code uses a scipy-derived local `differential_evolution` with
default `best1bin` donor, mutation dithering `(0.5, 1)`, `recombination=1`, random
initialization, and no polishing.

```python
import numpy as np
import torch
import torch.nn.functional as F

from torchattacks.attack import Attack
from torchattacks.attacks._differential_evolution import differential_evolution


class OnePixel(Attack):
    def __init__(self, model, pixels=1, steps=10, popsize=10, inf_batch=128):
        super().__init__("OnePixel", model)
        self.pixels = pixels
        self.steps = steps
        self.popsize = popsize
        self.inf_batch = inf_batch
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        batch_size, channel, height, width = images.shape
        bounds = ([(0, height), (0, width)] + [(0, 1)] * channel) * self.pixels
        popmul = max(1, int(self.popsize / len(bounds)))

        adv_images = []
        for idx in range(batch_size):
            image, label = images[idx:idx + 1], labels[idx:idx + 1]

            if self.targeted:
                target_label = target_labels[idx:idx + 1]

                def func(delta):
                    return self._loss(image, target_label, delta)

                def callback(delta, convergence):
                    return self._attack_success(image, target_label, delta)
            else:
                def func(delta):
                    return self._loss(image, label, delta)

                def callback(delta, convergence):
                    return self._attack_success(image, label, delta)

            best = differential_evolution(
                func=func, bounds=bounds, callback=callback,
                maxiter=self.steps, popsize=popmul,
                init="random", recombination=1, atol=-1, polish=False,
            ).x
            best = np.split(best, len(best) / len(bounds))
            adv_images.append(self._perturb(image, best))
        return torch.cat(adv_images)

    def _loss(self, image, label, delta):
        adv_images = self._perturb(image, delta)
        prob = self._get_prob(adv_images)[:, label]
        return 1 - prob if self.targeted else prob

    def _attack_success(self, image, label, delta):
        adv_image = self._perturb(image, delta)
        prob = self._get_prob(adv_image)
        pred = np.argmax(prob)
        if self.targeted and (pred == label):
            return True
        if (not self.targeted) and (pred != label):
            return True
        return False

    def _get_prob(self, images):
        with torch.no_grad():
            outs = []
            for batch in torch.split(images, self.inf_batch):
                outs.append(self.get_logits(batch))
        prob = F.softmax(torch.cat(outs), dim=1)
        return prob.detach().cpu().numpy()

    def _perturb(self, image, delta):
        delta = np.array(delta)
        if len(delta.shape) < 2:
            delta = np.array([delta])
        adv_image = image.clone().detach().to(self.device)
        adv_images = torch.cat([adv_image] * len(delta), dim=0)
        for idx in range(len(delta)):
            pixel_info = delta[idx].reshape(self.pixels, -1)
            for pixel in pixel_info:
                pos_x, pos_y = pixel[:2]
                channel_v = pixel[2:]
                for channel, v in enumerate(channel_v):
                    adv_images[idx, channel, int(pos_x), int(pos_y)] = v
        return adv_images
```

## Why it works

The `L0` budget makes the problem combinatorial (choose support) plus continuous (choose
amplitude); a metaheuristic that needs no gradients and keeps population diversity handles both
at once, while the 5-tuple encoding folds the discrete support choice into a continuous vector and
enforces sparsity structurally. Because each modified pixel can take any valid value, one pixel can
cross a nearby decision boundary when such a vulnerable axis exists, and the attack doubles as a
geometric probe: a `d`-pixel perturbation explores the classifier along a `d`-dimensional
axis-aligned slice of input space.
