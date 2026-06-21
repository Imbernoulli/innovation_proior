I propose the one-pixel attack, the adversarial method introduced by Su, Vargas and Sakurai for fooling deep image classifiers while touching only a tiny number of spatial pixels. The canonical name I will use is the one-pixel attack, and the algorithm is most naturally implemented with differential evolution. It solves the problem of crafting an adversarial image under an L0 budget: for a correctly classified image x, find a perturbation e(x) that changes the classifier prediction, with at most d spatial pixels modified and with each modified pixel allowed to take any valid color. The case d equals one is the namesake extreme. This inverts the usual Lp attacks that bound the magnitude of change and spread it over every pixel.

The motivation for counting pixels rather than norm is that a small total norm smeared across an image is not the same thing as an imperceptible change. The same norm concentrated into a few pixels produces visible speckle, and no Lp bound can promise that a human will not notice the result. Counting modified pixels is a more direct measure of localization and therefore of plausible hiding power. Pushing the count to its minimum also turns the attack into a geometric probe: it asks how close the decision boundary is along a sparse, axis-aligned slice of the input space.

The objective is simple. In a targeted attack I maximize the probability the model assigns to a chosen class adv, written maximize over e(x) of f_adv(x + e(x)) subject to the support constraint ||e(x)||_0 <= d. In an untargeted attack I minimize the probability of the true class t, written minimize over e(x) of f_t(x + e(x)) subject to the same constraint. The constraint is combinatorial: I must choose which pixels to touch, and then choose what value to write. A gradient is the wrong tool for selecting a sparse support, and greedy saliency methods that use the Jacobian are white-box, commit irrevocably to locally attractive pixels, and stall on surfaces that adversarial training has flattened.

To hand the problem to a continuous optimizer I encode a sparse perturbation as d concatenated five-tuples. Each tuple is (x, y, R, G, B): the spatial location and the new color of one modified pixel. A full candidate solution is therefore a real vector of length 5d for an RGB image. Because exactly d tuples are decoded and every other pixel is left untouched, the L0 budget is satisfied by construction. There is no penalty term, no projection step, and no constrained optimization. The discrete choice of which pixel to modify is carried by the continuous location coordinates, which are rounded to integer indices only when the candidate is applied to the image. This encoding dissolves the apparent discreteness into a continuous search.

I optimize the vector with differential evolution, a gradient-free population metaheuristic. Differential evolution maintains a population of candidate vectors. For each parent it forms a child by adding a scaled difference of two other population members to a third. In the classic DE/rand/1 form the donor is x_{r1} + F times (x_{r2} minus x_{r3}), with distinct random indices r1, r2, r3 and a scale factor F typically around 0.5. The child then competes one-to-one with its parent and survives only if it improves the fitness. This one-to-one selection preserves diversity, which is why the population can escape the local optima that trap greedy or first-order methods. The mutation step is self-adaptive: while the population is spread the differences are large and exploration is broad, and as the population converges the differences shrink so refinement happens automatically. No gradient is ever computed; the algorithm only evaluates the objective, so it works for non-differentiable or black-box classifiers.

The fitness function is deliberately direct. In targeted mode I maximize the target-class probability, so the objective I minimize is one minus that probability. In untargeted mode I minimize the true-class probability. Both values are read straight from the softmax output of the model. In the clean experimental setting I use a population of 400, up to 100 generations, F = 0.5, and no crossover. Locations are initialized uniformly over the image extent and colors from a wide Gaussian covering the full intensity range. I early-stop when the target class exceeds a high confidence threshold, for instance ninety percent in targeted CIFAR-style evaluations, or when the true class drops below a low threshold, for instance five percent in non-targeted ImageNet-style evaluations. The cost is the number of model evaluations, which equals population size times the number of generations actually run.

In the trajectory ladder this method appears as the rung after Pixle and JSMA. Pixle is purely random and achieves very low success rates. JSMA improves by using a per-pixel saliency map, but it remains greedy and local, and it stalls on L2-adversarially trained models where first-order signals are suppressed. The one-pixel attack replaces that greedy gradient step with a global population search, so it does not commit irrevocably to any pixel and does not require a gradient. The configuration used in that ladder is more query-starved than the canonical one: it allows 24 modified pixels, runs six generations, uses a population multiplier of eight, and batches forward passes with size 128. Even so it is expected to beat the JSMA floor by escaping local optima, though the small generation budget limits how far it can refine a 120-dimensional search space.

The attack is interesting beyond its success rate because a d-pixel perturbation moves the input along a d-dimensional axis-aligned slice of the input space, with arbitrary distance along each chosen axis. This is structurally different from a small step in all directions, and it reveals how close decision boundaries lie along sparse coordinate directions. When a vulnerable axis exists, a single pixel with an extreme color can cross the boundary.

The following Python snippet gives a small, self-contained illustration of the encoding, the differential evolution loop, and the probability-based fitness. It uses a dummy linear-plus-softmax model so the code can be run without any external attack library.

```python
import numpy as np


def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


class DummyModel:
    def __init__(self, height=32, width=32, n_classes=10, seed=0):
        rng = np.random.default_rng(seed)
        self.weights = rng.standard_normal((height, width, 3, n_classes)) * 0.05
        self.bias = np.zeros(n_classes)

    def __call__(self, img):
        scores = np.tensordot(img, self.weights, axes=3) + self.bias
        return softmax(scores.reshape(1, -1))


def one_pixel_attack(image, true_label, model, pixels=1,
                     popsize=200, steps=60, F=0.5, seed=0):
    rng = np.random.default_rng(seed)
    H, W, C = image.shape
    dim = 5 * pixels
    lower = np.tile(np.array([0.0, 0.0, 0.0, 0.0, 0.0]), pixels)
    upper = np.tile(np.array([H, W, 1.0, 1.0, 1.0]), pixels)

    def decode(vec):
        adv = image.copy()
        for i in range(pixels):
            x, y, r, g, b = vec[5 * i:5 * i + 5]
            xi = int(np.clip(x, 0, H - 1))
            yi = int(np.clip(y, 0, W - 1))
            adv[xi, yi] = np.clip([r, g, b], 0.0, 1.0)
        return adv

    def fitness(vec):
        adv = decode(vec)
        prob = model(adv)[0, true_label]
        return -prob  # minimize true-class probability

    pop = rng.uniform(lower, upper, size=(popsize, dim))
    fits = np.array([fitness(p) for p in pop])

    for _ in range(steps):
        for i in range(popsize):
            r1, r2, r3 = rng.choice(popsize, size=3, replace=False)
            child = np.clip(pop[r1] + F * (pop[r2] - pop[r3]), lower, upper)
            f_child = fitness(child)
            if f_child < fits[i]:
                pop[i] = child
                fits[i] = f_child
                if -f_child < 0.05:  # true-class probability below 5%
                    return child
    return pop[np.argmin(fits)]


# Demonstration: fool a dummy model by changing one pixel.
rng = np.random.default_rng(7)
img = rng.uniform(0, 1, (32, 32, 3))
model = DummyModel(seed=42)
true_label = 0
best = one_pixel_attack(img, true_label, model, pixels=1,
                        popsize=100, steps=40)
adv = img.copy()
x, y, r, g, b = best[:5]
adv[int(np.clip(x, 0, 31)), int(np.clip(y, 0, 31))] = np.clip([r, g, b], 0, 1)
print("clean true-class prob:", model(img)[0, true_label])
print("adv  true-class prob:", model(adv)[0, true_label])
```

I call the result the one-pixel attack. It is a black-box, gradient-free, L0-bounded adversarial method that turns the sparse support constraint into a continuous differential-evolution search by encoding each modified pixel as a location-color five-tuple. The encoding enforces the pixel budget structurally, the optimizer preserves diversity to escape local optima, and the fitness is the raw model probability being driven toward the attack goal.
