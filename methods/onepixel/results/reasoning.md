Let me start from what actually bothers me about the adversarial examples everyone is making. The recipe is always the same shape: take a correctly classified image, add a perturbation `e(x)` whose *norm* is small — `L2` or `L-inf` — and watch the label flip. Szegedy's box-constrained L-BFGS minimizes `||r||_2` to the boundary; Goodfellow's fast gradient sign takes one `eps`-sized step of `sign(grad)` along every coordinate; DeepFool walks iteratively to the nearest linearized boundary with minimal `L2`. They differ in how they find `r`, but they agree on the currency: keep the *amount* of change small, spread it wherever it helps. And it does help — these perturbations are usually invisible. But "usually invisible" is exactly the thing that nags me. Nobody can promise imperceptibility from a small-norm constraint. A small total norm smeared over a thousand pixels is one kind of change; the same norm dumped into a few pixels is speckle you can see. If what I really care about is that a human wouldn't notice, then total norm is the wrong quantity to be bounding.

So what *is* the right quantity? The most direct measure of "localized, hence plausibly hidden" is: how many pixels did I touch at all. Not how far each moved — *how many moved*. That's the `L0` "norm," the count of nonzero entries of `e(x)`. And if I'm going to make the count small, I might as well ask the extreme question: how few pixels can possibly suffice? One? Three? Five? The interesting limit is the smallest count that still flips a label. So I want to flip the whole framing. Instead of `||e(x)|| <= L` with the support free, I want `||e(x)||_0 <= d` with `d` tiny — and crucially, *no* limit on how far each of those `d` pixels moves. Let a touched pixel go to any color it wants; just don't touch more than `d` of them. That's the opposite trade from everyone else: they cap amplitude and let count run free; I cap count and let amplitude run free.

There's a second thing bugging me, and it might be related. Every one of those attacks is built by back-propagating through `f`. They *are* gradients. That means white-box access and a differentiable model, and it means the attacker has to know the network's internals. I'd like an attack that needs only what a user of the model sees: the output probabilities. Black-box. Then it works on non-differentiable models too, and it's a fairer test of real-world robustness.

Let me write the objective and stare at it. Targeted version: I have `x` correctly classified as `t`, I pick a target class `adv`, and I want

  maximize over `e(x)`:  `f_adv(x + e(x))`   subject to   `||e(x)||_0 <= d`.

Non-targeted is the mirror: minimize `f_t(x + e(x))`, drive the true-class probability down until something else wins. Compare this to what the dense attacks solve — `maximize f_adv` subject to `||e(x)|| <= L`. The only change is the constraint, `L0` instead of `Lp`. But that one change rewrites the character of the problem completely. The dense constraint is a continuous ball; projected gradient steps live happily inside it. My constraint is *combinatorial*: `||e||_0 <= d` says "at most `d` of the `n` coordinates are nonzero," and choosing *which* `d` is a discrete selection over `C(n, d)` supports, on top of choosing the real-valued amplitude on each chosen pixel. There are two intertwined sub-problems — which dimensions to perturb, and by how much — and the first one is discrete.

Can I just gradient my way through this? That's the reflex, so let me try it and see where it breaks. The gradient `grad_x f_adv` tells me, to first order, how the target probability responds to nudging each pixel. So the greedy move is: compute the Jacobian, find the most "salient" pixels, push those. That's essentially Papernot's saliency-map attack — build a saliency map from the forward derivatives and greedily perturb the highest-saliency pixels toward the target. And it *is* sparse-ish, the closest prior art to what I want. But walk through what it costs me and where it stalls. It needs the Jacobian — white-box, differentiable — so I lose the black-box property I just said I wanted. It's greedy: it commits to the locally most salient pixel and never reconsiders, so it can wander into a local optimum where no single next pixel looks good even though a *different* pair of pixels would have worked. And empirically it ends up touching a few percent of the image — order forty pixels on a 32x32 — which is back to visible speckle, not the one-or-three-pixel regime I'm after. The discreteness is the real wall: a gradient is a statement about an infinitesimal move in *all* coordinates at once, and my constraint is about *selecting* a tiny set of coordinates. Gradients are the wrong tool for picking a sparse support. Wall.

So drop gradients entirely. What I have is the ability to *evaluate* the objective — feed in `x + e(x)`, read back `f_adv` — and I want to optimize a function that is non-differentiable in its support choice, multimodal, and that I can only query as a black box. That's precisely the setting metaheuristics were built for. Differential evolution, in particular: Storn and Price's population method for nonlinear, non-differentiable, multimodal objectives over continuous spaces. It keeps a population of candidate vectors and, each generation, makes a new candidate from each parent by adding a *scaled difference of two other population members*, then keeps whichever of parent and child scores better. It never asks for a gradient; it only ever calls the objective. And it has two properties I want for this specific problem. It's relatively robust to local minima — the population spreads out and the difference-based mutation lets it jump — which directly answers the greedy-saliency local-optimum failure. And it needs nothing but black-box evaluations, which answers the white-box failure. The strict one-pixel constraint makes the landscape hard and multimodal, exactly where a diversity-keeping population beats greedy descent.

But DE optimizes a *real vector*. My problem has a discrete part — which pixel. How do I hand a "which pixel plus what color" object to a continuous optimizer? This is the crux, and it's where the whole thing either works or doesn't. Let me think about what a single perturbation actually is. For one modified pixel I need to specify: its location, two numbers `(x, y)`, and its new color, three numbers `(R, G, B)`. Five numbers. So one perturbation is a 5-tuple. If I'm allowed `d` pixels, a full candidate solution is just `d` of these tuples concatenated — a flat real vector of length `5d` (for RGB images). And now look at what that encoding *does to the constraint*: by writing the perturbation as exactly `d` tuples and leaving every other coordinate of `e(x)` at zero, the `L0 <= d` budget is satisfied *by construction*. I never have to add a penalty term, never have to project back onto the `L0` ball, never have to do constrained optimization at all. The support size is baked into the length of the vector. The discrete "which pixels" choice is smuggled into the continuous `(x, y)` coordinates — DE evolves them as reals and I round to a pixel index when I apply them. So the combinatorial sub-problem and the amplitude sub-problem are handled by the *same* continuous search over `5d` reals. That's the move: don't fight the discreteness, encode it as continuous coordinates that get rounded at apply-time. DE never even knows it's looking at an image.

Let me make the DE concrete, because the parameter choices matter. A population of candidate solutions — say 400. Each generation I produce a child for each parent using the canonical DE/rand/1 mutation:

  `x_i(g+1) = x_{r1}(g) + F * (x_{r2}(g) - x_{r3}(g))`, with `r1`, `r2`, and `r3` all distinct random population indices, distinct from `i`,

where `F` is the scale factor and `g` is the generation index. Why this form? The mutation step is a *difference of two existing candidates* scaled by `F`. That's the elegant self-adaptation at the heart of DE: early on, when the population is spread across the search space, those differences are large, so the search radius is large and exploratory; as the population converges toward good solutions, the differences shrink, so the steps shrink and the search refines itself. I never have to schedule a step size — the population's own spread is the step size. I'll take `F = 0.5`: large enough to keep exploring under the brutal one-pixel constraint, small enough not to fling children out of any useful region. And I'll use *rand1* — mutate around a *random* member `x_{r1}` — rather than mutating around the current best. With a target this constrained, anchoring everything to the single best-so-far would collapse diversity too fast and walk me into exactly the local optimum that sank greedy saliency; a random base keeps the population exploring. I'll also drop crossover entirely — no recombination of parent and child coordinates, each child is a pure mutant. Crossover is a knob I don't need; the difference-mutation already mixes information across the population, and leaving it out keeps the method simple, which was part of the point.

Selection. After I make the child, it competes with *its own parent only* — index-wise, one-to-one — and survives into the next generation if and only if its fitness is at least as good. Why one-to-one rather than, say, keeping the global top-400 of parents-plus-children? Because the one-to-one tournament is what *preserves diversity*. If I always kept the global best half, a few strong candidates would clone themselves across the population and crush variety; by making each child only displace the specific parent it descended from, a mediocre-but-different candidate in one slot can't be wiped out by a great candidate in another slot. The population stays spread, fitness is monotone non-worsening, and — comparing only parent to child — I get diversity-keeping and improvement at the same time. That's the property that lets DE escape the local optima greedy methods get stuck in.

Now the fitness function, and here I want to resist over-engineering. The dense attacks abstract the goal into some surrogate — a margin loss, a constrained Lagrangian, a target function with penalty terms. I don't need any of that. I want the target class to be likely, so let the fitness just *be* the probability of the target class: for a targeted attack, the candidate's fitness is `f_adv(x + e(x))`, which DE maximizes; for non-targeted, it's `f_t(x + e(x))`, the true-class probability, which DE minimizes. That's it — I optimize the actual quantity I care about, read straight off the softmax. It's classifier-agnostic (I only need the probability vector, the black-box feedback), and it's simpler than any surrogate. No explicit target function, no constraints to enforce — the constraint is already in the encoding, and the objective is just the number the model hands me.

Initialization. Each candidate is `d` tuples of `(x, y, R, G, B)`. The coordinates should cover the image, so draw `x, y` uniformly over the spatial extent — `U(1, 32)` for a 32x32 CIFAR image, `U(1, 227)` for a 227x227 ImageNet image. The colors should cover the full intensity range, including the extremes, because a single pixel that flips a label is often pushed to an extreme value — so draw RGB from `N(mu = 128, sigma = 127)`, a Gaussian centered in the middle of the byte range but wide enough to reach 0 and 255 routinely. When I apply or bound a candidate, the pixel values have to stay in the valid image range.

How many evaluations does this cost, and when do I stop? Each generation evaluates the whole population once, so the number of model queries is population size times number of generations — pop 400 times up to 100 generations in the clean setup. But I don't need to run all 100 if I've already won. Add an early-stop suited to the evaluation setting: in a targeted CIFAR-style attack, the moment the target-class probability clears a high threshold — 90% in that setup — I'm clearly done, stop and save the queries; in a non-targeted ImageNet-style attack, the moment the true-class probability falls below a low threshold — 5% there — the true class has surely lost, stop. The cost is then `pop * generations_until_success`, and I'd measure it in evaluations because that's the currency of a black-box attack — that's what an attacker actually pays.

Is DE even buying me anything over the dumbest possible thing? The honest control is *random* one-pixel search: repeatedly pick a random pixel and random color, keep whatever flips the label, report the best over the same query budget. If random search did as well, the population and the difference-mutation would be pointless. The reason I expect DE to win is structural, not magic: random search throws away everything it learns — each probe is independent, it never uses the current population's geometry to move around a promising region, and it never concentrates probing where the boundary turned out to be close. DE's difference-mutation does use that geometry, and its diversity-keeping selection preserves good locations instead of forgetting them. With the same number of evaluations, the directed population search should find boundaries the scattershot misses — especially for a harder-to-fool network where the vulnerable pixels are rare and random probing rarely lands on one. That's the comparison I'd want to run to justify the machinery.

Let me also hold onto the geometric reading, because it's why this is worth doing beyond just "a sneakier attack." In the abstraction where a spatial pixel is treated as one input dimension, a `d`-pixel perturbation moves the image only along `d` of the `n` coordinate axes, with arbitrary distance along each — so it cuts the input space along a `d`-dimensional axis-aligned slice. One pixel is a single line through the image along one axis; three pixels, a 3-D cube; and so on. In RGB code, each chosen spatial location carries its channel values with it, but the support size is still `d` locations. Contrast the universal-perturbation picture, which adds a small step to *every* coordinate — a move inside a small sphere around the image. Mine is the dual: a *large* move along very few axes instead of a *small* move along all of them. So the experiment isn't only "can I fool the net" — it's "what do the decision boundaries look like along these low-dimensional axis-aligned slices," a probe of the input geometry that the dense-sphere attacks structurally can't perform. That's why one pixel, the extreme case, is the interesting one and not just a stunt.

Now let me turn all of this into the code I'd actually run, filling the three empty slots in the score-based harness — the representation, the score, the decode — and wrapping them with the population-based optimizer. Images in the PyTorch attack live in `[0, 1]`, so the channel-value bounds are `(0, 1)` rather than byte-scale `(0, 255)`. One implementation detail matters: the clean derivation above uses the DE/rand/1 donor with fixed `F = 0.5`, while the torchattacks implementation keeps the same `5d` encoding and probability fitness but delegates to a scipy-derived minimizer whose default donor is `best1bin` with mutation dithering `(0.5, 1)`. Setting `recombination=1` means the trial takes all coordinates from the donor, so there is no parent/child coordinate mixing. The targeted branch also has to replace the original labels with target labels before defining the loss and success test. With those details pinned down, the bounds are `[(0, H), (0, W)]` plus one `(0, 1)` value bound per channel, repeated `d` times; the minimization objective is `1 - prob_target` in targeted mode and `prob_true` in non-targeted mode; the callback stops when the predicted class has flipped the desired way; and decoding writes each tuple into a copy of the image.

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

Let me retrace the causal chain. I started uneasy that everyone bounds the *norm* of a perturbation and spreads it across all pixels, which neither guarantees imperceptibility nor works without white-box gradients. So I flipped the budget to `L0` — bound the *count* of changed pixels, push it to one, leave amplitude free — which directly measures localization but turns the problem combinatorial: I must choose *which* pixels (discrete) and *how much* (continuous) together. Gradient and greedy-saliency methods hit a wall there — they need the Jacobian (white-box), they're greedy into local optima, and they end up touching too many pixels to stay hidden — because a gradient is the wrong instrument for selecting a sparse support. That pushed me to a black-box, gradient-free, diversity-keeping optimizer: differential evolution, whose difference-scaled mutation self-adapts the search radius and whose one-to-one selection preserves the population diversity that lets it escape the local optima greedy search dies in. The discreteness dissolved once I encoded each modified pixel as a 5-tuple `(x, y, R, G, B)` and a candidate as `d` such tuples concatenated — a `5d`-real vector in which the `L0 <= d` budget holds by construction, so no penalty or projection is ever needed, and the discrete pixel choice rides along as continuous coordinates rounded at apply-time. The fitness is just the raw target/true probability off the softmax — the exact quantity I care about, classifier-agnostic, no surrogate. In the clean derivation, DE/rand/1 with `F = 0.5`, no crossover, population a few hundred, and confidence early-stopping gives the search rule; in the PyTorch attack, the same representation and probability objective drop into the scipy-derived DE wrapper with its all-donor `best1bin` trial. Either way, the attack doubles as a probe of decision-boundary geometry along low-dimensional axis-aligned slices of the input space. And it all drops into a standard score-based-attack harness as a representation, a one-line fitness, and a decode wrapped around the existing population optimizer.
