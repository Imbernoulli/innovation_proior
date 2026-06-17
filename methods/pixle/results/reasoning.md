OK, let me think this through from scratch. I want to fool an image classifier, but under the harshest realistic conditions: I cannot see its weights or gradients, I can only feed it an image and read back the class probabilities, and every such query costs me — a real deployed model will rate-limit or charge me, so I have to win in as few queries as possible. And the perturbation has to be sparse in the L0 sense: I may touch only a handful of *pixels*, though each pixel I touch I can change however I like, anywhere in [0,1]. So the question is concretely: which few pixels do I change, and to what, so that the predicted class flips, using as few forward passes as I can manage, and on images that might be tiny CIFAR thumbnails or full ImageNet frames.

Let me first be honest about why this is awkward. The gradient-based attacks I know — FGSM, PGD — are out twice over. They need gradients, which I don't have in black-box, and they produce *dense* perturbations: a gradient step spreads a little change across every pixel, which is exactly the wrong shape for an L0 budget where I want almost all pixels untouched. The white-box L0 methods don't transfer either. JSMA builds a saliency map from the Jacobian of the outputs w.r.t. the input and greedily fixes the most influential pixels to extreme values — but that Jacobian is gradient access I don't have. SparseFool linearizes the decision boundary DeepFool-style and sparsifies the resulting perturbation — again, it queries the boundary geometry through gradients. Both are white-box. I have to throw the gradient away entirely and search.

So this is a black-box search problem, and there's exactly one prior method that lives in my regime: the One-Pixel attack. Let me really understand what it does, because if I'm going to beat it I need to see precisely where it spends its money. One-Pixel encodes a perturbation as a candidate solution — a fixed-length array of five-tuples, each `(x, y, R, G, B)`: for each pixel it wants to modify, *where* it is and *what new color* to write there. Then it optimizes this array with Differential Evolution. DE keeps a population — on the order of four hundred candidates — and every generation forms children by `x_i' = x_{r1} + F (x_{r2} - x_{r3})` from random members, scale `F` around 0.5, keeping a child only if its fitness beats its parent. The fitness is the one number I'm allowed to read: the probability the model assigns the true class (for untargeted), which DE drives down. It's genuinely black-box — DE only ever *evaluates* the objective, never differentiates it — and on a 32x32 CIFAR image it can flip the label by changing a single pixel. Beautiful in the small.

But now stare at the cost. Two things bother me. The first is that DE evaluates an *entire population* every generation: four hundred forward passes before it even takes one step, and many generations to converge. On big images its success rate collapses and it burns thousands of queries. The reported behavior is exactly that — fine on CIFAR, failing on ImageNet, needing thousands of interrogations. The second thing bothers me more, because it's structural. Look at what the search space *is*. Each modified pixel contributes a discrete position `(x, y)` and a *continuous color* `(R, G, B)` over the full `[0,255]^3` cube. So the search is jointly over "which pixels" and "what values to paint them." The position part is unavoidable — I have to choose which pixels. But the color part is an enormous continuous space, three real dimensions per pixel, and DE has to explore it. That's where the queries go: not so much into finding the right pixels as into finding the right *colors* to put there.

Let me sit with that. Do I actually need to search over colors at all? What would it take to flip the label — do I need some special, carefully-tuned RGB value that doesn't appear anywhere in the image? Think about the picture in front of me: a natural image already contains hundreds or thousands of distinct pixel values — bright pixels, dark pixels, every hue the scene happens to hold. The decision boundary near a natural image is mostly flat with only a few sensitive directions, and what makes a few-pixel change work is putting *high-contrast, out-of-place* structure where the network is sensitive. But "out of place" is relative to a location — a dark pixel dropped into a bright region is jarring whether or not its exact value was synthesized fresh or borrowed from the shadowed corner of the same image. So maybe the right values are *already in the image*. Maybe the image contains, somewhere, all the pixel values I'd ever need; the only thing I really have to decide is which existing value goes to which location.

If that's true, it changes everything, because I can just delete the color search. Instead of asking "what new color do I paint pixel P," I ask "which *existing* pixel's value do I copy onto P." I'm no longer synthesizing colors; I'm *rearranging* pixels that are already there. The continuous `[0,255]^3` per-pixel dimension collapses to a discrete choice among existing positions. The search space shrinks from positions-times-a-color-cube to positions-times-positions. And I get two things for free that I was going to have to enforce by hand otherwise: every value I write is, by construction, a legal pixel value already in `[0,1]`, so I never have to clip or worry about range; and the L0 count is clean, because the only pixels that change are the destinations I overwrite. Rearranging, not repainting. That's the move.

Let me set it up properly. Let `f(x)` give the class probabilities; for the clean image `x` of true label `y`, I want an adversarial `xbar` with `arg max_i f_i(xbar) != y`, and `||x - xbar||_0 <= epsilon`. As an objective I'll minimize the probability of the true class directly, `L(xbar, y) = f_y(xbar)` for untargeted; if I wanted a targeted attack onto `ybar` I'd instead minimize `1 - f_{ybar}(xbar)`, i.e. drive the target probability up. Both are read straight off the output vector, nothing else needed.

Now I need a search over rearrangements, and I want it cheap. DE is heavy precisely because of the population — hundreds of evaluations per step. The far simpler gradient-free template is plain random search: keep a current iterate, sample a random update, accept it only if it lowers the loss, otherwise discard and resample, and stop the instant the image is adversarial. One evaluation per step, no population. Random search has a reputation for being surprisingly effective on exactly these non-convex black-box objectives, and there's a strong precedent in the dense-attack world: a query-efficient black-box `L_inf`/`L2` attack works by sampling random *localized* updates — small contiguous squares — and keeping each one only if the margin loss drops. Localized is the operative word: convolutional networks are unusually sensitive to a few neighboring pixels changed together with high contrast, far more than to the same number of scattered pixels. That tells me my random updates shouldn't be random scattered pixels — they should be a small *contiguous block*.

So here's the candidate I'll sample. Take a small rectangular *patch* of the image as the *source*: an origin `(o_x, o_y)` and small side lengths `w_p, h_p`, giving the set of source coordinates `P = {(o_x + i, o_y + j)}` of size `w_p * h_p` (and if the patch would run off the edge I just slide it back inside). These are the pixels whose *values* I'm going to relocate. Then for each source pixel I need a *destination* — where its value gets written. Let me call that a mapping function `m: (i,j) in P -> (z,k)` returning, for each source pixel, a target coordinate in the image. The adversarial image is then

      xbar_{z,k} = x_{i,j}   for (i,j) in P with m(i,j) = (z,k),   and xbar = x elsewhere.

I overwrite the destination pixels with the source pixels' values; everything else is untouched. Because the patch is small, only a few pixels change per step, so I stay sparse, and because I copy existing values I stay in range. Good — that's the parameterization. A whole candidate rearrangement is compactly `[o_x, o_y, w_p, h_p]` plus the list of destinations, one per source pixel.

Wait — there's a variant worth a beat of thought. If I only *overwrite* the destination, the source pixels still sit unchanged at their original spots, so I've duplicated some values and the original destination values are gone. That's fine if I care about keeping the rest of the image pristine. But if I purely want speed, I could *swap* source and destination simultaneously — move the destination's old value back into the source slot — so no value is duplicated and no information is injected that wasn't there. Swapping should converge faster because it's a richer move; overwriting keeps the image closer to a strict "few pixels relocated" picture and lets overlapped pixels be reused later. I'll keep overwrite as the default and swap as an option; the sampling-a-patch-and-mapping skeleton is identical either way.

Now, how do I choose the destinations — what is `m`? The cheapest thing is just *random*: for each source pixel pick a uniformly random target coordinate (not equal to the origin, so I'm actually moving it). This injects high-contrast out-of-place pixels with essentially no extra queries to decide where, which is what I want for raw speed, and it'll be my default. But I can be cleverer if I care about *imperceptibility*. If my goal is the smallest visible change, I'd rather drop each source pixel onto the *most similar* pixel elsewhere — then the overwrite barely changes the image visually. Concretely, for a source pixel I compute the per-pixel difference to every other pixel, `diff = |x_other - x_source|` averaged over channels, and pick the location with the smallest diff — call that the *similarity* mapping. The mirror image is *distance*: pick the *most different* pixel, maximizing the visual jolt. Let me predict how those two will behave before I commit. Similarity makes each move tiny and unobtrusive, which is great for stealth but means each move barely perturbs the logits, so I'll need more iterations and I'm prone to getting stuck — many near-identical moves that each do almost nothing. Distance makes each move maximally disruptive, which sounds powerful but is so blunt it tends to overshoot and thrash. Neither extreme alone is ideal.

The fix for both is to stop being deterministic. Instead of always taking the single most-similar (or most-distant) pixel, turn the diffs into a *distribution* and sample from it: a softmax over the transformed diffs so that similar pixels get higher probability — for similarity I map `diff -> 1/(1+diff)` so small differences become large scores, and I down-weight exact same-color matches — and then *sample* the destination rather than taking the argmax. For distance I softmax the raw diffs directly so large differences dominate, and sample. The point of sampling is that pure determinism keeps proposing the same handful of moves and walks straight into a local minimum; the randomness forces exploration of the search space and escapes it. So the family is: random (fast default), similarity / distance (deterministic, stealthy vs. blunt), and similarity-random / distance-random (the sampled versions that actually explore). For pure speed and query count, random is the one to reach for.

Now the outer search loop. I have a way to propose a candidate (sample patch, compute `m`, overwrite/swap) and a way to score it (`f_y` of the result). The simplest loop is greedy iterative: start from the clean image as the running best; each iteration, sample a candidate built off the current best, and if it lowers the loss, adopt it as the new best; stop when the prediction flips. Each accepted move is permanently baked into the image. Let me think about whether that's good enough... The worry is that there's no control over *where* greedy acceptance wanders. A move can lower `f_y` a little while dragging the image into a region of the search space that's actually a dead end — a sub-optimal basin from which the remaining cheap patches can't recover. Greedy has no way to back out, and worse, every accepted move spends part of my pixel budget, so a sequence of mediocre accepted moves both wastes queries and inflates L0.

So let me add structure: restarts. Wrap the iterative loop in an outer loop of `R` restarts, each running `T` inner iterations, but with one crucial discipline — within an inner loop I freeze the currently committed image as the base, sample several candidates around that base, and remember only a candidate that lowers the best loss seen so far. I do not let every small improvement immediately become the base for the next inner trial, because that would collapse back into greedy accumulation. At the restart boundary I move the committed image to the recorded best candidate before sampling the next batch. This is more conservative than greedy: it throws away mediocre patches instead of permanently baking them in, so it is less eager to spend the pixel budget on moves that happen to lower `f_y` but drag the image into a poor region. Both versions still early-stop the moment the argmax flips, so easy images pay only a few queries.

Let me write out the restart-iterative procedure carefully, because the base image and the saved candidate are easy to mix up. Start with `xbar <- x` and record `l <- f_y(x)`. For each restart `r`, use the current `xbar` as the fixed destination image for that restart's trials. For each of `T` iterations sample a patch `p = (o_x, o_y, w_p, h_p)`, build `P`, build a candidate by copying each source pixel `x_{i,j}` to `m(i,j)` on top of that fixed destination image, and evaluate `f_y` of the candidate. If the candidate beats the best loss `l` seen so far, record it and update `l`. After the inner loop, commit `xbar` to the recorded best candidate; if no candidate has ever improved the clean loss yet, carrying the last trial forward keeps the search from freezing at the initial image. Check for misclassification during the trials and stop immediately if it happens. The clean image remains the source of copied pixel values, while the committed image is the destination whose already accepted rearrangements are preserved between restarts.

There's one subtlety in the bookkeeping I want to get right: improvements compound only at restart boundaries, not after every inner iteration. Inside one restart, the base image is fixed; across restarts, the base becomes the best recorded candidate. In the greedy iterative version I remove that boundary and update the base immediately whenever the loss decreases. That gives fewer layers of control, which is exactly why it can accumulate small, locally helpful but globally wasteful moves.

Let me now also pin down the patch size, because it controls the whole tradeoff. The patch side lengths `w_p, h_p` are sampled within `[min, max]` bounds. Bigger patches move more pixels per step, so the loss drops faster and I need fewer iterations — but more pixels move at once, so L0 grows and I overshoot the minimal sparse perturbation. Smaller patches (down to a single pixel) keep L0 tiny and find genuinely minimal attacks, at the cost of more iterations. So the patch-size bound is the knob that trades query count against sparsity, and for a strict L0 budget I want it small — a side length of one or a few pixels. In the strict-budget configuration that's exactly what I set: patch sides drawn from the smallest range, random mapping, and a modest restarts-times-iterations budget with early stopping. The early-stop callback is what keeps both queries and L0 low: the moment a candidate is misclassified I return it, no matter how few pixels I've spent.

Let me make sure the loss and the stop condition are doing exactly what I think. The loss I minimize is `f_y(xbar)`, the probability of the true class — minimizing it directly pushes probability mass off the correct class. But "loss decreased" is not the same as "attack succeeded": `f_y` can drop while `y` is still the argmax. So I keep them separate — a `loss` function returning `f_y` for the accept/reject decision, and a `callback` that recomputes the output and checks whether `arg max f != y` (or `== ybar` for targeted) for the stop decision. I evaluate the loss to decide whether to keep a candidate, and the callback to decide whether to halt. For a targeted attack the loss is `1 - f_{ybar}` and the callback checks `arg max f == ybar`. Both read only the probability vector — fully black-box throughout.

Let me convince myself the candidate construction itself is right at the tensor level, because the index juggling is where a bug would hide. A solution is `[x, y, x_offset, y_offset]` followed by a flat list of destination coordinates, one per source pixel. The human notation uses `(x, y)`, but a PyTorch image is indexed as `(channel, row, col)`, so once I turn destinations into tensor indices I store them as `(row, col)`. To apply a solution, take the source slice over all channels and the rows/cols `y..y+y_offset`, `x..x+x_offset` from the clean image — that's the patch's pixel values, flattened to shape `(C, n)` where `n` is the number of source pixels. Then write those values into the destination image at `dest[:, row_list, col_list] = source_values`. For the swap variant I first read out the destination's current values at those `(row, col)` coordinates, do the overwrite, then reshape those saved values back to `(C, y_offset, x_offset)` and write them into the source slot — a genuine exchange. The destination image I write into is a clone of the current committed image, so accepted moves persist between restarts or greedy iterations. That matches the parameterization exactly: source values are existing pixels, destinations are coordinates, only the destination pixels change.

Step back and trace the whole causal chain. I needed a black-box, query-cheap, sparse attack that scales to big images. The only black-box L0 predecessor, One-Pixel, was expensive because Differential Evolution searches a continuous color cube per pixel and evaluates a whole population per step. The unlock was noticing the image already contains every value I need, so I don't search colors at all — I *rearrange* existing pixels, which kills the continuous dimension, guarantees valid in-range values, and gives a clean L0 count. To search rearrangements cheaply I borrowed the accept-if-improves random-search template (one evaluation per step, no population) and its lesson that *localized* updates work best, so each candidate copies a small contiguous *source patch* to destinations chosen by a mapping function — random for speed, similarity/distance and their sampled variants for stealth and exploration. To keep the search from becoming immediate greedy accumulation, I move the committed image only at restart boundaries, while the iterative variant updates it after every improving candidate. I early-stop the instant the prediction flips, so easy images cost only a few queries. That's the method; here it is as code.

```python
import numpy as np
import torch
from torch.nn.functional import softmax


class Pixle:
    """Black-box L0 attack by rearranging existing pixels. Samples a small source
    patch, maps each source pixel to a destination, overwrites (or swaps), and keeps
    a candidate only if it lowers the chosen probability loss. Random search with restarts; only
    output probabilities are used."""

    def __init__(self, model, x_dimensions=(2, 10), y_dimensions=(2, 10),
                 pixel_mapping="random", restarts=20, max_iterations=10,
                 swap=False, update_each_iteration=False, device="cpu"):
        self.model = model
        self.device = device
        self.swap = swap
        self.update_each_iteration = update_each_iteration   # False -> restart-iterative
        self.max_patches = max_iterations
        self.restarts = restarts
        self.pixel_mapping = pixel_mapping.lower()
        # patch side bounds; ints = fixed pixel counts, floats in [0,1] = fraction of side
        if isinstance(x_dimensions, (int, float)):
            x_dimensions = [x_dimensions, x_dimensions]
        if isinstance(y_dimensions, (int, float)):
            y_dimensions = [y_dimensions, y_dimensions]
        self.p1_x_dimensions = x_dimensions
        self.p1_y_dimensions = y_dimensions

    def __call__(self, images, labels):
        # False -> update the committed image at restart boundaries; True -> greedy iterative
        if not self.update_each_iteration:
            return self.restart_forward(images, labels)
        return self.iterative_forward(images, labels)

    # -- query-only primitives: everything reads off the probability vector -------
    @torch.no_grad()
    def _get_prob(self, image):
        out = self.model(image.to(self.device))
        return softmax(out, dim=1).detach().cpu().numpy()

    def _get_fun(self, img, label, target_attack=False):
        img = img.to(self.device)
        if isinstance(label, torch.Tensor):
            label = label.detach().cpu().numpy()
        label = np.asarray(label)
        if label.ndim == 0:
            label = label[None]

        @torch.no_grad()
        def loss(solution, solution_as_perturbed=False):
            pert = solution if solution_as_perturbed else self._perturb(img, solution)
            p = self._get_prob(pert)[np.arange(len(label)), label]
            return (1 - p).sum() if target_attack else p.sum()

        @torch.no_grad()
        def callback(solution, solution_as_perturbed=False):
            pert = solution if solution_as_perturbed else self._perturb(img, solution)
            mx = np.argmax(self._get_prob(pert)[0])
            target = int(label[0])
            return bool(mx == target) if target_attack else bool(mx != target)

        return loss, callback

    # -- candidate construction: sample a contiguous source patch -----------------
    def get_patch_coordinates(self, image, x_bounds, y_bounds):
        c, h, w = image.shape[1:]
        x, y = np.random.uniform(0, 1, 2)
        x_offset = np.random.randint(x_bounds[0], x_bounds[1] + 1)
        y_offset = np.random.randint(y_bounds[0], y_bounds[1] + 1)
        x, y = int(x * (w - 1)), int(y * (h - 1))
        if x + x_offset > w:           # slide back inside the image
            x_offset = w - x
        if y + y_offset > h:
            y_offset = h - y
        return (x, y), (x_offset, y_offset)

    # -- the mapping function m: where each source pixel goes ----------------------
    def get_pixel_mapping(self, source_image, x, x_offset, y, y_offset,
                          destination_image=None):
        if destination_image is None:
            destination_image = source_image
        destinations = []
        c, h, w = source_image.shape[1:]
        source_image = source_image[0]

        if self.pixel_mapping == "random":
            # cheapest: each source pixel -> a uniformly random target row/col
            for _ in range(x_offset):
                for _ in range(y_offset):
                    row, col = np.random.uniform(0, 1, 2)
                    destinations.append([int(row * (h - 1)), int(col * (w - 1))])
        else:
            # similarity/distance: rank targets by per-pixel difference, then either
            # take the extreme (deterministic) or sample from a softmax (exploration)
            for i in np.arange(y, y + y_offset):
                for j in np.arange(x, x + x_offset):
                    pixel = source_image[:, i:i + 1, j:j + 1]
                    diff = (destination_image - pixel)[0].abs().mean(0).view(-1)
                    if "similarity" in self.pixel_mapping:
                        diff = 1 / (1 + diff)     # small difference -> large score
                        diff[diff == 1] = 0       # down-weight exact same-color matches
                    probs = torch.softmax(diff, 0).cpu().numpy()
                    indexes = np.arange(len(diff))
                    ranked = iter(sorted(zip(indexes, probs),
                                         key=lambda t: t[1], reverse=True))
                    while True:
                        if "random" in self.pixel_mapping:
                            index = np.random.choice(indexes, p=probs)   # sample
                        else:
                            index = next(ranked)[0]                      # argmax-down
                        _y, _x = np.unravel_index(index, (h, w))
                        if _y == i and _x == j:    # must actually move the pixel
                            continue
                        destinations.append((_y, _x))
                        break
        return destinations

    def _perturb(self, source, solution, destination=None):
        if destination is None:
            destination = source
        c, h, w = source.shape[1:]
        x, y, xl, yl = solution[:4]
        destinations = solution[4:]
        src_pixels = np.ix_(range(c), np.arange(y, y + yl), np.arange(x, x + xl))
        idx = torch.tensor(destinations, dtype=torch.long, device=self.device)
        destination = destination.clone().detach().to(self.device)
        s = source[0][src_pixels].view(c, -1)              # source patch values (C, n)
        rows, cols = idx[:, 0], idx[:, 1]
        if self.swap:                                      # exchange: no duplicated info
            d = destination[0, :, rows, cols].clone()
            destination[0, :, rows, cols] = s
            destination[0][src_pixels] = d.view(c, yl, xl)
        else:                                              # overwrite destinations only
            destination[0, :, rows, cols] = s
        return destination

    def _bounds(self, images):
        x_b = tuple(max(1, d if isinstance(d, int) else round(images.size(3) * d))
                    for d in self.p1_x_dimensions)
        y_b = tuple(max(1, d if isinstance(d, int) else round(images.size(2) * d))
                    for d in self.p1_y_dimensions)
        return x_b, y_b

    # -- search: restart-iterative (update committed image at restart boundaries) --
    def restart_forward(self, images, labels):
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
        x_bounds, y_bounds = self._bounds(images)
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv_images = []
        for idx in range(images.size(0)):
            image, label = images[idx:idx + 1], labels[idx:idx + 1]
            best_image, pert_image = image.clone(), image.clone()
            loss, callback = self._get_fun(image, label)
            best_solution = None
            best_p = loss(image, solution_as_perturbed=True)
            for _ in range(self.restarts):
                stop = False
                for _ in range(self.max_patches):
                    (x, y), (xo, yo) = self.get_patch_coordinates(image, x_bounds, y_bounds)
                    dests = self.get_pixel_mapping(image, x, xo, y, yo,
                                                   destination_image=best_image)
                    solution = [x, y, xo, yo] + dests
                    pert_image = self._perturb(image, solution, destination=best_image)
                    p = loss(pert_image, solution_as_perturbed=True)
                    if p < best_p:                              # record the best seen so far
                        best_p, best_solution = p, pert_image
                    if callback(pert_image, solution_as_perturbed=True):  # flipped -> stop
                        best_solution, stop = pert_image, True
                        break
                best_image = pert_image if best_solution is None else best_solution
                if stop:
                    break
            adv_images.append(best_image)
        return torch.cat(adv_images)

    # -- search: greedy iterative (commit every improving move) -------------------
    def iterative_forward(self, images, labels):
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
        x_bounds, y_bounds = self._bounds(images)
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv_images = []
        for idx in range(images.size(0)):
            image, label = images[idx:idx + 1], labels[idx:idx + 1]
            best_image = image.clone()
            loss, callback = self._get_fun(image, label)
            best_p = loss(image, solution_as_perturbed=True)
            for _ in range(self.max_patches):
                (x, y), (xo, yo) = self.get_patch_coordinates(image, x_bounds, y_bounds)
                dests = self.get_pixel_mapping(image, x, xo, y, yo,
                                               destination_image=best_image)
                solution = [x, y, xo, yo] + dests
                pert_image = self._perturb(image, solution, destination=best_image)
                p = loss(pert_image, solution_as_perturbed=True)
                if p < best_p:                                  # adopt immediately
                    best_p, best_image = p, pert_image
                if callback(pert_image, solution_as_perturbed=True):
                    best_image = pert_image
                    break
            adv_images.append(best_image)
        return torch.cat(adv_images)
```
