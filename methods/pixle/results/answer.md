# Pixle, distilled

Pixle is a black-box, query-efficient `L0` adversarial attack that fools an image classifier by
**rearranging existing pixels** of the image rather than synthesizing new colors. It samples a small
contiguous *source patch*, maps each source pixel to a *destination* coordinate via a mapping
function, **overwrites** (or optionally **swaps**) the destination pixel with the source pixel's
value, and accepts the candidate only if it lowers the probability the model assigns to the true
class. The search is plain random search wrapped in restarts, and it uses only the model's output
probabilities — no weights, no gradients.

## Problem it solves

Craft an adversarial image under a strict pixel (`L0`) budget in the black-box setting where only
output probabilities are observable and queries are costly, on images of any size. The dominant
black-box `L0` predecessor searches both *which* pixels and *what RGB values* to write, via
Differential Evolution over a continuous color cube per pixel — expensive (a whole population per
generation), and it fails on large images. Pixle removes the color-search dimension entirely.

## Key idea

A natural image already contains all the pixel values needed to misclassify it, so there is no need
to search over colors — only over *where existing values go*. Reframe the attack as a **rearrangement
of pixels**:

- Collapses the continuous per-pixel `[0,1]^3` color search to a discrete choice among existing
  positions, shrinking the search space dramatically.
- Every written value is, by construction, a legal in-range pixel — no clipping needed.
- Only the destination pixels change, so the `L0` count is clean.

Search is gradient-free **random search**: hold a committed image, propose localized patch
rearrangements, record candidates that lower the loss, and stop the instant the prediction flips.
Updates are a small *contiguous* patch (convolutional nets are most sensitive to localized changes),
keeping `L0` low per step.

## Method

Loss read off the probability vector: untargeted `L(xbar, y) = f_y(xbar)` (minimize true-class
probability); targeted `L(xbar, ybar) = 1 - f_{ybar}(xbar)`.

Candidate construction. Sample a patch `p = (o_x, o_y, w_p, h_p)` giving source coordinates
`P = {(o_x+i, o_y+j)}`, `|P| = w_p*h_p` (slid inside the image if it overflows). A mapping function
`m: (i,j) in P -> (z,k)` gives each source pixel a destination. The candidate is

```
xbar_{z,k} = x_{i,j}   for (i,j) in P, m(i,j) = (z,k);   xbar = x  elsewhere
```

i.e. overwrite each destination with its source value (or swap the pair, injecting no duplicated
information and converging faster).

Mapping functions `m`:
- **random** — each source pixel goes to a uniformly random target coordinate (fastest; default for
  speed/queries).
- **similarity / distance** — deterministically pick the most similar (`min |x_other - x_src|`,
  smallest visible change) or most different pixel.
- **similarity-random / distance-random** — turn the per-pixel differences into a softmax
  (`1/(1+diff)` for similarity, raw `diff` for distance) and *sample* the destination; the randomness
  prevents repeated deterministic choices and improves exploration.

Search structure:
- **Restart-Iterative** (default): `R` restarts of `T` iterations each; each restart samples
  candidates from the currently committed image, records only candidates that improve the best loss
  seen so far, and moves the committed image at the restart boundary. Conservative — mediocre
  candidates do not immediately become the next base image — so it spends the `L0` budget more
  carefully than greedy acceptance.
- **Iterative**: a single greedy loop that adopts every improving move immediately; fewer iterations
  but can wander into sub-optimal regions with no control.

Both early-stop the moment the prediction flips (or hits the target), so easy images cost only a few
queries. Patch size is the sparsity/query knob: larger patches converge in fewer iterations but move
more pixels (higher `L0`); a side length of one or a few pixels keeps the perturbation tight for a
strict budget.

## Restart-Iterative algorithm

```
xbar <- x ;  l <- f_y(x)
best <- None
for r = 1..R:
    base <- xbar
    for t = 1..T:
        cand <- base
        sample patch p = (o_x, o_y, w_p, h_p);  build P
        for (i,j) in P:  (row,col) <- m(i,j);  cand_{row,col} <- x_{i,j}   # over base
        if f_y(cand) < l:   l <- f_y(cand);  best <- cand                 # best seen so far
        if arg max f(cand) != y:   xbar <- cand;  return xbar      # early stop
    xbar <- cand if best is None else best                         # canonical fallback
return xbar
```

## Working code

```python
import numpy as np
import torch
from torch.nn.functional import softmax


class Pixle:
    """Black-box L0 attack by rearranging existing pixels. Samples a contiguous
    source patch, maps each source pixel to a destination, overwrites (or swaps),
    and keeps a candidate only if it lowers the chosen probability loss. Random search with
    restarts; only output probabilities are used."""

    def __init__(self, model, x_dimensions=(2, 10), y_dimensions=(2, 10),
                 pixel_mapping="random", restarts=20, max_iterations=10,
                 swap=False, update_each_iteration=False, device="cpu"):
        if restarts < 0 or not isinstance(restarts, int):
            raise ValueError("restarts must be an integer >= 0")
        if pixel_mapping.lower() not in ["random", "similarity", "similarity_random",
                                         "distance", "distance_random"]:
            raise ValueError("invalid pixel_mapping")
        self.model = model
        self.device = device
        self.swap = swap
        self.update_each_iteration = update_each_iteration   # False -> restart-iterative
        self.max_patches = max_iterations
        self.restarts = restarts
        self.pixel_mapping = pixel_mapping.lower()
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

    def get_patch_coordinates(self, image, x_bounds, y_bounds):
        c, h, w = image.shape[1:]
        x, y = np.random.uniform(0, 1, 2)
        x_offset = np.random.randint(x_bounds[0], x_bounds[1] + 1)
        y_offset = np.random.randint(y_bounds[0], y_bounds[1] + 1)
        x, y = int(x * (w - 1)), int(y * (h - 1))
        if x + x_offset > w:
            x_offset = w - x
        if y + y_offset > h:
            y_offset = h - y
        return (x, y), (x_offset, y_offset)

    def get_pixel_mapping(self, source_image, x, x_offset, y, y_offset,
                          destination_image=None):
        if destination_image is None:
            destination_image = source_image
        destinations = []
        c, h, w = source_image.shape[1:]
        source_image = source_image[0]
        if self.pixel_mapping == "random":
            for _ in range(x_offset):
                for _ in range(y_offset):
                    row, col = np.random.uniform(0, 1, 2)
                    destinations.append([int(row * (h - 1)), int(col * (w - 1))])
        else:
            for i in np.arange(y, y + y_offset):
                for j in np.arange(x, x + x_offset):
                    pixel = source_image[:, i:i + 1, j:j + 1]
                    diff = (destination_image - pixel)[0].abs().mean(0).view(-1)
                    if "similarity" in self.pixel_mapping:
                        diff = 1 / (1 + diff)
                        diff[diff == 1] = 0
                    probs = torch.softmax(diff, 0).cpu().numpy()
                    indexes = np.arange(len(diff))
                    ranked = iter(sorted(zip(indexes, probs),
                                         key=lambda t: t[1], reverse=True))
                    while True:
                        if "random" in self.pixel_mapping:
                            index = np.random.choice(indexes, p=probs)
                        else:
                            index = next(ranked)[0]
                        _y, _x = np.unravel_index(index, (h, w))
                        if _y == i and _x == j:
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
        s = source[0][src_pixels].view(c, -1)
        rows, cols = idx[:, 0], idx[:, 1]
        if self.swap:
            d = destination[0, :, rows, cols].clone()
            destination[0, :, rows, cols] = s
            destination[0][src_pixels] = d.view(c, yl, xl)
        else:
            destination[0, :, rows, cols] = s
        return destination

    def _bounds(self, images):
        x_b = tuple(max(1, d if isinstance(d, int) else round(images.size(3) * d))
                    for d in self.p1_x_dimensions)
        y_b = tuple(max(1, d if isinstance(d, int) else round(images.size(2) * d))
                    for d in self.p1_y_dimensions)
        return x_b, y_b

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
                    if p < best_p:
                        best_p, best_solution = p, pert_image
                    if callback(pert_image, solution_as_perturbed=True):
                        best_solution, stop = pert_image, True
                        break
                best_image = pert_image if best_solution is None else best_solution
                if stop:
                    break
            adv_images.append(best_image)
        return torch.cat(adv_images)

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
                if p < best_p:
                    best_p, best_image = p, pert_image
                if callback(pert_image, solution_as_perturbed=True):
                    best_image = pert_image
                    break
            adv_images.append(best_image)
        return torch.cat(adv_images)
```
