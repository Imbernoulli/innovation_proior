The problem is to fool an image classifier under the harshest realistic conditions at once. I have no access to weights or gradients — I can only feed the model an image and read back the predicted class probabilities, and every such query costs me, since a deployed model will rate-limit or charge, so I must win in as few forward passes as possible. The perturbation must also be sparse in the $L_0$ sense: for a correctly-classified image $x$ of true label $y$ I want an adversarial $\bar x$ with $\arg\max_i f_i(\bar x) \neq y$ (untargeted) or $=\bar y$ (targeted) and $\|x - \bar x\|_0 \le \epsilon$, where the budget $\epsilon$ counts *modified spatial pixels* and each modified pixel may change by any amount in $[0,1]$. Concretely, then: which few pixels do I change, and to what, so the prediction flips, on images that might be $32\times32$ CIFAR thumbnails or full ImageNet frames.

The existing options all fall short. Gradient methods (FGSM, PGD) are out twice over — they need gradients I cannot see, and a gradient step spreads a tiny change across *every* pixel, the opposite of what an $L_0$ budget wants. The white-box $L_0$ methods do not transfer: JSMA builds a saliency map from the Jacobian of the outputs with respect to the input and greedily fixes the most influential pixels to extreme values, but that Jacobian is gradient access; SparseFool linearizes the decision boundary DeepFool-style and sparsifies the result, but it too queries boundary geometry through gradients. That leaves exactly one black-box $L_0$ predecessor, the One-Pixel attack. It encodes a perturbation as a fixed-length array of five-tuples $(x, y, R, G, B)$ — one per modified pixel, specifying *where* it is and *what new color* to write — and optimizes that array with Differential Evolution, whose fitness is the one number I am allowed to read, the true-class probability. It is genuinely black-box and on $32\times32$ images can flip the label by changing a single pixel. But its cost is structural. DE evaluates an entire population — on the order of four hundred candidates — every generation before it even takes a step, and on larger images its success rate collapses and it burns thousands of queries. More importantly, look at *what* it searches: each modified pixel carries a discrete position plus a *continuous color* over the whole $[0,255]^3$ cube, three real dimensions per pixel. The position part is unavoidable, but the color cube is an enormous continuous space, and that is where the queries drain away — into finding the right colors to paint, not the right pixels to touch.

So I asked whether I need to search over colors at all. A natural image already contains hundreds or thousands of distinct pixel values — every brightness and hue the scene happens to hold. What makes a few-pixel change work is dropping *high-contrast, out-of-place* structure where the convolutional network is sensitive, and "out of place" is relative to a *location*: a dark pixel dropped into a bright region is jarring whether its value was synthesized fresh or copied from a shadowed corner of the same image. The image already holds every value I would ever need; the only real decision is which existing value goes to which location. That observation is the whole method, and I call it Pixle: a black-box, query-efficient $L_0$ attack that fools a classifier by *rearranging existing pixels* instead of synthesizing new colors. Deleting the color search collapses the continuous per-pixel $[0,1]^3$ dimension to a discrete choice among existing positions, shrinking the space from positions-times-a-color-cube to positions-times-positions; it guarantees every written value is already a legal in-range pixel, so I never clip; and it gives a clean $L_0$ count, because the only pixels that change are the destinations I overwrite.

The objective is read straight off the probability vector. Untargeted, I minimize the true-class probability $L(\bar x, y) = f_y(\bar x)$, which directly pushes probability mass off the correct class; targeted onto $\bar y$, I instead minimize $L(\bar x, \bar y) = 1 - f_{\bar y}(\bar x)$, driving the target probability up. Crucially, "loss decreased" is not the same as "attack succeeded" — $f_y$ can drop while $y$ is still the argmax — so I keep two separate readouts: a `loss` returning $f_y$ for the accept/reject decision, and a `callback` recomputing the output and checking $\arg\max f \neq y$ (or $=\bar y$ for targeted) for the stop decision. Both read only probabilities, fully black-box throughout.

The candidate is a *rearrangement*. I sample a small rectangular *source patch* — an origin $(o_x, o_y)$ and side lengths $w_p, h_p$ giving source coordinates $P = \{(o_x + i,\, o_y + j)\}$ of size $w_p \cdot h_p$, slid back inside the image if it overflows the edge. These are the pixels whose *values* I relocate. A mapping function $m: (i,j) \in P \to (z,k)$ gives each source pixel a destination, and the candidate is

$$\bar x_{z,k} = x_{i,j} \quad \text{for } (i,j) \in P,\ m(i,j) = (z,k); \qquad \bar x = x \ \text{elsewhere.}$$

I overwrite each destination with its source value and leave everything else untouched; because the patch is small, only a few pixels change per step (sparse), and because I copy existing values they stay in range. A whole candidate is compactly $[o_x, o_y, w_p, h_p]$ plus the list of destinations. The patch is *contiguous* on purpose: convolutional networks are unusually sensitive to a few neighboring pixels changed together with high contrast, far more than to the same number of scattered pixels, so localized updates do more damage per query. There is a sibling move worth noting — instead of merely overwriting, I can *swap* source and destination, reading out the destination's old value and moving it back into the source slot, so no value is duplicated and no information is injected that was not already there. Swapping is a richer move that tends to converge faster; overwriting keeps the rest of the image pristine and lets overlapped pixels be reused later. Overwrite is the default, swap an option, and the sample-a-patch-and-map skeleton is identical either way.

For the mapping $m$ itself, the cheapest choice is *random*: for each source pixel pick a uniformly random target coordinate (not equal to the origin, so the pixel actually moves). This injects high-contrast out-of-place pixels with essentially no extra queries to decide where, and it is the default for raw speed. If imperceptibility matters I can compute, for a source pixel, its per-channel-averaged absolute difference to every other pixel, $\text{diff} = |x_{\text{other}} - x_{\text{src}}|$, and pick the most *similar* destination (smallest visible change) or the most *different* one (maximal jolt). But pure determinism keeps proposing the same handful of moves and walks straight into a local minimum — similarity barely perturbs the logits and gets stuck, distance overshoots and thrashes. The fix is to stop being deterministic: turn the diffs into a distribution and *sample* the destination. For similarity I map $\text{diff} \mapsto 1/(1+\text{diff})$ so small differences become large scores, zero out exact same-color matches, softmax, and sample; for distance I softmax the raw diffs so large differences dominate, and sample. The randomness forces exploration and escapes the basin the argmax falls into. The family is therefore: random (fast default), similarity / distance (deterministic, stealthy vs. blunt), and similarity-random / distance-random (the sampled, exploring versions).

The outer search is gradient-free random search — hold a current iterate, sample one update, accept it only if the loss drops, otherwise discard and resample — which costs one evaluation per step and needs no population, unlike DE. The naive form is greedy iterative: start from the clean image as the running best, sample a candidate off it each step, adopt any improving move immediately, and stop when the prediction flips. The danger is that greedy acceptance has no control over where it wanders: a move can lower $f_y$ a little while dragging the image into a dead-end basin from which the remaining cheap patches cannot recover, and since every accepted move spends part of the pixel budget, a sequence of mediocre accepts wastes both queries and $L_0$. So the default wraps the loop in restarts. Across $R$ restarts of $T$ iterations each, within an inner loop I *freeze* the currently committed image as the base, sample several candidates around it, and record only a candidate that lowers the *best loss seen so far* — I do not let every small improvement immediately become the next base, which would collapse back into greedy accumulation. Improvements compound only at the restart boundary, where I move the committed image to the recorded best candidate before the next batch; if nothing has yet beaten the clean loss, carrying the last trial forward keeps the search from freezing at the initial image. This is more conservative, throwing away mediocre patches rather than baking them in, so it spends the $L_0$ budget more carefully.

Both variants early-stop the instant the argmax flips, so easy images pay only a few queries. The patch-size bound is the knob that trades query count against sparsity: larger patches move more pixels per step, so the loss drops faster and fewer iterations are needed, but $L_0$ grows and the minimal sparse perturbation is overshot; a side length of one or a few pixels keeps $L_0$ tiny at the cost of more iterations, which is exactly what a strict budget calls for. One bookkeeping subtlety governs correctness at the tensor level: the human notation is $(x,y)$, but a PyTorch image is indexed $(\text{channel}, \text{row}, \text{col})$, so destinations are stored as $(\text{row}, \text{col})$; to apply a solution I take the source slice over all channels at rows/cols $y..y+y_{\text{off}}$, $x..x+x_{\text{off}}$ from the clean image, flatten it to $(C, n)$, and write it into a *clone* of the committed destination image at the destination coordinates — for swap I first save the destination's current values, overwrite, then reshape and write them back into the source slot. The clone is what lets accepted moves persist between restarts.

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
