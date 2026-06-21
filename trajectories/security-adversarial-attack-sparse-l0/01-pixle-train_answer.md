The scaffold's default `run_attack` returns the clean image untouched, an ASR of zero by construction, and the whole climb starts from there: fool an adversarially-robust `L2` CIFAR-10 classifier by moving at most $24$ spatial pixels, amplitude free. Before I spend any complexity I want to know what the *floor* looks like — the cheapest thing that is still a genuine sparse attack — so the rest of the ladder has a measured bar to clear. The cheapest realistic threat model never differentiates the network at all; it only feeds in an image and reads back the probability vector. So the bottom rung is the black-box, query-only attack that needs nothing but `model(x)`, and among such sparse attacks I want the structurally leanest one: no population, no gradient estimation, one forward pass per probe.

I propose **Pixle**, a pixel-rearrangement attack. The defining observation is that a natural image already contains, somewhere inside it, a huge palette of pixel values — bright pixels, dark pixels, every hue the scene happens to hold. So to drop a high-contrast, out-of-place pixel onto a sensitive location I do not need to *synthesize* a new color; I can *copy* an existing pixel's value onto the target. That one move collapses the search. The other black-box sparse attack in this regime, the differential-evolution one-pixel attack, has to search a continuous $[0,1]^c$ color cube *per* modified pixel on top of choosing the location — three real color dimensions for every pixel — and it evaluates a whole population each generation. Rearranging deletes the color search entirely: the question is no longer "what new color do I paint this pixel" but only "which existing pixel's value do I copy here." The continuous per-pixel color dimension vanishes, the search becomes positions-times-positions, and two properties come for free that I would otherwise have to enforce by hand — every value written is by construction already legal in $[0,1]$, so I never clip, and the $L0$ count stays clean because the only pixels that change are the destinations I overwrite.

Concretely, a candidate is built like this: take a small contiguous *source patch* of the image (an origin plus tiny side lengths), and for each source pixel pick a *destination* coordinate where its value gets written; overwrite the destinations, leave everything else untouched. Because the patch is small only a few pixels change per step, which keeps me sparse; because I copy existing values I stay in range. The destinations come from the cheapest possible map, `pixel_mapping="random"`: each source pixel goes to a uniformly random target, no extra queries spent deciding where. There is a deliberate structural prior in moving a *contiguous* source patch rather than scattered pixels — convolutional networks are unusually sensitive to a few *neighboring* pixels changed together with high contrast, the same lesson the query-efficient dense black-box attacks exploit — so the patch keeps the baseline honest while staying the leanest instantiation of the idea. The outer search is plain accept-if-improves random search with restarts: hold a committed image, sample a candidate built off it, evaluate the true-class probability, record a candidate only if it lowers that probability, move the committed image to the best recorded candidate at each restart boundary, and stop the instant the prediction flips — one evaluation per probe, no population.

What makes this the floor *by design* is how thin the configured search is against this target. The patch side lengths are drawn from $(1,2)$ in both dimensions, so each sampled patch is one to four pixels — a single pixel up to a $2\times2$ block. With `restarts=3` and `max_iterations=5` the attack gets on the order of fifteen candidate evaluations per image, committing the running best only at the three restart boundaries (`update_each_iteration=False`, the restart-iterative variant, not the greedy one). Fifteen random tiny patches copied to random destinations, on a *robust* model: the budget allows $24$ changed pixels but this configuration will rarely approach it. Random search throws away everything it learns — each probe is independent, it never uses the geometry of past probes to concentrate where the boundary turned out to be close — and these are `L2`-adversarially-trained models whose entire training objective was to *flatten* the loss surface near each input, making exactly the local, few-pixel sensitivities a random-search attack relies on as scarce as possible. So a scattershot of fifteen tiny random patches is hunting rare needles in a haystack deliberately built to hide them. The weakest link is the random mapping: it spends no queries deciding *where* to put the out-of-place pixel, so against a flattened surface it is essentially hoping to hit a sensitive location by luck, fifteen times. I expect this rung to almost entirely fail — a near-zero ASR across `Rebuffi-R18-L2`, `Augustin-L2`, and `Engstrom-L2` — and that is the point: it gives the measured floor every later rung is judged against, and it already names the fix, which is to *compute* per-pixel importance and spend the budget on it rather than guessing locations.

The literal scaffold edit instantiates `torchattacks.Pixle` with this tiny-patch, random-mapping, restart-iterative configuration and returns `attack(images, labels)`. The `pixels`, `device`, and `n_classes` arguments are unused — Pixle's own patch-size and budget knobs govern sparsity, and the harness validates the $L0$ count after the fact, rejecting any sample that exceeds $24$.

```python
def run_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    pixels: int,
    device: torch.device,
    n_classes: int,
) -> torch.Tensor:
    import torchattacks

    _ = (pixels, device, n_classes)
    model.eval()
    attack = torchattacks.Pixle(
        model,
        x_dimensions=(1, 2),
        y_dimensions=(1, 2),
        pixel_mapping="random",
        restarts=3,
        max_iterations=5,
        update_each_iteration=False,
    )
    return attack(images, labels)
```
