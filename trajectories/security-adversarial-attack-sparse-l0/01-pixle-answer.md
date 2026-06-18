**Problem.** Fill `run_attack` with the cheapest real sparse attack as the ladder's floor: black-box,
query-only, no gradients, against `L2`-robust CIFAR-10 models under an `L0` budget of 24 pixels.

**Key idea (Pixle, pixel-rearrangement).** A natural image already contains every pixel value the attack
needs, so do not *synthesize* colors — *copy* existing ones. Sample a small contiguous source patch, map
each source pixel to a destination, overwrite the destinations. This kills the per-pixel color search
that makes differential-evolution one-pixel attacks expensive, guarantees in-range values for free, and
keeps the `L0` count clean (only destinations change). The outer search is accept-if-improves random
search with restarts; stop the instant the label flips.

**Why it is the floor.** The `torchattacks` call the harness uses is configured minimally:
one-to-four-pixel patches (`x_dimensions=(1,2)`, `y_dimensions=(1,2)`), `pixel_mapping="random"` (no
queries spent choosing *where*), and only `restarts=3` × `max_iterations=5` ≈ 15 evaluations per image,
committing at restart boundaries (`update_each_iteration=False`). A blind, query-starved search for rare
fragile pixels on surfaces that adversarial training deliberately flattened should almost entirely fail.

**Scaffold edit / hyperparameters.** Thin wrapper around `torchattacks.Pixle`; `pixels`/`device`/
`n_classes` unused (Pixle's own knobs govern sparsity; the harness validates `L0 ≤ 24` after the fact).

**What to watch.** Near-zero ASR across `Rebuffi-R18-L2`, `Augustin-L2`, `Engstrom-L2`, establishing the
floor and pointing the next rung at *using* per-pixel importance instead of guessing locations.

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
