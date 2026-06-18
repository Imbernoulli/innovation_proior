**Problem (from step 2).** JSMA's greedy first-order saliency cleared the floor (mean ASR ≈ 0.047) but
stalled low — local, gradient-based, no backtracking, exactly what `L2` adversarial training flattens.
Escape it with a search that maintains diversity and only *evaluates* the objective.

**Key idea (One-Pixel, differential evolution).** Optimize a sparse perturbation with DE, a gradient-free
population metaheuristic. Encode one modified pixel as a 5-tuple `(x, y, R, G, B)`; a candidate is `d`
such tuples concatenated — a `5d`-real vector — so `||e||_0 <= d` holds *by construction* (no penalty, no
projection; the discrete pixel choice rides as continuous coordinates rounded at apply-time). Each
generation, every parent gets a child via DE difference-mutation `x_{r1} + F*(x_{r2}-x_{r3})`, kept only
if it improves (one-to-one tournament). Fitness is the raw true-class probability `f_t(x+e)`, minimized.

**Why it beats greedy saliency.** The difference-scaled mutation self-adapts the search radius (large
while spread, small as it converges) and the one-to-one selection preserves diversity, so DE jumps out of
the local optima a greedy first-order method dies in — JSMA's exact failure mode.

**Scaffold edit / hyperparameters.** Thin wrapper around `torchattacks.OnePixel`, untargeted, `pixels=24`
(a `120`-dim DE search per image), `steps=6` generations, `popsize=8`, `inf_batch=128` (batches forward
passes only). `device`/`n_classes` unused.

**What to watch.** Mean ASR clears 0.047 into the low-mid teens (population search escapes local optima),
but its own *query starvation* — six generations over a 120-dim space on flattened surfaces — caps it well
under the strongest sparse attacks, pointing the next rung at far more query-efficient search.

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

    _ = (device, n_classes)
    model.eval()
    attack = torchattacks.OnePixel(
        model,
        pixels=pixels,
        steps=6,
        popsize=8,
        inf_batch=128,
    )
    return attack(images, labels)
```
