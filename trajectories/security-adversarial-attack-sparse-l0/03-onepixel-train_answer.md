JSMA cleared the Pixle floor but only just: mean ASR $\approx 0.047$ — `0.0267` on `Rebuffi-R18-L2`, `0.0467` on `Augustin-L2`, `0.0667` on `Engstrom-L2`. Roughly quadrupling Pixle's $0.011$ confirms that *using* a per-pixel importance signal beats guessing locations, but the flat, low magnitude across all three models is the tell. JSMA's saliency is a *local, first-order* quantity read at the clean image, and the greedy walk commits to the best-scoring pair, saturates them, and never reconsiders — and `L2`-adversarial training is precisely hardening against that first-order signal. The both-signs gate clears few features, the greedy search picks among weak candidates, and it stalls in local optima where some *other* support would have flipped the label. The gradient is honest, but the surface lies to it locally, and greedy with no backtracking cannot recover. So I drop the gradient entirely and search.

I propose the **One-Pixel** attack: optimize the sparse perturbation with **differential evolution**, a gradient-free population metaheuristic. The point is not that gradients are useless — it is that a *greedy, local, first-order* use of them gets trapped on a flattened surface, and a method that maintains a diverse population and only ever *evaluates* the objective sidesteps both the locality and the greediness. Differential evolution has exactly the two properties JSMA lacked. It keeps a *population* and forms each child from a scaled difference of two other members,

$$ x_i' = x_{r_1} + F\,(x_{r_2} - x_{r_3}), $$

so the search radius *self-adapts* — large while the population is spread (exploration), small as it converges (refinement) — and a child competes only with *its own parent* in a one-to-one tournament, which preserves diversity instead of letting a few strong candidates clone themselves across the population. JSMA stalled because once it saturated a pixel it could never undo that choice; DE never commits irrevocably, a candidate that looks good early can be displaced by its own descendants, and the population holds *many* supports at once, so a promising-but-different combination is not crushed by the current leader. And JSMA leaned on a local first-order signal that robust training suppressed; DE reads only the *global* objective value — the actual probability the model assigns the true class on the full perturbed image — so it is immune to the gradient being small. It does not care that the surface is locally flat; it cares only whether a candidate, as a whole, lowers the true-class probability. That is the right instrument for a surface engineered to defeat first-order attacks.

The unlock that makes DE fit a sparse problem is the encoding. DE optimizes a real vector, but my problem has a discrete part — which pixel. So I encode one modified pixel as a $5$-tuple $(x, y, R, G, B)$ — a location and a color — and a candidate with budget $d$ is $d$ such tuples concatenated, a real vector of length $5d$. By writing exactly $d$ tuples and leaving every other coordinate at zero, $\lVert e\rVert_0 \le d$ holds *by construction* — no penalty, no projection, no constrained optimization. The discrete "which pixel" choice rides along as continuous $(x,y)$ coordinates rounded to indices at apply-time; DE never knows it is looking at an image, it searches $5d$ reals and I decode. The fitness is the one number the black box hands me: for the untargeted attack here, the true-class probability $f_t(x+e)$, which DE *minimizes* — no surrogate loss, the exact quantity I care about read straight off the softmax.

The configuration here diverges sharply from textbook DE and it bounds what I should expect. The fill is untargeted `OnePixel` with `pixels=24`, so the encoding is $24$ five-tuples, a $120$-dimensional DE search per image — a large search space relative to the budget the optimizer is given. The `torchattacks` implementation delegates to a scipy-derived differential-evolution minimizer whose population is `popsize` times the parameters per member, so `popsize=8` over a $120$-dim problem is a real population, but `steps=6` means only six generations of evolution before it stops. Six generations of DE over a $120$-dim space is a *coarse* search — enough to find the obviously-fragile pixels, nowhere near enough to refine $24$ well-placed ones on a robust model. (`inf_batch=128` only batches the forward passes for speed; it buys no extra search.) There is a structural reason $24$ pixels is hard for this configuration: DE evolves the whole $120$-dimensional vector jointly, and the difference-mutation perturbs many coordinates at once. Early generations, population spread, take large exploratory steps that scatter all $24$ pixels; only as the population converges do the steps shrink enough to refine individual placements, and six generations barely reaches that refinement phase. This is the opposite failure from JSMA — JSMA placed each pixel carefully but greedily and locally, OnePixel places all pixels by global search but coarsely. Neither has both careful placement and a global, reconsiderable view.

DE's structural advantage over JSMA is real and on the exact axis JSMA failed: it does not commit greedily and it does not rely on a local first-order signal, so it should not stall in the same local optima. With $24$ pixels of budget and a diversity-keeping population, it should find more flips than greedy saliency — I expect it to clear $0.047$ comfortably and land in the low-to-mid teens of percent. But I do not expect it to approach the strongest sparse attacks, because six generations is too few to evolve a good $24$-pixel support against a flattened surface: DE will harvest the easy fragile-pixel flips and run out of generations before solving the harder samples. Its ceiling is *query starvation*, not the wrong-tool problem JSMA had — which points the next rung at spending queries far more efficiently.

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
