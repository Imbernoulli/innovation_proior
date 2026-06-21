The Pixle floor came back almost exactly as feared: mean ASR $\approx 0.011$ — `0.0067` on `Rebuffi-R18-L2`, `0.020` on `Augustin-L2`, `0.0067` on `Engstrom-L2`, one or two successes out of $150$ on two models. The diagnosis is precise: with `pixel_mapping="random"` Pixle spends *no* information deciding which pixels to touch, so fifteen blind throws almost never land on a fragile pixel on surfaces that adversarial training flattened. The lesson is unambiguous — against robust models I cannot guess locations; I have to *compute* which pixels matter and spend the budget on them. The harness grants full model access including gradients, so I no longer have to stay black-box.

I propose **JSMA**, the adversarial saliency-map attack. The reflex from the dense world is to read one loss gradient, but that collapses the whole output vector and the label into a single scalar $J$ and gives one direction over all pixels — useful for a dense $L_\infty$ step, useless for *selecting* a tiny support. What I want instead is the full sensitivity of *every* output to *every* input feature: the forward derivative, the Jacobian $\partial F_j/\partial x_i$. It is per-feature *and* per-class, and crucially it keeps the *sign*, so for each individual pixel I can ask the targeted question a single loss gradient cannot express — does pushing this pixel raise some chosen class while dragging the others down? That is exactly the question a sparse, label-flipping attack lives on. The forward derivative is computable under this threat model: each entry threads the chain rule through every layer, but in practice each row $j$ of the Jacobian is one backward pass with the output seeded at logit $j$, so the whole $\text{n\_classes}\times\text{num\_features}$ matrix costs ten backward passes on CIFAR-10 — cheap, and exactly the white-box capability Pixle declined to use.

The both-signs structure of the saliency is the whole point. To flip the label toward a target class $t$, a useful feature must, when increased, do two things at once: raise the target output, $\partial F_t/\partial x_i > 0$, and lower the rest, $\sum_{j\neq t}\partial F_j/\partial x_i < 0$. A feature failing either test scores zero; among features passing both, the score is the product

$$ S(i,t) = \left(\frac{\partial F_t}{\partial x_i}\right)\cdot\left|\sum_{j\neq t}\frac{\partial F_j}{\partial x_i}\right|, $$

large only when target-help and others-hurt are both large. The product, not a sum, is deliberate: a sum would let a huge target-derivative paper over a near-zero others-effect, selecting a feature that pushes the target up while doing nothing to suppress competitors — on a robust model precisely a wasted pixel. The product demands *both* be substantial, and the absolute value turns "more negative (more helpful)" into "higher score" once the sign gate has passed. One subtlety pins down a real choice: the two conditions carry independent information only if I differentiate the *logits*, not the softmax probabilities — the probabilities sum to one, so raising one mechanically drops the rest and the gate becomes vacuous, and the softmax's saturated derivatives flatten the ranking. On the logits the outputs are unconstrained, so requiring both signs genuinely selects the rare favorable features. Because a single feature is rarely favorable on both axes — most pixels strongly help the target but slightly help a competitor too, and get gated out — the search modifies *two* features at a time, letting one pixel's strongly-negative others-sum compensate the other's slightly-positive one. Each selected feature is saturated to its extreme in one shot ($\theta = +1$, increasing intensity being more reliably misclassified than decreasing it), then dropped from the search domain, the Jacobian is recomputed every iteration because the network is non-linear and the sensitivities shift after every move, and the loop stops when the prediction reaches the target or the budget is spent.

Two configuration choices specific to this task carry as much weight as the saliency itself, and both are bug-fixes a naive fill would miss. First, `torchattacks.JSMA` is targeted-only and needs a target class per sample. The textbook choice is a fixed shift like $(y+1)\bmod n$, but forcing the image toward an arbitrary neighbor class can be far harder than just pushing it off its own class, and on a robust model "harder target" means "fails more." I instead call `set_mode_targeted_least_likely`, which picks per sample the class the model currently rates *least* likely. That is the strongest possible *untargeted* proxy: driving probability mass toward the class the model is most confident is wrong is the most aggressive way to collapse the true class, and it works for any $n$. Second — and this is the failure mode the harness explicitly guards against — the $L0$ budget. JSMA counts in *feature* space: it computes $\text{num\_features} = C\cdot H\cdot W = 3\cdot32\cdot32 = 3072$, sets $\text{max\_iters} = \lceil \text{num\_features}\cdot\gamma/2\rceil$, and modifies two features per iteration, so it touches at most $\text{num\_features}\cdot\gamma$ features. A spatial pixel counts as changed if *any* of its three channels moves, so the feature count upper-bounds the distinct spatial pixels. A carelessly set $\gamma$ — say a constant meant as "10/1024" — perturbs $\sim30$ features and therefore up to $\sim30$ distinct pixels, blowing past the budget, and the harness then rejects *every* such sample, collapsing ASR to zero. The fix is $\gamma = \text{pixels}/(C\cdot H\cdot W) = 24/3072$, so $\text{max\_iters} = \lceil 24/2\rceil = 12$ iterations, $12\times2 = 24$ features touched at most, which upper-bounds distinct spatial pixels by $24$ — exactly the budget. Getting $\gamma$ right is the difference between a valid attack and one the harness throws out wholesale.

JSMA is a genuine step up on the axis Pixle failed — it *uses* a per-pixel importance signal and the gradient the harness now lets me read — so it should clear $0.011$. But I do not expect a large number, and the reason is structural: the saliency is a *first-order, local* quantity computed at the clean image, and the search is greedy with no backtracking, committing to the best pair, saturating it, recomputing, never reconsidering. On a robust `L2` model the first-order signal is exactly what training hardened against, so the both-signs gate finds few features that clear it and the greedy walk can stall in a local optimum where no next pair looks good even though a different support would have worked. My expectation is a mean ASR in the few-percent range — above the floor because it uses saliency, but well short of a search that can escape local optima, which is the next weakness to attack.

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

    # gamma bounds total perturbed features (C*H*W space) to `pixels`, which
    # is a sufficient upper bound on the number of distinct spatial pixels.
    num_features = int(images.shape[1] * images.shape[2] * images.shape[3])
    gamma = float(pixels) / float(num_features)

    attack = torchattacks.JSMA(model, theta=1.0, gamma=gamma)
    # Least-likely class as target -> strong untargeted proxy, works for any
    # n_classes.
    attack.set_mode_targeted_least_likely(quiet=True)
    return attack(images, labels)
```
