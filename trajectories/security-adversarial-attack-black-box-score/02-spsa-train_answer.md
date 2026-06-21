The floor came back at $\text{asr} = 0.474$ over a flat 65 queries, and the shape of that result names exactly what to fix. It worked on the easier pairs but collapsed on VGG11-BN (0.255 on CIFAR-10, 0.300 on CIFAR-100), and every row read $\text{avg\_queries} = 65$ — handed a budget of 1000, it spent 65 and walked away from the rest. So the floor has two separable diseases: an *economic* one (it quits at 65 queries when 1000 are available) and a *directional* one (isotropic interior nudges in roughly 3000 dimensions are almost orthogonal to any descent direction, so even the steps it takes crawl, and on the harder VGG boundary they crawl to nothing). Curing the economic disease is trivial — run more iterations. The directional one is the real problem, and the question sharpens to: using only the forward queries the floor was throwing away, can I reconstruct an actual *descent direction* instead of guessing one?

I propose SPSA — simultaneous-perturbation gradient estimation, descending the estimate with Adam. Before estimating anything I fix what objective to estimate the gradient *of*, because the floor's plain $f_y$ quietly hurt it. Cross-entropy saturates: once the model is confident the softmax pins at 0 or 1 and the loss goes flat, so its value barely moves as I perturb the input, while the logits stay roughly linear far from the clean point. So I descend the **margin**
$$J(x) = f_y(x) - \max_{k \neq y} f_k(x),$$
positive while $y$ is on top, zero exactly at misclassification, and keeping a usable slope long after cross-entropy has died. I want that slope, because all I will ever have are differences in $J$.

The textbook way to descend an objective you can only measure is finite differences: probe each coordinate to both sides and difference, $\hat{g}_i = (J(x + c e_i) - J(x - c e_i)) / (2c)$, unbiased to $O(c^2)$ because the symmetric difference cancels the even Taylor term. But assembling one full gradient costs two probes per coordinate, $2D$ queries for $D = C \cdot H \cdot W = 3072$ on CIFAR — over six times the entire budget for a *single* step. The right idea, the wrong economics; what I need is an estimate whose cost does not grow with $D$.

The trick that buys that is to drop the coordinate isolation. Perturb *all* coordinates at once by a single random vector $v \in \mathbb{R}^D$ and take the two-sided difference along it, $df = J(x + cv) - J(x - cv)$. Taylor-expanding, $J(x \pm cv) = J(x) \pm c\,(v \cdot \nabla J) + \tfrac{c^2}{2} v^\top H v \pm O(c^3)$, so the symmetric difference is $df = 2c\,(v \cdot \nabla J) + O(c^3)$ — the quadratic terms cancel. That single scalar is the directional derivative along $v$, mixing all $D$ partials. To pull coordinate $i$ back out, divide by $v_i$:
$$\hat{g}_i = \frac{df}{2c\,v_i} = \partial_i J + \sum_{j \neq i} \frac{v_j}{v_i}\,\partial_j J + O(c^2).$$
The $j = i$ term gives exactly $\partial_i J$; every other partial appears too, but carried by a *random* coefficient $v_j / v_i$. This is only useful if that cross-talk averages to zero. If $v$ has independent, mean-zero components, then for $j \neq i$, $\mathbb{E}[v_j / v_i] = \mathbb{E}[v_j]\,\mathbb{E}[1/v_i] = 0$ — *provided* $\mathbb{E}[1/v_i]$ is finite — so $\mathbb{E}[\hat{g}_i] = \partial_i J + O(c^2)$, almost unbiased, from exactly **two** function evaluations regardless of $D$. That is the economics the floor needed.

The finiteness caveat decides everything, so I cannot wave it past. The instinctive $v \sim \mathcal{N}(0, I)$ *fails* it: the Gaussian density is bounded away from zero near $v_i = 0$ while $1/v_i$ blows up there, so the inverse moment diverges and the cross-talk has no usable mean — in practice, occasional enormous estimates whenever some $v_i$ lands near zero. The cure is a mean-zero distribution whose components stay *away* from zero, and the clean one is the symmetric Bernoulli (Rademacher), $v_i \in \{+1, -1\}$ with probability $\tfrac12$ each: mean zero, and since $v_i = \pm 1$, $1/v_i = v_i$ is bounded with finite moments of every order, so the cross-talk truly cancels. A bonus the implementation exploits: because $v_i = \pm 1$, dividing by $v_i$ equals multiplying by it, so $\hat{g} = (df / 2c)\cdot v$ needs no per-coordinate division. The Rademacher choice is not aesthetic; it is forced by the finite-inverse-moment condition.

One simultaneous estimate is correct on average but jittery — each single-direction estimate points partly sideways because of surviving cross-talk, and a defended model's own query noise adds to that. The cure is the same averaging the asymptotics rely on, done *within* a step: draw $n$ independent Rademacher vectors, form $n$ independent two-query estimates, and average, $\bar{g} = \tfrac{1}{n} \sum_i (df_i / 2c)\,v_i$. Variance falls like $1/n$ and the cross-talk is knocked toward its zero mean. This costs $2n$ queries per step, run as one batch (here $n = \text{nb\_sample} = 128$, so $256$ queries per step, evaluated in chunks of $\text{max\_batch\_size} = 64$). That $n$ is the dial between a clean, reliable direction and a cheap noisy one.

Now I have a noisy gradient and I am in stochastic-approximation territory. The plain update $x' = x - a\,\bar{g}$ works, but $\bar{g}$ is noisy and its coordinates are unevenly scaled — some pixels have large derivatives, some tiny, while the noise floor is roughly uniform across them — so a single global step $a$ is the wrong tool, the same per-coordinate-scaling problem that plagues noisy network training. The fix there is **Adam**: keep per-coordinate exponential moving averages of the estimate and its square and step by $\hat{m} / (\sqrt{\hat{v}} + \epsilon)$. The first moment smooths noise across steps, a second layer of averaging on top of the within-step averaging; the $1/\sqrt{\hat{v}}$ rescaling gives each pixel its own effective step so a few large-derivative pixels do not dominate. The estimator is unbiased enough that Adam cannot tell it from a real gradient, and its robustness to noisy, unevenly-scaled gradients is exactly what this directional estimate needs. I run Adam on the perturbation at $\text{lr} = 0.01$.

I keep the variable as the *perturbation* $\delta x$ rather than the image, because the constraint lives on $\delta x$. After each Adam step I project back into the set the harness enforces: clamp $\delta x$ to $[-\varepsilon, \varepsilon]$ (the Euclidean projection onto the $L_\infty$ box, since a box projects coordinate-wise), then clamp $x_0 + \delta x$ to $[0,1]$ and fold the result back into $\delta x$. This is precisely the projection the harness re-checks on return, so getting it right is what keeps a sample from being scored a failure on a validity violation rather than on the model's prediction.

The probe radius $c = \delta = 0.01$ and step $\text{lr} = 0.01$ set the bias–noise trade. Smaller $c$ means smaller Taylor bias ($O(c^2)$), but the *signal* in $df = 2c\,(v \cdot \nabla J)$ scales with $c$ while the model's query-noise floor is fixed, so as $c \to 0$ the per-probe signal-to-noise *degrades* like $c$. The sweet spot is small enough that the directional-derivative approximation is faithful, large enough that $df$ rises above the noise floor; $0.01$ on $[0,1]$-scaled pixels sits there — well inside the $\varepsilon = 8/255 \approx 0.031$ ball, and big enough to produce a measurable $df$. The modest $\text{lr} = 0.01$ keeps the noisy estimate from yanking the iterate out of the productive region in one step.

Now the economics for this harness. A step costs $2 \cdot \text{nb\_sample}$ queries, so the fill sets $\text{nb\_iter} = \max(1, n_{\text{queries}} // (2 \cdot \text{nb\_sample}))$ — it spends *exactly* the budget rather than capping at 64. With $n_{\text{queries}} = 1000$ and $\text{nb\_sample} = 128$ that is $1000 // 256 = 3$ Adam steps, near the full budget, so I expect $\text{avg\_queries} \approx 768$ against the floor's 65. That inverts the floor's economic disease — where random search quit early, SPSA spends the whole allowance reconstructing a real direction — but it also exposes the SPSA bargain: paying $2n$ queries for *one* descent step, even spending the full budget gives only a handful of steps. So SPSA should beat the floor everywhere, lifting the mean from 0.474 well into the 0.8s, doubling the VGG11-BN rows the floor collapsed on into the 0.6 range if their failure was directional, and saturating the ResNet20 pairs toward the low-to-mid 0.9s — at the cost of $\text{avg\_queries}$ climbing from 65 to about 768, with the residual weak spots exactly the rows where 3 noisy steps may still be too few.

```python
import torch
import torch.nn as nn


def run_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float,
    n_queries: int,
    device: torch.device,
    n_classes: int,
) -> torch.Tensor:
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    import torchattacks

    _ = (device, n_classes)
    model.eval()

    nb_sample = 128
    nb_iter = max(1, int(n_queries) // (2 * nb_sample))

    attack = torchattacks.SPSA(
        model=model,
        eps=eps,
        delta=0.01,
        lr=0.01,
        nb_iter=nb_iter,
        nb_sample=nb_sample,
        max_batch_size=64,
    )
    adv_images = attack(images, labels)
    delta = torch.clamp(adv_images - images, min=-eps, max=eps)
    return torch.clamp(images + delta, 0.0, 1.0).detach()
```
