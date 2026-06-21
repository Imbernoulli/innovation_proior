SPSA reached $\text{asr} = 0.816$ — a real estimated descent direction beat blind noise everywhere, and the VGG11-BN pairs the floor collapsed on roughly doubled, confirming that failure was directional. But every row read $\text{avg\_queries} = 768$: SPSA spends the whole budget yet gets only three Adam steps, because one descent step costs $2 \cdot \text{nb\_sample} = 256$ queries. And the VGG pairs only reached about 0.61 — three noisy steps is simply not enough on the harder boundaries. The SPSA bargain is a genuine direction at $O(n)$ queries per step, and on hard boundaries the number of steps is exactly what I am short of. So the question sharpens: can I get many cheap, *effective* moves instead of a few expensive ones? If a single query bought me a whole candidate move I either keep or discard, that is one query per *step* rather than $256$ — hundreds of steps at the same budget — *if* the move is good enough that greedy accept-if-better converges in tens of steps. There is a deeper reason to abandon gradient estimation entirely: SPSA, like any finite-difference method, follows the *local gradient*, and a large class of defenses defeat exactly that by shattering, randomizing, or flattening the gradient signal. A method that never touches a gradient is immune to masking by construction. That points back to pure random search — but with the proposal distribution carrying all the structure the floor lacked.

I propose Square Attack: greedy random search on the margin, with structured square updates, one query per candidate. The objective is the same margin SPSA descended, $J = f_y - \max_{k \neq y} f_k$, used both as the loss to minimize and as the success test ($\texttt{loss="margin"}$). The whole game is now in *what distribution I draw the proposed update from*, and each design choice is forced.

The first is **corners, not interiors**. The constraint is a box, $|x_{\text{adv},i} - x_i| \le \varepsilon$ componentwise, and successful $L_\infty$ perturbations almost always sit at a *corner* of that box, every touched component at $\pm\varepsilon$ — if a component can move by up to $\varepsilon$ and moving it helps, you want it all the way out, not halfway. This is exactly where the floor bled, making interior moves with $\text{step} = \varepsilon/2$. So I use updates of value $\pm 2\varepsilon$ followed by re-projection onto the box: a component already at $x_i + \varepsilon$ plus $+2\varepsilon$ clips back to $x_i + \varepsilon$, and $-2\varepsilon$ clips to $x_i - \varepsilon$. Every touched component lands back on a corner — the full per-component budget spent every step — and, unlike SimBA's orthogonal small steps which can never be undone, a later square can freely overwrite a region committed earlier.

The second is **square support**, and this is where I use the structure SPSA never did: the model is *convolutional*. Its first layer correlates small $s \times s$ patches against learned filters $w$, so the change I induce in a first-layer activation at $(u,v)$ is $z_{u,v} = (\delta * w)_{u,v}$ over the $s \times s$ receptive window. I do not know $w$, but I can bound how much I could move that activation, $|z_{u,v}| \le \varepsilon \sum_{i,j} |w_{i,j}| \cdot \mathbf{1}[\text{index in support of } \delta]$, and the bound is maximal when the receptive window is *fully covered* by my nonzero entries. So for a fixed number $k$ of changed pixels I want the shape that maximizes the count of fully-covered $s \times s$ windows. Building greedily and tracking the number $N$ of fully-covered squares: extending a shape as a long thin strip spends about $s$ cells per newly covered window, while keeping it near-square and adding a strip along the longer side creates many covered windows at once. The optimum for area $k$ is the near-square rectangle — for $k = \ell^2$, a literal $\ell \times \ell$ square. The convolutional structure thus *forces* the update's support to be a square: the shape that maximizes the worst-case change in first-layer activations per pixel of budget. And unlike fixed-grid corner-search attacks, I sample the square's *position* freely anywhere each iteration; freezing a grid throws away the freedom to choose where to spend budget.

The third is the **sign pattern inside the square**. The cheap default is an independent random sign per pixel, but a *spatially shared* sign does better, and the reason is alignment. Greedy random search makes progress when the proposed update correlates with the direction the loss responds to — call it $v$, which behaves like a gradient — and image gradients are approximately piecewise constant, so neighboring pixels want to move the same way. Compare $\mathbb{E}|\langle \delta, v\rangle|$ over one channel of the square. Independent signs make $\langle \delta, v\rangle$ a sum of independent signed terms, and by Khintchine its expected magnitude is $\Theta(\|v_{\text{block}}\|_2)$ — the signs partially cancel, random-walk style. One shared sign $\rho$ across the square gives $\langle \delta, v\rangle = \rho \sum_{\text{block}} v = \rho\,\|v_{\text{block}}\|_1$ for a constant-sign block, so $\mathbb{E}|\langle \delta, v\rangle| = \Theta(\|v_{\text{block}}\|_1)$. And $\|v_{\text{block}}\|_1 \gg \|v_{\text{block}}\|_2$ for a constant block ($h^2$ versus $h$), an entire factor of $h$ better alignment. So I share one sign across the whole square *within* each channel, but keep separate signs *per channel*, since different first-layer color filters can want different channel directions and the implementation keeps that freedom for free.

The fourth is **coarse-to-fine size**. Let $p \in [0,1]$ be the fraction of spatial pixels touched this step; the side is $s = \text{round}(\sqrt{p \cdot n_{\text{features}} / c})$, clamped to at least 1. Early in the search I am far from a solution and want big, coarse moves that can change the prediction outright — large $p$. As I close in, large squares overshoot and are more likely to be rejected, wasting a query, so $p$ shrinks over the budget — the direct analogue of step-size decay, the very thing Adam gave SPSA, here built into the move size. The fill starts at $p_{\text{init}} = 0.8$ and halves $p$ at fixed iteration breakpoints, with $\texttt{resc\_schedule=True}$ rescaling those breakpoints to the actual budget by mapping $\text{it} \to \text{int}(\text{it} / n_{\text{queries}} \cdot 10000)$, so the same coarse-to-fine shape stretches to whatever $n_{\text{queries}}$ is given. And rather than start at the clean image and spend the first moves finding the boundary, I initialize *already* on the boundary at a structured high-frequency point CNNs are known to be sensitive to — random width-1 vertical stripes at full $L_\infty$ radius.

The accept rule is greedy accept-if-better on $J$ with one extra clause: if a candidate is already misclassified ($\text{margin} \le 0$) it is force-accepted regardless of the loss comparison, because crossing the boundary is the actual goal and a marginally worse loss reading must not throw away a successful flip. Once a sample's margin hits zero I stop spending queries on it — the same early exit the earlier rungs used, but it matters far more here because steps are cheap and many samples flip in the first dozen.

The economics are the axis on which SPSA bled, and they now invert. SPSA cost 256 queries per step and got three steps; Square costs **one** query per candidate (one initial query of the stripe init, then one per iteration on the still-unfooled samples), so a 1000 budget buys hundreds of structured steps. More importantly, because each step is one query and a sample exits the instant it flips, the *average* queries over the batch fall far below the budget — easy samples flip in a handful of moves and stop consuming queries, pulling the mean down toward the easy tail. This is the opposite of SPSA, where every sample paid the full 768 regardless. So I expect Square to win on *both* axes at once, which is rare: higher $\text{asr}$ from many better-aligned steps on the hard boundaries, and lower $\text{avg\_queries}$ from cheap steps plus early exit. Against SPSA's numbers, the mean should climb from 0.816 toward the mid-0.9s; the weakest SPSA rows, the VGG11-BN pairs at 0.605 and 0.620, are the sharpest test — if their residual failure was step-starvation, hundreds of cheap structured steps should lift both into the high-0.8s / low-0.9s; the ResNet20 pairs should near-saturate around 0.99; and $\text{avg\_queries}$ should invert from a flat 768 to roughly 70–260 depending on scenario difficulty.

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

    attack = torchattacks.Square(
        model=model,
        norm="Linf",
        eps=eps,
        n_queries=max(1, int(n_queries)),
        n_restarts=1,
        p_init=0.8,
        seed=int(os.environ.get("SEED", "42")),
        verbose=False,
        loss="margin",
        resc_schedule=True,
    )
    adv_images = attack(images, labels)
    delta = torch.clamp(adv_images - images, min=-eps, max=eps)
    return torch.clamp(images + delta, 0.0, 1.0).detach()
```
