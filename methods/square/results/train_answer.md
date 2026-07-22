I am locked out of a classifier's internals and can only poke it from outside: I send an image $x_{\hat{}}$, it returns the scores $f(x_{\hat{}}) \in \mathbb{R}^K$, and that is all — no weights, no gradients, no backprop. For a correctly classified pair $(x, y)$ I want a nearby image the model gets wrong, $\arg\max_k f_k(x_{\hat{}}) \neq y$ with $\|x_{\hat{}} - x\|_\infty \le \epsilon$ and $x_{\hat{}} \in [0,1]^d$. Equivalently I minimize the margin $L(f(x_{\hat{}}), y) = f_y(x_{\hat{}}) - \max_{k \neq y} f_k(x_{\hat{}})$, which stays positive while $y$ is the top class and crosses zero exactly when another class overtakes it. The binding constraint is not the box but the fact that every query is rationed: there is a hard per-sample budget $N$, because in any real deployment queries are rate-limited, billed, or logged. So the goal is not "can I find an adversarial example" — with unlimited queries, of course — but "find one in as few forward passes as possible, on as many images as possible." Success rate and average queries are the only numbers that matter.

The standard recipe is to estimate the gradient by sampling: evaluate $L$ at $x \pm \sigma u$ for many random directions $u$, assemble a NES/SPSA finite-difference estimate of $\nabla_x L$, take a PGD step on it, and repeat. It works eventually, but the variance of that estimate scales with the dimension of the space being probed, and for an ImageNet image $d \approx 150{,}000$; getting one usable direction costs many queries, so a single step is expensive and the whole attack runs into the tens of thousands of queries. There is a deeper failure too. A black-box robustness probe must be trustworthy — it must not call a model robust when it is not. But a finite-difference attack follows the *local gradient*, and a large class of defenses do not actually remove adversarial examples; they merely wreck the local gradient signal, shattering it, randomizing it, or flattening it to zero. Those defenses defeat white-box PGD and finite-difference black-box attacks for the same reason: both worship the local gradient. The orthonormal-basis random search SimBA avoids gradient estimation but crawls — it adds small $L_2$ moves one basis direction at a time, and because the basis directions are orthogonal, a move it later regrets can never be undone. The discrete corner-search and tiling attacks exploit useful structure but freeze a coarse grid of allowed positions and sizes in advance, throwing away the freedom to choose where to spend budget. I want something that never touches a local gradient, costs one query per candidate, and is free to overwrite any region of the image it committed to earlier.

I propose Square Attack: a score-based black-box attack that is pure greedy random search on the margin, with a proposal distribution designed from the structure of the problem so that accept-if-better converges in a few dozen to a few hundred queries instead of a few hundred thousand. Random search has the two properties I am chasing for free — it uses only forward evaluations, so it is immune to gradient masking by construction, and it costs one query per proposed candidate, not $O(d)$. The entire game is the proposal distribution, because vanilla random search samples updates uniformly on a sphere and in 150k dimensions a random direction is almost orthogonal to anything useful; I would accept almost nothing and burn the whole budget. Three design choices fix this, each forced by a piece of structure.

First, stay on the boundary. The $L_\infty$ constraint is the box $|x_{\hat{},i} - x_i| \le \epsilon$, and successful $L_\infty$ perturbations almost always have every component at $\pm\epsilon$ — at a *corner* of the box, because if a component helps when moved and it can move by up to $\epsilon$, you want it all the way out. So I make updates of magnitude $2\epsilon$: if a component already sits at $x_i + \epsilon$, adding $-2\epsilon$ and re-projecting clips it to $x_i - \epsilon$, while $+2\epsilon$ keeps it at $x_i + \epsilon$; near the image edge the two attainable extremes are $\max(0, x_i - \epsilon)$ and $\min(1, x_i + \epsilon)$, and the same clipping logic always lands the touched component on one of them. Updates live in $\{-2\epsilon, 0, 2\epsilon\}$ and every touched component returns to a feasible corner rather than wasting budget in the interior — and unlike SimBA's orthogonal moves, a later square can overwrite a region committed to earlier.

Second, make the update a square. A uniform scatter of $\pm 2\epsilon$ is back to the hopeless random direction, so *where* the nonzero entries go matters, and the model is a convolutional network whose first layer correlates small $s \times s$ patches against learned filters $w$. The change induced in a first-layer activation $z = \delta * w$ at position $(u,v)$ satisfies $|z_{u,v}| \le \epsilon \sum_{i,j} |w_{i,j}| \, \mathbf{1}[\text{index falls on a nonzero entry of } \delta]$, which is largest when the entire $s \times s$ receptive window is covered by nonzero entries. So for a fixed budget of $k$ changed pixels, $\|\delta\|_0 = k$, I should shape the support to maximize the number of output positions whose whole window is covered. Building the support greedily one cell at a time while tracking the number $N$ of contained $s \times s$ squares: a thin strip spends about $s$ new cells per new covered window, whereas extending the shorter side of a near-square rectangle of sides $a \ge b$ creates $a - s + 1$ covered windows at once. Keeping the shape as close to square as possible therefore wins; the completed near-square rectangle with $a = \lfloor\sqrt{k}\rfloor$, $b = \lfloor k/a \rfloor$, $r = k - ab$ contains
$$N^* = (a - s + 1)(b - s + 1) + (r - s + 1)^+$$
windows, and when $k = l^2$ the optimum is exactly $a = b = l$, a literal square. The convolutional structure, not an arbitrary preference, pushes the update into a square block — and unlike the fixed-grid attacks I let the block's position be sampled uniformly anywhere each step, keeping the freedom the derivation says to keep.

Third, share one sign per channel across the square. The cheap default is an independent random sign for every pixel, but a spatially constant sign does strictly better against the kind of direction the loss actually responds to, because image gradients are approximately piecewise constant — neighboring pixels in a region want to move the same way. Modeling the relevant direction $v$ over one channel of the square as a constant-sign block: with independent Rademacher signs per pixel, $\langle \delta, v \rangle$ is a sum of independent signed terms and the Khintchine inequality gives $\mathbb{E}|\langle \delta, v \rangle| = \Theta(\|v_{\text{block}}\|_2)$, the signs partially cancelling random-walk style; with one shared sign $\rho$ across the block, $\langle \delta, v \rangle = \rho \sum_{\text{block}} v = \rho \, \|v_{\text{block}}\|_1$ since $v$ has constant sign, so $\mathbb{E}|\langle \delta, v \rangle| = \Theta(\|v_{\text{block}}\|_1)$. For a constant $h \times h$ block $\|v\|_1 = h^2$ against $\|v\|_2 = h$, so the shared sign is aligned with the gradient an entire factor of $h$ better. I do not collapse the color channels into one sign, though — different first-layer color filters can want different channel directions, and the implementation keeps that freedom essentially for free, sampling one $\pm 2\epsilon$ sign per channel ($[c,1,1]$) shared spatially over the square.

How far does the convergence theory reach? For an $L$-smooth objective $g$, with the update $x_{t+1} = x_t + \delta_t$ taken only if it lowers $g$, smoothness gives $g(x_{t+1}) \le g(x_t) + \min\{0, \langle \nabla g(x_t), \delta_t \rangle + \tfrac{L}{2}\|\delta_t\|^2\}$. Applying the identity $2\min\{a,b\} = a + b - |a-b|$ with $a = 0$, then $|A+B| \ge |A| - |B|$, the two $\tfrac{L}{4}\|\delta_t\|^2$ terms combine and
$$g(x_{t+1}) \le g(x_t) + \tfrac{1}{2}\langle \nabla g, \delta_t \rangle + \tfrac{L}{2}\|\delta_t\|^2 - \tfrac{1}{2}|\langle \nabla g, \delta_t \rangle|.$$
Conditioning on $x_t$ with $\mathbb{E}[\delta_t \mid x_t] = 0$ kills the signed term, and given a variance bound $\mathbb{E}\|\delta_t\|^2 \le \gamma_t^2 C$ and a correlation lower bound $\mathbb{E}|\langle \delta_t, v \rangle| \ge \tilde{C}\gamma_t \|v\|$ for every $v$, taking $v = \nabla g(x_t)$ yields $\mathbb{E}\,g(x_{t+1}) \le \mathbb{E}\,g(x_t) - \tfrac{\tilde{C}\gamma_t}{2}\mathbb{E}\|\nabla g(x_t)\| + \tfrac{LC\gamma_t^2}{2}$. Summing $t = 0 \dots T$ with $\gamma_t = \gamma/\sqrt{T}$ and telescoping gives
$$\min_{t \le T} \mathbb{E}\|\nabla g(x_t)\| \le \frac{2}{\gamma \tilde{C}\sqrt{T}}\Big(g(x_0) - \mathbb{E}\,g(x_{T+1}) + \tfrac{\gamma^2 C L}{2}\Big),$$
i.e. $O(1/\sqrt{T})$ to a critical point. I have to be honest about which variant this covers. The correlation bound holds for the *independent*-sign square — there $\mathbb{E}\|\delta\|^2 = 4c\epsilon^2 h^2$ exactly and Khintchine plus convexity of the norm give $\mathbb{E}|\langle \delta, v \rangle| \ge (\sqrt{2}\,\epsilon h^2 / w^2)\|v\|_2$ — but it fails for the per-channel shared-sign square: take $h = 2$ and $v^i_{k,l} = (-1)^{k+l}$ in every channel (a checkerboard), and each channel contributes its own sign times $(+1 - 1 - 1 + 1) = 0$, so $\langle v, \delta \rangle = 0$ and the bound collapses. So the theory anchors the random-search family through the independent-sign square, while the very $L_1$-alignment that makes the per-channel shared-sign square a better image attack is exactly why it sits outside the worst-case $L_2$ bound; the shared-sign version is the image-structured heuristic.

Two remaining slots. The move-size schedule sets $p \in [0,1]$, the fraction of spatial pixels touched, so the side is $s = \mathrm{round}(\sqrt{p \cdot n_{\text{features}} / c})$ clamped to at least 1. Early on I want big coarse squares that can flip the prediction outright; as I close in, big squares overshoot and are more likely to be rejected, wasting queries, so $p$ should shrink over the budget — the direct analogue of step-size decay. I start at $p_{\text{init}} = 0.8$ and halve $p$ at fixed iteration breakpoints, rescaling the current iteration to a 10,000-query reference via $\mathrm{int}(it / N \cdot 10000)$ so the same coarse-to-fine shape stretches to any budget. For initialization, starting from the clean image wastes the first moves finding the boundary; instead I start already on the boundary in a configuration the network is sensitive to. CNNs are disproportionately sensitive to structured high-frequency patterns, and width-1 vertical stripes — each column independently colored $\pm\epsilon$ per channel, broadcast down every row — are exactly such a pattern, so I initialize with random vertical stripes at full $L_\infty$ radius, clipped to $[0,1]$. The margin doubles as the optimization signal and the success test, since I must compute it anyway; I accept a candidate if its loss dropped, but force-accept it if its margin has crossed zero, because a flip is the actual goal and a marginally worse loss reading must not throw away a success. Once a sample's margin is nonpositive I stop spending queries on it, and the loop stops when every sample is fooled or the budget runs out.

```python
import math
import torch
import torch.nn.functional as F


class SquareLinf:
    def __init__(self, model, eps=8 / 255, n_queries=5000, p_init=0.8,
                 loss="margin", resc_schedule=True, seed=0):
        self.model = model
        self.eps = eps
        self.n_queries = n_queries
        self.p_init = p_init
        self.loss = loss
        self.rescale_schedule = resc_schedule
        self.seed = seed

    def margin_and_loss(self, x, y):
        logits = self.model(x)                              # one query per row
        xent = F.cross_entropy(logits, y, reduction="none")
        u = torch.arange(x.shape[0], device=x.device)
        y_corr = logits[u, y].clone()
        logits[u, y] = -float("inf")
        y_other = logits.max(dim=-1)[0]
        margin = y_corr - y_other
        if self.loss == "margin":
            return margin, margin
        return margin, -1.0 * xent

    def p_selection(self, it):
        if self.rescale_schedule:
            it = int(it / self.n_queries * 10000)
        if   10 < it <= 50:    p = self.p_init / 2
        elif 50 < it <= 200:   p = self.p_init / 4
        elif 200 < it <= 500:  p = self.p_init / 8
        elif 500 < it <= 1000: p = self.p_init / 16
        elif 1000 < it <= 2000: p = self.p_init / 32
        elif 2000 < it <= 4000: p = self.p_init / 64
        elif 4000 < it <= 6000: p = self.p_init / 128
        elif 6000 < it <= 8000: p = self.p_init / 256
        elif 8000 < it:         p = self.p_init / 512
        else:                   p = self.p_init
        return p

    def random_sign(self, shape, device):
        return torch.sign(2 * torch.rand(shape, device=device) - 1)

    def random_int(self, low, high, shape, device):
        t = low + (high - low) * torch.rand(shape, device=device)
        return t.long()

    @torch.no_grad()
    def perturb(self, x, y):
        torch.manual_seed(self.seed)
        if x.is_cuda:
            torch.cuda.random.manual_seed(self.seed)
        c, h, w = x.shape[1:]
        n_features = c * h * w

        # vertical-stripe boundary init
        adv = torch.clamp(
            x + self.eps * self.random_sign([x.shape[0], c, 1, w], x.device), 0.0, 1.0)
        margin_min, loss_min = self.margin_and_loss(adv, y)        # 1 query / sample

        for it in range(self.n_queries):
            idx = (margin_min > 0.0).nonzero().flatten()           # unfooled only
            if len(idx) == 0:
                break
            x_curr, x_best_curr, y_curr = x[idx], adv[idx], y[idx]

            p = self.p_selection(it)
            s = max(int(round(math.sqrt(p * n_features / c))), 1)
            vh = int(self.random_int(0, h - s, [1], x.device).item())
            vw = int(self.random_int(0, w - s, [1], x.device).item())

            deltas = torch.zeros([c, h, w], device=x.device)
            deltas[:, vh:vh + s, vw:vw + s] = (
                2.0 * self.eps * self.random_sign([c, 1, 1], x.device))  # one spatial sign/channel

            x_new = x_best_curr + deltas
            x_new = torch.min(torch.max(x_new, x_curr - self.eps),
                              x_curr + self.eps)                   # Linf projection
            x_new = torch.clamp(x_new, 0.0, 1.0)                   # image box

            margin, loss = self.margin_and_loss(x_new, y_curr)     # 1 query / sample

            improved = (loss < loss_min[idx]).float()
            loss_min[idx] = improved * loss + (1.0 - improved) * loss_min[idx]
            miscl = (margin <= 0.0).float()                        # force-accept a flip
            improved = torch.max(improved, miscl)
            margin_min[idx] = improved * margin + (1.0 - improved) * margin_min[idx]
            sel = improved.reshape([-1, *([1] * (x.ndim - 1))])
            adv[idx] = sel * x_new + (1.0 - sel) * x_best_curr

            if (margin_min <= 0.0).all():
                break

        return adv
```
