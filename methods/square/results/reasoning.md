Let me start from what actually hurts. I have a classifier I can only poke from outside: I send it an image, it sends back the scores `f(x_hat)` in `R^K`, and that is all. No weights, no gradients, no backprop. I want, for a correctly classified `(x, y)`, a nearby image that it gets wrong — formally `argmax_k f_k(x_hat) != y` with `||x_hat - x||_inf <= eps` and `x_hat in [0,1]^d`. So I am minimizing the margin `L(f(x_hat), y) = f_y(x_hat) - max_{k != y} f_k(x_hat)`, which is positive while `y` is still the top class and crosses zero exactly when some other class overtakes it. And the constraint that really bites is not the box — it is that every single query costs me. There is a hard budget `N` per image, because in any realistic setting queries are rate-limited or billed or logged. So the objective is not "can I find an adversarial example" — of course I can, given infinitely many queries — it is "find one in as few forward passes as possible, on as many images as possible." Success rate and average queries are the only two numbers that matter.

What does everyone do today? They estimate the gradient. I can't differentiate the model, but I can probe it: evaluate `L` at `x +/- sigma u` for a bunch of random directions `u` and assemble a finite-difference or NES estimate of `grad_x L`, then take a PGD step on that estimate, and repeat. It works, in the sense that it eventually flips the image. But stare at the cost. The variance of that estimate scales with the dimension of the space I'm probing, and for an image `d` is enormous — a hundred and fifty thousand for ImageNet. To get one usable gradient direction I need many directions, so one PGD step costs many queries, and the whole attack runs into the tens of thousands of queries per image. That is the exact opposite of what I want. And there's a second, deeper problem I can't ignore. The whole point of a black-box robustness probe is to be *trustworthy* — to not be fooled into calling a model robust when it isn't. But a finite-difference attack is estimating and following the *local gradient*, and a large class of defenses don't actually remove adversarial examples, they just wreck the local gradient signal — shatter it, randomize it, flatten it to zero. Those defenses defeat white-box PGD and they defeat finite-difference black-box attacks for exactly the same reason: both worship the local gradient. So an attack that estimates the gradient inherits the single failure mode I most want to avoid. I want something that never touches a local gradient at all.

OK, so drop the gradient entirely. What's the most primitive thing I can do with only function values? Propose a random change to my current image, query the model, and keep the change only if the margin went down — pure greedy hill-climbing on the scores. This is random search, the old Rastrigin idea, and it has the two properties I'm chasing for free: it uses *only* forward evaluations, so it's immune to gradient masking by construction, and it costs *one* query per proposed candidate, not `O(d)`. The catch — and this is the whole game — is that vanilla random search samples updates uniformly on a sphere of fixed radius, which in 150k dimensions is hopeless: a random direction is almost orthogonal to anything useful, so I'd accept almost nothing and burn the entire budget. The leverage is entirely in *what distribution I draw the proposed update from*. So the real question becomes: what proposal makes greedy accept-if-better converge in a few dozen to a few hundred queries instead of a few hundred thousand?

Let me look for structure I can exploit. First piece: the constraint is `Linf`, `|x_hat_i - x_i| <= eps` componentwise. That's a box. And there's a well-documented empirical fact about `Linf` adversarial perturbations — the successful ones almost always have every component sitting at `+/- eps`, i.e. at a *corner* of the box. That makes sense: if a component can move by up to `eps` and moving it helps, you want it all the way out, not halfway. So I should not be making tiny continuous nudges like a shrunk gradient step; I should be at the boundary, spending the full per-component budget, all the time. This is the thing that's wrong with the orthonormal-basis random search like SimBA — it adds small `L2`-norm moves one basis direction at a time, so it crawls, and worse, since the basis directions are orthogonal, a move it later regrets can never be undone by subsequent moves; it has spent budget in a direction and can't take it back. I don't want orthogonal small steps. I want to be on the boundary and I want to be able to overwrite a region I committed to earlier.

So the move-design constraint sharpens up: I start at the boundary of the feasible set, and every iteration I want to *stay* on the boundary. The question is what update magnitude keeps me there after the inevitable re-projection. Suppose a component is currently at `x_i + eps`, the upper corner. If I add `+2eps` and clip back into `[x_i - eps, x_i + eps]`, I land at `x_i + eps` again; if I add `-2eps`, I land at `x_i - eps`. So an update of magnitude `2eps` flips a corner component to the *other* corner or keeps it pinned, never leaving it in the interior. Let me check this survives the image-box clipping at the edges, where the two corners are `max(0, x_i - eps)` and `min(1, x_i + eps)`. Take `eps = 0.05` and three pixels — a mid-gray one and two near the box edges:

```
xi=0.50, currently at upper corner 0.550:  +2eps -> 0.550 ;  -2eps -> 0.450   corners [0.450, 0.550]
xi=0.02, currently at upper corner 0.070:  +2eps -> 0.070 ;  -2eps -> 0.000   corners [0.000, 0.070]
xi=0.98, currently at upper corner 1.000:  +2eps -> 1.000 ;  -2eps -> 0.930   corners [0.930, 1.000]
```

In every row the result is exactly one of the two corner values, even when `[0,1]` clipping has already cut one corner short. So updates of magnitude `2eps` are the right quantum: after projection every touched component is at a feasible corner instead of wasting budget in the interior, and a sign flip swaps it to the opposite corner.

Now, *where* do I put the nonzero entries of the update, and *how many*? This is where I have to think about what the model actually is, because a uniformly random scatter of `+/- 2eps` over the image is back to the hopeless high-dimensional random direction. The model is a convolutional network. Its very first layer takes small `s x s` patches of the input and correlates them against learned filters `w`. So the change I induce in a first-layer activation `z = delta * w` at position `(u,v)` is `z_{u,v} = sum_{i,j} delta_{u-., v-.} w_{i,j}`, summed over the `s x s` receptive window. I don't know the filter weights — they're inside the black box — but I can bound how much I could *possibly* move that activation. Taking absolute values, `|z_{u,v}| <= eps * sum_{i,j} |w_{i,j}| * 1[the index falls on a nonzero entry of delta]`. So for a given output location `(u,v)`, the most I can move it is governed by how many of my nonzero perturbation entries fall inside its `s x s` receptive window `C(u,v)`. The best case for an activation is that the window is *completely* covered by my nonzero entries — then the indicator is 1 on all `s^2` of them and the bound is maximal.

So suppose I get to set exactly `k` nonzero entries of `delta` — a fixed budget of changed pixels, `||delta||_0 = k`. I want to *shape* those `k` entries to maximize the number of output positions `(u,v)` whose entire `s x s` receptive window is covered, because each such position is an activation I'm hitting at full strength. Phrase it cleanly: among all shapes of area `k` I could carve out of the image, which shape contains the largest number of fully-covered `s x s` sub-windows? My instinct is that a compact blob beats a long thin strip, but instinct is cheap here, so let me just count. Build the shape up greedily, one unit cell at a time, tracking `N` = number of fully-covered `s x s` squares. Start from an `s x s` block: that's `N = 1`. To get `N = 2` I extend to `s x (s+1)` — one more column gives one more fully-covered window. If I keep extending as a long strip, I spend about `s` new cells for each new covered window. If instead the current completed rectangle has sides `a >= b` and I add cells along the longer side so the shorter side becomes `b+1`, that new strip creates `a - s + 1` covered windows at once. So the greedy way to spend cells is to keep the shape as close to square as possible, not to make a thin rectangle.

Let me put a number on it before I commit, because the strip-vs-blob gap could easily be tiny. Take `k = 12`, `s = 2`, and count `(A-s+1)(B-s+1)` full windows for the rectangles of area `<= 12`:

```
1 x 12 strip : (0)(11)  = 0 full 2x2 windows
2 x 6  rect  : (1)(5)   = 5
3 x 4  rect  : (2)(3)   = 6      <- the near-square
```

The compact `3 x 4` gives 6, the `2 x 6` gives 5, the degenerate strip gives 0. The gap is real, and the winner is the most square-like rectangle. Carrying the greedy argument through, the completed near-square rectangle has sides `a = floor(sqrt(k))` and `b = floor(k/a)`, with `r = k - ab` cells left over; the count is `N* = (a-s+1)(b-s+1) + (r-s+1)^+`, the leftover strip contributing only if it is long enough to complete extra windows. For `k=12` that formula gives `a=3, b=4, r=0`, hence `N* = 2*3 = 6` — matching the direct count above. I want to be sure the formula isn't just lucky on one input, so I brute-force the best rectangle for a range of `(k, s)` and compare:

```
s=2:  k=6 ->4? no, formula 2 vs best 2 ;  k=8 ->3=3 ; k=9 ->4=4 (3x3) ; k=16 ->9=9 (4x4) ; k=20 ->12=12 (4x5) ; k=25 ->16=16 (5x5)
s=3:  k=9 ->1=1 (3x3) ; k=12 ->2=2 (3x4) ; k=16 ->4=4 (4x4) ; k=25 ->9=9 (5x5)
```

The formula equals the brute-force best in every case, and crucially whenever `k = l^2` the maximizer is the literal `l x l` square (`k=9 -> 3x3`, `k=16 -> 4x4`, `k=25 -> 5x5`). So the shape of the update isn't an arbitrary choice; the convolutional structure pushes me to squares. That's why I'll concentrate each step's changed pixels into a square block placed somewhere on the image, rather than scattering them.

And — unlike the tiling and grid attacks that accept "use square-ish regions" but then *freeze* a coarse grid of allowed positions and sizes at the start — I'll let the square's *position* be sampled freely anywhere each iteration, and let its size shrink on a schedule. Freezing the grid throws away exactly the freedom the count above says I should keep: the optimizer should be able to choose where to spend budget.

Now the sign pattern inside the square. The cheap default would be an independent random sign `+/- 2eps` for every pixel and every color channel in the square. But let me ask whether a spatially constant sign does better, because that feels like it throws away degrees of freedom and I want to know if that's actually bad. Greedy random search makes progress when the proposed update `delta` is well-correlated with the direction I'd actually want to move, and the relevant direction, the thing the model's loss responds to, behaves like a gradient `v`. There's a known property of these image gradients: they're approximately *piecewise constant*, so neighboring pixels in a region tend to want to move the same way. I can model `v` over one color channel of my square as roughly a constant-sign block and compare `E|<delta, v>|` for two sampling schemes. If I draw an *independent* Rademacher sign for every pixel, call it `delta^multiple`, then over the block `<delta, v>` is a sum of `h^2` independent signed terms, and by the Khintchine inequality its expected magnitude is `Theta(||v_block||_2)`, the `L2` norm of the block; the signs partially cancel, random-walk style. If instead I draw one sign `rho` and share it across all spatial locations of that channel's square, call it `delta^single`, then `<delta, v> = rho * sum over block of v`, and since `v` has constant sign over the block, that sum is `||v_block||_1`, so `E|<delta, v>| = Theta(||v_block||_1)`.

The whole argument rests on `||v_block||_1 >> ||v_block||_2`, so let me not just assert it — let me draw the two schemes on a constant `h x h` block (take `v = all +1`, entries `+/- 1`) and measure `E|<delta, v>|` for each, for a few `h`:

```
h=2: ||v||_1=4,  ||v||_2=2.00, ratio 2.0  | E|<shared,v>|=4.00  E|<indep,v>|=1.50  shared/indep=2.67
h=3: ||v||_1=9,  ||v||_2=3.00, ratio 3.0  | E|<shared,v>|=9.00  E|<indep,v>|=2.46  shared/indep=3.66
h=5: ||v||_1=25, ||v||_2=5.00, ratio 5.0  | E|<shared,v>|=25.0  E|<indep,v>|=4.03  shared/indep=6.21
h=8: ||v||_1=64, ||v||_2=8.00, ratio 8.0  | E|<shared,v>|=64.0  E|<indep,v>|=6.36  shared/indep=10.1
```

The shared-sign alignment is exactly `||v||_1 = h^2`, with zero variance — every draw is the same magnitude `h^2`. The independent-sign alignment tracks `||v||_2 = h` up to the Khintchine constant (`E|<indep,v>| ~ 0.8 h`, the `sqrt(2/pi)` factor). So the shared-to-independent advantage grows like `h`, an entire extra factor of the square's side, on a coherent block — and that's precisely the regime the piecewise-constant property says I'm in. A spatially shared sign it is. I should not collapse the color channels into one sign, though; different first-layer color filters can want different channel directions, and the implementation can keep that freedom essentially for free. So the update is: a square block, all spatial entries within each channel set to `+/- 2eps`, with one random sign per channel, placed at a uniformly random location.

I should be honest about how far the convergence theory actually reaches, because I want to know whether this per-channel shared-sign square is *provably* a descent-in-expectation method or a heuristic tailored to images. Let me try to prove convergence for the general scheme first and see what assumptions it needs. Take an `L`-smooth objective `g` (think of a smooth-activation net), unconstrained for the analysis. My update is `x_{t+1} = x_t + delta_t` if that lowers `g`, else `x_{t+1} = x_t`. By `L`-smoothness, `g(x_t + delta_t) <= g(x_t) + <grad g(x_t), delta_t> + (L/2)||delta_t||^2`. Since I take the better of staying and moving, `g(x_{t+1}) <= g(x_t) + min{0, <grad g, delta_t> + (L/2)||delta_t||^2}`. Now use the identity `2 min{a,b} = a + b - |a - b|`. With `a = 0` and `q = <grad g, delta_t> + (L/2)||delta_t||^2`, this gives `min{0, q} = (q - |q|)/2`. So `g(x_{t+1}) <= g(x_t) + (1/2)<grad g, delta_t> + (L/4)||delta_t||^2 - (1/2)|<grad g, delta_t> + (L/2)||delta_t||^2|`. The triangle inequality gives `|A+B| >= |A| - |B|`; taking `A = <grad g, delta_t>` and `B = (L/2)||delta_t||^2`, I get `-|A+B|/2 <= -|A|/2 + |B|/2`. Substituting, the two `L||delta_t||^2/4` terms combine, so `g(x_{t+1}) <= g(x_t) + (1/2)<grad g, delta_t> + (L/2)||delta_t||^2 - (1/2)|<grad g, delta_t>|`. Now condition on `x_t`. If `E[delta_t | x_t] = 0`, the signed inner product vanishes. I need two more conditional properties: a variance bound `E[||delta_t||^2 | x_t] <= gamma_t^2 C`, and a correlation lower bound `E[|<delta_t, v>| | x_t] >= Ctilde gamma_t ||v||_2` for every `v`, applied at `v = grad g(x_t)`. Then `E[g(x_{t+1}) | x_t] <= g(x_t) - (Ctilde gamma_t / 2)||grad g(x_t)||_2 + (L C gamma_t^2 / 2)`. Taking total expectations, rearranging, summing over `t = 0..T`, and telescoping the `g` terms gives the average gradient-norm bound. With the constant choice `gamma_t = gamma/sqrt(T)`, the squared-step term contributes only `gamma^2`, so `min_{t<=T} E||grad g(x_t)||_2 <= (2/(gamma Ctilde sqrt(T)))(g(x_0) - E g(x_{T+1}) + gamma^2 C L / 2)`. So `O(1/sqrt(T))` to a critical point is available for any proposal distribution that really has those two moment properties.

But now the uncomfortable check: does my per-channel shared-sign square actually satisfy that correlation lower bound `E|<delta, v>| >= Ctilde gamma ||v||` for *every* `v`? The bound has to hold for the worst-case direction, not just the friendly constant blocks I used to motivate the shared sign. So let me hunt for a direction that the shared sign can't see. The shared sign couples all spatial locations of a channel through a single `rho`, so `<delta, v>` over the square is `2eps * rho * (sum of v over the square)`. If I can find a `v` whose sum over the square is *zero* while `v` itself is nonzero, the inner product is identically `0` for both signs of `rho`, and the lower bound collapses. A sign pattern that sums to zero on any `h x h` block — that's a checkerboard. Take `h = 2` and `v^i_{k,l} = (-1)^{k+l}` in every channel:

```
v = [[ +1, -1 ],
     [ -1, +1 ]]    sum over the 2x2 block = +1 -1 -1 +1 = 0,   ||v||_2 = 2 (nonzero)
```

For the shared sign, `<delta, v> = 2eps * rho * 0 = 0` whether `rho = +1` or `rho = -1`, in every channel — so `E|<delta, v>| = 0` while `||v||_2 = 2 > 0`. The correlation lower bound fails flat. The convergence proof does not cover the per-channel shared-sign square directly. That's a wall, but an informative one. (I have to be careful with the exponent here: I first wrote the checkerboard as `(-1)^{kl}`, but that gives entries `+1,+1,+1,-1` summing to `+2`, not `0`, so it would *not* break the bound — the pattern that actually sums to zero is `(-1)^{k+l}`.)

Let me check the *multiple*-sign variant — independent Rademacher per pixel and channel — against the same checkerboard, to confirm it survives where the shared sign died. With independent signs `<delta, v>` over the block is a sum of independent signed terms, and I measured above that its expected magnitude is `Theta(||v_block||_2)`; on the checkerboard `||v||_2 = 2 != 0`, so a quick draw confirms `E|<delta, v>| ~ 1.5 > 0` rather than collapsing to zero. So the independent-sign square keeps a nonzero correlation on the very direction that killed the shared sign. Working the general bound out, Khintchine gives `E|<delta, v>| >= (2eps/sqrt2) E_{r,s} ||V_{(r,s)}||_2`, convexity of the norm gives `E_{r,s}||V_{(r,s)}||_2 >= ||E_{r,s}V_{(r,s)}||_2`, and with wraparound every coordinate of `v` appears in an `h x h` window with probability `h^2/w^2`, so `||E_{r,s}V_{(r,s)}||_2 = (h^2/w^2)||v||_2` and `E|<delta, v>| >= (sqrt2 eps h^2 / w^2)||v||_2`. The variance is exact: each nonzero entry has magnitude `2eps`, there are `c h^2` of them, so `E||delta||^2 = 4 c eps^2 h^2`. So the *multiple*-sign square is the variant covered by the smoothness proof. The picture that leaves me with is exactly the trade I measured: the independent-sign square is the one with the worst-case guarantee, while the per-channel shared-sign square buys the larger `L1`-scale alignment on locally coherent gradients at the price of being blind to checkerboard directions the worst-case bound has to charge for. For real images, where the gradient is piecewise-constant and almost never a pixel-frequency checkerboard, I'll take the shared sign and keep the proof as the anchor for the family.

Now the move-size schedule. Each step I change a square of side `s`. How big? Let `p in [0,1]` be the fraction of the image's spatial pixels I touch this step; then the square's area is `p` times the spatial area per channel. In code I know the total feature count `n_features = c * H * W`, so the side is `s = round(sqrt(p * n_features / c))`, clamped to at least 1. Early in the search I'm far from a solution and want big, coarse moves that can change the prediction outright: large `p`, large squares. As I close in, large squares overshoot; they're too blunt to make the small final adjustments, and a big square is more likely to hurt the loss and get rejected, wasting a query. So I want `p` to shrink over the course of the budget, the direct analogue of decaying the step size in gradient optimization. I'll start at `p_init = 0.8` in this implementation and halve it at fixed schedule breakpoints. Calibrated to a 10,000-query run, the breakpoints `10, 50, 200, 500, 1000, 2000, 4000, 6000, 8000` walk `p` down through `p_init/2, p_init/4, ... , p_init/512`. And since the budget `N` isn't always 10,000, I rescale: map the current iteration `it` to `int(it / N * 10000)` before reading the schedule, so the same coarse-to-fine shape stretches or compresses to whatever budget I'm given.

The last free slot is the initialization, and for a non-convex greedy search the starting point matters as much as the moves. I could start from the clean image, with perturbation zero, but then my first useful move has to find the boundary from scratch. Better to start already on the boundary and already in a configuration the network is likely to be sensitive to. Here I lean on the second empirical fact about CNNs: they're disproportionately sensitive to certain structured high-frequency input patterns. Vertical stripes of width one, each column independently colored `+/- eps` per channel, are exactly such a high-frequency pattern. So I initialize the perturbation as random vertical stripes at full `Linf` radius, `x + eps * sign(random)` over `[N, c, 1, W]` (one sign per column per channel, broadcast down every row), clipped to `[0,1]`.

For the objective I optimize, the margin `f_y - max_{k != y} f_k` is the natural one and I'll use it both as the thing I minimize and as the success test, since I have to compute it anyway to check whether I'm done. It crosses zero exactly at misclassification. Once a sample has margin `<= 0`, I stop spending queries on it; the query budget should go to images that are not yet flipped. Cross-entropy can be used as an alternative optimization signal, but the success and stop test is still the margin.

Let me assemble the whole loop and make sure every accept/stop rule is right. Initialize `x_best` to the vertical-stripe boundary point, query once to get each sample's margin and loss. Then iterate up to `N` times: identify the still-unfooled samples, the ones with positive margin, and only spend queries on those; pick `p` from the rescaled schedule and `s = max(round(sqrt(p * n_features / c)), 1)`; sample a random top-left corner `(vh, vw)` for the square; build `new_deltas` as zeros with the `s x s` square set to `2*eps * sign(random([c,1,1]))`, one sign per channel shared spatially across the square; form the candidate `x_new = x_best + new_deltas`, project it back into the `Linf` ball around `x` by `min(max(x_new, x - eps), x + eps)`, then clip to `[0,1]`; query the model on `x_new`, one query per active sample. Accept-if-better on the loss, but with one extra rule: if the candidate is already misclassified, `margin <= 0`, I force it to be recorded as the new best regardless of the loss comparison, because crossing the boundary is the actual goal and I don't want a marginally worse loss reading to throw away a successful flip. The stored `loss_min` changes only on a true loss improvement; the stored `margin_min` and `x_best` change on loss improvement or success. Stop a sample as soon as its margin hits zero; stop the whole loop when every sample is fooled or the budget is exhausted. One forward evaluation of a candidate is one query, and the default run uses a single restart.

Here's the method as the code I'd actually run, filling the three slots — initialization, the per-iteration square size, and the single-sign square proposal — that the random-search harness left open:

```python
import math
import torch


def margin_and_loss(model, x, y):
    """Untargeted margin f_y - max_{k!=y} f_k; used as both the loss and the
    success test. margin <= 0  <=>  the sample is already misclassified."""
    logits = model(x)                                  # one query per row of x
    u = torch.arange(x.shape[0], device=x.device)
    y_corr = logits[u, y].clone()
    logits[u, y] = -float("inf")
    y_other = logits.max(dim=-1)[0]
    margin = y_corr - y_other
    return margin, margin                              # loss == margin for loss="margin"


def random_choice(shape, device):
    return torch.sign(2 * torch.rand(shape, device=device) - 1)


def random_int(low, high, shape, device):
    t = low + (high - low) * torch.rand(shape, device=device)
    return t.long()


def p_selection(p_init, it, n_queries, resc_schedule=True):
    """Coarse-to-fine square-area schedule: halve p at fixed iteration breakpoints,
    optionally rescaled to the actual query budget (analogue of step-size decay)."""
    if resc_schedule:
        it = int(it / n_queries * 10000)               # rescale schedule to budget
    if   10  < it <= 50:   p = p_init / 2
    elif 50  < it <= 200:  p = p_init / 4
    elif 200 < it <= 500:  p = p_init / 8
    elif 500 < it <= 1000: p = p_init / 16
    elif 1000 < it <= 2000: p = p_init / 32
    elif 2000 < it <= 4000: p = p_init / 64
    elif 4000 < it <= 6000: p = p_init / 128
    elif 6000 < it <= 8000: p = p_init / 256
    elif 8000 < it:         p = p_init / 512
    else:                   p = p_init
    return p


@torch.no_grad()
def square_attack_linf(model, x, y, eps, n_queries, p_init=0.8, resc_schedule=True):
    c, h, w = x.shape[1:]
    n_features = c * h * w

    # vertical-stripe initialization: [N, c, 1, w] broadcasts down the rows
    x_best = torch.clamp(x + eps * random_choice([x.shape[0], c, 1, w], x.device),
                         0.0, 1.0)
    margin_min, loss_min = margin_and_loss(model, x_best, y)   # 1 query / sample

    for it in range(n_queries):
        idx = (margin_min > 0.0).nonzero().flatten()           # only the unfooled
        if len(idx) == 0:
            break
        x_curr, x_best_curr = x[idx], x_best[idx]
        y_curr = y[idx]

        p = p_selection(p_init, it, n_queries, resc_schedule)
        s = max(int(round(math.sqrt(p * n_features / c))), 1)  # square side
        vh = int(random_int(0, h - s, [1], x.device).item())   # torchattacks helper
        vw = int(random_int(0, w - s, [1], x.device).item())

        # one sign per color channel, shared spatially over the square; value +/-2eps
        # so that after Linf projection the touched components land on a corner
        deltas = torch.zeros([c, h, w], device=x.device)
        deltas[:, vh:vh + s, vw:vw + s] = (
            2.0 * eps * random_choice([c, 1, 1], x.device)
        )

        x_new = x_best_curr + deltas
        x_new = torch.min(torch.max(x_new, x_curr - eps), x_curr + eps)  # Linf proj
        x_new = torch.clamp(x_new, 0.0, 1.0)                            # image box

        margin, loss = margin_and_loss(model, x_new, y_curr)   # 1 query / sample

        improved = (loss < loss_min[idx]).float()
        loss_min[idx] = improved * loss + (1.0 - improved) * loss_min[idx]
        miscl = (margin <= 0.0).float()
        improved = torch.max(improved, miscl)
        margin_min[idx] = improved * margin + (1.0 - improved) * margin_min[idx]
        sel = improved.view(-1, *([1] * (x.ndim - 1)))
        x_best[idx] = sel * x_new + (1.0 - sel) * x_best_curr

        if (margin_min <= 0.0).all():
            break

    return x_best
```

Let me retrace the causal chain. I'm locked out of the model's gradients and every query is rationed, and the two go-to families both fail me: finite-difference gradient estimation costs `O(d)` queries per step and, worse, follows the local gradient and so is fooled by gradient masking, the very thing a robustness probe must not be fooled by. So I went to pure random search — only forward values, one query per candidate, immune to masking by construction — and the entire problem collapsed to designing the proposal distribution so greedy accept-if-better actually converges in this enormous space. The `Linf` corner observation said spend the full per-component budget every step, and the three-pixel clipping check confirmed that updates of magnitude `2eps` re-project onto a corner even at the image edges, so later moves can overwrite earlier regions. The convolutional first layer, through a counting argument over which output windows my changed pixels fully cover — checked on `k=12` and against brute force across a range of `(k,s)` — pushed the changed pixels into a square, the shape that maximizes the number of fully-hit receptive fields for a fixed pixel budget, and told me not to freeze the position in advance. The `L1`-vs-`L2` Khintchine comparison against a piecewise-constant gradient direction, which I measured comes out to a factor of `h`, said a spatially shared sign correlates with the gradient an entire factor of `h` better than independent pixel signs, so I keep one sign per channel across the square; the smoothness convergence proof gives an `O(1/sqrt(T))` guarantee, but only for the independent-sign variant — a `(-1)^{k+l}` checkerboard makes the shared-sign correlation vanish, which is precisely the `L1`-alignment trade showing up as the proof's blind spot. A coarse-to-fine halving schedule on the square area mimics step-size decay and rescales to any budget, and a vertical-stripe boundary initialization starts the search from a structured high-frequency point. The margin objective doubles as the optimization signal and the stop test, and once the margin is nonpositive I stop spending queries on that sample. The whole thing is a few dozen lines: propose a per-channel constant square at a random spot, project to a corner, query once, keep it if the loss dropped or the margin crossed zero, shrink the squares as I go.
</content>
</invoke>
