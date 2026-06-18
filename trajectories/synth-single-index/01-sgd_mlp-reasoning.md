The recipe is the whole point, but it fills the empty slots of a fixed MLP harness, and with the most
naive fill that harness *is* the floor — so the place to start is the textbook recipe and the question
of whether plain gradient descent, left to its own devices, moves the first layer toward `theta*` at
all. I have a two-layer ReLU net, a single-index target `y = g(<theta*, x>) + noise`, isotropic
Gaussian inputs, and four callbacks to fill: how to initialise the two layers, what optimiser to
build, what one mini-batch update does, and an optional post-hoc refit. The default fill answers all
four in the most ordinary way, and I want to understand precisely what that ordinary choice can and
cannot do, because that is what every later rung will have to beat.

Initialisation first. The first layer is a `Linear(d, W)` with `W = 256`, `d = 100`. The standard
choice for a layer feeding a ReLU is Kaiming-normal: draw each weight `~ N(0, 2/d)` so the
pre-activation variance is preserved through the ReLU (the factor 2 compensates for ReLU killing half
its inputs). Biases at zero. The readout `fc2` is a `Linear(W, 1)`; I give it a small uniform init in
`[-1/sqrt(W), 1/sqrt(W)]`, the usual `1/sqrt(fan_in)` scale, so the network output starts at `O(1)`
variance rather than exploding. There is nothing single-index-aware here — the rows of `fc1` point in
`W` independent random directions on roughly the sphere of radius `sqrt(2)`, each one essentially
orthogonal to `theta*` (a random unit vector has correlation `~1/sqrt(d)` with any fixed direction, a
basic concentration fact). So at init the network has no idea where `theta*` is; every feature is a
random projection. The entire burden of finding the direction falls on the optimiser.

The optimiser is the second slot. Plain SGD with momentum 0.9, learning rate `1e-2`, no weight decay —
the config defaults. Momentum smooths the mini-batch gradient by an exponential average, which is
standard and helps on the easy, well-conditioned part of the problem; it does not change the
fundamental signal-to-noise story I am about to worry about. The third slot, `training_step`, is the
canonical supervised loop: zero the grads, forward the mini-batch, mean-squared error against the
(noisy) targets, backward, step. Batch size 256, a fresh i.i.d. index each of `8000` steps, so over a
run the optimiser sees `8000 x 256 ~ 2e6` sample-uses drawn from `n_train = 32768`. The optional
`finalize` does nothing — vanilla SGD has no closed-form refit stage.

Now the part that actually decides the outcome: does this find `theta*`? The honest answer depends
entirely on the information exponent `k` of the link, and I can see why from the gradient. Consider one
first-layer row `w_j`. Its gradient under the squared loss is, schematically,
`grad_{w_j} L = E[ 2 (f(x) - y) a_j x 1{<w_j,x> + b_j >= 0} ]`. The piece that carries information
about the direction is the correlation of the target with the gated input, `E[ g(<theta*,x>) x
1{...} ]`. Expand this against the Gaussian in Hermite polynomials and the leading term that points
along `theta*` is governed by the first non-vanishing Hermite coefficient of `g` — the information
exponent `k`. For `k = 1` (ReLU, sign) the first Hermite moment `E[y x] = mu_1 theta*` is already
nonzero, so the very first gradient on every row has an `O(1)`-detectable component pointing along
`theta*`, scaled by `mu_1`. For `k = 3` (the `He_3` link) the first *two* Hermite moments vanish; the
direction does not appear in the gradient until the third-order term, which after contracting against a
row that overlaps `theta*` by only `~1/sqrt(d)` is of size `~ d^{-(k-1)/2} = d^{-1}` — a thousandth,
buried under the per-mini-batch sampling noise, which for a batch of 256 is `~ sqrt(d / 256) ~ 0.6`.
That is the crux: for the hard link the per-step gradient signal is essentially noise, and chasing it
with a small learning rate just rattles the rows around their random init.

So I can predict the split before running it. On `relu-d100` (`k=1`) the first moment is large, the
direction signal is present every step, and ordinary SGD should drive `direction_recovery` close to 1
and `test_mse` to near the noise floor — this is the regime where the textbook recipe genuinely works,
and any later method will at best tie it here. On `sign-d100` (`k=1` but non-smooth) the first moment
is also nonzero, but the link is discontinuous, the readout has to approximate a step from ReLU
features, and the squared-loss landscape is rougher; I expect partial recovery, better than chance but
not reliably near 1, with high seed-to-seed variance depending on whether a given run's random rows
happened to carry enough early overlap. On `hermite-d100` (`k=3`) I expect the worst: the direction
signal is below the mini-batch noise, the rows barely move toward `theta*`, and recovery should land
well short of 1 — this is exactly the `n = Theta(d^{k-1})` wall, seen from the per-mini-batch side
rather than the sample-count side.

There is a second, subtler reason vanilla SGD underperforms on the hard link even when it does find
*some* direction: the first layer and the readout are coupled the whole way through. Every row is free
and trained jointly with the head, so the high-dimensional direction search and the one-dimensional
link fit happen at once and interfere — the head chases whatever the rows currently encode while the
rows are still near the equator, and the joint dynamics can stall in configurations that fit the noisy
targets locally without aligning to `theta*`. Nothing in the default recipe decouples these two jobs,
and nothing aggregates the weak per-step direction signal across the whole dataset; each step sees only
its 256-sample slice. Those are precisely the two levers the later rungs will pull — freezing part of
the net to collapse the landscape, and replacing the noisy mini-batch crawl with one full-batch step
that sums the signal over all `n_train` samples.

Let me make the noise count concrete, because it is the whole reason the hard link is hard and it sets
the bar for everything that follows. The relevant order parameter for any row is its correlation with
the truth, `m = <w_j/||w_j||, theta*>`. At a random init `m ~ 1/sqrt(d) ~ 0.1` for `d = 100` — every
row starts microscopically close to the equator `{m = 0}`. Near the equator the population gradient
that would move a row toward `theta*` scales like `m^{k-1}`: for `k = 1` that is `m^0 = O(1)`
(present from the first step), but for `k = 3` it is `m^2 ~ d^{-1} ~ 0.01`. Against that I have the
empirical gradient noise of a mini-batch of `B = 256` samples, which by a standard concentration count
is `~ sqrt(d/B) = sqrt(100/256) ~ 0.6`. So on the hard link the signal-to-noise ratio per step is
roughly `0.01 / 0.6 ~ 0.02` — the direction signal is essentially invisible inside the mini-batch
gradient, and momentum cannot manufacture a signal that is not there; it only averages noise. The
remedy implied by the same arithmetic is to *aggregate*: summing the gradient over the full
`n_train = 32768` samples cuts the noise to `~ sqrt(d/n_train) ~ 0.055`, finally below the `0.01`-scale
signal once the dimension factor is folded in (the third-order term needs `n ~ d^2 = 10^4 < 32768`
samples to be seen). But vanilla SGD never does that — it crawls in 256-sample steps — so it leaves the
hard-link signal on the table. That is the gap the giant-step rung will close, and the bias-freeze rung
will first attack the *other* half of the problem, the coupling, by collapsing the landscape so the
gradient on each row at least points cleanly along `theta*` when it does carry signal.

I should also be clear about why I do not simply crank the learning rate or run longer to compensate
on the hard link, since those are the obvious knobs and they do not work. A larger `base_lr` amplifies
the noise as much as the (absent) signal, so it makes the rows diffuse faster around their random init,
not converge toward `theta*`; the squared-loss landscape for a wide ReLU net is non-convex and a too-
large step on noisy gradients drives the head to overfit the noisy targets rather than the direction.
Running more steps does not help either, because each step is an independent draw of essentially the
same noise — there is no accumulation of a sub-threshold signal in plain mini-batch SGD with a fresh
i.i.d. batch each step; the per-step bias toward `theta*` is below the floor, so the long-run drift is
a random walk on the sphere, not a descent toward a pole. The only way to surface the signal is to
change the *estimator* (aggregate over the whole dataset) and the *landscape* (decouple direction from
link), which are exactly the structural moves the later rungs make. So I am not under-tuning the
control here; the control is at its honest ceiling, and that ceiling is what the ladder has to clear.

So at step 1 my edit is the trivial one: leave the scaffold at its default. Kaiming first layer, small
uniform readout, SGD with momentum on all parameters, mean-squared-error mini-batch updates, no
finalize. It is the floor by construction — no single-index structure is used, the direction is left
entirely to whatever the joint SGD dynamics extract from 256-sample gradients, and the harness's own
direction estimator `normalize(sum_j |a_j| w_j)` reads off whatever alignment those rows happen to
acquire (the distilled module is in the answer).

What I will watch is exactly the `k`-split. I expect `relu-d100` near-perfect (this is the easy regime
where vanilla SGD is already good), `sign-d100` middling and high-variance (`k=1` but non-smooth), and
`hermite-d100` clearly the weakest (`k=3`, the direction signal drowned by mini-batch noise). The mean
score is then dragged down by the two hard links while the easy one pins near 1. Whatever the precise
numbers, the diagnosis they will support is already pointed at step 2: this is a *landscape and
signal-aggregation* problem, not a learning-rate problem, and the first fix is to stop letting the
first layer and the head fight each other — to freeze the part of the net that does the non-parametric
link fit so the remaining high-dimensional search collapses onto the one direction that matters.
