The floor came back exactly as the information-exponent argument predicted, and the numbers tell me which failure I am facing. Vanilla joint SGD left $\mathrm{subspace\_err}$ at $1.95/2.37/2.74$ for $r2/r3/r4$ — essentially $\sqrt{2r}=2.0/2.4/2.8$, the value two rank-$r$ orthogonal projectors take when they share *no* directions, so the first layer did not move toward $V^*$ at all. And $\mathrm{test\_mse}$ sat at $10.4/9.8/9.1$, *above* the $\mathrm{Var}[g]=6$ label variance, meaning the untamed joint dynamics is not merely failing to fit but actively injecting prediction variance. Per-seed $\mathrm{subspace\_err}$ barely fluctuates, so this is a structural dead end, not bad luck. Two intertwined causes: the net conflates finding $V^*$ (nonconvex, hard) with fitting the link (convex, easy once features are good), so the readout chases moving near-random features the whole time; and the first-layer signal is third-order-weak, so a single $\eta=5\times10^{-2}$ step cannot move a row $O(1)$ into $V^*$.

I propose a **two-stage, layer-wise recipe** that attacks both causes by refusing to train the two jobs together. Stage 1, the first 80% of the budget (6400 steps), trains *only* the first layer to align with $V^*$. Stage 2, the last 20%, freezes that layer and fits the readout in *closed form* by ridge regression. The split is the whole idea: the nonconvex part gets a clean, well-conditioned alignment phase, and the convex part gets its exact global optimum rather than a co-adapting approximation.

The defining moves of stage 1 follow from looking at the first-layer gradient in isolation. With the readout frozen, a row $w_i$ feels $g_i = a_i\,(1/n)\sum_\nu x^\nu \sigma'(\langle w_i,x^\nu\rangle)(\hat f(x^\nu)-y^\nu)$; the nuisance is $\hat f$, the net's own output polluting the residual. The cleanest signal would be a residual of just $-y$, the pure correlation Stein's lemma turns into a contraction of the teacher's Hermite tensor against the row — the part that rotates $w_i$ into $V^*$. I cannot engineer the from-scratch zero-output symmetric net inside this harness, so I do the next best thing the contract allows: spherical init places each $w_i$ uniformly on $S^{d-1}$ (the clean $O(1/\sqrt{d})$-foot-in-$V^*$ start), and the readout is a *random $\pm1$ sign* vector scaled by $1/\sqrt{W}$ — symmetric in expectation and small in magnitude, so it gives stage 1 a usable signal without dominating it. To hold the readout fixed I set $\mathrm{lr\_outer}=10^{-8}$ (the normalizer rejects exactly zero) *and*, inside `training_step`, zero the output layer's weight and bias gradients after `loss.backward()` and before `optimizer.step()`. The literal enforcement of "freeze the readout" is what kills the co-adaptation that inflated the floor's MSE; a leaking readout would re-introduce it. This is a real departure from the abstract two-stage recipe: I run the *squared-loss* gradient on the live net with the readout pinned, so the residual is $(\hat f - y)$ with $\hat f$ the small-random-readout output, a slightly noisier alignment signal than the pure correlation but the honest one this harness exposes — and with the readout small the $\hat f$ contamination is second-order.

The step-size choice and the spherical reprojection are the load-bearing hygiene. A from-scratch analysis would take one *giant* step $\eta\sim p\sqrt{n/d}$ to cancel the $d^{-(s-1)/2}$ smallness of the leap term in one shot, but I cannot do that safely on a fixed $n=4096$-scale set reshuffled into 128-point batches: a giant step on a finite batch amplifies the sampling fluctuations in the empirical degree-3 correlation, not the signal. So instead of betting everything on one step I take *many moderate ones* — $\mathrm{lr\_inner}=10^{-1}$, twice the floor's rate, over 6400 steps, letting the alignment accumulate across thousands of mini-batch correlations. After each update I re-normalize every row back to the unit sphere, dividing by its norm. I care about a row's *orientation* relative to $V^*$, not its length; renormalizing stops repeated steps from inflating $\|w_i\|$ and muddying the $\|\Pi^* w_i\|/\|w_i\|$ story (and the $\mathrm{subspace\_err}$ SVD that reads it), so a long sequence of moderate steps acts like a clean rotation toward $V^*$ rather than a drift in norm.

I am clear-eyed about the ceiling. The leap order is 3 and the link is a *decoupled* $\sum \mathrm{He}_3$ with no cross terms, so there is no staircase: each teacher direction must be picked up cold from its own degree-3 correlation, with nothing beneath it to expose the next by conditioning. My stage 1 is really $r$ independent third-order alignment problems run in parallel across $256$ rows, and whether any row's empirical degree-3 correlation rises above the finite-batch sampling floor inside 6400 steps is genuinely uncertain. I expect *partial* alignment at best — better than the floor's zero, because freezing the readout and renormalizing removes the two things that were hurting joint SGD, but far from full recovery.

Stage 2 is the part I am confident in, and it is why separating the stages helps the MSE regardless of how well stage 1 does. With $W_\mathrm{in}$ frozen, the features $\phi(x)=\mathrm{ReLU}(W_\mathrm{in}x)$ are fixed, so $\hat f = a\cdot\phi(x)+b$ is *linear* in $(a,b)$ and least-squares fitting it is convex — ordinary ridge regression, no backprop, the global optimum of the readout given whatever subspace stage 1 found. I solve it once: I enlarge `make_dataset` to a cached $n=20{,}000$ set (the ridge wants more rows than the 128-point loop batch to condition the Gram matrix, and $\min(20\text{k}, \texttt{max\_train\_examples})$ keeps me inside the harness's $200\text{k}$ cap), form the post-ReLU feature matrix, append a ones column for the bias, and solve $(\Phi^\top\Phi + \lambda I)\,a = \Phi^\top y$ with $\lambda = 10^{-3}\cdot n$. After the solve the remaining steps just report the train loss without touching the net. The two stages are not independent knobs: the closed-form fit can only be as good as the subspace allows — the lower bound $E[(g-\hat f)^2]\ge E[\mathrm{Var}(g\mid P_U x)]$ is exact — so stage 1 sets the ceiling and stage 2 reaches it, but stage 2 will never make the MSE *worse* the way a co-adapting readout did.

The falsifiable expectations against the floor: $\mathrm{subspace\_err}$ should drop *below* $\sqrt{2r}$ once co-adaptation and norm-inflation are removed — somewhere in a $1.5$–$2.6$ band, most movement on $r2$, least on $r4$; $\mathrm{test\_mse}$ should fall below the floor's $9$–$10$ toward (not all the way to) $\mathrm{Var}[g]=6$; and $\mathrm{score}$ should rise about an order of magnitude into the $10^{-4}$ range, driven mostly by the MSE. The decisive diagnostic is this: if $\mathrm{subspace\_err}$ stays *pinned* at $\sqrt{2r}$ while only the MSE moves, stage 1's degree-3 climb found nothing and the entire gain is the convex ridge on random features — which would force the next rung to abandon gradient-based alignment and estimate the subspace by a *moment* method that sees degree-3 structure directly, rather than waiting for SGD to climb a ladder that is not there.

```python
# EDITABLE region of custom_strategy.py — step 2: two-stage (frozen-readout SGD + ridge)
_FULL_TRAIN_X: torch.Tensor | None = None
_FULL_TRAIN_Y: torch.Tensor | None = None


def init_model(model: nn.Sequential, config: TaskConfig) -> None:
    """Spherical init: rows of W_in on the unit d-sphere, output sign init."""
    in_layer = model[0]
    out_layer = model[2]
    W = torch.randn_like(in_layer.weight)
    W = W / W.norm(dim=1, keepdim=True).clamp(min=1e-8)
    with torch.no_grad():
        in_layer.weight.copy_(W)
        if in_layer.bias is not None:
            in_layer.bias.zero_()
        # Symmetric +/- 1/sqrt(W) output: lets stage 1 see a useful signal.
        signs = torch.randint(0, 2, (out_layer.weight.shape[1],),
                              device=out_layer.weight.device).float().mul_(2).sub_(1)
        out_layer.weight.copy_(signs.unsqueeze(0) / math.sqrt(out_layer.weight.shape[1]))
        if out_layer.bias is not None:
            out_layer.bias.zero_()
        out_layer._full_ridge_solved = False


def make_dataset(
    config: TaskConfig,
    teacher: torch.Tensor,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Larger Gaussian training set (n=20k) to support stage-2 ridge solve."""
    global _FULL_TRAIN_X, _FULL_TRAIN_Y
    g = torch.Generator().manual_seed(seed)
    num_examples = min(20_000, config.max_train_examples)
    x = torch.randn(num_examples, config.n_features, generator=g)
    y = teacher_outputs(x, teacher)
    _FULL_TRAIN_X = x.contiguous()
    _FULL_TRAIN_Y = y.contiguous()
    return x, y


def get_optimizer_config(config: TaskConfig) -> dict[str, object]:
    """SGD on the first layer; output layer is solved analytically later."""
    return {
        "optimizer": "sgd",
        "lr_inner": 1e-1,
        "lr_outer": 0.0 + 1e-8,  # output layer kept ~ frozen during stage 1
        "wd_inner": 0.0,
        "wd_outer": 0.0,
        "momentum": 0.0,
        "noise_std": 0.0,
    }


def training_step(
    model: nn.Sequential,
    optimizer: torch.optim.Optimizer,
    optimizer_config: OptimizerConfig,
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    step: int,
    config: TaskConfig,
) -> dict[str, float]:
    """Two-stage update with spherical projection of the first-layer rows.

    Stage 1 covers the first 80% of steps: gradient step on the squared
    loss while projecting each W_in row back to the unit sphere.
    Stage 2 (last 20%) freezes W_in and refits the output layer once by ridge
    regression on the post-ReLU features computed on the full cached training set.
    """
    model.train()
    stage1_steps = max(1, int(0.8 * config.max_steps))

    if step < stage1_steps:
        optimizer.zero_grad(set_to_none=True)
        preds = model(batch_x).view(-1)
        loss = ((preds - batch_y) ** 2).mean()
        loss.backward()
        # Project the output layer's gradient to zero -- keep it fixed.
        out_layer = model[2]
        if out_layer.weight.grad is not None:
            out_layer.weight.grad.zero_()
        if out_layer.bias is not None and out_layer.bias.grad is not None:
            out_layer.bias.grad.zero_()
        optimizer.step()
        # Re-normalize first-layer rows back to the unit sphere.
        with torch.no_grad():
            in_layer = model[0]
            norms = in_layer.weight.norm(dim=1, keepdim=True).clamp(min=1e-8)
            in_layer.weight.div_(norms)
        return {"loss": float(loss.item())}

    # Stage 2: ridge regression for output weights on the full training set.
    with torch.no_grad():
        in_layer = model[0]
        out_layer = model[2]
        if not getattr(out_layer, "_full_ridge_solved", False):
            if _FULL_TRAIN_X is None or _FULL_TRAIN_Y is None:
                raise RuntimeError("two_stage requires make_dataset to cache the full training set.")
            full_x = _FULL_TRAIN_X.to(device=batch_x.device, non_blocking=True)
            full_y = _FULL_TRAIN_Y.to(device=batch_y.device, non_blocking=True)
            full_features = torch.relu(in_layer(full_x))
            full_feats = torch.cat(
                [full_features, torch.ones_like(full_features[:, :1])],
                dim=1,
            )
            lam = 1e-3 * full_feats.shape[0]
            gram = full_feats.T @ full_feats + lam * torch.eye(
                full_feats.shape[1],
                device=full_feats.device,
                dtype=full_feats.dtype,
            )
            rhs = full_feats.T @ full_y
            sol = torch.linalg.solve(gram, rhs)
            out_layer.weight.copy_(sol[:-1].unsqueeze(0))
            if out_layer.bias is not None:
                out_layer.bias.copy_(sol[-1:].clone())
            out_layer._full_ridge_solved = True

        features = torch.relu(in_layer(batch_x))
        preds = out_layer(features).view(-1)
        loss = ((preds - batch_y) ** 2).mean()
    return {"loss": float(loss.item())}
```
