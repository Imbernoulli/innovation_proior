**Problem.** Vanilla joint SGD left `subspace_err` pinned at `√(2r)` (uncorrelated subspaces) and
`test_mse` *above* `Var[g] = 6` — an optimization dead end where the net conflates finding `V*` (hard,
nonconvex) with fitting the cubic link (easy, convex), so the readout chases moving near-random features
and inflates the MSE. The cure is to separate the two jobs.

**Key idea (two-stage / layer-wise).** *Stage 1* (first 80% of steps): train only the first layer to
align with `V*` — spherical init on `S^{d-1}`, a small random ±1 readout held fixed (output-layer
gradient zeroed each step, `lr_outer ≈ 0`), moderate `lr_inner = 1e-1` over thousands of mini-batch
correlations, and **re-projection of each row onto the unit sphere** after every step so the climb is a
clean rotation, not a norm drift. *Stage 2* (last 20%): freeze `W_in` and fit the readout in **closed
form** by ridge regression on the post-ReLU features of a cached `n = 20{,}000` set — the convex global
optimum given the learned subspace, which stops the readout from inflating variance.

**Why it should beat the floor.** Freezing the readout + renormalizing removes the two things hurting
joint SGD, so even partial degree-3 alignment pulls `subspace_err` below `√(2r)`; the ridge solve gives
the best possible fit on whatever features stage 1 found, pulling `test_mse` toward `Var[g] = 6`. But
the link is a *decoupled* `Σ He₃` with no lower-degree staircase, so stage 1 must find each direction
cold from a third-order signal — recovery is partial, and stage 1's ceiling bounds stage 2's MSE
(`E[(g−f̂)²] ≥ E[Var(g | P_U x)]`).

**Hyperparameters.** `optimizer="sgd"`, `lr_inner = 1e-1`, `lr_outer = 1e-8` (frozen), `wd = 0`,
`momentum = 0`, `noise_std = 0`; `n = min(20_000, max_train_examples)`; stage-1 fraction 0.8; ridge
`λ = 10⁻³·n`.

**What to watch.** `subspace_err` in the 1.5–2.6 band (below `√(2r)`, most movement on r2), `test_mse`
falling below 9–10 toward 6, `score` up roughly an order of magnitude into the `1e-4` range. If
`subspace_err` stays pinned at `√(2r)` while only MSE moves, stage 1 found nothing and the gain is pure
random-feature ridge — forcing a moment-based subspace estimator next.

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
