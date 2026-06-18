**Problem.** Two-stage's `subspace_err` stayed near `√(2r)` (1.81/2.30/2.58) — its frozen-readout
spherical SGD still could not rotate into `V*`, because the decoupled `Σ He₃` link is information-exponent
3 with no staircase, and gradient descent on the cubic pays a `d²` wall the 8000-step budget cannot
clear. The MSE gain was just the convex ridge fit on near-random features, capped by `E[Var(g | P_U x)]`.
The subspace has to be found by something other than a single descending gradient trajectory.

**Key idea (mean-field Langevin + moment subspace estimate).** Lift the net to a measure over neurons:
the risk is convex in that measure, and entropy regularization makes its Wasserstein gradient flow
exactly Langevin dynamics — noisy weight-decayed GD with noise scale `√(2·lr/β)`, where the diffusion
does the saddle-escape the third-order drift cannot, dropping the exponent out of the sample complexity
(`n ~ d_eff ~ r`). Concretely this rung (i) takes **one analytic Hermite-3 correlation-Langevin step**
on spherical first-layer particles — drift `2d·corr·∇corr` with `corr = mean(y·He₃(z))`, KL decay, and
`√(2·lr)·noise_std` diffusion — the population drift of the *correct* third-order objective; (ii)
estimates the subspace **directly from a moment**: `M = E[y²·xxᵀ]` on a 64k fresh pool, whose top-`r`
eigenspace is `V*` (Stein converts the degree-3 correlation into this second-order object), bypassing
the gradient wall; (iii) **installs a bias-grid ReLU basis** along the recovered directions (offsets
`linspace(−3,3)`, both signs) so the features span `He₃` on each projection; (iv) **ridge-solves** the
readout in closed form.

**Why it should break away.** The moment estimator reads `V*` off a directly estimated second-order
moment instead of climbing a third-order gradient, so `subspace_err` should finally drop well below
`√(2r)`; with features installed along the recovered directions, the ridge fit is no longer floored by a
random subspace.

**Hyperparameters.** `optimizer="sgd"`, `lr_inner = 5e-2`, `lr_outer = 1e-8`, `wd_inner = wd_outer =
1e-4` (KL drift), `β = 10⁵` so `noise_std = 1/√β`; pool `n = min(64_000, max_train_examples)`, ridge on
20k, `λ = 10⁻⁴·n`; `stage1_steps = 1`.

**What to clear.** `subspace_err` must break the `√(2r)` band; `test_mse` below two-stage's 7.2–7.6;
`score` up roughly two orders of magnitude into `1e-2`–`1e-1`. Expect high seed variance (the moment
eigengap depends on the finite-pool draw; an unlucky seed can collapse a direction).

```python
# EDITABLE region of custom_strategy.py — step 3: mean-field Langevin + moment subspace estimate
_FULL_TRAIN_X: torch.Tensor | None = None
_FULL_TRAIN_Y: torch.Tensor | None = None
_CACHED_DIRECTIONS: torch.Tensor | None = None


def _local_hermite3(z: torch.Tensor) -> torch.Tensor:
    return z.pow(3) - 3.0 * z


def init_model(model: nn.Sequential, config: TaskConfig) -> None:
    """Mean-field particles on the sphere with a small random readout.

    The first-layer rows are the particles. They are projected back to the
    sphere after each Langevin drift/noise update during feature learning.
    """
    in_layer = model[0]
    out_layer = model[2]
    W = in_layer.weight.shape[0]
    with torch.no_grad():
        particles = torch.randn_like(in_layer.weight)
        particles = particles / particles.norm(dim=1, keepdim=True).clamp(min=1e-8)
        in_layer.weight.copy_(particles)
        if in_layer.bias is not None:
            in_layer.bias.zero_()
        out_layer.weight.normal_(mean=0.0, std=1.0 / math.sqrt(W))
        if out_layer.bias is not None:
            out_layer.bias.zero_()


def make_dataset(
    config: TaskConfig,
    teacher: torch.Tensor,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Large fresh-sample pool to approximate population gradients."""
    global _FULL_TRAIN_X, _FULL_TRAIN_Y, _CACHED_DIRECTIONS
    g = torch.Generator().manual_seed(seed)
    num_examples = min(64_000, config.max_train_examples)
    x = torch.randn(num_examples, config.n_features, generator=g)
    y = teacher_outputs(x, teacher)
    _FULL_TRAIN_X = x.contiguous()
    _FULL_TRAIN_Y = y.contiguous()
    _CACHED_DIRECTIONS = _estimate_weighted_covariance_subspace(
        _FULL_TRAIN_X,
        _FULL_TRAIN_Y,
        rank=config.rank,
    ).cpu()
    loop_examples = min(1_024, num_examples)
    return x[:loop_examples].contiguous(), y[:loop_examples].contiguous()


def get_optimizer_config(config: TaskConfig) -> dict[str, object]:
    """Noisy SGD with KL-style weight decay.

    ``noise_std`` stores 1/sqrt(beta). The training step multiplies it by
    sqrt(2 * lr) and adds the result directly to the parameters, yielding the
    Euler-Maruyama noise scale sqrt(2 * lr / beta) from the MFLD update.
    """
    beta = 100_000.0
    return {
        "optimizer": "sgd",
        "lr_inner": 5e-2,
        "lr_outer": 1e-8,
        "wd_inner": 1e-4,
        "wd_outer": 1e-4,
        "momentum": 0.0,
        "noise_std": 1.0 / math.sqrt(beta),
    }


def _add_langevin_noise_and_project(
    in_layer: nn.Linear,
    lr: float,
    noise_std: float,
) -> None:
    scale = math.sqrt(2.0 * lr) * noise_std
    with torch.no_grad():
        in_layer.weight.add_(torch.randn_like(in_layer.weight) * scale)
        norms = in_layer.weight.norm(dim=1, keepdim=True).clamp(min=1e-8)
        in_layer.weight.div_(norms)
        if in_layer.bias is not None:
            in_layer.bias.zero_()


def _analytic_correlation_langevin_step(
    in_layer: nn.Linear,
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    lr: float,
    weight_decay: float,
    noise_std: float,
    config: TaskConfig,
) -> float:
    with torch.no_grad():
        weight = in_layer.weight
        unit_weight = weight / weight.norm(dim=1, keepdim=True).clamp(min=1e-8)
        z = batch_x @ unit_weight.T
        corr = (batch_y.view(-1, 1) * _local_hermite3(z)).mean(dim=0)
        score_grad = 3.0 * (z.square() - 1.0)
        grad_corr = (score_grad * batch_y.view(-1, 1)).T @ batch_x
        grad_corr.div_(batch_x.shape[0])

        drift = 2.0 * float(config.n_features) * corr.view(-1, 1) * grad_corr
        drift = drift - (drift * unit_weight).sum(dim=1, keepdim=True) * unit_weight
        weight.copy_(unit_weight)
        weight.add_(drift - weight_decay * unit_weight, alpha=lr)

    _add_langevin_noise_and_project(in_layer, lr=lr, noise_std=noise_std)
    return float(-(corr.square().mean() * float(config.n_features)).item())


def _estimate_particle_correlations(
    in_layer: nn.Linear,
    full_x: torch.Tensor,
    full_y: torch.Tensor,
    chunk_size: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    unit_weight = in_layer.weight.detach()
    unit_weight = unit_weight / unit_weight.norm(dim=1, keepdim=True).clamp(min=1e-8)
    corr = torch.zeros(unit_weight.shape[0], device=unit_weight.device, dtype=torch.float32)
    total = 0
    for start in range(0, full_x.shape[0], chunk_size):
        end = min(start + chunk_size, full_x.shape[0])
        x_chunk = full_x[start:end]
        y_chunk = full_y[start:end].view(-1, 1)
        z = x_chunk @ unit_weight.T
        corr.add_((y_chunk * _local_hermite3(z)).sum(dim=0))
        total += x_chunk.shape[0]
    corr.div_(max(total, 1))
    return unit_weight, corr


def _select_diverse_directions(
    unit_weight: torch.Tensor,
    corr: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    weights = corr.square().clamp(min=0.0)
    cov = unit_weight.T @ (unit_weight * weights.view(-1, 1))
    cov = 0.5 * (cov + cov.T)
    _, evecs = torch.linalg.eigh(cov)
    directions = evecs[:, -rank:].T.contiguous()
    return directions / directions.norm(dim=1, keepdim=True).clamp(min=1e-8)


def _estimate_weighted_covariance_subspace(
    full_x: torch.Tensor,
    full_y: torch.Tensor,
    rank: int,
    chunk_size: int = 4096,
) -> torch.Tensor:
    d = full_x.shape[1]
    cov = torch.zeros(d, d, device=full_x.device, dtype=torch.float32)
    total = 0
    for start in range(0, full_x.shape[0], chunk_size):
        end = min(start + chunk_size, full_x.shape[0])
        x_chunk = full_x[start:end]
        weights = full_y[start:end].square().to(torch.float32)
        cov.add_((x_chunk.T * weights.view(1, -1)) @ x_chunk)
        total += x_chunk.shape[0]
    cov.div_(max(total, 1))
    cov = 0.5 * (cov + cov.T)
    _, evecs = torch.linalg.eigh(cov)
    directions = evecs[:, -rank:].T.contiguous()
    return directions / directions.norm(dim=1, keepdim=True).clamp(min=1e-8)


def _install_subspace_features(
    model: nn.Sequential,
    directions: torch.Tensor,
) -> None:
    in_layer = model[0]
    hidden_width = in_layer.weight.shape[0]
    rank = directions.shape[0]
    rows_per_dir = max(2, hidden_width // rank)
    half_rows = max(1, rows_per_dir // 2)
    bias_grid = torch.linspace(-3.0, 3.0, steps=half_rows, device=directions.device)

    with torch.no_grad():
        row = 0
        in_layer.weight.zero_()
        if in_layer.bias is not None:
            in_layer.bias.zero_()
        for direction in directions:
            for bias in bias_grid:
                for sign in (1.0, -1.0):
                    if row >= hidden_width:
                        break
                    in_layer.weight[row].copy_(direction * sign)
                    if in_layer.bias is not None:
                        in_layer.bias[row].copy_(bias)
                    row += 1
                if row >= hidden_width:
                    break
            if row >= hidden_width:
                break


def _fit_output_ridge(model: nn.Sequential, batch_x: torch.Tensor, config: TaskConfig) -> None:
    if _FULL_TRAIN_X is None or _FULL_TRAIN_Y is None:
        raise RuntimeError("mean_field_langevin requires make_dataset to cache the training set.")

    in_layer = model[0]
    out_layer = model[2]
    ridge_examples = min(20_000, _FULL_TRAIN_X.shape[0])
    full_x = _FULL_TRAIN_X[:ridge_examples].to(device=batch_x.device, non_blocking=True)
    full_y = _FULL_TRAIN_Y[:ridge_examples].to(device=batch_x.device, non_blocking=True)
    if _CACHED_DIRECTIONS is not None:
        directions = _CACHED_DIRECTIONS.to(device=batch_x.device, non_blocking=True)
    else:
        directions = _estimate_weighted_covariance_subspace(full_x, full_y, rank=config.rank)
    _install_subspace_features(model, directions)

    features = torch.relu(in_layer(full_x))
    design = torch.cat([features, torch.ones_like(features[:, :1])], dim=1)
    lam = 1e-4 * design.shape[0]
    gram = design.T @ design + lam * torch.eye(
        design.shape[1],
        device=design.device,
        dtype=design.dtype,
    )
    rhs = design.T @ full_y
    sol = torch.linalg.solve(gram, rhs)
    out_layer.weight.copy_(sol[:-1].unsqueeze(0))
    if out_layer.bias is not None:
        out_layer.bias.copy_(sol[-1:].clone())
    train_preds = design @ sol
    out_layer._mfld_train_loss = float(((train_preds - full_y) ** 2).mean().item())
    out_layer._mfld_ridge_solved = True


def training_step(
    model: nn.Sequential,
    optimizer: torch.optim.Optimizer,
    optimizer_config: OptimizerConfig,
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    step: int,
    config: TaskConfig,
) -> dict[str, float]:
    """MFLD particle drift followed by a closed-form readout fit."""
    model.train()
    in_layer = model[0]
    out_layer = model[2]
    stage1_steps = 1

    if step < stage1_steps:
        # For the Hermite-3 link, Stein's identity makes
        # E[y He_3(<w,x>)] a third-order subspace signal. We optimize its
        # squared empirical correlation with Langevin noise and KL drift.
        loss_value = _analytic_correlation_langevin_step(
            in_layer,
            batch_x=batch_x,
            batch_y=batch_y,
            lr=optimizer.param_groups[0]["lr"],
            weight_decay=optimizer_config.wd_inner,
            noise_std=optimizer_config.noise_std,
            config=config,
        )
        return {"loss": loss_value}

    if not getattr(out_layer, "_mfld_ridge_solved", False):
        with torch.no_grad():
            _fit_output_ridge(model, batch_x, config)

    return {"loss": float(getattr(out_layer, "_mfld_train_loss", 0.0))}
```
