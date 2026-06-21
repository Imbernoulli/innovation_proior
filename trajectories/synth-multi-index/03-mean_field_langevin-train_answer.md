Two-stage moved the needle but in the way that confirmed my worry rather than resolving it. The MSE fell from the floor's $9$–$10$ to $7.4/7.6/7.2$ and the score rose an order of magnitude into the $10^{-4}$ band — both attributable to the convex stage-2 ridge solve, which stopped the readout from inflating variance. But $\mathrm{subspace\_err}$ went only $1.95\!\to\!1.81$ ($r2$), $2.37\!\to\!2.30$ ($r3$), $2.74\!\to\!2.58$ ($r4$) — *barely* below $\sqrt{2r}$, with the $r2$ seed-42 run still pinned at $1.99$, identical to the floor. This is precisely the diagnostic I flagged: the MSE moved while the subspace did not, so stage 1's degree-3 climb found almost nothing and the gain is essentially the random-feature ridge fit, capped exactly where the lower bound $E[(g-\hat f)^2]\ge E[\mathrm{Var}(g\mid P_U x)]$ says it must be when $P_U$ is near-random. Moderate-step spherical SGD is still fighting the same $d^2$ wall: a decoupled $\sum\mathrm{He}_3$ with no staircase, each direction needing a degree-3 correlation to rise above the finite-batch sampling floor, and 6400 moderate steps are not enough. The conclusion is forced — I must stop asking *gradient descent on the cubic* to find the subspace, because the $d^2$ wall is a property of the gradient method, not of the problem (whose information floor is $n\sim d$).

I propose **mean-field Langevin training paired with a moment subspace estimate**. Two ideas combine, each carrying a different part of the job. The Langevin frame justifies *why* a population method escapes the exponent that strangled a single SGD trajectory; the moment estimate is *how* this implementation actually finishes the subspace in one pass. The link is then fit by a closed-form ridge on a feature basis I install along the recovered directions.

Take the mean-field lift first, because it reframes the nonconvexity. The width-$W$ net $\hat f(x)=\sum_j a_j\sigma(\langle w_j,x\rangle)$ and any norm penalty depend on the weights only through the empirical measure $\mu=(1/W)\sum_j\delta_{w_j}$. The prediction is *linear* in $\mu$ and squared loss is convex in its argument, so the risk is a *convex* functional of $\mu$: the spurious basins I fought in weight space were an artifact of pinning the measure to finitely many atoms. Each neuron should drift by the negative gradient of the first variation $J'[\mu]$, and adding an entropy regularizer $(1/\beta)H(\mu\mid\tau)$ makes the free energy strictly convex with a unique minimizer whose Wasserstein gradient flow is *exactly* Langevin dynamics, $dw = -\nabla J'[\mu](w)\,dt + \sqrt{2/\beta}\,dB$. The entropy term *is* an injected Gaussian noise of scale $\sqrt{2/\beta}$, and that diffusion does the saddle-escape the third-order drift cannot: near the flat equator where the pull is $O(\mathrm{overlap}^2)$ and almost nothing, the noise keeps the population exploring, and the information exponent drops out of the *statistics* — sample complexity becomes $n\sim d_{\mathrm{eff}}\sim r$, independent of the exponent. Conveniently, $L_2$ weight decay together with entropy equals relative entropy to a Gaussian base measure, so the abstract objective *is* noisy weight-decayed gradient descent with the Euler–Maruyama noise scale fixed at $\sqrt{2\cdot\mathrm{lr}/\beta}$. The harness implements exactly this — it stores $\mathrm{noise\_std}=1/\sqrt\beta$ and adds $\mathrm{noise\_std}\cdot\sqrt{2\cdot\mathrm{lr}}\cdot\mathcal N(0,I)$ to the parameters — so I set $\beta=10^5$, $\mathrm{wd}=10^{-4}$ on both layers (the KL drift), and keep the rows on the sphere by a Riemannian retraction after each step, because on the positively-curved sphere the log-Sobolev constant is polynomial in $d$ rather than $\exp(\beta)$ — the difference between fast and hopeless.

But a long Langevin run on the squared-loss gradient would pay the same finite-sample noise that capped two-stage, so I take a sharper move the Langevin frame *licenses* and that finishes the subspace in essentially one shot. The realization is about the drift itself. For the Hermite-3 link the population correlation a neuron feels is $E[y\cdot\mathrm{He}_3(\langle w,x\rangle)]$, and the natural feature-learning objective is to maximize the *square* of this correlation — how much each particle aligns with the third-order subspace signal. I compute its gradient analytically rather than through autograd on the squared loss: with $z=x\hat w^\top$ for unit rows $\hat w$, $\mathrm{corr}=\mathrm{mean}(y\cdot\mathrm{He}_3(z))$, the score-gradient is $3(z^2-1)$ (from $\mathrm{He}_3'$), and the drift on the rows is $2d\cdot\mathrm{corr}\cdot\nabla\mathrm{corr}$, projected onto the sphere's tangent and renormalized, with the KL decay and the $\sqrt{2\cdot\mathrm{lr}}\cdot\mathrm{noise\_std}$ diffusion folded in. This single analytic correlation-Langevin step ($\mathrm{stage1\_steps}=1$) is the population drift of the *correct* third-order objective, not the third-order-weak gradient of the squared loss — it nudges the particles, but it is not what carries the recovery.

What carries the recovery is the moment estimate computed alongside it, and it is the heart of why this rung should finally crack the subspace. Stein's identity, $E[x\cdot h(x)]=E[\nabla h]$, converts the degree-3 correlation into a *second-order* object I can read directly from the data. Form the label-weighted input covariance $M = E[y^2\cdot xx^\top]$. Because $y=g(U^{*\top}x)$ depends on $x$ only through its $V^*$-projection, the directions in which $y^2$ modulates the input variance are exactly the teacher directions: $M$ is a perturbation of $I_d$ whose top eigenspace is $V^*$. So I estimate $M$ in one chunked pass over a *large fresh pool* — `make_dataset` builds $n=\min(64{,}000,\texttt{max\_train\_examples})$ points, the fresh-sample regime that makes the empirical $M$ track its population value — symmetrize it, and take the top-$r$ eigenvectors by `eigh`; those *are* the recovered subspace directions, computed without ever paying the $d^2$ gradient wall. This is the moment method the information-exponent literature points to as the route to near-optimal $\Theta(d)$ sample complexity. The harness caches the result as `_CACHED_DIRECTIONS` at dataset-construction time so the readout solve can use it directly.

With the subspace in hand, the link is again a convex closed-form solve, but now on *good* features. I do not trust the noisy first layer to be the feature map; I *install* a basis explicitly along the recovered directions. For each of the $r$ estimated directions I place rows at a grid of biases ($\mathrm{linspace}(-3,3)$) and both signs, so the ReLU features tile each teacher direction with thresholded bumps at a range of offsets — a deliberate univariate basis along every direction the link reads, which is exactly what is needed to represent $\mathrm{He}_3$ on each projection. Then I ridge-fit the readout on $n=20{,}000$ of the pool: post-ReLU design matrix, a ones column for the bias, $(\Phi^\top\Phi+\lambda I)a=\Phi^\top y$ with $\lambda=10^{-4}\cdot n$. Same convex move as two-stage, but now the features actually span the cubic on $V^*$, so the residual is no longer floored by $\mathrm{Var}(g\mid P_U x)$ with random $U$ — it is floored only by how well the bias-grid bumps approximate $\mathrm{He}_3$, which with a fine grid is small.

So the rung, in order, is: one analytic Hermite-3 correlation-Langevin step on the spherical first-layer particles; a moment estimate $M=E[y^2xx^\top]$ whose top-$r$ eigenspace recovers $V^*$ directly from the large fresh pool; an explicit bias-grid ReLU basis installed along those directions; and a closed-form ridge solve for the readout. Against two-stage's numbers I expect $\mathrm{subspace\_err}$ to *break away* from the $\sqrt{2r}$ band both prior rungs were stuck near — the moment estimator reads $V^*$ off a directly estimated second-order moment rather than climbing a third-order gradient — $\mathrm{test\_mse}$ to fall below $7.2$–$7.6$ (a *missed* direction still leaves a full $6/r$ chunk of the variance unexplained, so not to zero), and $\mathrm{score}$ to jump roughly two orders of magnitude into the $10^{-2}$–$10^{-1}$ band as both exponentials improve together. I also expect this to be high-variance across seeds, because the moment eigengap depends on how cleanly $E[y^2xx^\top]$ separates the teacher directions on a finite pool, and an unlucky draw can collapse a direction. But the entire point of the $E[y^2xx^\top]$ move is that it sees the degree-3 structure the gradient could not, so this should be the first rung to actually find the subspace.

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
