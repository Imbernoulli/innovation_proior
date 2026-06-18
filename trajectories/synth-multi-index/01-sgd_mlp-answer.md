**Problem.** A fixed two-layer ReLU MLP must recover the unknown `r`-dimensional teacher subspace `V* =
span(U*)` and the cubic link `g(z) = (1/√r) Σ_i He₃(z_i)` from isotropic Gaussian inputs. The link has
**information exponent 3**, and there is no lower-degree term to bootstrap from — each teacher direction
must be found cold from a degree-3 correlation.

**Key idea (the floor).** Train the obvious way: vanilla joint SGD on both layers, Kaiming init, fixed
`n = 4096` Gaussian set, no momentum/decay/noise, one squared-loss step. This is the bare gradient flow
the information-exponent and leap analyses predict will fail: a random row's overlap with a teacher
direction is `O(1/√d)`, the drift toward it scales like `overlap^{s-1} = overlap²`, so escaping the
uninformative equator costs `≈ d^{s-1} = d² ≈ 16{,}384` steps **per direction** — twice the 8000-step
budget — and the decoupled `Σ He₃` link offers no staircase to shortcut the climb.

**Why it is the weakest rung.** With the first layer stuck near its random start, `subspace_err` should
sit near `‖P_Û − P_{U*}‖_F ≈ √(2r)` (≈ 2.0/2.4/2.8 for r2/r3/r4), and the readout — fitting near-random,
cubic-blind ReLU features — should leave `test_mse` near `Var[g] = 6`, collapsing
`score = exp(−subspace_err²/r)·exp(−test_mse)` to near zero. It is the floor by construction: no
mechanism manufactures the missing third-order signal inside budget.

**Hyperparameters.** `optimizer="sgd"`, `lr_inner = lr_outer = 5e-2`, `wd = 0`, `momentum = 0`,
`noise_std = 0`; `n = 4096`; both layers trained jointly for all 8000 steps.

**What to watch.** Subspace essentially unmoved across all three ranks, MSE near the label variance,
near-zero score — an *optimization* failure (the net can represent `g`; it cannot reach `V*` by
descending a third-order-flat landscape). That is what forces separating the two jobs at step 2.

```python
# EDITABLE region of custom_strategy.py — step 1: vanilla joint SGD (the floor)
def init_model(model: nn.Sequential, config: TaskConfig) -> None:
    """Default Kaiming-uniform initialization for both linear layers."""
    for layer in model:
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


def make_dataset(
    config: TaskConfig,
    teacher: torch.Tensor,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed Gaussian training set of size n=4096."""
    g = torch.Generator().manual_seed(seed)
    num_examples = 4_096
    x = torch.randn(num_examples, config.n_features, generator=g)
    y = teacher_outputs(x, teacher)
    return x, y


def get_optimizer_config(config: TaskConfig) -> dict[str, object]:
    """Plain SGD on both layers, no momentum, no weight decay."""
    return {
        "optimizer": "sgd",
        "lr_inner": 5e-2,
        "lr_outer": 5e-2,
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
    """Single squared-loss update applied to both layers jointly."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    preds = model(batch_x).view(-1)
    loss = ((preds - batch_y) ** 2).mean()
    loss.backward()
    optimizer.step()
    return {"loss": float(loss.item())}
```
