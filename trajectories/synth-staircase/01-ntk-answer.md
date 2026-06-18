**Problem.** Learn a latent sparse target `f*(x)=h*(x_I)` on `{+1,-1}^{100}` with a two-layer net under
online batch-SGD, across leap-1 (`h1`), leap-2 (`h2`), leap-3 (`h3`) targets. The floor every adaptive
strategy must beat is the strategy that learns *no* features.

**Key idea (the lazy / NTK baseline).** Freeze the first layer at its random initialization and train
only the readout. The frozen first layer is a bank of `M` random features
`phi_j(x)=relu(<w_j,x>+c_j)`, `w_j~N(0,1/d)` (so `<w,x>` stays `O(1)`), `c_j~N(0,1)`; the output is the
linear readout `(1/sqrt(M)) sum_j a_j phi_j(x)`. Square-loss SGD on the readout is convex online
least-squares onto a fixed feature map — a random-features approximation to kernel ridge regression. The
`1/sqrt(M)` normalization (vs the mean-field `1/M`) is the choice that keeps the network in the lazy
regime where features do not move.

**Why it is the floor.** A degree-`k` sparse parity over an unknown subset needs `min(n,q)=Omega(d^k)`
for any fixed feature map. With `d=100`, `n=b·T=6·10^5`, `M=100`: degree-1 is reachable (the `z1` term of
`h1`), degree-2 needs `Omega(d^2)=10^4` features (unreachable at `M=100`), degree-3 needs
`Omega(d^3)=10^6` (above both budget and feature count). So the lazy machine grabs at most the degree-1
piece of `h1` and otherwise sits at the trivial predictor. Watch the metrics: `score_h3≈exp(-1)` looks
high only because `h3` has unit target variance — Fourier recovery is the honest tell.

**Hyperparameters.** ReLU random features, `w~N(0,1/d)`, `b~N(0,1)`, first layer frozen; readout init `0`,
`1/sqrt(M)` output normalization; SGD on trainable params only, `lr=0.5/d`, no momentum; square loss on
fresh batches.

```python
def build_model(config: TaskConfig) -> nn.Module:
    """Random-feature / NTK-style network: freeze first layer, train head."""

    class RandomFeaturesNTK(nn.Module):
        def __init__(self, d: int, M: int) -> None:
            super().__init__()
            self.fc1 = nn.Linear(d, M, bias=True)
            self.fc2 = nn.Linear(M, 1, bias=False)
            # He / NTK-style init for ReLU: w ~ N(0, 1/d), b ~ N(0, 1).
            nn.init.normal_(self.fc1.weight, mean=0.0, std=(1.0 / d) ** 0.5)
            nn.init.normal_(self.fc1.bias, mean=0.0, std=1.0)
            nn.init.zeros_(self.fc2.weight)
            # Freeze first-layer weights (the "lazy" / NTK regime where features
            # are fixed at initialization).
            self.fc1.weight.requires_grad_(False)
            self.fc1.bias.requires_grad_(False)
            self.M = M

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = torch.relu(self.fc1(x))
            # 1/sqrt(M) normalization for NTK-style output.
            return self.fc2(h).view(-1) / (self.M ** 0.5)

    return RandomFeaturesNTK(d=config.d, M=config.width)


def get_optimizer(model: nn.Module, config: TaskConfig) -> torch.optim.Optimizer:
    """SGD on the trainable parameters (just the readout in lazy training)."""
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(params, lr=0.5 / config.d, momentum=0.0)


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    pred = model(x).view(-1)
    loss = ((pred - y) ** 2).mean()
    loss.backward()
    optimizer.step()
    return float(loss.item())
```
