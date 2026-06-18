**Problem.** Mean-field SGD climbs only the leap-1 staircase (`h1`) and stalls at the trivial predictor on
the leap-2 chain (`h2`) and leap-3 monomial (`h3`), because its flow starts at the origin and a coordinate
gets *zero* first-order gradient until the supports beneath it are lit. The cure must manufacture the
staircase the non-MSP targets lack — escape a flat saddle.

**Key idea (curriculum / layer-wise saddle-to-saddle SGD).** Three composing mechanisms on the same
two-layer net: (1) **layer-wise alternation** — each `train_step` does a feature step (update the first
layer, readout frozen) then a readout step (update the readout, features frozen), forcing the
saddle-to-saddle dynamics that picks up monomials in leap order instead of idling on the plateau with both
layers frozen; (2) **Adam with per-layer LRs** — Adam normalizes each coordinate by its running gradient
magnitude, so the tiny-but-nonzero saddle-escape gradient is amplified to an `O(lr)` step (first layer
`lr=1e-2` to escape, readout `lr=1e-2/sqrt(M)` to stay stable); (3) **mu-P-style init** — `w~N(0,1/d)`
(pre-activation `O(1)`, signal weights near the saddle), readout `N(0,1)`, `1/M` output normalization, so
feature learning is preserved under width-scaling. Shifted sigmoid keeps `sigma^{(r)}(0)!=0` so the
cascade stays alive.

**Why it works (and its ceiling).** Saddle-to-saddle SGD learns a leap-`k` function in `~d^{max(k,2)}`
steps. With budget `n=6·10^5`, `d=100`: `h1`/`h2` sit at `d^2=10^4` (reachable if the saddle-escape fires)
while `h3` sits at `d^3=10^6` (above budget — expect only partial recovery). So this is the natural
strongest baseline on top of the leap-1-only mean-field rung, with the leap-3 monomial as the hard ceiling.

**Hyperparameters.** mu-P init (`fc1~N(0,1/d)`, bias 0, `fc2~N(0,1)`); shifted sigmoid; `1/M` output;
Adam `betas=(0.9,0.999)`, `eps=1e-8`, `lr_fc1=1e-2`, `lr_fc2=1e-2/sqrt(M)`; layer-wise alternating
feature-then-readout step per batch.

```python
def build_model(config: TaskConfig) -> nn.Module:
    """mu-P-style two-layer net with shifted-sigmoid activation.

    The shifted sigmoid used by Abbe-Boix-Adsera-Misiakiewicz has nonzero low
    derivatives around the origin, avoiding the zero even derivatives of tanh.
    """

    class MuPTwoLayer(nn.Module):
        def __init__(self, d: int, M: int) -> None:
            super().__init__()
            self.fc1 = nn.Linear(d, M, bias=True)
            self.fc2 = nn.Linear(M, 1, bias=False)
            # mu-P-style init: w ~ N(0, 1/d), readout small.
            nn.init.normal_(self.fc1.weight, mean=0.0, std=(1.0 / d) ** 0.5)
            nn.init.zeros_(self.fc1.bias)
            nn.init.normal_(self.fc2.weight, mean=0.0, std=1.0)
            self.M = M

        @staticmethod
        def _shifted_sigmoid(u: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(u - 0.5)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self._shifted_sigmoid(self.fc1(x))
            # mu-P-style 1/M readout normalization.
            return self.fc2(h).view(-1) / self.M

    return MuPTwoLayer(d=config.d, M=config.width)


def get_optimizer(model: nn.Module, config: TaskConfig) -> torch.optim.Optimizer:
    """Adam with mu-P-style per-layer learning rates.

    First layer gets a larger LR so that it picks up the low-frequency monomial
    quickly (the saddle-to-saddle "leap-1" warm-up); the readout uses a
    moderate LR proportional to 1/sqrt(M).
    """
    fc1_params = []
    fc2_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("fc1"):
            fc1_params.append(p)
        else:
            fc2_params.append(p)
    width = config.width
    return torch.optim.Adam(
        [
            {"params": fc1_params, "lr": 1e-2},
            {"params": fc2_params, "lr": 1e-2 / (width ** 0.5)},
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    """Layer-wise alternating Adam: feature step, then readout step.

    The alternation drives the saddle-to-saddle dynamics: each layer in turn
    is updated to follow the next-easiest monomial in the leap ordering.
    """
    fc1_params = [p for n, p in model.named_parameters() if n.startswith("fc1") and p.requires_grad]
    fc2_params = [p for n, p in model.named_parameters() if n.startswith("fc2") and p.requires_grad]

    # 1) Feature step (freeze readout).
    for p in fc2_params:
        p.requires_grad_(False)
    optimizer.zero_grad(set_to_none=True)
    pred = model(x).view(-1)
    loss1 = ((pred - y) ** 2).mean()
    loss1.backward()
    optimizer.step()
    for p in fc2_params:
        p.requires_grad_(True)

    # 2) Readout step (freeze features).
    for p in fc1_params:
        p.requires_grad_(False)
    optimizer.zero_grad(set_to_none=True)
    pred = model(x).view(-1)
    loss2 = ((pred - y) ** 2).mean()
    loss2.backward()
    optimizer.step()
    for p in fc1_params:
        p.requires_grad_(True)

    return float(0.5 * (loss1.item() + loss2.item()))
```
