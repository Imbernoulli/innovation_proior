Mean-field SGD did exactly what its mechanism predicted, and the numbers split by leap with surgical cleanliness. On $h_1$ it beat the lazy floor — $\text{test\_mse}_{h_1}$ fell from NTK's $2.78$ to $2.45$, recovery from $0.96$ to $0.89$, $\text{score}_{h_1}$ rose from $0.062$ to $0.086$, as feature learning climbed the degree-1 and part of the degree-2 stair no frozen kernel could reach. But on $h_2$ it landed at $\text{test\_mse} = 3.00$, recovery $1.00$, $\text{score} = 0.050$, *identical* to NTK's trivial-predictor row; and on $h_3$ at $\text{test\_mse} = 1.00$, recovery $0.9995$, the deceptive $\exp(-1)$ of a unit-variance target. The diagnosis is precise: the mean-field flow starts at the origin in the signal directions, and a coordinate gets *zero* first-order gradient until the supports beneath it are lit. $h_1$ has a degree-1 entry point that starts the cascade; $h_2$ leaps by 2 and $h_3$ leaps straight to degree 3, so both leave coordinates frozen at a flat saddle where plain SGD goes nowhere because the gradient is genuinely zero. The cure must *manufacture* the staircase the non-MSP targets lack.

I propose curriculum / layer-wise saddle-to-saddle SGD: three composing mechanisms on the same two-layer net, each attacking a different reason the bare flow stalled. The first is **layer-wise alternation**. The bare flow freezes on $h_2$/$h_3$ because the readout $a$ and the features $u$ move *simultaneously*, and at the origin neither has a gradient in the leftover coordinates. The saddle-to-saddle picture says a hierarchical target should be learned as a sequence of plateaus — sit near a saddle, slowly align one layer, then a fast transition lifts the next component — and the way to *force* that structure rather than wait for it is to alternate which layer moves. Each `train_step` does two sub-updates on the *same* batch: a feature step that updates only the first layer with the readout frozen, then a readout step that updates only the second layer with the features frozen. The feature step, with the readout held at its current nonzero, diverse values, gets a gradient on $u$ even where the joint flow had none, because the residual $(h^* - \hat f)$ projected through the *fixed* readout exposes correlations the simultaneous update averaged away; the readout step then recombines the freshly-aligned features into the output. I report the mean of the two sub-losses as the training loss.

The second mechanism is **Adam with per-layer learning rates**, to actually escape once a gradient appears. Even with alternation the gradient lifting a leftover coordinate is tiny — it is the high-order correlation the homogeneity argument showed vanishes to order $2^{k-1}$ at the saddle — and a fixed-$\mathrm{lr}$ SGD step on a gradient that small moves nowhere in $T = 4000$ steps. Adam normalizes each coordinate's update by a running estimate of its gradient magnitude, so a *consistently small but nonzero* gradient — exactly the signal that escapes a saddle — is amplified to an $O(\mathrm{lr})$ step regardless of its raw scale. I use $\beta = (0.9, 0.999)$, $\epsilon = 10^{-8}$. But per-coordinate normalization means the effective step is now set by $\mathrm{lr}$, not the self-correcting $\eta = 1/2$ of the mean-field flow, so I split it by layer: the first layer, which must escape the saddle, gets the full $\mathrm{lr} = 10^{-2}$ (the leap-1 warm-up that picks up the low-frequency monomial quickly), while the readout gets $\mathrm{lr} = 10^{-2}/\sqrt M$ so its update scale matches the $1/M$-normalized output and the linear readout phase stays stable instead of oscillating.

The third mechanism is **$\mu$P-style initialization**, so width-scaling preserves feature learning under the new optimizer. I take $w \sim N(0, 1/d)$ — which keeps the pre-activation $\langle w,x\rangle$ at $O(1)$ over $x\in\{+1,-1\}^d$ so the activation's low-order derivatives $m_r$ that drive the cascade are actually exercised, and puts the signal block at $O(1/\sqrt d)\approx 0.1$, i.e. near the saddle the saddle-to-saddle story needs — a zero first-layer bias, and a readout drawn $N(0,1)$ with the $1/M$ output normalization carried over from the mean-field scaling. Under this parametrization the readout and feature updates stay balanced as width changes. I keep the shifted sigmoid $\sigma(x) = \mathrm{sigmoid}(x-0.5)$ unchanged, because the cascade-keeps-alive argument is untouched: I still need $\sigma^{(r)}(0)\neq0$ for all low $r$, and a symmetric activation would zero half the chain no matter how cleverly the layers alternate.

I should not overclaim the reach. The leap-complexity picture says saddle-to-saddle SGD learns a leap-$k$ function in roughly $d^{\max(k,2)}$ steps. With budget $n = 6\cdot10^5$, $d = 100$: $h_1$ (leap-1) sits at $d^2 = 10^4$, comfortably inside budget; $h_2$ (leap-2) also at $d^2 = 10^4$ (the $\max(\text{leap},2)$ floor is $2$), inside budget *if* the saddle-escape actually fires; but $h_3$ (leap-3) sits at $d^3 = 10^6$, *above* my budget, so even the strongest baseline should expect only partial recovery of the single leap-3 monomial. The honest claim is therefore: layer-wise saddle-to-saddle training should learn any low-leap target in $d^{\max(\text{leap},2)}$ steps, the natural strongest method on top of the leap-1-only mean-field rung, with the leap-3 monomial as the hard ceiling. The decisive test is $h_2$: mean-field SGD was flat at the trivial $0.050$, and if the alternation + Adam + $\mu$P combination genuinely escapes the saddle, $h_2$ should move off that floor for the first time on the ladder — even partial recovery of one of its three degree-2 monomials would show it fired.

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
