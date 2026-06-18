**Problem.** The lazy baseline cannot adapt: its features are frozen before they know the latent subset
`I`, so it sits at the trivial predictor on all three targets. The fix is to put the network in the
parametrization where the first-layer weights *move* and the features can rotate toward `I`.

**Key idea (mean-field two-layer SGD).** Use the `1/M` (mean-field) output normalization, not the
`1/sqrt(M)` lazy one. With `1/M` each neuron contributes `O(1/M)` while its weights travel an `O(1)`
distance, so the dynamics stays nonlinear and SGD concentrates onto a Wasserstein gradient flow on the
neuron measure. For a sparse target this flow collapses (as `d` grows) to a dimension-free flow on
`(a,u,s) in R^{P+2}` that *starts with the signal weights at the origin*. A coordinate leaves the origin
only once the lower-degree supports beneath it are lit, so components are learned in increasing degree at
rate `|u_k| ~ t^{2^{k-1}}` — climbing the staircase.

**Why it works (and which targets fail).** This succeeds exactly when the target's Fourier supports can
be ordered to add one new coordinate at a time (the merged-staircase property, MSP / leap-1). `h1` is a
vanilla staircase: `z1` lights `u_1` linearly, which lights `u_2~t^2`, which lights `u_3~t^4` — learnable.
`h2` leaps by 2 at its first support and `h3` leaps straight to degree 3 — both leave coordinates frozen
at the origin (a homogeneous-linear / Gronwall argument), so the flow is stuck at strictly positive risk,
independent of step size or activation. Plain SGD climbs only one new coordinate at a time.

**Hyperparameters.** Two-layer net, no bias, `w~N(0,I_d)` (signal weights start near the origin),
readout signs `a~Unif({+1,-1})` (neuron diversity makes the readout-phase kernel full rank), shifted
sigmoid `sigma(x)=(1+e^{-x+0.5})^{-1}` (so `sigma^{(r)}(0)!=0` for all low `r`, keeping the cascade
alive), `1/M` output normalization, plain SGD `lr=0.5`, no momentum, square loss on fresh batches.

```python
def build_model(config: TaskConfig) -> nn.Module:
    """Two-layer mean-field network (Abbe et al. Fig. 1 default)."""

    class TwoLayerMeanField(nn.Module):
        def __init__(self, d: int, M: int) -> None:
            super().__init__()
            self.fc1 = nn.Linear(d, M, bias=False)
            self.fc2 = nn.Linear(M, 1, bias=False)
            nn.init.normal_(self.fc1.weight, mean=0.0, std=1.0)
            # No trainable bias in the Abbe et al. mean-field setup.
            with torch.no_grad():
                signs = torch.randint(0, 2, (M, 1), dtype=torch.float32) * 2.0 - 1.0
                self.fc2.weight.copy_(signs.t())
            self.M = M

        @staticmethod
        def _shifted_sigmoid(u: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(u - 0.5)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self._shifted_sigmoid(self.fc1(x))
            return self.fc2(h).view(-1) / self.M

    return TwoLayerMeanField(d=config.d, M=config.width)


def get_optimizer(model: nn.Module, config: TaskConfig) -> torch.optim.Optimizer:
    """Plain SGD with eta_k = 1/2."""
    return torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.0)


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
