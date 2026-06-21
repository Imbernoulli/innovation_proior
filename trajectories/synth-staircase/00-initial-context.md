## Research question

A two-layer fully connected network of width `M` is trained by online batch-SGD on the hypercube
`{+1,-1}^d`. It has to recover a latent sparse target `f*(x) = h*(x_{j_1},...,x_{j_P})` that depends
on only `P=4` of the `d=100` coordinates. The single design choice is the **learning strategy**:
model architecture, initialization, parametrization, optimizer, and one-step update. The uniform
`{+1,-1}^d` sampler, the three targets, the squared loss, and the evaluation are fixed.

Three targets span the hierarchy: the leap-1 staircase `h1(z)=z1+z1z2+z1z2z3`, the leap-2 non-MSP
chain `h2(z)=z1z2+z2z3+z3z4`, and the leap-3 monomial `h3(z)=z1z2z3`. A strategy that succeeds on
`h1` but fails on `h3` scores poorly, because the aggregate is a geometric mean over the three
environments.

## Prior art / Background / Baselines

Which sparse Boolean functions plain SGD finds depends on the *hierarchical structure* of the
target's Fourier support.

- **Fixed-feature lower bounds.** Any linear method on a *fixed* feature map needs `n = Omega(d^k)`
  samples to fit a `k`-sparse parity. With the budget here (`n = b·T = 150·4000 = 6·10^5`, `d = 100`)
  this covers degree-1 and degree-2 components but not the degree-3 monomial. Gap: a frozen feature
  map cannot adapt to the unknown latent subset, so it pays `d^k` for the high-degree pieces of every
  target.
- **Mean-field limit of two-layer SGD.** With the `1/M` output normalization the weights travel an
  `O(1)` distance, so the network stays in the *feature-learning* regime and SGD concentrates onto a
  Wasserstein gradient flow on the empirical measure of neurons — features can move toward latent
  coordinates. Gap: the dynamics is nonlinear and adaptive, but it does not identify which targets
  the flow can drive to zero risk.
- **Saddle-to-saddle / hierarchical pickup.** Starting from a near-origin init, the mean-field flow
  leaves signal directions at saddles; a coordinate's weight escapes only once lower-degree supports
  beneath it are active, so components are learned in increasing degree, with higher stairs orders
  of magnitude slower. This makes a staircase learnable in `O(d)` and a non-staircase leap provably
  stuck. Gap: the sequential degree-by-degree mechanism stalls when a single step must introduce
  more than one new coordinate.

## Fixed substrate

The harness is fixed and must not be touched. It sets: ambient dimension `d=100`, latent dimension
`P=4`, width `M=100`, batch size `b=150`, total online-SGD steps `T=4000` (one-pass, every batch
freshly sampled from `Unif({+1,-1}^d)`), and a test set of `8192` fresh samples. For each top-level
seed it draws a fresh random latent subset `I` of size `P`, builds `f*(x)=h*(x_I)`, trains one run,
and reports test MSE and Fourier recovery. The data sampler, the three targets `h1/h2/h3`, the
squared loss, the Fourier-recovery evaluation, and the `train_one_run`/`run_benchmark` driver are all
fixed. The driver calls the three editable functions: it builds the model once per run via
`build_model`, wraps it with `get_optimizer`, and at every step samples a fresh `(x,y)` and calls
`train_step`.

## Editable interface

Only one region of `pytorch-examples/synth_staircase/custom_strategy.py` is editable — the three
functions `build_model(config)`, `get_optimizer(model, config)`, and
`train_step(model, optimizer, x, y)`. The contract: `build_model` returns a fresh `nn.Module`
consuming `[B,d]` (entries in `{+1,-1}`) and returning `[B]` or `[B,1]`; `get_optimizer` wraps only
`model.parameters()` (or a filtered subset); `train_step` performs one optimizer step on a fresh
batch and returns the scalar training loss as a Python `float` (it must call
`optimizer.zero_grad()` itself). `config` is the fixed `TaskConfig` (`config.d`, `config.width`,
`config.batch_size`, ...). Every method is a fill of exactly this contract.

The starting scaffold is a two-layer mean-field network with shifted-sigmoid activation, random
`+-1` readout signs, `N(0,I_d)` first layer, no bias, plain SGD at `lr=0.5`. Each method replaces
exactly these three definitions and nothing else.

```python
# EDITABLE region of custom_strategy.py — default fill (mean-field two-layer SGD)
def build_model(config: TaskConfig) -> nn.Module:
    class TwoLayerMeanField(nn.Module):
        def __init__(self, d: int, M: int) -> None:
            super().__init__()
            self.fc1 = nn.Linear(d, M, bias=False)
            self.fc2 = nn.Linear(M, 1, bias=False)
            # Mean-field init: w_j ~ N(0, I_d); no trainable first-layer bias.
            nn.init.normal_(self.fc1.weight, mean=0.0, std=1.0)
            with torch.no_grad():
                signs = torch.randint(0, 2, (M, 1), dtype=torch.float32) * 2.0 - 1.0
                self.fc2.weight.copy_(signs.t())          # a_j ~ Unif({+1,-1})
            self.M = M

        @staticmethod
        def _shifted_sigmoid(u: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(u - 0.5)                  # sigma(x) = (1 + e^{-x+0.5})^{-1}

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self._shifted_sigmoid(self.fc1(x))
            return self.fc2(h).view(-1) / self.M           # 1/M mean-field normalization

    return TwoLayerMeanField(d=config.d, M=config.width)


def get_optimizer(model: nn.Module, config: TaskConfig) -> torch.optim.Optimizer:
    return torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.0)  # plain SGD, eta = 1/2


def train_step(model, optimizer, x, y) -> float:
    optimizer.zero_grad(set_to_none=True)
    pred = model(x).view(-1)
    loss = ((pred - y) ** 2).mean()                        # square loss, fresh batch
    loss.backward()
    optimizer.step()
    return float(loss.item())
```

## Evaluation settings

Three environments, one per latent target — `h1` (leap-1 / MSP staircase), `h2` (leap-2 non-MSP
chain), `h3` (leap-3 monomial) — each over three top-level seeds `{42, 123, 456}`, one run per seed
with a fresh latent subset `I`. For each environment three quantities are reported:
`test_mse_{env}` (squared error of `model(x)` against `h*(x_I)` on the held-out test set, lower is
better), `fourier_recovery_{env}` (mean `|hat_S(model) - hat_S(h*)|` over the latent monomials, lower
is better), and the primary leaderboard metric `score_{env} = exp(-test_mse_{env})`, bounded in
`(0,1]`, higher is better. The aggregate task score is the geometric mean of the three
per-environment scores, so a catastrophic failure on any one environment is not hidden by a strong
result on another.
