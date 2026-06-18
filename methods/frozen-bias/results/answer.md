# Frozen-bias shallow network for single-index models, distilled

## Problem it solves

Single-index regression has

```text
y = f_*(<theta*, x>) + xi,        x ~ N(0,I_d),        theta* in S^{d-1},
```

with both the high-dimensional direction `theta*` and the one-dimensional link `f_*` unknown. The goal is one shallow network trained by gradient methods that recovers the direction and fits the link without hard-wiring `f_*` as the activation and without a bespoke Hermite/slicing estimator.

## Theorem-level construction

The clean model ties every hidden unit to one shared direction and randomizes only the scalar feature dictionary:

```text
G(x;c,theta) = (1/sqrt(N)) sum_{i=1}^N c_i phi(eps_i <theta,x> - b_i),
phi(u)=max(0,u), theta in S^{d-1}.
```

The signs `eps_i` are Rademacher, the biases `b_i ~ N(0,tau^2)` with `tau > 1`, and both are fixed after initialization. The trained variables are `theta` and `c`. The frozen biases define a one-dimensional random-feature dictionary in `u=<theta,x>`; the shared `theta` carries the high-dimensional search.

## Population algebra

Write `f_* = sum_j alpha_j h_j` in normalized Hermite polynomials, let `s=min{j: alpha_j != 0}`, let `m=<theta,theta*>`, let `T_j=T h_j`, and set `Q_lambda=Q+lambda I`. With `||f_*||_gamma=1`,

```text
L(c,theta) = 1 + c^T Q_lambda c
             - 2 <c, sum_{j>=s} alpha_j m^j T_j> + sigma^2.
```

Thus

```text
nabla_c L = 2(Q_lambda c - sum_{j>=s} alpha_j m^j T_j),
c*(m) = Q_lambda^{-1} sum_{j>=s} alpha_j m^j T_j.
```

Differentiating the displayed loss makes the `theta` derivative colinear with `theta*` (up to the harmless global factor 2 that is divided out in the stationary equation). The optimized readout gives

```text
Lbar(theta) = - <g_m, P_hat_lambda g_m>_gamma + 1 + sigma^2,
g_m = sum_{j>=s} alpha_j m^j h_j.
```

The critical equation is

```text
sum_{j>=s} alpha_j^2 j m^{2j-1}
  = <(I-P_hat_lambda) g_m, gbar_m>_gamma,
gbar_m = sum_{j>=s} j alpha_j m^{j-1} h_j.
```

In the ideal infinite-feature, unregularized limit the right side is zero and `Lbar = 1+sigma^2 - sum_j alpha_j^2 m^{2j}`, strictly decreasing in `|m|`. For finite random features, the Bach-style approximation bound

```text
N >= C/lambda * log(1/(lambda delta)),
||(I-P_hat_lambda)f||_gamma^2 <= 4 A(f,lambda)
```

plus the ReLU RKHS approximation rate

```text
A(f,lambda) <= C(tau^{1+beta} ||f''||_4^2 lambda^beta + lambda C_f^2),
beta = (1 - 1/tau^2)/(3 + 1/tau^2),
```

makes the right side strictly smaller than `alpha_s^2 s |m|^{2s-1}` whenever `m != 0`, provided `lambda < lambda*`. Hence the population critical directions are only the equator `m=0` and the poles `theta=+-theta*`, with a unique readout `c` for each critical direction.

## Optimization and risk

The empirical loss concentrates uniformly despite ReLU kinks by controlling activation-pattern flips with Gaussian anti-concentration. The gradient flow uses a time-scale schedule

```text
dot c(t) = - zeta(t) nabla_c L_n(c,theta),     zeta(t)=1{t>T_0},
dot theta(t) = - nabla_theta^S L_n(c,theta),
```

with sparse small-norm `c(0)` for the first phase. The precise theorem requires

```text
n = Otilde(max{ (d+N)d^{s-1}/lambda^4, d^{(s+3)/2}/lambda^2 })
```

and gives

```text
1 - |<theta_T,theta*>|
  = Otilde(lambda^{-4} max{(d+N)/n, d^4/n^2}).
```

For ReLU with `lambda=Theta(1)` and `s>2`, this is the advertised `Otilde(d^s)` batch-gradient-flow recovery rate. The ReLU cases `s=1` and `s=2` are `d^2` and `d^2.5`; replacing ReLU by the smooth activation analyzed in the appendix can improve those cases to `d^s`.

After direction recovery, a fresh-sample ridge refit of `c` with a separate `lambda_{n'}` yields

```text
E[||F_hat-F_*||^2 | theta_hat]
  <= C ||f_*''||_gamma^{2/(beta+1)} (sigma^2 tau^2/n')^{beta/(beta+1)}
     + C ||f_*'||_gamma^2 (1-|<theta_hat,theta*>|).
```

The first term is one-dimensional and does not scale with ambient `d`; the second term is the price of imperfect direction recovery.

## MLS-Bench scaffold implementation

The local benchmark code cannot instantiate the exact tied-direction theorem model because `TwoLayerMLP` exposes independent first-layer rows. Its reference `frozen_bias.edit.py` is therefore a scaffold adaptation: initialize those rows on the unit sphere, sample `fc1.bias` uniformly in `[-1,1]`, freeze only that bias vector, train all remaining parameters with SGD and MSE, and leave the optional ridge refit unused. The block below matches the local reference edit.

```python
class Strategy:
    """Frozen-bias shallow network (Bietti et al., NeurIPS 2022, Thm 1)."""

    def __init__(self, config: TaskConfig) -> None:
        self.config = config

    def init_two_layer(self, net: TwoLayerMLP, config: TaskConfig) -> None:
        # Random first-layer weights on the unit sphere; biases sampled
        # uniformly in [-1, 1] and frozen (see Bietti et al. 2022, eqn. 3).
        with torch.no_grad():
            W = torch.randn_like(net.fc1.weight)
            W = W / W.norm(dim=1, keepdim=True).clamp_min(1e-12)
            net.fc1.weight.copy_(W)
            net.fc1.bias.uniform_(-1.0, 1.0)
        net.fc1.bias.requires_grad_(False)

        bound = 1.0 / math.sqrt(config.width)
        nn.init.uniform_(net.fc2.weight, -bound, bound)
        nn.init.zeros_(net.fc2.bias)

    def make_optimizer(
        self,
        net: TwoLayerMLP,
        config: TaskConfig,
    ) -> torch.optim.Optimizer:
        params = [p for p in net.parameters() if p.requires_grad]
        return torch.optim.SGD(
            params,
            lr=config.base_lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )

    def training_step(
        self,
        net: TwoLayerMLP,
        optimizer: torch.optim.Optimizer,
        x: torch.Tensor,
        y: torch.Tensor,
        step: int,
        config: TaskConfig,
    ) -> StepMetrics:
        net.train()
        optimizer.zero_grad(set_to_none=True)
        preds = net(x)
        loss = torch.mean((preds - y) ** 2)
        loss.backward()
        optimizer.step()
        return StepMetrics(loss=float(loss.item()), extra={})

    def finalize(
        self,
        net: TwoLayerMLP,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        config: TaskConfig,
    ) -> None:
        return


def build_strategy(config: TaskConfig) -> Strategy:
    return Strategy(config)

```
