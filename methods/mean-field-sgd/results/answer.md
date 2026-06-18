# Mean-field two-layer SGD for sparse Boolean functions (the merged-staircase setting)

## Problem

Learn a sparse target `f*(x) = h*(z)` on `x ~ Unif({+1,-1}^d)`, where `z = x_I` are `P` latent
(unknown) coordinates and `h* : {+1,-1}^P -> R`, using one-pass batch-SGD on a two-layer network,
with `O(d)` samples and `d` arbitrarily large. The question is *which* `h*` are learnable this way.

## Key idea

Use the **mean-field** (`1/N`) parametrization so the network is in the feature-learning regime
rather than the lazy/NTK linear regime. In that regime, for a sparse target, one-pass SGD concentrates
(as `d, N, 1/eta` grow) onto a **dimension-free Wasserstein gradient flow** on `R^{P+2}` that starts
with the first-layer signal weights at the origin. The flow learns the Fourier components of `h*` in
order of increasing degree — a coordinate's weight `u_k` only leaves zero once the lower-degree supports
are nonzero, and grows like `t^{2^{k-1}}` (climbing the staircase). This succeeds exactly when the
non-zero Fourier supports `{S_1,...,S_m}` of `h*` can be ordered so each adds at most one new coordinate.

## Main result (the characterization)

`h*` is **strongly `O(d)`-SGD-learnable** iff the dimension-free flow from the origin reaches zero risk.

- **Necessity (Merged-Staircase Property).** Define MSP: the supports can be ordered with
  `|S_i \ ∪_{j<i} S_j| <= 1` for all `i`. If `h*` does *not* satisfy MSP, then some signal coordinates
  stay frozen at `u_i^t = 0` for all `t` (a homogeneous-linear / Gronwall argument), so every monomial
  touching them is unlearnable and `inf_t R(rhobar_t) >= sum_{S unlearnable} hhat(S)^2 > 0`,
  independent of activation, step schedule, regularization, and the choice of `mu_a` and `m_2^w`
  within the `u^0 = 0` dimension-free initialization. MSP is necessary.
- **Near-sufficiency.** For *generic* MSP `h*` (almost surely over the Fourier coefficients) the
  result holds in the discrete-time regime with a sufficiently high-degree polynomial activation;
  the smooth-activation extension needs a one-time random perturbation of the activation for the
  second phase. This caveat is unavoidable in spirit because symmetric MSPs can collapse the weights
  into a subspace with no zero-risk solution. For *vanilla staircases*
  `h*(z) = alpha_1 z1 + alpha_2 z1 z2 + ... + alpha_P z1...zP` (all `alpha != 0`) learnability holds
  with no genericity and no activation perturbation, for any activation with
  `sigma^{(r)}(0) != 0`, `r = 0,...,P`.
- **Separation.** Linear methods (kernel / random features, dimension `q`, `n` samples) need
  `min(n,q) = Omega(d^k)` to learn a target with a degree-`k` sparse component, and
  `d^{omega(1)}` for vanilla staircases of
  slowly-growing degree `P` — superpolynomially more than the `O(d)` (resp. `d^{O(1)}`) of this SGD.

`z1 z2 z3` (a leap straight to degree 3) is non-MSP and not learnable; `z1 + z1 z2 + z1 z2 z3` is a
vanilla staircase and is learnable.

## Why it works (the mechanism, stated cleanly)

- **Mean-field reduction.** With `fhat(x;Theta) = (1/N) sum_j a_j sigma(<w_j,x>)`, one-pass batch-SGD
  with anisotropic step `H_k = diag(eta^a_k, eta^w_k I_d)` and reg `Lambda` converges to a continuity
  equation whose particle drift is the population version of the finite update:
  `dot a = xi^a E[(f* - fhat) sigma] - xi^a lambda^a a` and
  `dot w = xi^w a E[(f* - fhat) sigma' x] - xi^w lambda^w w`.
- **Sparsity => dimension-free flow.** Split `x=(z,r)`, `w=(u,v)`. Symmetric iid init makes `fhat`
  independent of the noise block (preserved by the PDE), and `<v,r> ≈ ||v||_2 G`, `G ~ N(0,1)` (CLT /
  Berry-Esseen). Collapsing `v` to `s = ||v||_2` gives effective params `(a, u, s) in R^{P+2}`,
  `fhat(z;rhobar) = int a E_G[sigma(<u,z> + s G)] rhobar`, and as `d -> infinity` the init is
  `a ~ mu_a`, `u = 0`, `s = m_2^w`. So the flow starts at the saddle `u = 0`.
- **Quantitative.** `||fhat(.;Theta^k) - fhat(.;rhobar_{k eta})|| <= e^{K T^7}{ sqrt((P+log d)/d) +
  sqrt(log N/N) + sqrt((d+log N)/b) sqrt(eta) }`, valid for `k <= T/eta`, `T = eta n/b` bounded.
- **Hierarchical pickup.** For vanilla staircases, layerwise training (train `u` then `a`) gives the
  simplified first-layer cascade
  `d/dt huk = a alpha_{1..k} m_k prod_{j<k} huj`, `huk(0)=0`, `m_r = sigma^{(r)}(0)`, which yields
  `u_k ~ t^{2^{k-1}}` (degrees `1, 2, 4, ...`). The second (linear) phase is kernel regression with
  `K^{T1}(z,z') = E_{a~mu_a}[sigma(<u^{T1}(a),z>) sigma(<u^{T1}(a),z'>)]`; in Fourier basis
  `K^{T1} = D(M + Delta)D` with `M = (E_a[a^{beta(S)+beta(S')}])`, `beta(S) = sum_{k in S} 2^{k-1}`.
  Since `beta` hits all of `{0,...,2^P-1}`, `M` is the Gram matrix of `1,X,...,X^{2^P-1}` (lin. indep.),
  so `lambda_min(M) > 0`; for small `T1`, `lambda_min(K^{T1}) > 0` and the risk decays as
  `e^{-lambda_min(K^{T1}) t}`. Random `a ~ mu_a` (neuron diversity) is what makes `M` full rank;
  `sigma^{(r)}(0) != 0` is what keeps the cascade from stalling at degree `r`.

## Final form (training recipe)

The finite Figure 1 style recipe: a two-layer network in the `1/M` mean-field scaling, no bias,
scaled Gaussian first layer with `sqrt(d) w_{j,l} ~ N(0,1)`, readout weights from `Unif([-1,1])`,
shifted-sigmoid activation `sigma(x) = (1 + e^{-x+0.5})^{-1}`, trained by plain SGD with
`eta = 1/2` on the square loss with fresh batches.

```python
import torch
import torch.nn as nn


def build_model(config) -> nn.Module:
    class TwoLayerMeanField(nn.Module):
        def __init__(self, d: int, M: int) -> None:
            super().__init__()
            self.fc1 = nn.Linear(d, M, bias=False)   # no bias
            self.fc2 = nn.Linear(M, 1, bias=False)
            nn.init.normal_(self.fc1.weight, mean=0.0, std=d ** -0.5)
            with torch.no_grad():
                self.fc2.weight.uniform_(-1.0, 1.0)               # a_j ~ Unif([-1,1])
            self.M = M

        @staticmethod
        def _shifted_sigmoid(u: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(u - 0.5)            # sigma(x) = (1 + e^{-x+0.5})^{-1}

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self._shifted_sigmoid(self.fc1(x))
            return self.fc2(h).view(-1) / self.M     # 1/M mean-field normalization

    return TwoLayerMeanField(d=config.d, M=config.width)


def get_optimizer(model: nn.Module, config) -> torch.optim.Optimizer:
    return torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.0)   # plain SGD, eta = 1/2


def train_step(model, optimizer, x, y) -> float:
    optimizer.zero_grad(set_to_none=True)
    pred = model(x).view(-1)
    loss = ((pred - y) ** 2).mean()                  # square loss, one-pass fresh batch
    loss.backward()
    optimizer.step()
    return float(loss.item())
```

This is the mean-field two-layer SGD recipe used for the sparse-hypercube experiments: it realizes the
feature-learning scaling, learns vanilla staircase targets in `O(d)` samples under the theorem's
conditions, and stays stuck on non-MSP targets such as `z1 z2 z3`.
