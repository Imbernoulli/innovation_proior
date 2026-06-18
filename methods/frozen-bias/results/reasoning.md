Let me start from the actual obstruction. I observe `y = f_*(<theta*,x>) + xi` with Gaussian `x`, a unit vector `theta*` I do not know, and a one-dimensional link `f_*` I also do not know. If I knew the link, this would mostly be a hidden-direction recovery problem. If I knew the direction, this would be ordinary one-dimensional nonparametric regression. The difficulty is that a normal shallow network mixes those two jobs: every hidden row tries to choose a direction while the readout tries to assemble a function, and near a random start the direction signal is tiny.

The scale of that tiny signal is already encoded by the Hermite expansion. I write `f_* = sum_j alpha_j h_j`, and define the information exponent `s` as the first nonzero Hermite degree. For Gaussian inputs,

```text
<h_j(<theta,.>), h_k(<theta',.>)> = 1{j=k}<theta,theta'>^j.
```

So if my current direction has correlation `m=<theta,theta*>`, each degree-`j` component only sees the teacher through a factor `m^j`. A random unit vector starts at `|m| ~ d^{-1/2}`. If the first usable coefficient is degree `s`, the landscape is flat around the equator, and escaping that equator is the statistical bottleneck. The Ben Arous-Gheissari-Jagannath picture tells me the hard part is search, not the final descent: once I have macroscopic correlation, alignment is comparatively cheap.

Now I need an architecture that preserves that scalar `m` structure instead of letting `N` separate hidden directions wander independently. The target has one direction, so the cleanest model should have one trainable direction. I tie the hidden rows to a shared `theta` and let the neurons differ only by a scalar bias and a sign:

```text
G(x;c,theta) = (1/sqrt(N)) sum_i c_i phi(eps_i <theta,x> - b_i).
```

The signs and biases should not move. If `theta` is fixed, those `b_i,eps_i` are just a random-feature dictionary for a one-dimensional kernel in the scalar variable `u=<theta,x>`. Moving them would turn the link-fitting dictionary into another active part of the high-dimensional search, which is exactly the coupling I am trying to remove. So the split is: `theta` is the rich feature-learning variable, while the biases and signs are the lazy random-feature scaffold for the univariate link.

Let me check the algebra. Define the feature operator `T` by

```text
(Tf)_i = (1/sqrt(N)) E_z[f(z) phi(eps_i z - b_i)],     T_j = T h_j.
```

Then the student expansion is `G(x;c,theta)=sum_j <c,T_j> h_j(<theta,x>)`. Using the Hermite correlation identity, the regularized population loss becomes

```text
L(c,theta) = 1 + c^T Q_lambda c
             - 2 <c, sum_{j>=s} alpha_j m^j T_j> + sigma^2,
Q_lambda = Q + lambda I.
```

This is the first key point: all dependence on the ambient `theta` enters through the single scalar `m`. The readout gradient is exact:

```text
nabla_c L = 2(Q_lambda c - sum_{j>=s} alpha_j m^j T_j),
```

so for fixed `theta` the optimum is unique,

```text
c*(m) = Q_lambda^{-1} sum_{j>=s} alpha_j m^j T_j.
```

The raw derivative in `theta` is colinear with `theta*`; differentiating the displayed square loss gives a global factor 2, while the source's gradient display suppresses that factor. I do not want to make that harmless constant do any work. The critical set is governed by the projected loss after substituting `c*(m)`, and there the source derives the stationary equation after dividing by 2:

```text
Lbar(theta) = - <g_m, P_hat_lambda g_m> + 1 + sigma^2,
g_m = sum_{j>=s} alpha_j m^j h_j,
gbar_m = sum_{j>=s} j alpha_j m^{j-1} h_j,

sum_{j>=s} alpha_j^2 j m^{2j-1}
  = <(I-P_hat_lambda)g_m, gbar_m>.
```

In the infinite-feature, unregularized limit, `P_hat_lambda` is the identity and the right side vanishes. Then `Lbar = 1+sigma^2 - sum_j alpha_j^2 m^{2j}` is strictly decreasing in `|m|`; the only stationary directions are the equator and the two poles. The finite-feature problem is to prove the right side is too small to create a new stationary point.

The right side is an approximation error:

```text
|<(I-P_hat_lambda)g_m,gbar_m>|
  <= ||(I-P_hat_lambda)g_m|| ||gbar_m||.
```

Bach's random-feature equivalence gives the finite-to-infinite reduction with `N >= C/lambda log(1/(lambda delta))`:

```text
||(I-P_hat_lambda)f||^2 <= 4 A(f,lambda)
```

for zero-mean `f`. The zero-mean condition matters, and here it holds because `g_m` starts at Hermite degree `s>=1`. The ReLU random-feature RKHS then supplies

```text
A(f,lambda) <= C(tau^{1+beta} ||f''||_4^2 lambda^beta + lambda C_f^2),
beta = (1 - 1/tau^2)/(3 + 1/tau^2).
```

This is why `tau>1` is not cosmetic: it makes `beta` positive. For `g_m`, the norms carry the same `m` powers as the left side, so the comparison becomes constant in `m`:

```text
|right side| <= 4 lambda^{beta/2} |m|^{2s-1}
                sqrt(C tau^{1+beta}) Ktilde C_{f_*}^2,
left side >= alpha_s^2 s |m|^{2s-1}.
```

Choosing `lambda < lambda*` makes the right side strictly smaller for every nonzero `m`. That contradiction rules out all critical points with `0<|m|<1`. The population landscape has the same topology as the ideal Hermite basis: equator plus poles, no spurious middle directions.

The empirical part has to survive two hazards. First, the equator is degenerate, not a strict saddle, so generic saddle-escape theorems are not enough. Second, ReLU gradients are discontinuous at activation flips, so a naive Lipschitz net argument fails. The concentration proof handles this by showing that only a small fraction of Gaussian samples can sit close enough to a ReLU kink for a small perturbation of `theta` to change the activation pattern. Averaged over samples, the bad discontinuity contribution is controlled, and the uniform gradient deviation has the form

```text
Delta = max{ sqrt(D log(n/delta)/n), (d log(n/delta))^2/n },     D=max{d,N}.
```

With that, approximate empirical critical points split into a bad equatorial cluster with `|m|` small and a good polar cluster with `1-|m|` small. To avoid returning to the bad cluster, I need the early trajectory to escape the equator before the readout starts chasing a wrong one-dimensional fit. That is the reason for the time-scale schedule:

```text
dot c(t) = - zeta(t) nabla_c L_n(c,theta),     zeta(t)=1{t>T_0},
dot theta(t) = - nabla_theta^S L_n(c,theta).
```

I initialize `c(0)` sparse and small, move only `theta` until `T_0`, and then turn on joint training. This is a proof device, but it captures the design principle: direction search first, link fitting second. The theorem's sample requirement is

```text
n = Otilde(max{ (d+N)d^{s-1}/lambda^4, d^{(s+3)/2}/lambda^2 }),
```

and the recovery guarantee is

```text
1 - |<theta_T,theta*>|
  = Otilde(lambda^{-4} max{(d+N)/n, d^4/n^2}).
```

So with constant `lambda` and `s>2`, the ReLU model has `Otilde(d^s)` recovery. I should not overstate the low-exponent cases: the same ReLU theorem gives `d^2` for `s=1` and `d^2.5` for `s=2`; the smooth-activation appendix is what improves those to `d^s`.

Once the direction is good, the nonparametric part should use its own regularization. I therefore refit only `c` on a fresh sample:

```text
chat = argmin_c (1/n') sum_i (c^T Phi(<theta_hat,x_i'>)-y_i')^2
                 + lambda_{n'} ||c||^2.
```

The fresh sample breaks the dependence between the learned direction and the kernel features. The risk splits cleanly:

```text
E[||F_hat-F_*||^2 | theta_hat]
  <= C ||f_*''||^{2/(beta+1)} (sigma^2 tau^2/n')^{beta/(beta+1)}
     + C ||f_*'||^2 (1-|<theta_hat,theta*>|).
```

The first term is the one-dimensional kernel-ridge rate; the second is only the residual direction error. That is the decoupling I wanted: high-dimensional search pays the information-exponent price, and the link fit pays a one-dimensional price.

Now I have to land this in the actual benchmark scaffold, and this is where I need to be honest. The scaffold's `TwoLayerMLP` has independent first-layer rows; it does not expose the exact tied-direction model, Gaussian bias law, Rademacher signs, spherical projection, two-phase schedule, or fresh-sample ridge refit. The local reference edit keeps the implementable core: put first-layer rows on the sphere, sample hidden biases once, freeze `fc1.bias`, train the remaining parameters with SGD on MSE, and use the task's standard direction estimator `normalize(sum_j |a_j| w_j)`. The code I ship should match that reference, not a fictional tied model:

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

So the final chain is: the information exponent tells me why random initialization near the equator is hard; tying the theorem model to one trainable direction and freezing the one-dimensional random-feature dictionary collapses the population loss to a scalar function of `m`; the random-feature/RKHS bounds keep the finite-width critical set from growing spurious middle points; empirical concentration plus a time-scale schedule lets gradient flow escape the degenerate equator and reach a pole at `Otilde(d^s)` for ReLU when `s>2`; and a separate ridge readout refit recovers a dimension-free link-estimation rate. The MLS-Bench implementation is the fixed-MLP version of the same bias-freezing idea, and its faithful code artifact is the local `frozen_bias.edit.py` block above.
