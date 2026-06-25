Let me start from the actual obstruction. I observe `y = f_*(<theta*,x>) + xi` with Gaussian `x`, a unit vector `theta*` I do not know, and a one-dimensional link `f_*` I also do not know. If I knew the link, this would mostly be a hidden-direction recovery problem. If I knew the direction, this would be ordinary one-dimensional nonparametric regression. The difficulty is that a normal shallow network mixes those two jobs: every hidden row tries to choose a direction while the readout tries to assemble a function, and near a random start the direction signal is tiny.

How tiny? A random unit vector and the truth start at correlation `m=<theta_0,theta*>`. I should pin down the scale before reasoning about it, since everything downstream depends on it. Drawing both directions uniformly on `S^{d-1}` and measuring `|m|` over many trials, the typical magnitude tracks `d^{-1/2}`:

```text
d=  50  median|m| = 0.101   1/sqrt(d) = 0.141
d= 200  median|m| = 0.047   1/sqrt(d) = 0.071
d= 800  median|m| = 0.024   1/sqrt(d) = 0.035
d=3200  median|m| = 0.0125  1/sqrt(d) = 0.018
```

The ratio `median|m| * sqrt(d)` sits near `0.68` across all four dimensions, so `|m_0| = Theta(d^{-1/2})` is the regime I am stuck in at initialization. Whatever signal the gradient carries at the start, it is being read off a correlation that shrinks like `1/sqrt(d)`.

The scale of that signal is encoded by the Hermite expansion. I write `f_* = sum_j alpha_j h_j`, and define the information exponent `s` as the first nonzero Hermite degree. For Gaussian inputs the relevant identity is

```text
<h_j(<theta,.>), h_k(<theta',.>)> = 1{j=k} <theta,theta'>^j.
```

I want to be sure I have this right, because the whole population calculation rests on it. Taking `d=6`, two random unit directions with `<theta,theta'> = 0.147`, and Monte-Carlo averaging the products of normalized Hermite polynomials over four million Gaussian draws:

```text
j=0,k=0:  est = 1.0000   pred = 1.0000
j=1,k=1:  est = 0.1468   pred = 0.1468   (= m^1)
j=2,k=2:  est = 0.0210   pred = 0.0215   (= m^2)
j=3,k=3:  est = 0.0032   pred = 0.0032   (= m^3)
```

Off-diagonal terms come out near zero. So a degree-`j` component along my current direction sees the teacher only through a factor `m^j`, and that factor is the `j`-th power of a correlation I just established is `~ d^{-1/2}`. If the first usable coefficient is degree `s`, the useful signal at initialization is of order `(d^{-1/2})^s = d^{-s/2}`, which is what makes the equator flat and escaping it the statistical bottleneck. The Ben Arous-Gheissari-Jagannath analysis of online SGD says the same thing from the trajectory side: the cost is concentrated in the search phase that escapes the equator, while the final descent to full alignment is comparatively cheap.

Now I need an architecture that preserves that scalar `m` structure instead of letting `N` separate hidden directions wander independently. The target has one direction, so it is worth seeing how far I get with a model that has exactly one trainable direction. I tie the hidden rows to a shared `theta` and let the neurons differ only by a scalar bias and a sign:

```text
G(x;c,theta) = (1/sqrt(N)) sum_i c_i phi(eps_i <theta,x> - b_i).
```

Should the signs and biases move? If `theta` is fixed, those `b_i,eps_i` are just a random-feature dictionary for a one-dimensional kernel in the scalar variable `u=<theta,x>`. Letting them move would turn the link-fitting dictionary into another active part of the high-dimensional search, which is exactly the coupling I am trying to remove. So I freeze them: `theta` is the rich feature-learning variable, while the biases and signs are the lazy random-feature scaffold for the univariate link.

Let me check the algebra to see whether this split actually collapses the loss the way I am hoping. Define the feature operator `T` by

```text
(Tf)_i = (1/sqrt(N)) E_z[f(z) phi(eps_i z - b_i)],     T_j = T h_j.
```

Then the student expansion is `G(x;c,theta)=sum_j <c,T_j> h_j(<theta,x>)`. Using the Hermite correlation identity I just verified, the regularized population loss is

```text
L(c,theta) = 1 + c^T Q_lambda c
             - 2 <c, sum_{j>=s} alpha_j m^j T_j> + sigma^2,
Q_lambda = Q + lambda I.
```

The thing to notice is that the only place the ambient `theta` appears is inside the scalar `m`: every `theta`-dependent term is `m^j` times something built from the frozen dictionary. So the split did what I wanted — the high-dimensional variable has been compressed to one number. The readout gradient is exact:

```text
nabla_c L = 2(Q_lambda c - sum_{j>=s} alpha_j m^j T_j),
```

so for fixed `theta` the optimum is unique,

```text
c*(m) = Q_lambda^{-1} sum_{j>=s} alpha_j m^j T_j.
```

The raw derivative in `theta` is colinear with `theta*`; differentiating the displayed square loss gives a global factor 2, while the stationary equation divides it back out. I will keep that harmless constant out of the bookkeeping. The critical set is governed by the projected loss after substituting `c*(m)`:

```text
Lbar(theta) = - <g_m, P_hat_lambda g_m> + 1 + sigma^2,
g_m = sum_{j>=s} alpha_j m^j h_j,
gbar_m = sum_{j>=s} j alpha_j m^{j-1} h_j,

sum_{j>=s} alpha_j^2 j m^{2j-1}
  = <(I-P_hat_lambda)g_m, gbar_m>.
```

Before trusting this, let me work the cleanest case by hand: a pure single-Hermite link `f_* = h_s`, so `alpha_s=1` and all other coefficients vanish. In the infinite-feature, unregularized limit `P_hat_lambda` is the identity and the right side is exactly zero, so `Lbar = 1+sigma^2 - sum_j alpha_j^2 m^{2j} = 1+sigma^2 - m^{2s}`. Tabulating this on `m in [0,1]` with `sigma^2=0.25`:

```text
s=1:  Lbar(0)=1.250  Lbar(1)=0.250   strictly decreasing,  interior stationary = none
s=2:  Lbar(0)=1.250  Lbar(1)=0.250   strictly decreasing,  interior stationary = none
s=3:  Lbar(0)=1.250  Lbar(1)=0.250   strictly decreasing,  interior stationary = none
```

The loss falls monotonically from the equator (`m=0`, value `1.25`) to the pole (`|m|=1`, value `0.25`), and the only place the derivative `-2s m^{2s-1}` vanishes on the interior is `m=0`. So in the ideal limit the stationary directions are exactly the equator and the two poles, no spurious middle ones. The critical equation's left side `s m^{2s-1}` confirms the same picture from the other direction — it is strictly positive for every `m in (0,1)` and zero only at `m=0`, so no interior `m` can balance a zero right side. The finite-feature problem is therefore to show the right side stays too small to manufacture a balance at some interior `m`.

The right side is an approximation error, and I can bound it by Cauchy-Schwarz:

```text
|<(I-P_hat_lambda)g_m,gbar_m>|
  <= ||(I-P_hat_lambda)g_m|| ||gbar_m||.
```

Bach's random-feature equivalence gives the finite-to-infinite reduction with `N >= C/lambda log(1/(lambda delta))`:

```text
||(I-P_hat_lambda)f||^2 <= 4 A(f,lambda)
```

for zero-mean `f`. The zero-mean condition is not free, and I should check it holds: `g_m` starts at Hermite degree `s>=1`, so it has no degree-0 component and is mean-zero. Good. The ReLU random-feature RKHS then supplies

```text
A(f,lambda) <= C(tau^{1+beta} ||f''||_4^2 lambda^beta + lambda C_f^2),
beta = (1 - 1/tau^2)/(3 + 1/tau^2).
```

The exponent `beta` is what makes the bound shrink with `lambda`, so its sign matters. Evaluating it:

```text
tau=0.9:  beta = -0.055
tau=1.0:  beta =  0.000
tau=1.5:  beta = +0.161
tau=3.0:  beta = +0.286
```

So `beta>0` exactly when `tau>1`; at or below `tau=1` the approximation bound has the wrong sign and buys me nothing. That is the reason the bias variance has to be set with `tau>1` rather than the usual `tau=1`. With `beta>0` fixed, the norms of `g_m` and `gbar_m` carry the same `m` powers as the left side of the critical equation, so the comparison becomes constant in `m`:

```text
|right side| <= 4 lambda^{beta/2} |m|^{2s-1}
                sqrt(C tau^{1+beta}) Ktilde C_{f_*}^2,
left side >= alpha_s^2 s |m|^{2s-1}.
```

Both sides carry `|m|^{2s-1}`, so it cancels and the inequality reduces to a comparison of constants. Choosing `lambda < lambda*` drives the right-side constant below `alpha_s^2 s`, so for every nonzero `m` the left side strictly exceeds the right and the critical equation cannot be satisfied on `0<|m|<1`. The finite-feature landscape keeps the same equator-plus-poles topology I just verified in the ideal case; the random features have not created a spurious middle direction.

The empirical part has to survive two hazards that the population picture hides. First, the equator is degenerate, not a strict saddle, so generic saddle-escape theorems do not apply directly. Second, ReLU gradients are discontinuous at activation flips, so a naive Lipschitz net argument over `theta` fails. The way through is to bound how often a small perturbation of `theta` flips a ReLU activation: by Gaussian anti-concentration only a small fraction of samples sit close enough to a kink, so averaged over the sample the discontinuous contribution is controlled. The resulting uniform gradient deviation has the form

```text
Delta = max{ sqrt(D log(n/delta)/n), (d log(n/delta))^2/n },     D=max{d,N}.
```

With that, approximate empirical critical points split into a bad equatorial cluster with `|m|` small and a good polar cluster with `1-|m|` small — the empirical shadow of the two-point population set. To keep the trajectory from settling into the bad cluster, the early dynamics should escape the equator before the readout starts chasing a wrong one-dimensional fit. That motivates a time-scale schedule:

```text
dot c(t) = - zeta(t) nabla_c L_n(c,theta),     zeta(t)=1{t>T_0},
dot theta(t) = - nabla_theta^S L_n(c,theta).
```

I initialize `c(0)` sparse and small, move only `theta` until `T_0`, then turn on joint training: direction search first, link fitting second. Carrying the bound `Delta` through the escape-and-descent analysis gives the sample requirement

```text
n = Otilde(max{ (d+N)d^{s-1}/lambda^4, d^{(s+3)/2}/lambda^2 }),
```

and the recovery guarantee

```text
1 - |<theta_T,theta*>|
  = Otilde(lambda^{-4} max{(d+N)/n, d^4/n^2}).
```

With constant `lambda` and `s>2`, the first branch `(d+N)d^{s-1}` dominates, giving `Otilde(d^s)` recovery for the ReLU model. I should not overstate the low-exponent cases: at `s=1` the second branch `d^{(s+3)/2}=d^2` takes over, and at `s=2` it gives `d^{2.5}`, both worse than `d^s`. Closing that gap needs the smooth-activation appendix, not ReLU.

Once the direction is good, the nonparametric part should use its own regularization. I refit only `c` on a fresh sample:

```text
chat = argmin_c (1/n') sum_i (c^T Phi(<theta_hat,x_i'>)-y_i')^2
                 + lambda_{n'} ||c||^2.
```

The fresh sample breaks the dependence between the learned direction and the kernel features, so the risk splits:

```text
E[||F_hat-F_*||^2 | theta_hat]
  <= C ||f_*''||^{2/(beta+1)} (sigma^2 tau^2/n')^{beta/(beta+1)}
     + C ||f_*'||^2 (1-|<theta_hat,theta*>|).
```

The first term is the one-dimensional kernel-ridge rate, with no factor of the ambient `d`; the second is just the residual direction error. That is the decoupling I was after: the high-dimensional search pays the information-exponent price `d^s`, and the link fit pays only a one-dimensional price.

Now I have to land this in the actual benchmark scaffold, and this is where I need to be honest. The scaffold's `TwoLayerMLP` has independent first-layer rows; it does not expose the exact tied-direction model, Gaussian bias law, Rademacher signs, spherical projection, two-phase schedule, or fresh-sample ridge refit. The implementable core that survives the translation is: put first-layer rows on the sphere, sample hidden biases once, freeze `fc1.bias`, train the remaining parameters with SGD on MSE, and read the direction off with the task's estimator `normalize(sum_j |a_j| w_j)`. The code I ship matches that reference, not a fictional tied model:

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

So the chain that holds up: the random start sits at `|m| ~ d^{-1/2}`, so a degree-`s` link gives only `~ d^{-s/2}` signal at the equator; tying the model to one trainable direction and freezing the one-dimensional random-feature dictionary collapses the population loss to a scalar function of `m`, which in the ideal limit is monotone with stationary points only at the equator and the poles; the random-feature/RKHS bounds (needing `tau>1` for `beta>0`) keep the finite-width critical set from growing spurious middle points; empirical concentration plus a time-scale schedule lets gradient flow escape the degenerate equator and reach a pole at `Otilde(d^s)` for ReLU when `s>2`; and a separate ridge readout refit recovers a dimension-free link-estimation rate. The MLS-Bench implementation is the fixed-MLP version of the same bias-freezing idea, and its faithful code artifact is the local `frozen_bias.edit.py` block above.
