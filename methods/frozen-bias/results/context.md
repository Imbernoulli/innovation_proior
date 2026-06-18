# Context: learning single-index models with shallow networks (circa 2021-2022)

## Research question

A single-index model is a high-dimensional regression target that depends on the input only
through one unknown direction:

```
y = f_*(<theta*, x>) + xi,   x ~ N(0, I_d),   theta* in S^{d-1},   xi ~ N(0, sigma^2),
```

where *both* the unit direction `theta*` and the univariate "link" `f_*` are unknown. The
projection `<theta*, x>` is a single-number summary (an "index") of the high-dimensional point
`x`. This is the simplest target on which a learner must do genuine *feature learning* — it
cannot succeed by treating the input features as fixed, because the informative one-dimensional
subspace is hidden and must be discovered. The problem is *semi-parametric*: a high-dimensional
*parametric* part (recover `theta*` in `R^d`) sits next to a low-dimensional *non-parametric*
part (estimate the univariate `f_*`).

The precise goal is a single learner that, trained by plain gradient descent, simultaneously
(1) recovers the hidden direction `theta*` with *near-optimal* sample complexity in the
dimension `d`; (2) approximates the link `f_*` non-parametrically with a rate that does *not*
degrade with `d`; (3) does this *without knowing `f_*` in advance* (so the activation cannot be
chosen to equal the link, as in teacher-student studies); and (4) does it with a standard,
practitioner-style shallow network and gradient method rather than a bespoke estimator built
for this one problem. Each existing approach below achieves a subset; none achieves all four
with a network trained end to end. Closing that gap is the problem.

## Background

By this time the central theoretical question is *when do neural networks, trained by
gradient methods, provably beat fixed-kernel methods on high-dimensional data with hidden
low-dimensional structure?* Approximation-theoretic advantages of shallow nets over
non-adaptive kernels have been known for decades (Barron 1993; Pinkus 1999), and Bach (2017a)
showed that infinite-width shallow networks approximate single- and multi-index models *without
a curse of dimensionality*, hinting at a statistical advantage — but that work gives no
tractable training algorithm. The missing piece is a *computational* guarantee: a concrete
gradient procedure that finds the hidden structure efficiently. Hardness results rule this out
in full generality, so progress comes from structured target classes; single-index models are
the cleanest such class.

The load-bearing concepts:

- **Hermite analysis of Gaussian data.** With `x ~ N(0, I_d)`, the normalized Hermite
  polynomials `{h_j}` form an orthonormal basis of `L^2(gamma)`. Two facts drive everything:
  `h_j' = sqrt(j) h_{j-1}`, and the *correlation identity*
  `<h_j(<theta, .>), h_{j'}(<theta', .>)>_{gamma_d} = delta_{j,j'} <theta, theta'>^j`. The
  second says that the overlap between a degree-`j` Hermite feature along two directions is
  exactly the `j`-th power of their inner product. Writing `f_* = sum_j alpha_j h_j` exposes the
  target's structure through its Hermite coefficients `alpha_j`.

- **Information exponent (Ben Arous, Gheissari, Jagannath, JMLR 2021).** The information
  exponent `s` of `f_*` is the index of its *first nonzero* Hermite coefficient,
  `s = min{j : alpha_j != 0}`. It governs how flat the population landscape is near the
  "equator." For the rank-one estimation problem with correlation `m = <theta, theta*>`, the
  population loss behaves near `m = 0` like `Phi(m) ~ m^s`: the loss is flat to order `s` at the
  equator, so the *gradient signal* there scales like `m^{s-1}`. Their analysis of online SGD
  from a random start gives sample-complexity thresholds that depend *only* on `s`:
  weak recovery (escaping the equator to reach a macroscopic correlation) needs
  `n = Theta(1)·d` samples for `s = 1`, `n = Theta(d log d)` for `s = 2`, and
  `n = Theta(d^{s-1})` for `s >= 3`, with a matching lower bound up to logarithmic factors.
  They also establish the *search-vs-descent* picture: for `s >= 2` essentially all the data is
  spent in the *search* phase that escapes the high-entropy equatorial region; once a
  macroscopic correlation is reached, the *descent* to full alignment `|m| -> 1` is fast,
  costing only `O(d)` further samples. This is the diagnostic that frames the whole difficulty:
  the hard part is leaving the neighborhood of a random initialization, *not* the final descent.

- **Random initialization sits dangerously near the equator.** A direction drawn uniformly on
  `S^{d-1}` has correlation `|<theta_0, theta*>| = Theta(1/sqrt(d))` with the truth (a standard
  high-dimensional-probability fact). So at initialization the learner is at `m ~ 1/sqrt(d)`,
  *inside* the flat equatorial region whose gradient signal is `~ m^{s-1}`. The signal there is
  tiny, and it must be told apart from the finite-sample fluctuation of the empirical gradient.

- **Random features and the degrees-of-freedom equivalence (Rahimi & Recht 2007; Bach 2017b).**
  If a direction is *fixed*, fitting only the readout layer of a shallow net is a *random-feature*
  model approximating a fixed kernel. Bach (2017b) shows the number of random features needed to
  match the kernel/RKHS approximation error is governed by the kernel's *degrees of freedom* —
  roughly `1/lambda` for a Tikhonov level `lambda` — rather than by the ambient dimension: with
  `N >~ (1/lambda) log(1/(lambda delta))` features the finite-feature approximation error is
  within a constant factor of the infinite-width RKHS error.

- **Lazy vs. rich training regimes (Chizat, Oyallon, Bach 2019).** Whether a differentiable
  model *learns features* or merely behaves like its linearization (a fixed kernel) is set by a
  relative scaling between its parts. A "lazy" part stays near initialization and acts as a
  kernel; a "rich" part moves and learns features. The relative scale between layers is the
  knob that interpolates between the two regimes.

- **Benign non-convex landscapes and uniform concentration.** A line of work (Ge et al.;
  Mei, Bai & Montanari 2016; Foster et al. 2018; Dudeja & Hsu 2018) studies non-convex
  objectives whose *population* landscape has only well-understood critical points, and transfers
  this structure to the *empirical* landscape via uniform convergence of the empirical gradient
  to the population gradient, of the form `||nabla L_n - nabla L|| = O(sqrt(d/n))`. A recurring
  subtlety is that some of these landscapes lack the *strict-saddle* property usually used to
  guarantee escape, so the flat equatorial saddle needs a dedicated argument.

## Baselines

The prior approaches a new method would be measured against.

**Teacher-student gradient analyses (Goldt et al.; Ge et al.; Zhong et al.; Ben Arous et al.
2021).** Study a "student" network trained to fit a "teacher," where the student's *activation
is set equal to the teacher's link* `f_*`. Under this match, learning reduces to estimating the
hidden direction(s), and the population landscape (in the Hermite basis) has exactly the
information-exponent structure above. **Limitation:** the link is *assumed known* and hard-wired
into the architecture. The non-parametric half of the semi-parametric problem is removed by
fiat, so these results do not cover the situation where `f_*` is unknown and must itself be
estimated.

**Dedicated semi-parametric / single-index estimators (projection pursuit, Friedman & Stuetzle;
sliced inverse regression, Li 1991; moment methods, Dalalyan et al.; Hermite-feature methods,
Dudeja & Hsu 2018).** These recover `theta*` (and sometimes `f_*`) with `n = O(d^s)` samples by
*explicitly* exploiting the model: e.g. building estimators around individual Hermite
polynomials or inverse-regression slices. Dudeja & Hsu (2018) in particular characterize the
population landscape of such an objective over Gaussian data and obtain the benign,
`m^s`-near-the-equator topology. **Limitation:** each is a *tailored procedure* engineered for
the single-index structure, not a standard network trained by gradient descent; the recovery of
`theta*` and the estimation of `f_*` are handled by separate, problem-specific machinery rather
than emerging from one end-to-end gradient method.

**Infinite-width approximation results (Bach 2017a).** Show a *representational* advantage:
shallow networks can represent single- and multi-index targets without a curse of dimensionality.
**Limitation:** purely about approximation/representation — no algorithm, no optimization or
sample-complexity guarantee for actually *finding* the representing network by gradient descent.

**One-giant-step / first-layer-then-readout analyses (Ba et al. 2022; Damian, Lee &
Soltanolkotabi 2022), concurrent.** Take a *single* large gradient step on the first layer, then
fit the readout, and show this already separates networks from fixed kernels — e.g. learning the
relevant direction with `n = O(d^2)` samples. **Limitation:** a single step suffices only for a
*separation*; it does not optimize both layers to convergence, and the resulting sample
complexity does not track the information exponent `s` of a general link. What happens when the
two layers are trained *jointly to convergence* — and whether that recovers the
information-exponent-optimal rate for arbitrary `s` — is left open.

**Plain shallow ReLU net + SGD with standard initialization.** The natural practitioner baseline:
a two-layer ReLU MLP, Kaiming-initialized, both layers trained by SGD with momentum.
**Limitation:** with all first-layer weights free and untied, the high-dimensional search and the
non-parametric fit are entangled; near a random start the direction signal is buried in the flat
equatorial region, and there is no clean theory that the joint dynamics recover `theta*` at the
information-exponent rate — naive analyses of the entangled dynamics give pessimistic
complexities far above `d^s`.

## Evaluation settings

The natural yardsticks for this problem, all pre-existing.

- **Synthetic single-index data with controllable information exponent.** Gaussian inputs
  `x ~ N(0, I_d)` in `d` dimensions; a hidden unit direction `theta*` drawn at random; a link
  `f_*` chosen so its information exponent `s` can be set (e.g. a piecewise-linear or polynomial
  link, or one with its low-order Hermite components removed to raise `s`). Label noise
  `xi ~ N(0, sigma^2)` added to training targets; the test target is noise-free.
- **Link families spanning the difficulty axis.** An easy monotone link (`s = 1`), a higher
  information-exponent link beyond the kernel regime (`s >= 3`), and a non-smooth link (e.g.
  `sign`) to probe robustness.
- **Metrics.** *Direction recovery* `|<theta_hat, theta*>|` (the correlation `|m|`, with `1`
  perfect), measured as the primary signal; and held-out *test mean-squared error*
  `||F_hat - F_*||^2` on a fresh noise-free sample, for the function-estimation quality. Reported
  as a function of sample size `n` and dimension `d`, to read off the empirical sample-complexity
  rate and check whether the excess-risk rate is independent of `d`.
- **Protocol.** Train from random initialization (direction uniform on the sphere); run gradient
  descent for a fixed budget of iterations; optionally re-fit the readout layer by ridge
  regression at the end. Repeat over several random seeds (different `theta*` and data draws),
  reporting central tendency and spread.

## Code framework

The method plugs into a fixed shallow-network training harness. The model is a two-layer ReLU
MLP `Linear(d, W) -> ReLU -> Linear(W, 1)`; the data generator, the link functions, the
evaluation, and the outer driver are all fixed. The learner only controls four callbacks:
how to *initialize* the two layers (any in-place scheme; parameters may be frozen), how to build
the *optimizer* over the trainable parameters, how to run one *training step* on a minibatch,
and an optional *finalize* hook for a post-hoc re-fit. Nothing about how the direction is to be
discovered or how the readout dictionary should be chosen is settled — that is exactly what is
to be designed, so the slots are left empty.

```python
import torch


class TwoLayerMLP(torch.nn.Module):
    """Fixed model: Linear(d, W) -> ReLU -> Linear(W, 1). Given, not designed here."""
    def __init__(self, d, W):
        super().__init__()
        self.fc1 = torch.nn.Linear(d, W)     # first layer: weight (W, d), bias (W,)
        self.fc2 = torch.nn.Linear(W, 1)     # readout layer
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x))).squeeze(-1)


class Strategy:
    """The learner. Decides initialization, optimizer, the per-step update, and any refit."""

    def init_two_layer(self, net: TwoLayerMLP, config) -> None:
        # TODO: how to initialize fc1 (weights and biases) and fc2, and which
        #       parameters, if any, to train vs. hold fixed. To be designed.
        pass

    def make_optimizer(self, net: TwoLayerMLP, config) -> torch.optim.Optimizer:
        # TODO: an optimizer over whichever parameters we decide to train.
        pass

    def training_step(self, net, optimizer, x, y, step, config):
        # one gradient update on a minibatch; return the loss
        net.train()
        optimizer.zero_grad(set_to_none=True)
        preds = net(x)
        loss = torch.mean((preds - y) ** 2)        # squared-error objective
        loss.backward()
        optimizer.step()
        return loss

    def finalize(self, net, x_train, y_train, config) -> None:
        # TODO: optional post-hoc re-fit of the readout, if useful.
        pass


def direction_estimate(net: TwoLayerMLP) -> torch.Tensor:
    # standard direction read-out from the trained first/second layers
    w = net.fc1.weight                  # (W, d) first-layer rows
    a = net.fc2.weight.squeeze(0)       # (W,)   readout weights
    theta_hat = (a.abs().unsqueeze(1) * w).sum(0)
    return theta_hat / theta_hat.norm().clamp_min(1e-12)
```

The empty `init_two_layer`, `make_optimizer`, and `finalize` slots are where the method's
choices about which parameters move, which are held fixed, and how the readout is fit will go.
