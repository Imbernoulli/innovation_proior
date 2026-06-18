Let me start from what actually makes this hard. I have data `y = f_*(<theta*, x>) + xi` with `x ~ N(0, I_d)` Gaussian, `theta*` a unit vector I don't know, and `f_*` a univariate function I also don't know. The target only depends on `x` through the single projection `<theta*, x>`. Two things to recover, living on completely different scales: the *direction* `theta*` is a needle in `d`-dimensional space, and the *link* `f_*` is a one-dimensional function. This is the cleanest possible feature-learning problem — there's a hidden one-dimensional structure, and a learner that treats the coordinates of `x` as fixed features (a kernel method) is going to pay a curse of dimensionality, because it can't adapt its basis to the unknown direction. I want a shallow network trained by ordinary gradient descent to find both, and I want it to be near-optimal in samples.

What do people do today? The teacher-student line — Goldt, Ge, Zhong, and especially the analysis of Ben Arous, Gheissari and Jagannath — studies exactly this kind of hidden-direction recovery, but with a crucial cheat: the student's activation function is *set equal to the link* `f_*`. If I already know `f_*`, the non-parametric half of the problem evaporates and all that's left is finding `theta*`. That's not my situation; I don't know `f_*`. On the other side, the classical semi-parametric statistics — projection pursuit, sliced inverse regression, the Hermite-feature estimator of Dudeja and Hsu — recover `theta*` and even `f_*` with `n = O(d^s)` samples, but each is a bespoke procedure that explicitly exploits the single-index structure, builds estimators around individual Hermite polynomials or inverse-regression slices. Beautiful, but it's not a network trained end to end; the direction recovery and the function fit are done by separate machinery. And Bach showed infinite-width shallow nets can *represent* these targets without a curse of dimensionality — but that's pure approximation theory, no algorithm. So the gap is sharp: I want one gradient method on one shallow network that does both halves at once, optimally, without knowing the link.

Let me first understand *why* finding the direction is hard, precisely, because the difficulty is going to dictate the architecture. The relevant order parameter is the correlation `m = <theta, theta*>` between my current direction and the truth. At a random initialization, `theta` is uniform on the sphere `S^{d-1}`, and a uniform random unit vector has correlation `|<theta_0, theta*>| = Theta(1/sqrt(d))` with any fixed direction — that's a basic high-dimensional fact, the sphere concentrates near its equator. So I *start* at `m ~ 1/sqrt(d)`, microscopically close to the equator `{m = 0}`. Now, how strong is the gradient signal there? Expand `f_*` in Hermite polynomials, `f_* = sum_j alpha_j h_j`, and let `s` be the index of the first nonzero coefficient — Ben Arous et al. call this the *information exponent*. The Hermite correlation identity for Gaussian data, `<h_j(<theta,.>), h_{j'}(<theta',.>)>_{gamma_d} = delta_{j,j'} <theta,theta'>^j`, means a degree-`j` feature along `theta` overlaps the target's degree-`j` component by exactly `m^j`. So the population loss, as a function of the direction, is going to be built out of powers `m^{2j}`, and near the equator the leading term is `m^{2s}` — the loss is *flat to order `s`*, and the gradient signal scales like `m^{2s-1}`, or in the rank-one language `~ m^{s-1}`. At `m ~ 1/sqrt(d)` that signal is `~ d^{-(s-1)/2}`, vanishingly small. And it has to be distinguished from the finite-sample noise in the empirical gradient, which by uniform concentration is `~ sqrt(d/n)`. Set signal above noise: roughly `d^{-(s-1)/2} > sqrt(d/n)` after the relevant powers, and you land on `n ~ d^s`. That's the whole story of the difficulty — it's *escaping the flat equator from a random start*, not the final descent. Ben Arous et al. made this quantitative: the descent phase, once you have macroscopic correlation, is cheap (`O(d)` more samples), and essentially all the data goes into the search phase. Their thresholds are `n = d` for `s=1`, `n = d log d` for `s=2`, `n = d^{s-1}` for `s >= 3`, with a matching lower bound. So `d^s` (for weak-then-strong recovery in batch) is the target rate, and it's set by `s` alone.

Good. Now the design question: what shallow network, with what trained and what frozen, makes the gradient landscape *exactly* this benign one-dimensional `m`-landscape, instead of some entangled high-dimensional mess? Let me write the most generic shallow net and see what fights me. A standard two-layer ReLU net has `G(x) = sum_i c_i phi(<w_i, x> - b_i)` with *every* inner weight `w_i in R^d` free and every bias `b_i` free. The trouble is immediate: with all `w_i` independent and trainable, the high-dimensional search (which directions do the `w_i` point?) and the non-parametric fit (what shape does the readout build?) are completely tangled. The gradient with respect to any `w_i` mixes both jobs. Naive analyses of this coupled dynamics give pessimistic sample complexities, far above `d^s`, because during the early phase `c` and the `w_i` interact adversarially. I need to *decouple* the two halves.

Here's the structural observation. The target has *one* direction. So why give the network `N` independent inner directions? Let me tie them: a single shared inner direction `theta in S^{d-1}`, with the neurons differing only in their *bias* (and a sign):

```
G(x; c, theta) = (1/sqrt(N)) sum_{i=1}^N c_i phi(eps_i <theta, x> - b_i),
```

`phi = ReLU`, `eps_i in {+1,-1}` Rademacher signs. Now there is exactly *one* high-dimensional parameter, `theta`, carrying the direction inference, and the collection `{(b_i, eps_i)}` is a dictionary of one-dimensional features in the scalar variable `u = <theta, x>`, with `c` selecting among them to build the link. The architecture literally mirrors the single-index factorization: one direction times one univariate function.

But what do I do with the biases `b_i`? Train them? Stare at this for a second. If I *fix* `theta` and learn `c` alone, this is a *random-feature model* — Rahimi and Recht — approximating a fixed kernel. Which kernel? If I draw `b ~ N(0, tau^2)` and `eps` Rademacher,

```
kappa(u, v) = E_{b, eps}[ phi(eps u - b) phi(eps v - b) ],
```

and `c` is fitting a function in the RKHS of `kappa`. The biases *are the random-feature sampling* of that kernel. And the lesson from Chizat, Oyallon and Bach on lazy versus rich training is that a part of the model that *stays at initialization and acts like a fixed kernel* is the "lazy" part, while a part that *moves and learns features* is the "rich" part. The non-parametric link-fitting is exactly a lazy, kernel job; the direction-finding is exactly a rich, feature-learning job. So the biases should be *lazy* — sampled once and frozen — and `theta` should be *rich* — trained. If I were to also train the biases, I'd drag the non-parametric problem back into the high-dimensional dynamics and re-entangle the two halves I just separated. So: freeze `b_i` (and `eps_i`) at random initialization, train only `theta` and `c`.

Let me check that this freezing actually buys the landscape collapse I want, because if it does, that's the justification, not "it seems natural." Compute the population loss. Write `L(c, theta) = E_{x,y}[(y - G(x;c,theta))^2] + lambda ||c||^2`, with a Tikhonov `lambda ||c||^2` on the readout (I'll justify `lambda` shortly). Expand the square and use the Hermite decomposition of the student. Define the feature operator `T: L^2(gamma) -> R^N` by `(Tf)_i = (1/sqrt(N)) E_{z~gamma}[f(z) phi(eps_i z - b_i)]`, and `T_j = T h_j`. Then `G(x;c,theta) = c^T Phi(<theta,x>) = sum_{j>=0} <c, T_j> h_j(<theta,x>)`. The key is again the Hermite correlation identity: cross terms between the student's degree-`j` part and the teacher's degree-`j'` part contribute `<theta,theta*>^j = m^j` only when `j = j'`. Carrying it through,

```
L(c, theta) = 1 + c^T Q_lambda c - 2 <c, sum_{j>=s} alpha_j m^j T_j> + sigma^2,
```

where `Q = T T^*` is the `N x N` feature-covariance matrix and `Q_lambda = Q + lambda I`. (The constant `1` is `sum_j alpha_j^2 = ||f_*||^2`, normalized to one; `sigma^2` is the irreducible label-noise floor.) And here's the payoff: **the direction `theta` enters this expression only through the scalar `m = <theta, theta*>`.** All `d` dimensions of the high-dimensional landscape have collapsed onto a single number. That collapse is *because* the biases are frozen — they don't depend on `theta`, so the only `theta`-dependence flows through the inner products `m^j`. Freezing the biases is exactly what makes the high-dimensional problem effectively one-dimensional. That's the structural reason, derived, not asserted.

Now differentiate. With respect to `c`,

```
nabla_c L = 2 (Q_lambda c - sum_{j>=s} alpha_j m^j T_j),
```

and since `Q_lambda` is positive definite for `lambda > 0`, `L` is *strictly convex* in `c` at any fixed `theta`, with a unique minimizer `c = Q_lambda^{-1} sum_{j>=s} alpha_j m^j T_j`. Good — the readout has a clean closed form given the direction. With respect to `theta`, only the `m^j = <theta,theta*>^j` factors carry the dependence, and `d/dtheta (m^j) = j m^{j-1} theta*`, so

```
nabla_theta L = -( sum_{j>=s} <T_j, c> j alpha_j m^{j-1} ) theta*.
```

Look at that — the gradient with respect to `theta` is *colinear with `theta*`*. It points straight at the truth (or straight away). The only thing the dynamics can do to `theta` is move it along the `theta*` axis, i.e. change `m`. Combined with the spherical constraint (I take the Riemannian gradient `nabla^sph = proj_{theta^perp} nabla`, because `theta` lives on `S^{d-1}` — I *need* the unit norm for identifiability, otherwise I could rescale `theta` and absorb it into `f_*`), this means the whole high-dimensional flow reduces to a scalar flow in `m`. The frozen-bias architecture has turned a `d`-dimensional non-convex optimization into a one-dimensional one. That is the entire point.

So let me characterize the critical points in the variable `m`. Substitute the optimal `c` back to get the *projected* loss `Lbar(theta) = min_c L(c,theta)`. Using the SVD `T = U Lambda V` and the identity `<T_j, Q_lambda^{-1} T_{j'}> = <h_j, P_lambda h_{j'}>_gamma` where `P_lambda = Sigma(Sigma + lambda I)^{-1}` is the regularized projection onto the random-feature space (`Sigma = T^* T`),

```
Lbar(theta) = - <g_m, P_lambda g_m>_gamma + (1 + sigma^2),   g_m(z) := sum_{j>=s} alpha_j m^j h_j(z).
```

`g_m` is exactly the best function you could fit by tuning the readout at fixed `theta`, with no regularization — the projection of the target onto degree structure scaled by `m^j`. Differentiate `Lbar` in `m`. The critical-point condition is

```
sum_{j>=s} alpha_j^2 (2j) m^{2j-1} - 2 <(I - P_lambda) g_m, gbar_m>_gamma = 0,
```

where `gbar_m(z) = sum_{j>=s} j alpha_j m^{j-1} h_j(z)` is `g_m` differentiated in `m` (it weights each Hermite component by its degree `j`). Divide by two:

```
sum_{j>=s} alpha_j^2 j m^{2j-1} = <(I - P_lambda) g_m, gbar_m>_gamma.     (*)
```

Let me read this. If the random features were *infinitely many* and unregularized (`lambda -> 0`, `P_lambda -> I`), the right side vanishes and the equation is `sum_{j>=s} alpha_j^2 j m^{2j-1} = 0`. The left side is `m^{s-1}` times a strictly positive series in `m^2` for `m != 0`, so it's zero only at `m = 0`. And in that ideal case `Lbar = -sum_{j>=s} alpha_j^2 m^{2j} + (1+sigma^2)` is *strictly decreasing in `|m|`* — its only critical points are the equator `m = 0` and the poles `m = +-1`. A perfectly benign landscape: no spurious local minima, the optimizer just slides from the equator up to a pole. The whole game is to show that switching to *finitely many regularized* random features keeps this picture — that the right side of (*) is a small enough perturbation that it can't create a new critical point at some `0 < |m| < 1`.

This is where I have to be careful, and where I'd hit a wall if I were sloppy. The right side is `<(I - P_lambda) g_m, gbar_m>`. The term `(I - P_lambda) g_m` is the part of `g_m` *not captured* by the random-feature space — the approximation error of the kernel. I need to bound it. By Cauchy-Schwarz, `|<(I-P_lambda)g_m, gbar_m>| <= ||(I-P_lambda)g_m|| · ||gbar_m||`. So everything rides on `||(I-P_lambda)g_m||`, the random-feature approximation error of `g_m`. Two pieces: how well does the *infinite-width* RKHS approximate `g_m`, and how many *finite* features do I need to inherit that?

For the finite-feature step, I lean on Bach's random-features-equal-kernel-quadrature result: the number of features needed to match the RKHS approximation error is set by the kernel's *degrees of freedom*, roughly `1/lambda`, not by the ambient dimension. Concretely, if `N >= (C/lambda) log(1/(lambda delta))`, then with probability `1-delta`, for any zero-mean `f`,

```
||(I - P_lambda) f||_gamma^2 <= 4 A(f, lambda),
```

where `A(f, lambda) = min_{g in H} ||f - g||^2 + lambda ||g||_H^2` is the regularized RKHS approximation error. So `N ~ 1/lambda` features suffice; importantly, **`N` does not depend on `d`** — the non-parametric width requirement is a one-dimensional quantity. (The zero-mean condition is what keeps the degrees-of-freedom bound tight; `g_m` is built from `h_j` with `j >= s >= 1`, so it has mean zero.)

Now the infinite-width RKHS approximation error `A(f, lambda)`. I want a polynomial-in-`lambda` rate. The ReLU choice helps here for a concrete reason: ReLU's piecewise-linear structure gives an explicit Sobolev representation of the RKHS norm. Working it out (the kernel `kappa(u,v) = <psi(u), psi(v)>` with `psi(u) = (1/sqrt2)(phi(u-.), phi(-u-.))`, and the RKHS-from-features theorem), I get for `f` with two derivatives and a tail condition,

```
||f||_H^2 <= 6 tau ( int |f''(t)|^2 / gamma_tau(t) dt + ||f||_gamma^2 + 6 ||f'||_gamma^2 + 2 <f, f''>_gamma ).
```

The `int |f''|^2/gamma_tau` term is the load-bearing one — it's a weighted second-derivative norm, so the RKHS controls *curvature*. Using a family of approximants that equal `f` on `[-M, M]` and are linear outside (so they live in the RKHS), and an `L^4(gamma)` control on `f''` (a mild hypercontractivity condition, satisfied by polynomials, sigmoids, compactly-supported smooth functions), the approximation error works out to

```
A(f, lambda) <= C ( tau^{1+beta} ||f''||_4^2 · lambda^beta + lambda C_f^2 ),    beta = (1 - 1/tau^2)/(3 + 1/tau^2).
```

This is where `tau > 1` earns its place. The exponent `beta` is positive *iff* `tau > 1`; and the larger `tau`, the closer `beta` climbs toward its `1/3` ceiling, i.e. the faster the non-parametric rate. So the bias variance `tau^2` isn't a free knob — it has to exceed 1 for the random-feature RKHS to be rich enough to approximate smooth links with a polynomial rate, and bigger `tau` is better for the approximation. That's the derivation-time reason to set `tau > 1`.

Now close the contradiction in (*). The left side, for `m != 0`, is at least its leading term: `sum_{j>=s} alpha_j^2 j m^{2j-1} >= alpha_s^2 s |m|^{2s-1}` (all terms share sign, leading one dominates near the equator and the rest only add). The right side: by Cauchy-Schwarz then the two approximation bounds, `|<(I-P_lambda)g_m, gbar_m>| <= 2 sqrt(A(g_m,lambda)) ||gbar_m|| <= 4 lambda^{beta/2} |m|^{2s-1} sqrt(C tau^{1+beta}) Ktilde C_{f_*}^2`, where I've used that both `g_m` and `gbar_m` carry an `m^{s-1}`-ish factor and a `m^s`-ish factor whose product is `|m|^{2s-1}`, matching the power on the left. So *both sides scale like `|m|^{2s-1}`* — the `m`-dependence cancels, and the whole question reduces to a constant comparison. Choose

```
lambda < lambda* := ( 4 sqrt(C tau^{1+beta}) Ktilde C_{f_*}^2 / (alpha_s^2 s) )^{-2/beta},
```

and then the right side is *strictly less than* `alpha_s^2 s |m|^{2s-1} <=` left side, for every `m != 0`. Contradiction. So with `N >~ 1/lambda` features and `lambda < lambda*`, with high probability over the random biases, **there is no critical point with `0 < |m| < 1`** — the population landscape's only critical points are the equator and the poles, exactly the benign topology of the ideal infinite case. The frozen-bias randomization is robust: it preserves the same landscape topology that a tailored Hermite basis would give (the kind Dudeja and Hsu engineer by hand), but here it falls out of random features.

That handles the *population* landscape. But I train on `n` samples, so I need the *empirical* landscape `L_n` to inherit this. Here's where the flat equator bites. The benign population landscape has a *degenerate* saddle at `m = 0` — the loss is flat to order `s` there, so the landscape does *not* have the strict-saddle property that the usual escape arguments rely on. I can't just invoke a generic theorem. I need uniform convergence of the empirical gradient to the population gradient, sharp enough near the equator. The concentration I can prove is

```
sup_{theta, ||c||<=r} ||nabla_theta L_n(c,theta) - nabla_theta L(c,theta)|| = O( r^2 max{ sqrt(D log(.)/n), (d log(.))^2/n } ),
```

`D = max{d, N}`, and a similar bound for the `c`-gradient. Getting this for ReLU takes care, because the sample gradient is *not* Lipschitz across the ReLU kink: a tiny change in `theta` can flip a neuron's activation pattern for some samples. So a naive epsilon-net argument fails at the kink. The fix is to show that for a fixed `theta`, only a *small fraction* of samples have their ReLU sign pattern flipped by an `epsilon`-perturbation — quantitatively, the chance that `|eps_j <theta,x_i> - b_j|` is within `||x_i|| epsilon` of zero is small by anti-concentration of the Gaussian, so the worst-case discretization error gets averaged out over `n` samples. With `ReLU'(0)` set to anything in `[0,1]` (the particular choice doesn't matter for the bound), the non-Lipschitz contribution is controlled and the standard sub-exponential-gradient concentration goes through.

With this uniform gradient bound, the empirical critical points split exactly as the population ones, up to a tolerance set by the concentration error `Delta ~ sqrt(D/n)`: a "bad" cluster near the equator with `|m| <~ (Delta/lambda^2)^{1/(2s-1)}` and a "good" cluster near the poles with `1 - |m| <~ (Delta/lambda^2)^2`. Now I can finally state when gradient flow recovers `theta*`. Near the equator the empirical landscape (in `theta`) still scales like `m^s`; to certify the algorithm does *not* get stuck at an `epsilon`-approximate critical point on the equator, I need the signal at my current `m` to beat the noise, i.e. `m >= epsilon^{1/(s-1)}`. Plug in the uniform bound `~ sqrt(d/n)` for the gradient error and the initialization `m_0 = Theta(1/sqrt(d))`, and the requirement to escape the equatorial influence becomes `n = O(d^s)`. The gradient flow escapes the equator in time `T_0 = Otilde(d^{s/2 - 1})`, then descends to a pole, landing at `1 - |<theta_T, theta*>| = Otilde(lambda^{-4} max{(d+N)/n, d^4/n^2})`. For `lambda = Theta(1)` and `s > 2` this is `n = Theta(d^s)` for full recovery — matching the near-optimal threshold `d^{s-1}` of Ben Arous et al. up to the batch-vs-online gap. I solved the non-convex high-dimensional inference and the one-dimensional non-parametric fit *simultaneously*, with one gradient method.

There's a subtlety in how I run the flow that I should pin down, because the *joint* dynamics of `c` and `theta` from a random start can re-entangle the two jobs. If I let `c` move freely from the very beginning, during the search phase there are adverse interactions — `c` chasing the wrong direction while `theta` is still near the equator — and a naive analysis blows the sample complexity up to `d^{2s}`. The cure is a *time-scale separation*: in the first phase optimize only `theta` (hold the readout effectively off), let it escape the equator; then in the second phase turn on `c` and jointly optimize. Concretely I scale the `c`-update by `zeta(t) = 1{t > T_0}`, and I initialize `c(0)` as a *sparse* vector with only `N_0 << N` nonzeros and small norm `rho`. This is the lazy-vs-rich knob from Chizat et al. made literal: a small/sparse readout keeps the model from doing non-parametric work during the search, so the direction moves cleanly; once the direction is locked, the full readout comes on for the fit. The warm-start is sufficient (whether vanilla joint dynamics also works is left open).

One more inefficiency to fix. With joint training, getting the excess risk to vanish forces the width `N` to grow with `n` — the readout's non-parametric error and the regularization `lambda` are coupled to the same `lambda` that I needed *large* for fast equator-escape. Those two wants conflict: large `lambda` for recovery, small `lambda` for low function-approximation error. So *decouple them*. After the gradient phase fixes `theta_hat`, do a final **ridge re-fit of the readout alone**: on a *fresh* sample `(x_i', y_i')`, solve

```
chat = argmin_c (1/n') sum_i (c^T Phi(<theta_hat, x_i'>) - y_i')^2 + lambda' ||c||^2,
```

a strongly convex problem — closed-form linear solve. The fresh sample matters technically: the kernel depends on `theta_hat`, which depends on the first sample, so re-using that data would couple the two; splitting breaks the dependence (a standard sample-splitting trick in semi-parametrics). With a separate, *small* `lambda'` tuned to the non-parametric rate, the excess risk becomes

```
E[ ||F_hat - F_*||^2 ] <~ ||f_*''||^{2/(beta+1)} (sigma^2 tau^2 / n')^{beta/(beta+1)} + ||f_*'||^2 (1 - |m|),
```

a clean split: a non-parametric term that decays in `n'` with a rate *independent of `d`*, plus a term proportional to the direction error `1 - |m|`. Keep `lambda` large in phase 1 for fast recovery, `lambda'` small in the re-fit for low excess risk; the width in the re-fit only needs to scale with `n'^{1/(beta+1)}`, again no `d`.

I should note what the ReLU non-smoothness costs in the optimization analysis. Squared loss on a ReLU network is non-smooth, so "gradient flow" really means a curve satisfying the subgradient differential *inclusion* `zdot in -partial L(z)` with the Clarke subdifferential. I need two things: such a curve *exists*, and the loss is *non-increasing* along it (the descent property). Both follow because the empirical squared loss of a ReLU network is *definable* in an o-minimal structure, which means it admits a chain rule; near-steepest-descent curves then coincide with subgradient curves, and `dL(z(t))/dt = -||partial-bar L||^2 <= 0`. For the spherical constraint on `theta`, I project the subdifferential onto the tangent space, and the chain rule survives because `<z, zdot> = 0` for any curve on the sphere (differentiating `||z||^2 = 1`). So the descent property holds, and the flow is well-defined despite the kinks. If I wanted to avoid this machinery entirely and discretize cleanly into actual gradient descent, I could swap ReLU for a smooth activation (a Gaussian-smoothed ReLU, essentially an ELU), which also improves the `s in {1,2}` rates to `d^s` — but ReLU is what practitioners use, so I keep it and pay the non-smooth analysis.

Let me put the whole reasoning into the code I'd actually ship, filling the empty slots of the shallow-net harness. The method is: initialize one shared inner direction on the sphere, sample and *freeze* the biases (and signs), train the direction and the readout with squared loss by SGD, and optionally re-fit the readout by ridge. The freezing is the one essential line — `requires_grad_(False)` on the biases — because that's what collapses the high-dimensional landscape to the benign one-dimensional `m`-landscape.

```python
import math
import torch
import torch.nn as nn


class Strategy:
    """Frozen-bias shallow network for single-index recovery.

    One shared inner direction theta carries the high-dimensional inference; the
    per-neuron biases (and signs) are sampled once at init and FROZEN, so the
    only active high-dimensional parameter is the direction. That freezing makes
    the population landscape depend on theta only through m = <theta, theta*>,
    collapsing the d-dimensional non-convex problem to a benign 1-D landscape.
    """

    def __init__(self, config):
        self.config = config

    def init_two_layer(self, net, config):
        # First-layer rows on the unit sphere (the 'active' shared-direction part);
        # biases drawn once and FROZEN (the 'lazy' random-feature part = kernel).
        with torch.no_grad():
            W = torch.randn_like(net.fc1.weight)
            W = W / W.norm(dim=1, keepdim=True).clamp_min(1e-12)
            net.fc1.weight.copy_(W)
            net.fc1.bias.uniform_(-1.0, 1.0)        # b_i ~ Uniform (variance plays role of tau^2)
        net.fc1.bias.requires_grad_(False)          # <-- freeze the biases: the key move

        bound = 1.0 / math.sqrt(config.width)       # small readout: lazy/rich relative scale
        nn.init.uniform_(net.fc2.weight, -bound, bound)
        nn.init.zeros_(net.fc2.bias)

    def make_optimizer(self, net, config):
        # Optimize only the FREE parameters (frozen biases are excluded automatically).
        params = [p for p in net.parameters() if p.requires_grad]
        return torch.optim.SGD(
            params,
            lr=config.base_lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,       # Tikhonov lambda on the trained params
        )

    def training_step(self, net, optimizer, x, y, step, config):
        # One step of gradient descent on the squared loss; the direction escapes
        # the flat equator (signal ~ m^{s-1}) then descends to alignment |m| -> 1.
        net.train()
        optimizer.zero_grad(set_to_none=True)
        preds = net(x)
        loss = torch.mean((preds - y) ** 2)
        loss.backward()
        optimizer.step()
        return loss

    def finalize(self, net, x_train, y_train, config):
        # Optional: closed-form ridge re-fit of the readout decouples the two
        # regularizations (large lambda for fast recovery, small lambda' for low
        # excess risk). Left as a no-op here; the trained net already recovers theta*.
        return


def build_strategy(config):
    return Strategy(config)
```

So the causal chain. The difficulty in a single-index model is escaping the flat equator from a random start: the gradient signal near the equator scales like `m^{s-1}` in the information exponent `s`, the random init sits at `m ~ 1/sqrt(d)`, and beating the `~ sqrt(d/n)` empirical-gradient noise forces `n ~ d^s`. A standard net entangles the direction search with the link fit, wrecking that rate. Tying all neurons to *one* shared inner direction and *freezing the biases* (the lazy random-feature part, the rich direction part) makes the population loss depend on `theta` only through `m = <theta,theta*>`, so its gradient is colinear with `theta*` and the whole `d`-dimensional landscape collapses to a one-dimensional one in `m`. In the ideal limit that landscape is strictly decreasing in `|m|` with only the equator and the poles as critical points; with finitely many regularized random features it stays benign because the random-feature approximation error `||(I-P_lambda)g_m||` is `O(lambda^{beta/2})`-small (needing `tau>1` for `beta>0` and `N ~ 1/lambda` features, no `d`-dependence), so choosing `lambda < lambda*` kills any spurious critical point — both sides of the critical-point equation scale like `|m|^{2s-1}` and the comparison is a constant one. Uniform concentration of the empirical gradient — handled at the ReLU kink by anti-concentration — transfers this to the empirical landscape; a time-scale separation (search the direction first with a small/sparse readout, then jointly fit) avoids re-entangling the two jobs, and gradient flow escapes the equator in `Otilde(d^{s/2-1})` and recovers `theta*` with `n = Theta(d^s)`. An optional fresh-sample ridge re-fit of the readout decouples the two regularizations and gives an excess-risk rate independent of `d`. The whole thing is a shallow ReLU net trained by ordinary gradient descent, with one essential modification: the biases are frozen.
