Let me start from what actually hurts. I have observational data `(X, A, Y)` — covariates, a binary treatment, an outcome — and I want the conditional average treatment effect `tau(x) = E(Y^1 - Y^0 | X = x)`, the whole heterogeneous function of who benefits and by how much. Under no-unmeasured-confounding, consistency, and overlap, this counterfactual contrast is identified as a difference of observed regressions, `tau(x) = mu_1(x) - mu_0(x)` with `mu_a(x) = E(Y | X=x, A=a)`. So the obvious thing — the plug-in, the T-learner — is to fit `mu_1` on the treated, `mu_0` on the controls, and subtract. And I keep getting burned by it, so let me actually look at *why* before I reach for anything fancier.

Picture the meanest version of the problem. Treatment is heavily confounded: say `pi(x) = P(A=1|X=x)` jumps with the sign of `x`, mostly untreated on the left, mostly treated on the right. The two response surfaces are some awkward non-smooth piecewise polynomial, and — here's the trap — they are *equal*, `mu_1 = mu_0`, so the true CATE is the constant zero, the simplest function imaginable. Now fit each arm separately. On the left there are barely any treated units, so `mu_1_hat` is starved of data and oversmooths; on the right there are barely any controls, so `mu_0_hat` undersmooths. Subtract them and I get a wandering spurious bump — large error, for a target that is identically zero. The plug-in inherited the difficulty of estimating two hard surfaces, when the thing I wanted was trivial. That's the whole disease in one picture: `tau` can be far simpler than `mu_0` and `mu_1`, and differencing the surfaces is structurally blind to that. Whatever I build has to be able to converge at the complexity of `tau`, not the complexity of the nuisances.

The clean way to think about "converge at the complexity of `tau`" is to imagine an oracle. Suppose I could actually *see* the individual contrast `Y^1 - Y^0` for each unit. Then I'd just regress that on `X` with a good nonparametric method and get an estimate whose rate is set by the smoothness or sparsity of `tau` alone — if `tau` is `gamma`-smooth in `d` dimensions, the pointwise minimax rate `n^{-1/(2 + d/gamma)}`; if it's constant, essentially a parametric rate. That oracle is my benchmark. The plug-in doesn't come close to it. So the real question is: can I manufacture a per-unit quantity, observable from `(X, A, Y)`, that behaves *like* the oracle's `Y^1 - Y^0` — same conditional mean `tau(x)`, so that regressing it on `X` recovers `tau` at `tau`'s own complexity?

What observable has conditional mean `tau(x)`? Here's one I know from weighting. Take `(A - pi(X)) Y / {pi(X)(1 - pi(X))}`. Let me check its conditional mean given `X`. Condition on `X=x`; `A` is Bernoulli(`pi`). When `A=1`, the weight is `(1-pi)/(pi(1-pi)) = 1/pi` and the outcome mean is `mu_1`; that branch contributes `pi · (1/pi) · mu_1 = mu_1`. When `A=0`, the weight is `(0-pi)/(pi(1-pi)) = -1/(1-pi)` and the outcome mean is `mu_0`; that branch contributes `(1-pi) · (-1/(1-pi)) · mu_0 = -mu_0`. Sum: `mu_1 - mu_0 = tau(x)`. So the inverse-probability-weighted transform has exactly the right conditional mean. If I knew `pi`, I could regress it on `X` and behave like the oracle, adapting to `tau`'s smoothness. And in that same nasty example, this *works* where the plug-in fails — it sidesteps estimating the surfaces.

But two things hurt. First, it's singly robust: it's only unbiased if `pi` is exactly right, and I have to estimate `pi` from data, with error. Second, and worse in practice, the variance. I'm dividing by `pi(1-pi)`, which blows up wherever the propensity gets near 0 or 1 — precisely the regions of weak overlap, which are unavoidable in high-dimensional confounding. The IPW transform throws away the outcome models entirely and pays full weighting variance for it. So I've traded the plug-in's bias problem for a variance problem and a fragility-to-`pi` problem. There has to be something that uses *both* the outcome models and the propensity, so that each can cover for the other.

Let me think about the scalar version first, because the average treatment effect — `psi = E{tau(X)} = E(Y^1) - E(Y^0)` — has a beautiful, fully worked-out theory I can lean on, and maybe I can lift its structure up to the function. Focus on one arm, `E(Y^1) = E{mu_1(X)}`. The semiparametric-efficiency story says this functional has an efficient influence function, and the influence function is exactly the recipe for an estimator that is first-order insensitive to errors in the nuisances. For the treated mean it is

  `phi(Z) = 1(A=1)/pi(X) · {Y - mu_1(X)} + mu_1(X) - psi`.

Read it: a regression-imputation piece `mu_1(X)`, augmented by an inverse-probability-weighted *residual* `1(A=1)/pi · (Y - mu_1)` that corrects the imputation only on the treated units. This is the augmented-IPW score. Why do I trust it? Because of the von Mises expansion that governs how the functional changes when the distribution it's evaluated at moves:

  `psi(Pbar) - psi(P) = ∫ phi(z; Pbar) d(Pbar - P)(z) + R_2(Pbar, P)`,

and the whole point is that `R_2` is a *second-order* remainder — it depends only on products or squares of the nuisance discrepancies. Let me actually see what `R_2` is for this `phi`, because the form of the remainder is the thing I care about. Set `Pbar` to be the law with estimated nuisances `(pibar, mubar)` and `P` the truth. Plug the estimated `phi` in and take its expectation under the truth; the part that doesn't cancel works out to

  `R_2 = ∫ {1/pibar(x) - 1/pi(x)} {mu_1(x) - mubar_1(x)} pi(x) dx`.

Stare at that. It's an integral of a *product*: `(1/pibar - 1/pi)`, which is an error in the propensity, times `(mu_1 - mubar_1)`, which is an error in the outcome model. The bias of the AIPW estimator of the ATE is a product of the two nuisance errors. That is the source of everything I want. Each factor can be individually large — a slow nonparametric rate, say `n^{-1/3}` each — and yet the product is `n^{-2/3}`, smaller order. And it's doubly robust on its face: if `pibar = pi` the first factor is zero, if `mubar_1 = mu_1` the second factor is zero — get *either* nuisance right and the bias vanishes. The averaging of `phi` is what gives an efficient *scalar* ATE. So `phi` is the magic per-unit object — for the average.

To estimate the *average* effect efficiently I *average* `phi`; conditioning suggests the parallel object should have `tau(x)` as its conditional mean, so the scalar average should turn into a regression on `X`. The conditional version of the same uncentered influence function (dropping the `-psi` and using both arms) is

  `phi(Z) = (A - pi(X)) / {pi(X)(1 - pi(X))} · {Y - mu_A(X)} + mu_1(X) - mu_0(X)`,

where `mu_A` means `mu_1` if `A=1` and `mu_0` if `A=0`. Let me unpack this to convince myself it's the natural two-arm AIPW pseudo-outcome. Expand the first term by cases. When `A=1`: `(1-pi)/(pi(1-pi)) = 1/pi`, so I get `(Y - mu_1)/pi`. When `A=0`: `(0-pi)/(pi(1-pi)) = -1/(1-pi)`, giving `-(Y - mu_0)/(1-pi)`. So equivalently

  `phi = mu_1 - mu_0 + A·(Y - mu_1)/pi - (1-A)·(Y - mu_0)/(1-pi)`:

a regression-difference `mu_1 - mu_0` plus an IPW correction applied to each arm's *residual*. Good — that's exactly the augmentation pattern, one piece per arm.

Before I get excited, the oracle check: does `E(phi | X=x) = tau(x)` when the nuisances are the truth? Condition on `X=x`. The `mu_1 - mu_0` term passes through. For the correction, when `A=1` (probability `pi`) the contribution is `pi · E[(Y-mu_1)/pi | X, A=1] = pi · (mu_1 - mu_1)/pi = 0`; when `A=0` similarly `0`. So `E(phi|X) = mu_1 - mu_0 = tau`. The true pseudo-outcome regresses to `tau`. So the oracle who could compute `phi` with true nuisances would regress it on `X` and get `tau` at `tau`'s own rate, just like the `Y^1 - Y^0` oracle, but now with the efficiency of the influence function baked in.

But I don't have true nuisances; I estimate `pihat, mu0hat, mu1hat`, form `phihat`, and regress *that*. So the real question — the one that decides whether this beats IPW and the plug-in — is what error I incur by using `phihat` instead of `phi`. The relevant quantity is the conditional bias of the estimated pseudo-outcome,

  `bhat(x) = E(phihat - phi | nuisances, X=x)`,

because regressing `phihat` is, conditional on the nuisance fits, regressing something whose mean is `tau + bhat`. Let me grind it out. I need `E(phihat | X=x)` with `pihat, muahat` held fixed (they were fit on other data) but `A` and `Y` random with their *true* law `(pi, mu_a)`. The difference term, when `A=1`: probability `pi`, weight `1/pihat`, residual `Y - mu1hat` with `E[Y|A=1]=mu_1`, contributing `pi · (mu_1 - mu1hat)/pihat`. When `A=0`: probability `1-pi`, weight `-1/(1-pihat)`, residual mean `mu_0 - mu0hat`, contributing `-(1-pi)(mu_0 - mu0hat)/(1-pihat)`. Add the `mu1hat - mu0hat` from the regression piece, subtract the truth `mu_1 - mu_0`:

  `E(phihat|X) - tau = pi(mu_1 - mu1hat)/pihat - (1-pi)(mu_0 - mu0hat)/(1-pihat) + (mu1hat - mu_1) - (mu0hat - mu_0)`.

Collect the arm-1 pieces: `pi(mu_1 - mu1hat)/pihat + (mu1hat - mu_1) = (mu_1 - mu1hat)(pi/pihat - 1) = (mu_1 - mu1hat)(pi - pihat)/pihat = (pihat - pi)(mu1hat - mu_1)/pihat`. And the arm-0 pieces: `-(1-pi)(mu_0 - mu0hat)/(1-pihat) - (mu0hat - mu_0) = (mu_0 - mu0hat){1 - (1-pi)/(1-pihat)} = (mu_0 - mu0hat)(pi - pihat)/(1-pihat) = (pihat - pi)(mu0hat - mu_0)/(1-pihat)`. So

  `bhat(x) = (pihat - pi)/pihat · (mu1hat - mu_1) + (pihat - pi)/(1-pihat) · (mu0hat - mu_0)`,

which I can write uniformly as

  `bhat(x) = sum_{a=0}^{1} { (pihat(x) - pi(x)) / [a·pihat(x) + (1-a)(1-pihat(x))] } · (muahat(x) - mu_a(x))`.

There it is, and it's the function-valued echo of the scalar `R_2`: the bias is a *product* of the propensity error `(pihat - pi)` and the outcome error `(muahat - mu_a)`, summed over arms, with the propensity entering only through a bounded reweighting `1/pihat` or `1/(1-pihat)`. And the double robustness is now manifest: if `pihat = pi`, every term carries `(pihat - pi) = 0`, so `bhat ≡ 0` regardless of how bad the outcome models are; if `muahat = mu_a` for both arms, every term carries `(muahat - mu_a) = 0`, so `bhat ≡ 0` regardless of `pihat`. Get *either* set of nuisances right and the pseudo-outcome is conditionally unbiased for `tau`. The outcome model and the propensity each insure the other. That's the property IPW and the plug-in each lacked, and it's why this should be far more stable than either.

So the picture is forming: build the AIPW pseudo-outcome from estimated nuisances and regress it on `X`. The second-stage error relative to the oracle is controlled by `bhat`, a product of nuisance errors, so it can be of *smaller order* than the nuisance errors themselves — meaning the final CATE estimate can converge at the oracle rate (the complexity of `tau`) even when `pi` and `mu_a` are estimated slowly. That's exactly the "converge at the complexity of `tau`" property I wanted and the plug-in couldn't give me.

But I'm being too quick. I derived `bhat` by treating the nuisance estimates as *fixed* — as if they were trained on data independent of the units I'm now plugging in. If I fit `pihat, muahat` on the same data I then evaluate `phihat` on and regress, that independence is a lie. Wall. The residual `Y - mu1hat` is *correlated* with `mu1hat` because `Y` helped fit `mu1hat`; the unit overfit its own model, so `E[(Y - mu1hat) | A=1]` is not `mu_1 - mu1hat` — there's an extra in-sample piece. More structurally, the error of `phihat - phi` then carries an empirical-process term `(P_n - P)(phihat - phi)` that does not vanish unless I impose strong complexity (Donsker) restrictions on the estimators — and modern flexible learners (boosting, forests, lasso in high dimensions) are exactly the things that *violate* such restrictions. So the clean product-bias `bhat` is not the whole story when I reuse data; there's a contamination term that can dominate and wreck everything.

I've seen this exact failure analyzed for the scalar case. A naively reused plug-in into an estimating equation picks up a term like `(1/sqrt(n)) sum_i m_0(X_i)(g_0 - ghat_0)(X_i)` whose summands don't have mean zero; with a nuisance rate `n^{-phi}`, `phi < 1/2`, it's of order `sqrt(n)·n^{-phi} -> infinity`. The cure that's known to work is *sample splitting*, refined into *cross-fitting*: estimate the nuisances on one part of the data, evaluate and regress the pseudo-outcome on a disjoint part. Then, conditional on the training part, the nuisance functions are genuinely fixed and independent of the evaluation units, so `phihat(Z_i)` really does have conditional mean `tau(X_i) + bhat(X_i)`, and the contamination term has conditional mean zero and small variance — no Donsker condition needed, because the dangerous coupling between the estimator and its own evaluation points has been severed. The cost is that each half of the data does less work; the standard fix is to swap the roles and average, recovering full-sample efficiency. With `K` folds: train nuisances out-of-fold, predict in-fold, so every unit's pseudo-outcome uses nuisances that never saw it. So sample splitting isn't a technicality I can skip — it's what makes my clean `bhat` derivation *true* rather than aspirational.

"Regress the pseudo-outcome" still hides a question: which second-stage regressors actually inherit the product-bias guarantee? I want a statement of the form: the fit using `phihat` equals the fit using the true `phi`, up to the smoothed bias plus something negligible relative to the oracle's own error. Let me define the oracle second-stage estimator `tau_tilde(x) = Ehat_n{phi(Z) | X=x}` — the same regression method, fed the *true* pseudo-outcome — and let its risk be `R_n*(x)^2 = E[{tau_tilde(x) - tau(x)}^2]`. I want to show `tau_hat - tau_tilde = (smoothed bhat) + o_P(R_n*)`.

Take the cleanest large class of second-stage regressors: linear smoothers, `Ehat_n{f(Z)|X=x} = sum_i w_i(x; X^n) f(Z_i)` — local polynomials, series, kernels, smoothing splines, k-NN, kernel ridge, even forests viewed as adaptive smoothers. For these I can compute the perturbation directly. The difference between feeding `phihat` and feeding `phi`, after subtracting the part I expect (the smoothed bias), is

  `T_n = sum_i w_i(x) { phihat(Z_i) - phi(Z_i) - bhat(X_i) }`.

Conditional on the training data and all covariates, each bracket has mean zero — that's the *definition* of `bhat` as the conditional bias, `E{phihat - phi | training, X_i} = bhat(X_i)` — so `E(T_n | training, X^n) = 0`. Now its conditional variance: the bracketed terms are independent across `i` given the training data (different evaluation units), so

  `E(T_n^2 | training, X^n) = sum_i w_i(x)^2 var{phihat(Z_i) - phi(Z_i) | training, X_i} <= ||phihat - phi||_{w^2}^2 · sum_i w_i(x)^2`,

where `||·||_{w^2}` is the natural weighted conditional `L_2` norm built from the squared smoother weights, and I used `var <= E[(·)^2]`. Meanwhile the oracle's own risk satisfies `R_n*(x)^2 >= E{||sigma||_{w^2}^2 sum_i w_i(x)^2}` for `sigma(x)^2 = var{phi(Z)|X=x}` — because the oracle's variance is `sum_i w_i^2 sigma(X_i)^2` plus a nonnegative squared-bias term. Divide: by Markov, for any `t`,

  `P( ||sigma||_{w^2} |T_n| / (||phihat - phi||_{w^2} R_n*) >= t ) <= (1/t^2 R_n*^2) E{ ||sigma||_{w^2}^2 sum_i w_i^2 } <= 1/t^2`,

so `T_n = O_P( ||phihat - phi||_{w^2} / ||sigma||_{w^2} · R_n* )`. As long as the pseudo-outcome is *consistent* in this norm, `||phihat - phi||_{w^2} -> 0` (no rate needed!), and `1/||sigma||_{w^2}` stays bounded, `T_n = o_P(R_n*)`. This is the stability property: replacing the true pseudo-outcome by the estimated one perturbs the second-stage fit only by the smoothed bias, up to `o_P(R_n*)`. I only needed consistency of the nuisances — not a fast rate — for the *structure* of the bound; the *rate* will come entirely from `bhat`. So the recipe is general: any stable second-stage smoother gives `tau_hat - tau_tilde = Ehat_n{bhat | X=x} + o_P(R_n*)`, and the estimator is oracle-efficient — asymptotically equivalent to regressing the true `Y^1-Y^0`-style pseudo-outcome — exactly when the *smoothed* bias is `o_P(R_n*)`.

Now I should get concrete about that smoothed bias, because `bhat` is a product `bhat = bhat_1 · bhat_2` (per arm: a propensity error times an outcome error). A linear smoother of a product is bounded by a Hölder split: with `1/p + 1/q = 1`,

  `|Ehat_n{bhat_1 bhat_2 | X=x}| = |sum_i w_i(x) bhat_1(X_i) bhat_2(X_i)| <= (sum_i |w_i|) · ||bhat_1||_{w,p} · ||bhat_2||_{w,q}`,

by Hölder's inequality on the weighted measure. For most decent smoothers `sum_i |w_i| <= C`, so the smoothed bias is at most a constant times a product of weighted norms of the two nuisance errors. Take `p = q = 2`: the smoothed bias is of the order of the product of the (local, weighted) `L_2` errors of the propensity and outcome estimators. Concretely, if `pi` is `alpha`-smooth and estimated at the minimax `n^{-1/(2+d/alpha)}`, and `mu_a` is `beta`-smooth at `n^{-1/(2+d/beta)}`, the smoothed bias is `O_P( n^{-(1/(2+d/alpha) + 1/(2+d/beta))} )`, and combining with the oracle's own `n^{-1/(2+d/gamma)}`,

  `tau_hat(x) - tau(x) = O_P( n^{-1/(2+d/gamma)} + n^{-(1/(2+d/alpha) + 1/(2+d/beta))} )`.

The first term is the oracle rate — set by `tau`'s smoothness alone, exactly the "converge at the complexity of `tau`" I demanded. The second is the product penalty, and oracle efficiency holds when it's the smaller of the two. Let me see when that is: I need `1/(2+d/alpha) + 1/(2+d/beta) >= 1/(2+d/gamma)`. Cross-multiplying and simplifying (write it out: `4 + d/alpha + d/beta >= (4 + 2d/alpha + 2d/beta + d^2/(alpha beta))/(2 + d/gamma)`, then clear the denominator and cancel), the condition collapses to

  `alpha·beta >= (d^2/4) / [ 1 + (d/gamma)(1 + d/(2·sbar)) ]`, equivalently `sqrt(alpha·beta) >= (d/2) / sqrt(1 + (d/gamma)(1 + d/(2·sbar)))`,

with `sbar = 2/(1/alpha + 1/beta)`, the harmonic mean of `alpha, beta`. This is illuminating against the classical ATE benchmark, where root-`n` estimation needs `sqrt(alpha·beta) >= d/2`. Two readings drop out. As `gamma -> infinity` (an arbitrarily smooth CATE), the denominator `-> 1` and my condition `-> sqrt(alpha·beta) >= d/2`, the ATE condition — an infinitely smooth effect should be nearly as easy as the average, and it is. And for finite `gamma`, the denominator exceeds 1, so the bar is *lowered*: because the oracle CATE rate is slower than root-`n`, I have more slack — the nuisances can be rougher than the ATE would tolerate and I still hit the oracle. The plug-in could never say anything like this; its error was just the nuisance error, no product, no lowered bar.

Let me sanity-check the construction against the alternatives one more time, since the test of a good idea is that the others fall out as deficient special cases. Drop the augmentation correction entirely and keep only `mu1hat - mu0hat`: that's the T-learner, and its bias is first-order in the outcome error with no product structure — no double robustness. Drop the outcome regression and keep only the weighted residual with `muahat ≡ 0`: the pseudo-outcome becomes `(A-pi)Y/(pi(1-pi))`, the IPW transform — singly robust, and high variance because nothing cancels the `1/(pi(1-pi))` blow-up; here the `mu_a` augmentation is exactly what reduces that variance by subtracting off the predictable part of `Y` before weighting the residual. And the residual-on-residual route — regress `(Y - m)` on `(A - pi)` à la Robinson — is orthogonal too, but its oracle guarantees demand both nuisances at `n^{-1/4}`; my product-bias analysis shows the doubly-robust pseudo-outcome reaches the oracle under the *weaker*, lowered bar above, and it's a plain mean-squared-error regression at the end rather than a weighted least-squares with an `(A-pi)` design that itself degrades near no-overlap. Each prior method is this construction with a piece removed or a robustness sacrificed.

A practical worry stops me before I write code: the `1/pihat` and `1/(1-pihat)` in the correction. In finite samples, and especially in high-dimensional confounding, `pihat` will sometimes land very near 0 or 1 — a near-positivity-violation — and a single unit can then get a gigantic weight, injecting enormous variance into the pseudo-outcome and letting one point dominate the second-stage fit. The theory assumes overlap `epsilon <= pi <= 1-epsilon`; the honest finite-sample enforcement is to *clip* the estimated propensity into `[epsilon, 1-epsilon]` for a small `epsilon` (say 0.05). That trades a controlled sliver of bias (where the true `pi` is genuinely beyond the clip) for a bounded inverse weight — the right trade, because an unclipped weight makes the variance unbounded while the clip's bias is local to the extreme-overlap region. A second, related guard: even with clipped propensities, heavy-tailed outcome residuals can produce a few extreme pseudo-outcomes that distort the regression; winsorizing `phihat` at a high quantile of its absolute value (say the 97th percentile) caps those without touching the bulk. Neither guard is a theorem requirement — they're variance-control on the empirical object — but skipping them is how this estimator gets a reputation for instability under weak overlap, so I'll keep both.

Now the choice of learners. The whole analysis was deliberately *agnostic* about methods — stability holds for any reasonable second-stage smoother, and the product-bias holds for any first-stage nuisance estimators — so I should pick flexible, off-the-shelf learners that handle nonlinear confounding and high-dimensional `X`. Gradient-boosted trees are a natural default for all three roles: a classifier with `predict_proba` for the propensity `pi`, regressors for the two outcome surfaces `mu_0, mu_1`, and a regressor for the final stage. Boosting captures interactions and nonlinearities without me specifying a form, and it's exactly the kind of regularized learner whose slow, biased-but-low-variance behavior the cross-fitting was built to accommodate. I'll give the outcome and propensity models enough depth and trees to be flexible (depth ~3–4, a couple hundred trees, a moderate learning rate, subsampling for variance), and the final-stage regressor a *shallower*, more strongly regularized setting (smaller depth, smaller learning rate) — because the second stage is estimating `tau`, which I'm betting is *simpler* than the nuisances, so I don't want it to chase noise in the pseudo-outcomes; a smoother final fit is the whole point of regressing a pseudo-outcome instead of differencing surfaces.

Let me assemble the procedure. Cross-fit the nuisances with `K` folds (5 is the standard choice — enough to keep each training fold large while making the out-of-fold predictions honest). For each fold: train `mu_0` on the control units of the training portion, `mu_1` on the treated units, and `pi` on all training units; then predict all three on the held-out fold, so each unit's `(pi_hat, mu0_hat, mu1_hat)` come from models that never saw it. After all folds, every unit has out-of-fold nuisance predictions. Clip the propensities. Form the AIPW pseudo-outcome per unit. Winsorize it. Then fit the final regressor mapping `X -> phihat` once, on all units (each unit's target already uses out-of-fold nuisances, so the cross-fitting discipline is satisfied; the final regressor learns the smooth `tau`). To predict on new `x`, just evaluate the final regressor.

```python
import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier


class CATEEstimator:
    """DR-Learner: build the doubly-robust (AIPW) pseudo-outcome from cross-fitted
    nuisances, then regress it on X. Consistent if either the outcome models or the
    propensity model is correct; converges at the complexity of tau, not of mu_a."""

    def __init__(self):
        self._seed = int(os.environ.get("SEED", "42"))

    # outcome models mu_a(x) = E[Y | X, A=a]: flexible, allowed to be slow/biased
    def _make_model_y(self):
        return GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            min_samples_leaf=20, subsample=0.8, random_state=self._seed,
        )

    # propensity model e(x) = P(A=1 | X): a classifier, predict_proba for pi_hat
    def _make_model_t(self):
        return GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            min_samples_leaf=20, subsample=0.8, random_state=self._seed + 1,
        )

    # final stage regresses X -> pseudo-outcome; shallower/slower since tau is simpler
    def _make_cate_model(self):
        return GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            min_samples_leaf=20, subsample=0.8, random_state=self._seed + 2,
        )

    def fit(self, X, T, Y):
        n = len(Y)

        # cross-fitting: nuisances trained out-of-fold, predicted in-fold, so each
        # unit's pseudo-outcome uses nuisances independent of that unit (severs the
        # own-observation coupling that would otherwise add a non-vanishing remainder)
        kf = KFold(n_splits=5, shuffle=True, random_state=self._seed)
        mu0_hat = np.zeros(n)
        mu1_hat = np.zeros(n)
        e_hat = np.zeros(n)

        for train_idx, val_idx in kf.split(X):
            mask0_train = T[train_idx] == 0          # controls in this training fold
            mask1_train = T[train_idx] == 1          # treated in this training fold

            m0 = self._make_model_y()
            m1 = self._make_model_y()

            # mu_0 fit on the control arm of the training fold, predicted out-of-fold
            if mask0_train.sum() > 5:
                m0.fit(X[train_idx[mask0_train]], Y[train_idx[mask0_train]])
                mu0_hat[val_idx] = m0.predict(X[val_idx])
            else:
                mu0_hat[val_idx] = Y[T == 0].mean() if (T == 0).sum() > 0 else Y.mean()

            # mu_1 fit on the treated arm of the training fold, predicted out-of-fold
            if mask1_train.sum() > 5:
                m1.fit(X[train_idx[mask1_train]], Y[train_idx[mask1_train]])
                mu1_hat[val_idx] = m1.predict(X[val_idx])
            else:
                mu1_hat[val_idx] = Y[T == 1].mean() if (T == 1).sum() > 0 else Y.mean()

            # propensity fit on the full training fold, predicted out-of-fold
            mt = self._make_model_t()
            mt.fit(X[train_idx], T[train_idx])
            e_hat[val_idx] = mt.predict_proba(X[val_idx])[:, 1]

        # enforce overlap: clip pi into [eps, 1-eps] so 1/pi, 1/(1-pi) stay bounded
        e_hat = np.clip(e_hat, 0.05, 0.95)

        # AIPW pseudo-outcome: regression difference + IPW correction on each arm's residual.
        # phi = mu1 - mu0 + T(Y - mu1)/e - (1-T)(Y - mu0)/(1-e);  E[phi|X]=tau, bias is a
        # product of (pi error)x(mu error), so it vanishes if EITHER nuisance is right.
        pseudo = (
            mu1_hat - mu0_hat
            + T * (Y - mu1_hat) / e_hat
            - (1 - T) * (Y - mu0_hat) / (1 - e_hat)
        )

        # winsorize extreme pseudo-outcomes (heavy residual tails near weak overlap)
        q = np.percentile(np.abs(pseudo), 97)
        pseudo = np.clip(pseudo, -q, q)

        # second stage: regress X -> pseudo-outcome to recover the (simpler) tau(x)
        self._cate_model = self._make_cate_model()
        self._cate_model.fit(X, pseudo)
        return self

    def predict(self, X):
        return self._cate_model.predict(X)
```

Let me retrace the causal chain so I'm sure the construction is forced, not assembled. I started stuck with the plug-in/T-learner, which differences two separately-fit response surfaces and so inherits their (large) error even when the effect `tau` is trivial — it cannot converge at the complexity of `tau`. The oracle who could regress the true contrast `Y^1 - Y^0` shows the target rate. IPW gives a per-unit observable with conditional mean `tau`, recovering the oracle's adaptivity, but is singly robust and explodes when overlap is weak. The semiparametric theory for the *average* effect hands me the augmented-IPW influence function, whose von Mises remainder is a *product* of the propensity and outcome errors — doubly robust and second-order. Lifting from average to conditional by *regressing* that influence function on `X` instead of averaging it, the oracle check confirms its conditional mean is `tau`, and grinding the conditional bias of the *estimated* pseudo-outcome reproduces the same per-arm product `(pihat - pi)(muahat - mu_a)` — so the second-stage error relative to the oracle is a product of nuisance errors, vanishing if either nuisance is right. That clean bias is only valid with the nuisances independent of the evaluation units, which forces sample-splitting / cross-fitting (train out-of-fold, predict in-fold, swap and average) to kill the empirical-process contamination without any Donsker restriction on the flexible learners. A stability argument for linear smoothers then makes "regress the pseudo-outcome" rigorous — the estimated-pseudo-outcome fit equals the true-pseudo-outcome (oracle) fit up to the smoothed product-bias plus `o_P(R_n*)` — and a Hölder split turns that smoothed bias into a product of weighted nuisance norms, giving oracle efficiency under a *lowered* bar `sqrt(alpha·beta) >= (d/2)/sqrt(1 + (d/gamma)(1 + d/(2 sbar)))` that relaxes the ATE condition as `tau` gets smoother. Propensity clipping and pseudo-outcome winsorizing tame the finite-sample variance under weak overlap; flexible boosted nuisances and a shallower, more-regularized final regressor instantiate the method-agnostic theory while betting that `tau` is the simplest of the functions in play.
