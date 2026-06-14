Let me start from what actually hurts. I have observational data `(X_i, W_i, Y_i)` — covariates, a binary treatment, an outcome — and I want the conditional average treatment effect `tau*(x) = E[Y(1) - Y(0) | X = x]`. Under unconfoundedness, `{Y(0),Y(1)}` is independent of `W` given `X`, so `tau*` is identified, and I'll assume overlap, `eta < e*(x) < 1 - eta` for the propensity `e*(x) = P(W=1|X)`, so both arms exist everywhere. The thing I keep wanting, and keep failing to get cleanly, is to point a modern black-box learner — boosting, a net, a penalized regression — at this and have it estimate the effect well. The obstacle is that two jobs are tangled together: I have to undo the fact that treatment correlates with `X` (confounding), and I have to flexibly model how the effect varies with `X` (heterogeneity), and every method I know does both jobs inside one piece of machinery, which means it does neither cleanly and comes with no guarantee that it actually isolated the causal part.

Let me look hard at why the obvious things fail, because the failure modes are going to tell me what the right object is. The most naive thing: fit one regression `f(x,w) = E[Y|X=x,W=w]` with `w` as just another feature, and read off `tau-hat(x) = f(x,1) - f(x,0)`. The problem is that `w` is one coordinate among `d+1`, and a regularized learner is free to barely use it — shrink its coefficient, rarely split on it — so `f` ends up almost flat in `w` and the estimated effect collapses toward zero. The learner has no reason to privilege the one direction I care about. So I want each arm to have its own model: fit `mu_0(x) = E[Y|X,W=0]` and `mu_1(x) = E[Y|X,W=1]` separately and subtract. But now I've created a different disease. The two models are trained and regularized independently, so the difference `mu_1 - mu_0` is a difference of two separately-shrunk objects, and the shrinkages don't cancel. Concretely, take a high-dimensional linear model and fit a lasso per arm: each `beta_(w)` is pulled toward 0 on its own, and `beta_(1) - beta_(0)` can be regularized *away* from zero even when the true effect is identically zero. That's backwards — the regularization that helps prediction within each arm is actively manufacturing a fake effect in the difference, and it's worst when the arms are different sizes. So differencing two response surfaces is structurally fragile.

There's a cleverer repair that imputes individual effects — on treated units form `D_i = Y_i - mu_0(X_i)`, regress those on `X`, do the symmetric thing on controls, and blend by the propensity. But step back and ask what its *error* depends on. `D_i` literally contains `mu_0(X_i)`, so an error in the arm model `mu_0` passes into `D_i` and hence into `tau-hat` at first order: perturb `mu_0` by a little function of size `delta`, and `tau-hat` moves by order `delta`. The accuracy of my effect is chained to how well I estimated the full arm surfaces — which include all the confounding structure, the hard part — not to how simple the effect itself is. That's exactly the coupling I'd love to break: I want the effect's error to depend on the complexity of `tau*`, not on the complexity of the confounding.

One more, because it's so close to clean. There's an identity I can write down immediately. Define the *marginal* outcome mean `m*(x) = E[Y|X=x]` (not the arm means — the overall conditional mean, marginalizing over treatment). Then consider the transformed outcome `U_i = (Y_i - m*(X_i)) / (W_i - e*(X_i))`. I claim `E[U_i | X_i = x] = tau*(x)`. If that's true I'm basically done — just regress `U_i` on `X_i` with any learner. Let me see if it holds, and in checking it I'll probably learn what the real structure is. I need to know `E[Y - m*(X) | X, W]`. Under unconfoundedness, `E[Y(w)|X] = E[Y|X,W=w] = mu_w*(X)`, and `m*(x) = E[Y|X=x] = mu_0*(x) + e*(x)(mu_1*(x) - mu_0*(x)) = mu_0*(x) + e*(x) tau*(x)`. Now `E[Y|X,W] = mu_0*(X) + W tau*(X)`. Subtract:

```
E[Y - m*(X) | X, W] = mu_0*(X) + W tau*(X) - mu_0*(X) - e*(X) tau*(X)
                    = (W - e*(X)) tau*(X).
```

So `E[Y - m*(X) | X, W] = (W - e*(X)) tau*(X)`, which means I can write, with a conditionally-mean-zero residual `eps`,

```
Y_i - m*(X_i) = (W_i - e*(X_i)) tau*(X_i) + eps_i,   E[eps_i | X_i, W_i] = 0.
```

Now divide by `W - e*` and take `E[.|X]`: indeed `E[U|X] = tau*(X)` — the transformed-outcome identity is real. But staring at it, the division is the disaster. Wherever the propensity drifts toward 0 or 1, `W - e*` goes to zero, and `U` is a tiny signal divided by a near-zero number — its variance explodes exactly in the low-overlap regions. Regressing such a wildly heteroskedastic target gives an unstable estimate. So the identity is right but the *estimator built by naive division* throws away the very structure that would stabilize it.

Now wait — look again at what I just derived, *before* I divided:

```
Y_i - m*(X_i) = (W_i - e*(X_i)) tau*(X_i) + eps_i,   E[eps_i | X_i, W_i] = 0.
```

I've seen this shape. This is Robinson's partially linear model, almost exactly. Robinson studied `E(Y|X,Z) = beta'X + theta(Z)` — a parametric part `beta'X` plus a nonparametric nuisance `theta(Z)` — and his move was: the model implies `Y - E(Y|Z) = beta'(X - E(X|Z)) + U` with `E(U|X,Z)=0`, so you residualize both `Y` and `X` against `Z` (with nonparametric kernel estimates of `E(Y|Z)` and `E(X|Z)`), then recover `beta` by no-intercept OLS of the `Y`-residual on the `X`-residual. The remarkable thing he proved is that `beta` comes out `sqrt(n)`-consistent and asymptotically normal *even though the kernel nuisances converge slower than `sqrt(n)`* — the target estimates fast while the nuisances estimate slow. The deeper precursor is just Frisch-Waugh-Lovell: in a linear regression, a coefficient equals the regression of the residualized outcome on the residualized regressor, the other variables projected out of both.

Line up my problem with his. My "regressor" is `W`; my "nuisance to partial out" is the function of `X` that predicts each of `Y` and `W`. My residuals are `Y - m*(X)` (outcome residualized against `X`) and `W - e*(X)` (treatment residualized against `X`). The structural identity I derived *is* Robinson's residual-on-residual form. The one difference — and it's the whole point — is that Robinson's coefficient `beta` is a *constant*. In his world the thing multiplying the regressor residual is a fixed number. In mine, the thing multiplying `W - e*(X)` is `tau*(X)`, a *function* of the covariates. So if I can generalize Robinson's partialling-out so that the "slope" is allowed to be a function instead of a constant, I get heterogeneous effects out of exactly the residual-on-residual structure that he showed is robust to slow nuisance estimation. That's the move: let the slope be `tau*(.)`.

How do I estimate a *function-valued* slope from `Y - m* = (W - e*) tau*(X) + eps`? Robinson minimized squared error of the residual relation to get a constant `beta`. The honest generalization is to keep the squared-error form but let `tau` range over functions. Since `eps` is conditionally mean-zero, the population least-squares projection picks out `tau*`:

```
tau*(.) = argmin_tau  E[ ( (Y - m*(X)) - (W - e*(X)) tau(X) )^2 ].
```

Let me verify this minimizer really is `tau*` and not something contaminated, because if it's biased the whole edifice falls. Expand the loss for a candidate `tau`. Write `Y - m*(X) = (W - e*(X)) tau*(X) + eps`. Then

```
(Y - m*) - (W - e*) tau = (W - e*)(tau* - tau)(X) + eps,
```

and squaring and taking expectation, the cross term `E[(W-e*)(tau*-tau)(X) eps]` vanishes because `E[eps | X, W] = 0` (condition on `X,W`, pull the deterministic factor out, the inner expectation of `eps` is zero). So

```
L(tau) = E[ (W - e*(X))^2 (tau*(X) - tau(X))^2 ] + E[eps^2],
```

and the second term doesn't depend on `tau`. The first term is a nonnegative weighted squared distance between `tau` and `tau*`. Conditional on `X=x`, the binary treatment gives `E[(W-e*(x))^2 | X=x] = e*(x)(1-e*(x))`, so overlap makes the weight strictly positive; looser but convenient bounds give `(1-eta)^{-2} R(tau) < E[(tau(X)-tau*(X))^2] < eta^{-2} R(tau)`, where `R(tau)=L(tau)-L(tau*)`. So the population objective is minimized uniquely (in `L2`) at `tau = tau*`, and regret is equivalent to the squared effect error, with the constants degrading as `eta -> 0`. Fine.

So the *oracle* procedure, if I knew `m*` and `e*`, is to minimize an empirical version of that loss plus a regularizer on `tau`:

```
tau-tilde = argmin_tau  (1/n) sum_i [ (Y_i - m*(X_i)) - (W_i - e*(X_i)) tau(X_i) ]^2  +  Lambda_n(tau).
```

Now the effect-estimation problem has become *plain regularized empirical loss minimization* in `tau`. That means any learner that minimizes a (weighted) squared loss — penalized regression, a neural net, boosting — can be the optimizer; I don't have to modify the learner's internals, I only have to feed it this loss. And the two jobs have separated: the loss itself is what kills confounding (through the residualization), while the *choice of learner* used to minimize it is what expresses the heterogeneity. I can use a black box to fit `tau` without auditing whether it controls for confounding, because the confounding control lives in the loss, not in the model class. That separation is the thing I was missing in every baseline.

Of course I don't know `m*` and `e*`. So the feasible method is two-step: first estimate the nuisances `m-hat(x) = E[Y|X]` and `e-hat(x) = P(W=1|X)` with whatever predictive learners I like, then plug them in and minimize

```
tau-hat = argmin_tau  (1/n) sum_i [ (Y_i - m-hat(X_i)) - (W_i - e-hat(X_i)) tau(X_i) ]^2  +  Lambda_n(tau).
```

But this is where I have to be careful, because now I've replaced the truth by estimates, and the whole reason to prefer this over the X-learner was supposed to be robustness to nuisance error. I need to actually check that error in `m-hat` and `e-hat` doesn't propagate badly into `tau-hat`. If it does, I've gained nothing.

Two things could go wrong, and I want to find both before they bite. First, an *overfitting* issue: if I estimate `m-hat(X_i)` using observation `i` itself, then the residual `Y_i - m-hat(X_i)` is artificially small — the model has partially memorized `Y_i` — and the same goes for `W_i - e-hat(X_i)`; this in-sample shrinkage biases the loss. The fix is forced by the structure of the bias: estimate the nuisance predictions for fold `i` using only data *outside* fold `i`. Split the data into `Q` folds (5 or 10), and for each `i` use `m-hat^{(-q(i))}` and `e-hat^{(-q(i))}` — predictions from models trained without `i`'s fold. This is cross-fitting, and it makes the held-out nuisance prediction statistically independent of `i`'s own residual, which I'll need in a moment for the second, deeper, issue.

The second issue decides whether the plug-in loss actually buys robustness. How does the error from `m-hat != m*` and `e-hat != e*` enter the feasible loss compared to the oracle loss? Let me write the feasible loss `L-hat(tau)` and define the nuisance errors in the direction that will keep the signs straight: `delta_m(X_i) = m*(X_i) - m-hat(X_i)` and `delta_e(X_i) = e*(X_i) - e-hat(X_i)`. The integrand is

```
[ (Y_i - m*(X_i)) + delta_m(X_i) - (W_i - e*(X_i)) tau(X_i) - delta_e(X_i) tau(X_i) ... ]^2,
```

wait, let me be careful with signs. `Y_i - m-hat = (Y_i - m*) + (m* - m-hat) = (Y_i - m*) + delta_m`. And `W_i - e-hat = (W_i - e*) + (e* - e-hat) = (W_i - e*) + delta_e`. So

```
(Y_i - m-hat) - (W_i - e-hat) tau(X_i)
   = [ (Y_i - m*) - (W_i - e*) tau(X_i) ]  +  delta_m(X_i)  -  delta_e(X_i) tau(X_i).
```

The first bracket is exactly the oracle integrand. Call it `r_i(tau)`. So the feasible integrand is `r_i(tau) + delta_m - delta_e tau`, and the feasible loss is

```
L-hat(tau) = (1/n) sum_i r_i(tau)^2
           + (1/n) sum_i ( delta_m - delta_e tau )^2
           + (2/n) sum_i r_i(tau)( delta_m - delta_e tau ).
```

The first sum is the oracle loss `L-tilde(tau)`. The interesting question is the other two. But I shouldn't look at `L-hat(tau)` alone — what the optimizer actually responds to is the loss *difference* between candidates, so let me track the regret-like object `L-hat(tau) - L-hat(tau_ref)` against some reference `tau_ref` (the best in-class approximation to `tau*`). When I difference, the `tau`-free pieces drop, and the surviving nuisance terms are exactly

```
-(2/n) sum_i delta_m_i delta_e_i (tau_i - tau_ref_i)
 +(1/n) sum_i delta_e_i^2 (tau_i^2 - tau_ref_i^2)
-(2/n) sum_i (Y_i - m*(X_i)) delta_e_i (tau_i - tau_ref_i)
-(2/n) sum_i (W_i - e*(X_i)) delta_m_i (tau_i - tau_ref_i)
 +(2/n) sum_i (W_i - e*(X_i)) delta_e_i (tau_i^2 - tau_ref_i^2).
```

Good, five terms, and the signs now line up with the expansion. The first one is the product channel. By Cauchy-Schwarz, `|(1/n) sum delta_m delta_e (tau - tau_ref)|` is bounded by `sqrt((1/n) sum delta_m^2) * sqrt((1/n) sum delta_e^2) * ||tau - tau_ref||_inf`, i.e. by `(RMSE of m-hat) * (RMSE of e-hat)` times a bounded factor. If each nuisance is `o(n^{-1/4})`, the product is `o(n^{-1/2})`. The second term is also second order, because it carries `delta_e^2` and `tau^2 - tau_ref^2 = 2 tau_ref(tau-tau_ref) + (tau-tau_ref)^2`, with `tau` capped in sup norm in the kernel analysis. So the purely deterministic nuisance pieces are not first-order leaks.

The remaining three terms are the dangerous ones because each contains a single nuisance error multiplied by an oracle residual. A single first-order nuisance error times an `O(1)` residual would dominate and ruin everything if it had a nonzero mean. This is exactly where cross-fitting earns its keep. Take the outcome-residual/e-error term, ignoring the harmless sign:

```
E[ (Y_i - m*(X_i)) (e*(X_i) - e-hat^{(-q)}(X_i)) (tau - tau_ref)(X_i) | I^{(-q)}, X_i ]
   = (e*(X_i) - e-hat^{(-q)}(X_i)) (tau - tau_ref)(X_i) * E[ Y_i - m*(X_i) | X_i ].
```

And `E[Y_i - m*(X_i) | X_i] = 0` by the definition of `m*` as the conditional mean of `Y` given `X`. So this whole cross term has *conditional mean exactly zero*. It's not bounded-and-small, it's *centered* — a mean-zero empirical process. Its magnitude is then controlled by fluctuation arguments (Talagrand's concentration inequality, generic chaining over the function class), which give `1/sqrt(n)`-type empirical-process factors rather than the raw `O(a_n)` size it would have had without centering. The treatment-residual/m-error term is centered the same way because `E[W_i - e*(X_i)|X_i] = 0`. The last term, the one with `(W-e*) delta_e (tau^2-tau_ref^2)`, is also centered by that same treatment-residual identity; its price is controlled by the same chaining machinery, with the capped `tau` class keeping `tau^2-tau_ref^2` in a manageable envelope. So the three single-nuisance-error channels survive only as fluctuations, because cross-fitting made the nuisance estimate independent of the held-out residual and let the conditional expectation factor onto a mean-zero residual.

Putting it together: the difference between the feasible regret and the oracle regret is bounded by second-order deterministic nuisance pieces plus centered empirical-process fluctuations — concretely `|R-hat_n(tau;c) - R-tilde_n(tau;c)| <= 0.125 R(tau;c) + o(rho_n(c))`, a small fraction of the regret itself plus a lower-order remainder. That is precisely the kind of "quasi-isomorphism" between the feasible and oracle loss that the empirical-risk-minimization machinery wants: if I have a high-probability quasi-isomorphism `(1/k) R-check_n(tau;c) - rho_n(c) <= R(tau;c) <= k R-check_n(tau;c) + rho_n(c)` for a loss, then minimizing that loss with a penalty `Lambda_n >= rho_n` gives a risk bound `L(tau-hat) <= inf_tau {L(tau) + kappa_2 Lambda_n(||tau||)}`. The oracle loss has such a quasi-isomorphism (this is the Bartlett-style isomorphic-coordinate-projection result, with `rho_n(c)` of the order `((c+1)^p log n / sqrt n)^{2/(1+p)}` up to logarithmic factors for an RKHS with eigenvalue decay `sigma_j ~ j^{-1/p}` and smoothness `alpha`), and I've just shown the feasible loss inherits the same quasi-isomorphism up to a slightly inflated `rho_n`. So I get, for `tau-hat` from penalized kernel regression on the feasible loss,

```
R(tau-hat) = Otilde_P( n^{-(1 - 2 alpha)/(p + (1 - 2 alpha))} ),
```

the *same* rate as the oracle who knew `m*` and `e*` a priori. The estimation error in the nuisances has dropped out of the leading-order bound: the rate of `tau-hat` depends only on the complexity of `tau*` (through `alpha, p`), not on the complexity of `m*` or `e*`. That's the quasi-oracle property, and it's the formal guarantee I was after — it holds provided each nuisance is `o(n^{-kappa})` with `kappa > 1/4`, and overlap holds. As `alpha, p -> 0` this collapses to the familiar semiparametric statement that you need fourth-root-consistent nuisances to get root-`n` inference on a target — the same threshold, recovered here for a function-valued target.

And I can now see *why the X-learner does not generally have this property*, which sharpens the contrast. The X-learner's estimate inherits `mu_0, mu_1` to first order: shift `mu_0(x) -> mu_0(x) - c/n^{0.25+xi}` and `mu_1(x) -> mu_1(x) + c/n^{0.25+xi}` and, by inspection of how it imputes and combines, its `tau-hat` shifts by `c/n^{0.25+xi}`. Those nuisance perturbations are vanishingly small on the `n^{-1/4}` scale — they'd satisfy any condition my theorem needs — yet they move `tau-hat` by an amount that can dominate the target rate `n^{-(1-2alpha)/(2(p+1-2alpha))}`. So no general guarantee on my scale can follow from those nuisance assumptions alone: its nuisance error enters first order. Here the deterministic drift is a product of the two nuisance errors, and the single-error residual channels are centered by cross-fitting; the X-learner has a raw first-order channel that nothing cancels.

Now let me make the feasible objective something I can actually hand to a standard learner, because "minimize this custom loss" should reduce to a call I already have. Look at the integrand again and complete the manipulation. For each `i`, with `Y-tilde_i = Y_i - m-hat(X_i)` and `W-tilde_i = W_i - e-hat(X_i)`,

```
[ Y-tilde_i - W-tilde_i tau(X_i) ]^2  =  W-tilde_i^2 [ Y-tilde_i / W-tilde_i - tau(X_i) ]^2.
```

That's just factoring `W-tilde_i^2` out of the bracket. So the R-loss is *identically* a weighted least-squares regression: regress the pseudo-outcome `Y-tilde_i / W-tilde_i` on `X_i`, with sample weight `W-tilde_i^2`. Operationally, that matters: every learner I care about (boosting, ridge, a net, a weighted random forest) accepts sample weights, so I can minimize the R-loss by a single ordinary weighted-regression call, no custom-loss surgery. And it reconciles with the failed U-learner: the U-learner *was* regressing the pseudo-outcome `Y-tilde/W-tilde`, but with weight 1, so the points where `W-tilde` is near zero — pseudo-outcomes with huge variance — got full say and blew up the variance. Here the weight is `W-tilde^2`, which is *exactly* small where `W-tilde` is small, so it downweights precisely the high-variance pseudo-outcomes. The `1/W-tilde` blowup in the pseudo-outcome is cancelled by the `W-tilde^2` weight; only the structurally informative `W-tilde^2` reweighting remains. The U-learner threw away the weight that the loss was telling it to use. Same identity, but reading it as weighted regression rather than as a transformed-outcome regression is what makes it stable.

I should think about the practical knobs the theory and the structure imply. Overlap: where `e*(x)` nears 0 or 1, the common treatment arm has `|W-tilde|` near zero, the pseudo-outcome `Y-tilde/W-tilde` is numerically explosive even with the weight, and the regret-to-error coupling `eta^{-2}` loosens. So I clip the estimated propensity into `[eta, 1-eta]` for a small `eta` (say 0.05), which enforces a practical overlap floor and also makes `|W_i - e-hat(X_i)| >= eta` for binary `W_i`. The bounded-`tau` device in the theory — restricting `||tau||_inf <= 2M` to rule out pathological minimizers — belongs in the learner or regularizer, not in a change to the R-loss itself, because the weighted-regression identity is exact only for the unmodified pseudo-outcome and weight.

For the nuisances themselves: I want `m-hat = E[Y|X]` and `e-hat = P(W=1|X)` estimated as accurately as possible for *prediction* — that's all the theory asks, `o(n^{-1/4})` RMSE — so I use a strong, well-regularized predictive learner for each and tune it for predictive accuracy, by cross-validation if I like. The `m`-model is a regressor of `Y` on `X`; the `e`-model is a classifier of `W` on `X` returning a probability. They're separate models with their own hyperparameters because they're predicting different things. Critically I fit them with cross-fitting: in each held-out fold, the predictions come from models trained on the other folds, so the residuals fed into the R-loss are out-of-sample. For the `tau`-model I again use a generic learner — here a gradient-boosted regressor — fed the pseudo-outcome and the `W-tilde^2` weights; I can and should tune *it* by cross-validating on the R-loss, since the R-loss is an ordinary held-out objective and the learner only has to find a generalizable minimizer of it, not also police confounding.

Let me write the whole thing as code, filling the one open slot in the CATE harness. Cross-fit the two nuisance models, form the two residuals, clip the propensity, build the pseudo-outcome and the squared-residual weights, and fit a weighted regression of the pseudo-outcome on `X`:

```python
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier


class CATEEstimator:
    """R-learner: generalize Robinson's residual-on-residual partialling-out so the
    'slope' is a function tau(x). Residualize Y and W against X (cross-fit), then
    minimize the R-loss sum_i [ (Y - m_hat) - (W - e_hat) tau(X) ]^2 as a weighted
    regression of the pseudo-outcome (Y - m_hat)/(W - e_hat) with weight (W - e_hat)^2."""

    def __init__(self, n_folds=5, seed=42, eta=0.05):
        self.n_folds = n_folds
        self.seed = seed
        self.eta = eta

    def _make_regressor(self):                    # generic predictive regressor
        return GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            min_samples_leaf=20, subsample=0.8, random_state=self.seed,
        )

    def _make_classifier(self):                   # generic predictive classifier
        return GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            min_samples_leaf=20, subsample=0.8, random_state=self.seed + 1,
        )

    def fit(self, X, W, Y):
        X, W, Y = np.asarray(X), np.asarray(W), np.asarray(Y)
        n = len(Y)

        # --- Cross-fit the nuisances: held-out predictions so residuals are
        #     out-of-sample (kills own-observation overfitting and centers the
        #     first-order cross terms to mean zero). ---
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        m_hat = np.zeros(n)                        # m(X) = E[Y|X]   (marginal mean)
        e_hat = np.zeros(n)                        # e(X) = P(W=1|X) (propensity)
        for tr, va in kf.split(X):
            my = self._make_regressor(); my.fit(X[tr], Y[tr])
            m_hat[va] = my.predict(X[va])
            mw = self._make_classifier(); mw.fit(X[tr], W[tr])
            e_hat[va] = mw.predict_proba(X[va])[:, 1]

        e_hat = np.clip(e_hat, self.eta, 1 - self.eta)

        Y_tilde = Y - m_hat                        # outcome residual using m_hat
        W_tilde = W - e_hat                        # treatment residual using e_hat

        # R-loss = sum (Y_tilde - W_tilde*tau)^2 = sum W_tilde^2 ((Y_tilde/W_tilde) - tau)^2
        weights = W_tilde ** 2
        pseudo = Y_tilde / W_tilde

        # Minimize the R-loss as a generic weighted regression of pseudo on X.
        self._cate_model = GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            min_samples_leaf=20, subsample=0.8, random_state=self.seed + 2,
        )
        self._cate_model.fit(X, pseudo, sample_weight=weights)
        return self

    def predict(self, X):
        return self._cate_model.predict(np.asarray(X))
```

Let me retrace the causal chain so I'm sure the pieces lock. I wanted heterogeneous effects from black-box ML, and every direct route failed for a structural reason: the single-model route lets the learner ignore the treatment; the two-arm-difference route lets independent regularization manufacture a fake effect; the imputed-effect route chains its error to the full arm surfaces at first order; the transformed-outcome route divides by a near-zero number and explodes. Computing `E[Y - m*(X) | X, W]` gave me `(W - e*(X)) tau*(X)`, which is Robinson's partially-linear residual-on-residual identity with the constant slope promoted to a function `tau*(X)` — so the population least-squares projection of the outcome residual on the treatment residual recovers `tau*` exactly, with regret equal to the overlap-weighted squared error. That turns CATE estimation into ordinary regularized loss minimization in `tau`, cleanly separating confounding control (in the loss) from effect modeling (in the learner). Substituting estimated nuisances and expanding the regret gives a product-of-errors term, a squared propensity-error term, and three single-error cross terms; cross-fitting centers the single-error terms to mean zero through `E[Y - m*|X] = 0` and `E[W - e*|X] = 0`, while the deterministic pieces are second order. Feeding that into the isomorphic-coordinate-projection ERM bound gives `tau-hat` the *same* rate as the oracle who knew the nuisances — the quasi-oracle property — which the X-learner does not generally have because its nuisance dependence is first-order and uncancelled. Finally, factoring `W-tilde^2` out of the squared loss reveals the R-loss is a weighted regression of the pseudo-outcome `(Y-m-hat)/(W-e-hat)` with weight `(W-e-hat)^2`, which is the U-learner's regression with the *right* weight restored — the weight that cancels the `1/W-tilde` variance blowup — so any sample-weight-aware learner minimizes it in one call.
