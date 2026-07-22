Let me start from what actually hurts. I have observational data `(X_i, W_i, Y_i)` — covariates, a binary treatment, an outcome — and I want the conditional average treatment effect `tau*(x) = E[Y(1) - Y(0) | X = x]`. Under unconfoundedness, `{Y(0),Y(1)}` is independent of `W` given `X`, so `tau*` is identified, and I'll assume overlap, `eta < e*(x) < 1 - eta` for the propensity `e*(x) = P(W=1|X)`, so both arms exist everywhere. The thing I keep wanting, and keep failing to get cleanly, is to point a modern black-box learner — boosting, a net, a penalized regression — at this and have it estimate the effect well. The obstacle is that two jobs are tangled together: I have to undo the fact that treatment correlates with `X` (confounding), and I have to flexibly model how the effect varies with `X` (heterogeneity), and every method I know does both jobs inside one piece of machinery, which means it does neither cleanly and comes with no guarantee that it actually isolated the causal part.

Let me look hard at why the obvious things fail, because the failure modes are going to tell me what the right object is. The most naive thing: fit one regression `f(x,w) = E[Y|X=x,W=w]` with `w` as just another feature, and read off `tau-hat(x) = f(x,1) - f(x,0)`. The problem is that `w` is one coordinate among `d+1`, and a regularized learner is free to barely use it — shrink its coefficient, rarely split on it — so `f` ends up almost flat in `w` and the estimated effect collapses toward zero. The learner has no reason to privilege the one direction I care about. So I want each arm to have its own model: fit `mu_0(x) = E[Y|X,W=0]` and `mu_1(x) = E[Y|X,W=1]` separately and subtract. But now I've created a different disease. The two models are trained and regularized independently, so the difference `mu_1 - mu_0` is a difference of two separately-shrunk objects, and the shrinkages don't cancel. Concretely, take a high-dimensional linear model and fit a lasso per arm: each `beta_(w)` is pulled toward 0 on its own, and `beta_(1) - beta_(0)` can be regularized *away* from zero even when the true effect is identically zero. That's backwards — the regularization that helps prediction within each arm is actively manufacturing a fake effect in the difference, and it's worst when the arms are different sizes. So differencing two response surfaces is structurally fragile.

There's a cleverer repair that imputes individual effects — on treated units form `D_i = Y_i - mu_0(X_i)`, regress those on `X`, do the symmetric thing on controls, and blend by the propensity. But step back and ask what its *error* depends on. `D_i` literally contains `mu_0(X_i)`, so an error in the arm model `mu_0` passes into `D_i` and hence into `tau-hat` at first order: perturb `mu_0` by a little function of size `delta`, and `tau-hat` moves by order `delta`. The accuracy of my effect is chained to how well I estimated the full arm surfaces — which include all the confounding structure, the hard part — not to how simple the effect itself is. That's exactly the coupling I'd love to break: I want the effect's error to depend on the complexity of `tau*`, not on the complexity of the confounding.

One more, because it's so close to clean. There's an identity I can write down immediately. Define the *marginal* outcome mean `m*(x) = E[Y|X=x]` (not the arm means — the overall conditional mean, marginalizing over treatment). Then consider the transformed outcome `U_i = (Y_i - m*(X_i)) / (W_i - e*(X_i))`. I suspect `E[U_i | X_i = x] = tau*(x)`. If that's true I'm basically done — just regress `U_i` on `X_i` with any learner. Let me see if it holds, and in checking it I'll probably learn what the real structure is. I need to know `E[Y - m*(X) | X, W]`. Under unconfoundedness, `E[Y(w)|X] = E[Y|X,W=w] = mu_w*(X)`, and `m*(x) = E[Y|X=x] = mu_0*(x) + e*(x)(mu_1*(x) - mu_0*(x)) = mu_0*(x) + e*(x) tau*(x)`. Now `E[Y|X,W] = mu_0*(X) + W tau*(X)`. Subtract:

```
E[Y - m*(X) | X, W] = mu_0*(X) + W tau*(X) - mu_0*(X) - e*(X) tau*(X)
                    = (W - e*(X)) tau*(X).
```

So `E[Y - m*(X) | X, W] = (W - e*(X)) tau*(X)`. Before I lean on this — it's about to become the backbone of everything — let me make sure I didn't drop a term, by checking it against numbers I control. Take a discrete `X in {a,b}` with `mu_0*(a)=1, mu_0*(b)=-2`, `tau*(a)=0.5, tau*(b)=3`, `e*(a)=0.3, e*(b)=0.8`. Then `m*` should be `mu_0* + e* tau*`, i.e. `m*(a)=1.15`, `m*(b)=0.40`. Simulating `Y = mu_0*(X) + W tau*(X) + N(0,1)` and reading off `E[Y - m*(X) | X=x, W=w]` arm-by-arm against the predicted `(w - e*(x)) tau*(x)`:

```
  x=a w=0: empirical -0.150   predicted (0-0.3)*0.5 = -0.150
  x=a w=1: empirical +0.351   predicted (1-0.3)*0.5 = +0.350
  x=b w=0: empirical -2.398   predicted (0-0.8)*3   = -2.400
  x=b w=1: empirical +0.598   predicted (1-0.8)*3   = +0.600
```

Four cells, four matches to two decimals. The identity is real and I have the signs right. So I can write, with a conditionally-mean-zero residual `eps`,

```
Y_i - m*(X_i) = (W_i - e*(X_i)) tau*(X_i) + eps_i,   E[eps_i | X_i, W_i] = 0.
```

Now divide by `W - e*` and take `E[.|X]`: `E[U|X] = tau*(X)`, so the transformed-outcome idea is genuinely valid. But staring at the division, it's the disaster. Wherever the propensity drifts toward 0 or 1, `W - e*` goes to zero, and `U` is a tiny signal divided by a near-zero number — its variance explodes exactly in the low-overlap regions. To feel how bad: on a continuous design where `e*` runs down to about 0.05, the raw pseudo-outcome `(Y-m*)/(W-e*)` has standard deviation around 8.6 and individual values reaching ~80, and when I sort the most extreme ones they all sit at the smallest `|W-e*|` (~0.05). In a poor-overlap window the unweighted mean pseudo-outcome (which is what regressing `U` on `X` is locally doing) lands at 0.049 when the true effect there is ~0.125, with sample variance over 200 — the estimate is both biased-looking and wildly noisy. So the identity is right but the *estimator built by naive division* throws away the very structure that would stabilize it.

Now wait — look again at what I derived, *before* I divided:

```
Y_i - m*(X_i) = (W_i - e*(X_i)) tau*(X_i) + eps_i,   E[eps_i | X_i, W_i] = 0.
```

I've seen this shape. This is Robinson's partially linear model, almost exactly. Robinson studied `E(Y|X,Z) = beta'X + theta(Z)` — a parametric part `beta'X` plus a nonparametric nuisance `theta(Z)` — and his move was: the model implies `Y - E(Y|Z) = beta'(X - E(X|Z)) + U` with `E(U|X,Z)=0`, so you residualize both `Y` and `X` against `Z` (with nonparametric kernel estimates of `E(Y|Z)` and `E(X|Z)`), then recover `beta` by no-intercept OLS of the `Y`-residual on the `X`-residual. The remarkable thing he proved is that `beta` comes out `sqrt(n)`-consistent and asymptotically normal *even though the kernel nuisances converge slower than `sqrt(n)`* — the target estimates fast while the nuisances estimate slow. The deeper precursor is just Frisch-Waugh-Lovell: in a linear regression, a coefficient equals the regression of the residualized outcome on the residualized regressor, the other variables projected out of both.

Line up my problem with his. My "regressor" is `W`; my "nuisance to partial out" is the function of `X` that predicts each of `Y` and `W`. My residuals are `Y - m*(X)` (outcome residualized against `X`) and `W - e*(X)` (treatment residualized against `X`). The structural identity I derived *is* Robinson's residual-on-residual form. The one difference — and it may be the whole point — is that Robinson's coefficient `beta` is a *constant*. In his world the thing multiplying the regressor residual is a fixed number. In mine, the thing multiplying `W - e*(X)` is `tau*(X)`, a *function* of the covariates. So if I can generalize Robinson's partialling-out so that the "slope" is allowed to be a function instead of a constant, I'd get heterogeneous effects out of exactly the residual-on-residual structure that he showed is robust to slow nuisance estimation. Let me see whether that generalization actually goes through.

How do I estimate a *function-valued* slope from `Y - m* = (W - e*) tau*(X) + eps`? Robinson minimized squared error of the residual relation to get a constant `beta`. The honest generalization is to keep the squared-error form but let `tau` range over functions:

```
tau*(.) =? argmin_tau  E[ ( (Y - m*(X)) - (W - e*(X)) tau(X) )^2 ].
```

I've written this with a question mark because I haven't earned it yet — I need to check the minimizer of that population least-squares projection really is `tau*` and not something contaminated, because if it's biased the whole edifice falls. Expand the loss for a candidate `tau`. Write `Y - m*(X) = (W - e*(X)) tau*(X) + eps`. Then

```
(Y - m*) - (W - e*) tau = (W - e*)(tau* - tau)(X) + eps,
```

and squaring and taking expectation, the cross term `E[(W-e*)(tau*-tau)(X) eps]` vanishes because `E[eps | X, W] = 0` (condition on `X,W`, pull the deterministic factor out, the inner expectation of `eps` is zero). So

```
L(tau) = E[ (W - e*(X))^2 (tau*(X) - tau(X))^2 ] + E[eps^2],
```

and the second term doesn't depend on `tau`. The first term is a nonnegative weighted squared distance between `tau` and `tau*`. Conditional on `X=x`, the binary treatment gives `E[(W-e*(x))^2 | X=x] = e*(x)(1-e*(x))`, so overlap makes the weight strictly positive.

I'd rather not trust this expansion on faith — it has two moving parts I could have botched, the cross-term cancellation and the `e(1-e)` weight. Back to the discrete population. First, does the loss actually bottom out at `tau*`? Holding `tau(b)` at its true value and sweeping `tau(a)`:

```
   tau(a)=0.00 -> L=1.0242
   tau(a)=0.30 -> L=1.0020
   tau(a)=0.50 -> L=0.9978   <- true tau(a)
   tau(a)=0.70 -> L=1.0019
   tau(a)=1.00 -> L=1.0239
```

The minimum sits right on 0.5, and the analytic scalar argmin `E[(Y-m*)(W-e*)|x=a]/E[(W-e*)^2|x=a]` comes out 0.5014 — `tau*` to the precision of the sample. Second, the weight `E[(W-e*)^2|X=x]`: I get 0.2102 at `x=a` versus `e(1-e)=0.3·0.7=0.21`, and 0.1595 at `x=b` versus `0.8·0.2=0.16`. Both the minimizer and the weight check out. And the regret identity itself — `L(tau) - L(tau*)` should equal `E[(W-e*)^2 (tau-tau*)^2]`:

```
   cand (1.0, 3.0): regret +0.0261   weighted-dist +0.0263
   cand (0.5, 0.0): regret +0.7162   weighted-dist +0.7180
   cand (-0.5, 4.0): regret +0.1857   weighted-dist +0.1848
```

They agree across three quite different candidates. So the population objective is minimized uniquely (in `L2`) at `tau = tau*`, and the regret is exactly the overlap-weighted squared effect error. Looser but convenient bounds give `(1-eta)^{-2} R(tau) < E[(tau(X)-tau*(X))^2] < eta^{-2} R(tau)`, where `R(tau)=L(tau)-L(tau*)`; with `eta=0.2` (the natural floor since my `e*` lives in `{0.3,0.8}`) and the first candidate, the lower bound `0.0408` and upper bound `0.6529` do indeed bracket the actual `E[(tau-tau*)^2]=0.125`. The constants degrade as `eta -> 0`, as the `eta^{-2}` warns.

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

The second issue decides whether the plug-in loss actually buys robustness. How does the error from `m-hat != m*` and `e-hat != e*` enter the feasible loss compared to the oracle loss? Let me write the feasible loss `L-hat(tau)` and define the nuisance errors in the direction that will keep the signs straight: `delta_m(X_i) = m*(X_i) - m-hat(X_i)` and `delta_e(X_i) = e*(X_i) - e-hat(X_i)`. I want to be careful with signs here, so I'll substitute directly. `Y_i - m-hat = (Y_i - m*) + (m* - m-hat) = (Y_i - m*) + delta_m`. And `W_i - e-hat = (W_i - e*) + (e* - e-hat) = (W_i - e*) + delta_e`. So

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

Five terms, and the signs line up with the expansion. The first one is the product channel. By Cauchy-Schwarz, `|(1/n) sum delta_m delta_e (tau - tau_ref)|` is bounded by `sqrt((1/n) sum delta_m^2) * sqrt((1/n) sum delta_e^2) * ||tau - tau_ref||_inf`, i.e. by `(RMSE of m-hat) * (RMSE of e-hat)` times a bounded factor. If each nuisance is `o(n^{-1/4})`, the product is `o(n^{-1/2})`. The second term is also second order, because it carries `delta_e^2` and `tau^2 - tau_ref^2 = 2 tau_ref(tau-tau_ref) + (tau-tau_ref)^2`, with `tau` capped in sup norm in the kernel analysis. So the purely deterministic nuisance pieces are not first-order leaks.

The remaining three terms are the dangerous ones, because each contains a single nuisance error multiplied by an oracle residual. A single first-order nuisance error times an `O(1)` residual would dominate and ruin everything *if it had a nonzero mean*. So the whole question is whether these terms are centered. Take the outcome-residual/e-error term and condition; with cross-fitting, `e-hat^{(-q)}` is fixed relative to fold `q`'s data, so it factors out:

```
E[ (Y_i - m*(X_i)) (e*(X_i) - e-hat^{(-q)}(X_i)) (tau - tau_ref)(X_i) | I^{(-q)}, X_i ]
   = (e*(X_i) - e-hat^{(-q)}(X_i)) (tau - tau_ref)(X_i) * E[ Y_i - m*(X_i) | X_i ].
```

And `E[Y_i - m*(X_i) | X_i] = 0` by the definition of `m*` as the conditional mean of `Y` given `X`. So this whole cross term has conditional mean exactly zero — at least on paper. Let me make sure I'm not fooling myself: this centering is the single most load-bearing claim in the argument, so I'll test the summand `(Y - m*)·delta_e·g(X)` numerically on a continuous design, with a deliberately nonzero deterministic `delta_e(X) = 0.1 cos(3X)` and `g = 1`. Binning by `X` and averaging:

```
   bin 1: mean +0.0001   bin 4: mean +0.0000
   bin 2: mean -0.0003   bin 5: mean +0.0003
   bin 3: mean -0.0001
   overall mean +0.00000   (vs typical |delta_e| ~ 0.062)
```

Every conditional bin averages to within `3e-4` of zero, and the overall mean is zero to five decimals, even though the per-observation summand has magnitude ~0.06. It's not bounded-and-small, it's *centered*. Its magnitude is then controlled by fluctuation arguments (Talagrand's concentration inequality, generic chaining over the function class), which give `1/sqrt(n)`-type empirical-process factors rather than the raw `O(a_n)` size it would have had without centering. The treatment-residual/m-error term is centered the same way because `E[W_i - e*(X_i)|X_i] = 0`. The last term, the one with `(W-e*) delta_e (tau^2-tau_ref^2)`, is also centered by that same treatment-residual identity; its price is controlled by the same chaining machinery, with the capped `tau` class keeping `tau^2-tau_ref^2` in a manageable envelope. So the three single-nuisance-error channels survive only as fluctuations, because cross-fitting made the nuisance estimate independent of the held-out residual and let the conditional expectation factor onto a mean-zero residual.

Putting it together: the difference between the feasible regret and the oracle regret is bounded by second-order deterministic nuisance pieces plus centered empirical-process fluctuations — concretely `|R-hat_n(tau;c) - R-tilde_n(tau;c)| <= 0.125 R(tau;c) + o(rho_n(c))`, a small fraction of the regret itself plus a lower-order remainder. That is precisely the kind of "quasi-isomorphism" between the feasible and oracle loss that the empirical-risk-minimization machinery wants: if I have a high-probability quasi-isomorphism `(1/k) R-check_n(tau;c) - rho_n(c) <= R(tau;c) <= k R-check_n(tau;c) + rho_n(c)` for a loss, then minimizing that loss with a penalty `Lambda_n >= rho_n` gives a risk bound `L(tau-hat) <= inf_tau {L(tau) + kappa_2 Lambda_n(||tau||)}`. The oracle loss has such a quasi-isomorphism (this is the Bartlett-style isomorphic-coordinate-projection result, with `rho_n(c)` of the order `((c+1)^p log n / sqrt n)^{2/(1+p)}` up to logarithmic factors for an RKHS with eigenvalue decay `sigma_j ~ j^{-1/p}` and smoothness `alpha`), and the feasible loss inherits the same quasi-isomorphism up to a slightly inflated `rho_n`. So I get, for `tau-hat` from penalized kernel regression on the feasible loss,

```
R(tau-hat) = Otilde_P( n^{-(1 - 2 alpha)/(p + (1 - 2 alpha))} ),
```

the *same* rate as the oracle who knew `m*` and `e*` a priori. The estimation error in the nuisances has dropped out of the leading-order bound: the rate of `tau-hat` depends only on the complexity of `tau*` (through `alpha, p`), not on the complexity of `m*` or `e*`. This is the property I was hoping the residual-on-residual structure would deliver, and it holds provided each nuisance is `o(n^{-kappa})` with `kappa > 1/4`, and overlap holds. As `alpha, p -> 0` it collapses to the familiar semiparametric statement that you need fourth-root-consistent nuisances to get root-`n` inference on a target — the same threshold, recovered here for a function-valued target.

And I can now see *why the X-learner does not generally inherit this*, which sharpens the contrast. The X-learner's estimate inherits `mu_0, mu_1` to first order: shift `mu_0(x) -> mu_0(x) - c/n^{0.25+xi}` and `mu_1(x) -> mu_1(x) + c/n^{0.25+xi}` and, by inspection of how it imputes and combines, its `tau-hat` shifts by `c/n^{0.25+xi}`. Those nuisance perturbations are vanishingly small on the `n^{-1/4}` scale — they'd satisfy any condition my theorem needs — yet they move `tau-hat` by an amount that can dominate the target rate `n^{-(1-2alpha)/(2(p+1-2alpha))}`. So no general guarantee on my scale can follow from those nuisance assumptions alone: its nuisance error enters first order. Here the deterministic drift is a product of the two nuisance errors, and the single-error residual channels are centered by cross-fitting; the X-learner has a raw first-order channel that nothing cancels.

Now let me make the feasible objective something I can actually hand to a standard learner, because "minimize this custom loss" should reduce to a call I already have. Look at the integrand again and complete the manipulation. For each `i`, with `Y-tilde_i = Y_i - m-hat(X_i)` and `W-tilde_i = W_i - e-hat(X_i)`,

```
[ Y-tilde_i - W-tilde_i tau(X_i) ]^2  =  W-tilde_i^2 [ Y-tilde_i / W-tilde_i - tau(X_i) ]^2.
```

That's just factoring `W-tilde_i^2` out of the bracket — algebraically trivial, but I want to be sure it's an *exact* identity and not an approximation, because the whole operational claim rests on it being equality term by term. Checked pointwise across all the simulated rows, the two sides differ by at most `1e-14` — machine precision, so it's an exact rewrite, not a near-one. So the R-loss is *identically* a weighted least-squares regression: regress the pseudo-outcome `Y-tilde_i / W-tilde_i` on `X_i`, with sample weight `W-tilde_i^2`. Operationally, that matters: every learner I care about (boosting, ridge, a net, a weighted random forest) accepts sample weights, so I can minimize the R-loss by a single ordinary weighted-regression call, no custom-loss surgery.

And it reconciles with the failed U-learner, which is worth pinning down because it explains the instability I saw earlier. The U-learner *was* regressing the pseudo-outcome `Y-tilde/W-tilde`, but with weight 1, so the points where `W-tilde` is near zero — pseudo-outcomes with huge variance — got full say. Here the weight is `W-tilde^2`, which is *exactly* small where `W-tilde` is small, so it downweights precisely the high-variance pseudo-outcomes; the `1/W-tilde` blowup in the pseudo-outcome is cancelled by the `W-tilde^2` weight, leaving only the structurally informative reweighting. I already have the numbers to see this pay off: in that poor-overlap window where the unweighted (U-learner) local mean was 0.049 against a true effect of ~0.125 with variance over 200, the `W-tilde^2`-weighted mean comes out 0.116 — close to the truth, with the explosive points correctly suppressed. Same identity, but reading it as weighted regression rather than as a transformed-outcome regression is what makes it stable.

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
