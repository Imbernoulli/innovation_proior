Let me start from what actually breaks. I have a trained classifier, say a support vector machine, that hands me a real number `f(x)` for every input, and `sign(f(x))` makes good decisions. The trouble is that `f(x)` is not a probability. It was produced by minimizing a margin loss, `C Σ (1 - y_i f_i)_+ + (1/2)||h||^2`, whose whole purpose is to separate the classes and keep the machine sparse, not to report how sure it is. So the ranking is good but the magnitudes are meaningless: there is no sense in which "the inputs scored at `f = 1.3`" are correct 87% of the time. And I need exactly that, because the classifier is rarely the end of the story. Its output feeds a Viterbi decoder, or a cost-sensitive decision where false positives and false negatives cost different amounts, or a combination with other modules. All of those are only correct if I hand them a calibrated `P(y = +1 | x)`, a number where, among all inputs I rate at confidence `p`, a fraction `p` really are positive. A margin is not that.

So the task is to learn a map `g` from the raw score `f` to a probability, without touching the classifier. The classifier is sparse and accurate; retraining it risks both. I just want a cheap post-processor I can fit on some labelled data.

The obvious first idea is the one that retrains: change the objective. Wahba's regularized-likelihood approach does this by putting a logistic link on the output, `P(y=1|x) = 1/(1 + exp(-f(x)))`, and training the kernel machine by minimizing a penalized negative log-likelihood, `-(1/m) Σ [ ((y+1)/2) log p + ((1-y)/2) log(1-p) ] + λ||h||^2`. Then the output is a posterior by construction; nothing needs to be post-processed. But watch what the likelihood loss does to sparseness. The hinge loss `(1-yf)_+` is exactly zero for every example comfortably on the right side of the margin, which is why most `α_i` come out zero and the machine is sparse. The log-likelihood loss is never exactly zero; every example, no matter how confidently correct, contributes a nonzero gradient. Every example wants a nonzero kernel term, and the machine goes dense. I would be throwing away the one property that made the SVM worth using, and retraining from scratch to boot. Whatever I do has to sit after the frozen classifier.

Fine, post-processing. The textbook way to get a posterior from a score is generative: estimate the two class-conditional densities `p(f | y = +1)` and `p(f | y = -1)`, then turn Bayes' rule:

  P(y=1 | f) = p(f|y=1) P(y=1) / [ p(f|y=1) P(y=1) + p(f|y=-1) P(y=-1) ].

So what do those densities actually look like? I should not guess. The cross-validated histograms for a linear SVM's out-of-sample scores are nowhere near Gaussian. There are kinks: the derivative of each density is discontinuous right at the margins, `f = +1` and `f = -1`. That is not a coincidence; the training cost `(1-yf)_+` is itself non-smooth exactly at the margin, so the score distribution can inherit a discontinuity there. And between the two margins, the densities decay roughly exponentially. Hold onto that: exponential tails between the margins.

The generative route in its naive form, fitting a Gaussian to each density, is already in trouble. Hastie and Tibshirani fit Gaussians to `p(f|y=±1)`. If I force a single tied variance on both, Bayes' rule collapses to a sigmoid in `f` because the quadratic terms cancel, and then the bias gets set so that `P(y=1|f) = 0.5` sits at `f = 0`. That is a one-parameter model: the slope comes from the shared variance, the location is pinned at zero. Two things are wrong with it. First, pinning `P=0.5` at `f=0` assumes the Bayes threshold is the classifier's own boundary, but the Bayes-optimal crossing depends on the class priors `P(y=-1)/P(y=+1)`, and when those are skewed the true posterior crosses `0.5` somewhere other than `f=0`. Second, if I untie the variances to get more flexibility, the quadratic no longer cancels and the posterior becomes `1/(1 + exp(a f^2 + b f + c))`, non-monotonic in `f`. That is a real problem, not a cosmetic one: the SVM was trained to push positives to large `f` and negatives to small `f`, so I have an extremely strong prior that the positive-class probability should increase with `f`, everywhere. A model that can bend back down and assign a lower positive probability to a higher score is fighting the very thing the classifier was built to do. Vapnik's alternative, a cosine expansion `a_0(u) + Σ a_n(u) cos(nt)`, has the same monotonicity problem, is not constrained to `[0,1]`, and needs a linear-system solve at every evaluation. Too heavy, and it can produce a posterior that wiggles.

So the generative densities are awkward, and forcing Gaussians on them gives me either too rigid or too loose. I do not actually want the densities; I want the posterior `P(y=1|f)`. Why estimate two whole distributions and combine them when I can model the one thing I care about directly? I switch to the discriminative frame: pick a parametric family for `P(y=1|f)` and fit its parameters to give the best probabilities on labelled data.

Which family? The data tells me. The positive density should rise as `f` moves upward inside the margin band, and the negative density should fall as `f` moves upward. Write the two exponential tails as `p(f|y=1) ∝ exp(γ_1 f)` and `p(f|y=-1) ∝ exp(-γ_0 f)` with `γ_0, γ_1 > 0`, and put priors `π_1, π_0` in Bayes' rule:

  P(y=1|f) = π_1 e^{γ_1 f} / (π_1 e^{γ_1 f} + π_0 e^{-γ_0 f}).

Divide top and bottom by the numerator:

  P(y=1|f) = 1 / (1 + (π_0/π_1) e^{-(γ_0+γ_1) f})
           = 1 / (1 + exp(-(γ_0+γ_1) f + log(π_0/π_1))).

There it is. Two exponential class-conditionals give a sigmoid in `f`, of the form

  P(y=1|f) = 1 / (1 + exp(A f + B)),

with `A = -(γ_0 + γ_1) < 0` controlling the slope and `B = log(π_0/π_1)` controlling the offset, with normalizing constants absorbed into `B`. The same form has a second reading I like even more: `1/(1+exp(Af+B))` is exactly the model "the score `f` is proportional, up to an affine transformation, to the log-odds of the positive class." Take logs: `log(P/(1-P)) = -(A f + B)`. The sigmoid decreases as `Af+B` grows, so positive-class probability increases with `f` exactly when `A < 0`. As a sign check, if `f` were already the calibrated log-odds of a probability `q`, the no-op would be `A = -1, B = 0`, because `1/(1+e^{-f}) = q`. The form honors the monotone prior and is automatically in `[0,1]`. Both of the things that broke the untied-Gaussian and the cosine model are fixed by the form itself.

And crucially I have two free parameters, `A` and `B`, both fit to data, not one parameter read off a generative variance. `A` is the slope: how fast confidence grows with the margin, how sharp the decision is. `B` is the offset: where `P = 0.5` actually sits in score space. Because `B` carries the class-prior ratio, the crossing point is free to move away from `f = 0` exactly when the priors are skewed. Fitting both discriminatively means I am tuning the sigmoid to the real posterior rather than inheriting it from a Gaussian assumption I already know is false.

Now, how do I fit `A` and `B`? I have labelled pairs `(f_i, y_i)` with `y_i ∈ {-1, +1}`. The model gives a Bernoulli probability `p_i = 1/(1+exp(A f_i + B))` for the positive class. The right objective for fitting a probability model is its negative log-likelihood. Encode the label as a target `t_i = (y_i + 1)/2 ∈ {0,1}`, and the cross-entropy is

  min_{A,B}  -Σ_i [ t_i log p_i + (1 - t_i) log(1 - p_i) ],   p_i = 1/(1+exp(A f_i + B)).

This is logistic regression of the label on the single feature `f`. Two parameters, a smooth convex-looking objective; any decent optimizer should be enough. Before I trust that, two questions nag at me: which `(f_i, y_i)` do I fit on, and what stops this from overfitting?

Take the data question first. The lazy choice is to reuse the very examples that trained the SVM. But the training scores are biased estimates of what the SVM will output out of sample, and biased in a structured way. The support vectors sitting on the margin are forced to `|y f| = 1`; the ones that violated the margin are pushed toward it by their `α_i`; only the examples comfortably beyond the margin have scores that look like genuine test-time scores. For a linear SVM the bias is usually mild, but for a non-linear SVM the support vectors can be a large fraction of the data, and then fitting the sigmoid on the training scores can be badly biased. I need an unbiased set of `(f_i, y_i)`: scores from data the classifier did not train on. A held-out fraction reserved just for fitting the sigmoid works; cross-validation works too, by training the SVM on folds and collecting the out-of-fold score for every point.

Now the overfitting question. Suppose my calibration set happens to be linearly separable in `f`: all positives above some threshold, all negatives below. What does the plain `{0,1}`-target maximum likelihood do? It tries to drive every `p_i` for a positive to exactly 1 and every `p_i` for a negative to exactly 0. The sigmoid can only do that by becoming infinitely steep: `A -> -inf`. The likelihood keeps improving the steeper it gets, with no finite optimum. I need to regularize.

I could put a prior on `(A, B)`, but then I have a hyperparameter to tune. I could smear the scores with a Parzen window, but then I have a bandwidth to tune. I want something with no extra knob. Instead of asserting that a positive example has target probability exactly 1, model the out-of-sample world as having the same density of scores but with a finite chance of the opposite label. A point observed as positive should be treated as positive, but with a small residual probability of being negative out of sample. That residual is precisely what stops the sigmoid from needing to reach 1.

I can pin it down with the rule of succession. With a uniform prior over the true probability `q` that the label is positive, seeing `N_+` positive examples gives a posterior whose mean target for a positive example is

  t_+ = (N_+ + 1) / (N_+ + 2).

Symmetrically, for a negative example the target probability of being positive is

  t_- = 1 / (N_- + 2).

So I replace the `{0,1}` targets with these soft targets for all the data and minimize the same cross-entropy:

  t_i = (N_+ + 1)/(N_+ + 2) if y_i = +1,   t_i = 1/(N_- + 2) if y_i = -1.

Now the separable case is no longer a runaway. The targets are strictly inside `(0,1)`: a positive's target is below 1, a negative's target is above 0. Pushing `A -> -inf` would send `p_i -> 1` for positives and overshoot `t_+ < 1`, increasing the cross-entropy. The optimum becomes finite. The regularization has no hyperparameter, and as `N_+, N_- -> inf`, the soft targets converge to `{0,1}`, so I recover plain maximum likelihood in the large-data limit. The objective is still just cross-entropy between `t_i` and `p_i`; non-binary targets do not require a different optimizer.

For the fit, I write `s_i = A f_i + B`, so `p_i = 1/(1 + e^{s_i})`. I want the gradient of

  F(A,B) = -Σ_i [t_i log p_i + (1-t_i) log(1-p_i)].

With `p = 1/(1+e^s)`, `dp/ds = -p(1-p)`. Also `dF/dp = -(t/p - (1-t)/(1-p)) = -(t-p)/(p(1-p))`. Multiplying gives `dF/ds = t - p`. Therefore

  ∂F/∂A = Σ_i f_i (t_i - p_i),    ∂F/∂B = Σ_i (t_i - p_i).

The Hessian follows because `∂(t-p)/∂s = p(1-p)`:

  H = [ Σ f_i^2 p_i(1-p_i)   Σ f_i p_i(1-p_i) ;
        Σ f_i p_i(1-p_i)     Σ p_i(1-p_i) ].

Now stare at this Hessian. Define `u_i = f_i sqrt(p_i(1-p_i))` and `v_i = sqrt(p_i(1-p_i))`. Then `H = [[u·u, u·v], [v·u, v·v]]`, a Gram matrix. Its determinant is

  det H = (Σ u_i^2)(Σ v_i^2) - (Σ u_i v_i)^2 >= 0

by Cauchy-Schwarz. So `H` is positive semidefinite always, hence `F` is convex. It is positive definite unless every `f_i` is identical, because equality in Cauchy-Schwarz requires `u` and `v` to be parallel, which means `f_i` is constant wherever `v_i` is nonzero. In the normal case the optimum is unique. A damped Newton method with a line search is enough: solve `(H_k + σI) δ_k = -∇F(z_k)` with a tiny `σ`, backtrack on `α` until the sufficient-decrease condition holds, and update `z <- z + αδ`. A quasi-Newton optimizer is fine too, because the problem is only two variables and convex.

The thing that actually bites is numerical evaluation. If `s_i` is large and positive, `e^{s_i}` overflows; if I compute `1 - p_i` directly when `p_i ≈ 1`, I subtract nearly equal floating-point numbers and lose precision, then feed garbage into `log(1-p_i)`. The clean fix is to reformulate one NLL term `ℓ = -(t log p + (1-t) log(1-p))` with `p = 1/(1+e^s)`:

  ℓ = (t - 1) s + log(1 + e^s),
  ℓ = t s + log(1 + e^{-s}).

The first identity comes from `log p = -log(1+e^s)` and `log(1-p) = s - log(1+e^s)`. The second comes from `log(1+e^s) = s + log(1+e^{-s})`. Pick the branch that keeps the exponent non-positive: if `s >= 0`, use `t s + log(1+e^{-s})`; if `s < 0`, use `(t-1)s + log(1+e^s)`. This never forms `1-p` and never takes `log(0)`. The same sign guard computes `p`: if `s >= 0`, use `e^{-s}/(1+e^{-s})`; otherwise use `1/(1+e^s)`.

The starting point should be the smoothed base rate. Set `A = 0`, a flat sigmoid, and choose `B` so `p = 1/(1+e^B)` equals `(N_+ + 1)/(N_+ + N_- + 2)`. Solving gives

  B = log((N_- + 1)/(N_+ + 1)),   A = 0.

Check: `1 + e^B = (N_+ + N_- + 2)/(N_+ + 1)`, so the flat prediction is exactly the Laplace-smoothed positive rate.

Now the concrete code is small: a calibrator that takes raw frozen-classifier scores, learns `A` and `B` on held-out labels, and returns `1/(1+exp(Af+B))` for new scores. If calibration examples carry weights, the same weights should enter the class counts, the loss sum, and the gradient; with no weights, every example has weight one. If the scores are very large, I can divide them by a positive constant before optimization and divide the fitted slope back afterward; the model is linear in the score, so this is only conditioning.

```python
import numpy as np
from scipy import optimize
from sklearn.base import BaseEstimator


class CalibrationMethod(BaseEstimator):
    """Score-level sigmoid calibrator: fit 1/(1+exp(A*f+B))."""

    def __init__(self, max_abs_prediction_threshold=30.0):
        self.max_abs_prediction_threshold = max_abs_prediction_threshold
        self.a_ = None
        self.b_ = None

    @staticmethod
    def _stable_sigmoid(s):
        out = np.empty_like(s, dtype=float)
        pos = s >= 0
        z = np.exp(-s[pos])
        out[pos] = z / (1.0 + z)
        z = np.exp(s[~pos])
        out[~pos] = 1.0 / (1.0 + z)
        return out

    def fit(self, scores, labels, sample_weight=None):
        f = np.asarray(scores, dtype=float).ravel()
        y = np.asarray(labels).ravel()
        if sample_weight is None:
            w = None
        else:
            w = np.asarray(sample_weight, dtype=float).ravel()
            if w.shape[0] != y.shape[0]:
                raise ValueError("sample_weight must have one entry per label")

        scale = 1.0
        max_abs = np.max(np.abs(f)) if f.size else 0.0
        if max_abs >= self.max_abs_prediction_threshold:
            scale = max_abs
            f = f / scale

        neg = y <= 0
        if w is None:
            prior0 = float(np.sum(neg))
            prior1 = float(y.shape[0] - prior0)
        else:
            prior0 = float(w[neg].sum())
            prior1 = float(w[~neg].sum())
        t = np.empty_like(f, dtype=float)
        t[y > 0] = (prior1 + 1.0) / (prior1 + 2.0)
        t[neg] = 1.0 / (prior0 + 2.0)

        def loss_grad(AB):
            A, B = AB
            s = A * f + B
            pos = s >= 0
            loss = np.empty_like(s)
            loss[pos] = t[pos] * s[pos] + np.log1p(np.exp(-s[pos]))
            loss[~pos] = (t[~pos] - 1.0) * s[~pos] + np.log1p(np.exp(s[~pos]))
            p = self._stable_sigmoid(s)
            d = t - p
            if w is None:
                weighted_d = d
                total_loss = loss.sum()
            else:
                weighted_d = w * d
                total_loss = np.dot(w, loss)
            grad = np.array([np.dot(weighted_d, f), weighted_d.sum()], dtype=float)
            return total_loss, grad

        AB0 = np.array([0.0, np.log((prior0 + 1.0) / (prior1 + 1.0))])
        result = optimize.minimize(
            loss_grad,
            AB0,
            jac=True,
            method="L-BFGS-B",
            options={"gtol": 1e-6, "ftol": 64 * np.finfo(float).eps},
        )
        self.a_ = float(result.x[0] / scale)
        self.b_ = float(result.x[1])
        return self

    def predict_proba(self, scores):
        f = np.asarray(scores, dtype=float)
        return self._stable_sigmoid(self.a_ * f + self.b_)
```

I needed calibrated probabilities from a frozen, margin-trained classifier whose scores rank well but mean nothing as probabilities, and I refused to retrain it because the regularized-likelihood route destroys sparseness. The generative path ran aground because the real class-conditional densities are non-Gaussian and kinked at the margins; tied Gaussians give a one-parameter sigmoid wrongly pinned to `f=0`, untied ones give a non-monotone posterior, and a cosine expansion wiggles out of `[0,1]`. So I modeled the posterior directly, and the empirical exponential tails between the margins, pushed through Bayes' rule, forced `1/(1+exp(Af+B))` with `A < 0`: monotone, bounded, with a free slope `A` and a free offset `B` that absorbs the class-prior shift. I fit `A,B` by cross-entropy on held-out or cross-validated scores because the classifier's own training scores are biased toward the margins. Plain `{0,1}` targets diverge on separable data (`A -> -inf`), so I replaced them with the Laplace-rule Bayesian targets `(N_+ +1)/(N_+ +2)` and `1/(N_- +2)`, which bound the targets inside `(0,1)`, regularize with no extra hyperparameter, and converge to `{0,1}` as data grows. The Hessian is a Gram matrix, positive definite unless all scores coincide, so the fit is strictly convex with a unique optimum reachable by damped Newton or a quasi-Newton solver. The stable NLL branches never form `1-p` or take `log(0)`, and the initialization `A=0, B=log((N_-+1)/(N_++1))` starts at the smoothed base rate. The code is just that score-level fit: raw `f_i` in, calibrated positive-class probabilities out.
