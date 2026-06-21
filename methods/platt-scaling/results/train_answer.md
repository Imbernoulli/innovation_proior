A trained binary classifier, say a support vector machine, hands me a real-valued score $f(x)$ for every input, and $\operatorname{sign}(f(x))$ makes good decisions. The trouble is that $f(x)$ is not a probability. It came out of minimizing a margin loss, $C\sum_i (1 - y_i f_i)_+ + \tfrac{1}{2}\lVert h\rVert^2$, whose entire purpose is to separate the classes and keep the machine sparse, not to report how sure it is. The ranking is good but the magnitudes are meaningless: there is no sense in which the inputs scored at $f = 1.3$ are correct $87\%$ of the time. Yet I need exactly that, because the classifier is rarely the end of the story. Its output feeds a Viterbi decoder, or a cost-sensitive decision where false positives and false negatives carry different prices, or a combination with other modules. All of those are only correct if I hand them a calibrated $P(y = +1 \mid x)$, a number where, among all the inputs I rate at confidence $p$, a fraction $p$ really are positive. A margin is not that. So the task is to learn a map from the raw score $f$ to a probability without touching the classifier, fit on a limited amount of labelled data.

The options on the table each fail in an instructive way. The route that retrains is Wahba's regularized-likelihood approach: put a logistic link on the output, $P(y=1\mid x) = 1/(1+\exp(-f(x)))$, and train the kernel machine by minimizing a penalized negative log-likelihood so the output is a posterior by construction. But the likelihood loss is never exactly zero — every example, however confidently correct, contributes a nonzero gradient and wants a nonzero kernel term — so the machine goes dense and I throw away the very sparseness that made the SVM worth using, and I would have to retrain from scratch. Whatever I do must sit *after* the frozen classifier. The textbook post-processing route is generative: estimate the two class-conditional densities $p(f\mid y=+1)$ and $p(f\mid y=-1)$ and combine them with Bayes' rule. But the cross-validated histograms of a linear SVM's out-of-sample scores are nowhere near Gaussian. They have kinks — the derivative of each density is discontinuous right at the margins $f=+1$ and $f=-1$, inherited from the non-smooth hinge cost — and between the two margins the densities decay roughly *exponentially*. Forcing Gaussians on them, as Hastie and Tibshirani do, gives either too rigid or too loose a model. With a single tied variance, Bayes' rule collapses to a sigmoid whose location is then pinned so that $P=0.5$ sits at $f=0$; but the Bayes-optimal crossing depends on the prior ratio $P(y=-1)/P(y=+1)$ and moves away from $f=0$ when the priors are skewed, so the pin is simply wrong. Untie the variances for more flexibility and the quadratic no longer cancels: the posterior becomes $1/(1+\exp(af^2+bf+c))$, *non-monotone* in $f$, capable of assigning a lower positive probability to a higher score — directly fighting the fact that the SVM was trained to push positives to large $f$. Vapnik's cosine expansion $a_0(u)+\sum_n a_n(u)\cos(nt)$ has the same monotonicity problem, is not constrained to lie in $[0,1]$, and needs a linear-system solve at every evaluation.

I propose Platt scaling. The lesson of the failures is that I should not estimate two whole densities and combine them; I should model the one thing I care about, the posterior $P(y=1\mid f)$, directly, and the empirical exponential tails dictate the family. Writing the two class-conditionals between the margins as $p(f\mid y=1)\propto e^{\gamma_1 f}$ and $p(f\mid y=-1)\propto e^{-\gamma_0 f}$ with $\gamma_0,\gamma_1>0$, and carrying priors $\pi_1,\pi_0$ through Bayes' rule,
$$P(y=1\mid f) = \frac{\pi_1 e^{\gamma_1 f}}{\pi_1 e^{\gamma_1 f} + \pi_0 e^{-\gamma_0 f}} = \frac{1}{1 + \exp\!\big(-(\gamma_0+\gamma_1)f + \log(\pi_0/\pi_1)\big)},$$
which is exactly a sigmoid in $f$. The method is therefore the two-parameter model
$$P(y=1\mid f) = \frac{1}{1 + \exp(A f + B)},$$
with $A = -(\gamma_0+\gamma_1) < 0$ setting the slope and $B = \log(\pi_0/\pi_1)$ the offset, normalizing constants absorbed into $B$. The same form has a second reading I like even more: taking logs, $\log\!\big(P/(1-P)\big) = -(Af+B)$, so the model says the classifier score is *affine in the positive-class log-odds*. This form fixes both things that broke the alternatives for free. It is automatically bounded in $[0,1]$, and it is monotone increasing in $f$ precisely when $A<0$, honoring the strong monotone prior the margin classifier induces. As a sign check, if $f$ were already the calibrated log-odds of a probability $q$, the no-op is $A=-1,\,B=0$, since $1/(1+e^{-f})=q$. Crucially I now have *two* free parameters fit to data rather than one read off a generative variance: $A$ controls how fast confidence grows with the margin, and $B$ controls where $P=0.5$ actually sits — and because $B$ carries the class-prior ratio, that crossing is free to move off $f=0$ exactly when the priors are skewed.

To fit $A$ and $B$ I use the right objective for a probability model, its Bernoulli negative log-likelihood, which here is just logistic regression of the label on the single feature $f$:
$$F(A,B) = -\sum_i \big[\, t_i \log p_i + (1-t_i)\log(1-p_i)\,\big], \qquad p_i = \frac{1}{1+\exp(A f_i + B)}.$$
Two data choices make this trustworthy. First, *which* scores: I must not reuse the SVM's own training scores, because they are biased in a structured way — support vectors are forced to $|y f|=1$, margin violators are pushed toward the margin, and for a non-linear SVM the support vectors can be a large fraction of the data. I fit instead on scores from data the classifier did not train on: a held-out fraction, or out-of-fold scores from cross-validation. Second, *what targets*: with plain $\{0,1\}$ targets a linearly separable calibration set is a disaster, because maximum likelihood drives every positive's $p_i\to 1$ and every negative's $p_i\to 0$, which the sigmoid can only achieve by becoming infinitely steep, $A\to-\infty$, with no finite optimum. Rather than add a prior or a Parzen bandwidth — each a new hyperparameter — I model the out-of-sample world as having a small residual chance of the opposite label, pinned down by Laplace's rule of succession. With a uniform prior over the true positive probability, observing $N_+$ positives gives a mean target $t_+ = (N_+ + 1)/(N_+ + 2)$ for a positive example, and symmetrically $t_- = 1/(N_- + 2)$ for a negative. These soft targets sit strictly inside $(0,1)$, so pushing $A\to-\infty$ would overshoot $t_+<1$ and *raise* the cross-entropy; the optimum becomes finite. The regularization has no extra knob, and as $N_+,N_-\to\infty$ the targets converge to $\{0,1\}$, recovering plain maximum likelihood in the large-data limit. The objective is still cross-entropy, so non-binary targets need no special optimizer.

What makes the fit clean is its convexity. Writing $s_i = Af_i+B$ and using $dp/ds = -p(1-p)$ together with $dF/dp = -(t-p)/(p(1-p))$, the gradient collapses to $dF/ds = t-p$, so
$$\frac{\partial F}{\partial A} = \sum_i f_i(t_i-p_i), \qquad \frac{\partial F}{\partial B} = \sum_i (t_i-p_i),$$
and since $\partial(t-p)/\partial s = p(1-p)$ the Hessian is
$$H = \begin{bmatrix} \sum_i f_i^2 p_i(1-p_i) & \sum_i f_i p_i(1-p_i) \\ \sum_i f_i p_i(1-p_i) & \sum_i p_i(1-p_i) \end{bmatrix}.$$
Defining $u_i = f_i\sqrt{p_i(1-p_i)}$ and $v_i = \sqrt{p_i(1-p_i)}$, this is the $2\times 2$ Gram matrix $\big[[u\cdot u, u\cdot v],[v\cdot u, v\cdot v]\big]$, whose determinant $(\sum u_i^2)(\sum v_i^2)-(\sum u_i v_i)^2 \ge 0$ by Cauchy-Schwarz. So $H$ is positive semidefinite always, hence $F$ is convex, and it is positive definite unless every $f_i$ is identical (equality in Cauchy-Schwarz needs $u\parallel v$, i.e. constant $f_i$). The optimum is therefore unique in the normal case, reachable by a damped Newton step or, since the problem is two-dimensional and convex, an off-the-shelf quasi-Newton solver. The remaining hazard is purely numerical: if $s_i$ is large and positive, $e^{s_i}$ overflows, and computing $1-p_i$ when $p_i\approx 1$ loses precision before $\log(1-p_i)$. I reformulate each NLL term as the two equivalent branches $\ell = (t-1)s + \log(1+e^s) = ts + \log(1+e^{-s})$ and pick the one whose exponent stays non-positive — $ts+\log(1+e^{-s})$ when $s\ge 0$, $(t-1)s+\log(1+e^s)$ when $s<0$ — which never forms $1-p$ and never takes $\log(0)$; the same sign guard evaluates $p$ stably. I initialize at the flat sigmoid sitting at the smoothed base rate, $A=0$ and $B=\log\!\big((N_-+1)/(N_++1)\big)$, so $1/(1+e^B)$ equals the Laplace-smoothed positive rate. If the scores are very large I divide them by a positive constant before optimizing and divide the fitted slope back afterward, which only conditions the problem since the model is linear in $f$. If calibration examples carry weights, the same weights enter the class counts, the loss sum, and the gradient.

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
