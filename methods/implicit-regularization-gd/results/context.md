# Context: optimizer bias in separable classification

## Research question

Modern over-parameterized classifiers are trained by minimizing a training loss with no explicit regularizer, are driven all the way to zero training error (and far beyond, to nearly zero training loss), and yet they generalize. This should not be possible from the capacity of the model class alone: when there are more parameters than examples, the empirical risk is wildly underdetermined — there is a continuum of weight vectors that achieve zero training error, and most of them generalize badly. Something other than the objective is choosing *which* zero-error solution we land on. The natural suspect is the optimization algorithm itself: gradient descent appears to carry an *implicit bias* toward a particular minimizer, and that bias — not any penalty term — is what controls generalization.

The precise question is: for the loss functions actually used in classification — the logistic loss and its multiclass generalization, the softmax cross-entropy — what is that bias? Concretely, take a linearly separable dataset and run plain gradient descent on the unregularized logistic loss until the loss goes to zero. *Which* separating hyperplane do we converge to, out of the infinitely many that achieve zero training error? A solution must (i) name the limiting object exactly, (ii) prove the iterates actually reach it, (iii) quantify how fast, and (iv) explain why this happens specifically for gradient descent and specifically for this family of losses.

There is a complication that makes the classification case qualitatively different from regression. For the squared loss with attainable finite minimizers, "the bias of gradient descent" has a clean answer: from the origin, gradient descent on an underdetermined least-squares problem converges to the minimum-Euclidean-norm interpolating solution. But the logistic loss has *no finite minimizer on separable data*. To send the loss to zero, every correctly-classified margin must be driven to $+\infty$, so the weight norm $\lVert\mathbf{w}\rVert$ itself diverges. "Minimum norm" is therefore vacuous here — the norm is unbounded. Only the *direction* $\mathbf{w}/\lVert\mathbf{w}\rVert$ is meaningful for prediction, and it is the asymptotics of that direction that must be characterized.

## Background

**The empirical phenomenon that frames everything.** It has become clear that the inductive bias of the optimizer, not the explicit objective, governs generalization in over-parameterized learning. Networks with far more parameters than training points, trained by gradient methods with no weight penalty, fit random labels perfectly yet generalize on real labels (Zhang, Bengio, Hardt, Recht, Vinyals 2017); search over the many zero-error minima shows most generalize poorly while gradient descent reliably finds good ones (Neyshabur, Tomioka, Srebro 2014; Neyshabur, Salakhutdinov, Srebro 2015). Early-stopped or one-pass stochastic gradient descent has a partial explanation via stability (Hardt, Recht, Singer 2016), but the puzzle persists for training run *to convergence* with no early stopping.

**Max-margin separation and the hard-margin SVM.** Among all hyperplanes separating a dataset $\{\mathbf{x}_n\}$ (relabeled so $y_n=1$, i.e. $\mathbf{w}^\top\mathbf{x}_n>0$ for all $n$), the one maximizing the minimal Euclidean distance to the data is the hard-margin support vector machine,
$$\hat{\mathbf{w}}=\arg\min_{\mathbf{w}}\lVert\mathbf{w}\rVert^2\ \ \text{s.t.}\ \ \forall n:\ \mathbf{w}^\top\mathbf{x}_n\ge 1.$$
By Lagrangian duality this solution is supported on the *support vectors* — the points achieving $\hat{\mathbf{w}}^\top\mathbf{x}_n=1$ — and satisfies the KKT conditions $\hat{\mathbf{w}}=\sum_n\alpha_n\mathbf{x}_n$ with, for each $n$, either ($\alpha_n\ge0$ and $\hat{\mathbf{w}}^\top\mathbf{x}_n=1$) or ($\alpha_n=0$ and $\hat{\mathbf{w}}^\top\mathbf{x}_n>1$). Large margin is the classical motivation for good generalization (it predates and underlies the SVM). The max-margin direction is unique.

**The smoothness/convergence machinery for gradient descent.** For a $\beta$-smooth non-negative objective $\mathcal{L}$ (gradient $\beta$-Lipschitz), gradient descent $\mathbf{w}(t+1)=\mathbf{w}(t)-\eta\nabla\mathcal{L}(\mathbf{w}(t))$ with step size $\eta<2/\beta$ satisfies $\mathcal{L}(\mathbf{w}(t+1))\le\mathcal{L}(\mathbf{w}(t))-\eta(1-\tfrac{\beta\eta}{2})\lVert\nabla\mathcal{L}(\mathbf{w}(t))\rVert^2$; summing the telescoping decrease gives $\sum_t\lVert\nabla\mathcal{L}(\mathbf{w}(t))\rVert^2<\infty$, hence $\nabla\mathcal{L}(\mathbf{w}(t))\to0$ (Ganti 2015, Thm 2). For the empirical loss $\mathcal{L}(\mathbf{w})=\sum_n\ell(\mathbf{w}^\top\mathbf{x}_n)$ with $\beta$-smooth $\ell$, the smoothness constant is $\beta\,\sigma_{\max}^2(\mathbf{X})$ where $\sigma_{\max}(\mathbf{X})$ is the largest singular value of the data matrix.

**The exponential-tail structure of classification losses.** The exponential loss $\ell(u)=e^{-u}$ and the logistic loss $\ell(u)=\log(1+e^{-u})$ are smooth, strictly decreasing, positive, and tend to zero at $+\infty$. Their negative derivatives share an *exponential tail*: $-\ell'(u)\to e^{-u}$ as $u\to\infty$ (for logistic, $-\ell'(u)=1/(1+e^u)$). On separable data the margins $\mathbf{w}^\top\mathbf{x}_n$ are all driven to $+\infty$, the regime in which this tail is the operative part of the loss; the two losses, which behave very differently on non-separable data, become indistinguishable there. A loss with a heavier (e.g. polynomial) tail does not have this structure.

**Where explicit-regularization analysis already pointed.** Rosset, Zhu, Hastie (2004) studied the *regularization path* $\mathbf{w}_\lambda=\arg\min_{\mathbf{w}}\sum_n\ell(\mathbf{w}^\top\mathbf{x}_n)+\lambda\lVert\mathbf{w}\rVert_p^p$ for exponential and logistic losses. Their Theorem 3 shows that on separable data $\mathbf{w}_\lambda/\lVert\mathbf{w}_\lambda\rVert_p\to$ the maximum-$L_p$-margin separator as $\lambda\to0$ (for $p=2$, exactly the SVM direction). The mechanism is a loss-comparison lemma: if direction $\mathbf{b}_1$ separates with strictly larger minimal margin than $\mathbf{b}_2$ (both unit norm), then for all scalings $d$ beyond a threshold, $\sum_n\ell(d\,\mathbf{b}_1^\top\mathbf{x}_n)<\sum_n\ell(d\,\mathbf{b}_2^\top\mathbf{x}_n)$, because the loss is dominated by its worst (smallest-margin) point and $e^{-d\,m_1}\ll e^{-d\,m_2}$ when $m_1>m_2$. So as the norm grows, the larger-margin direction is favored. But this is run on *explicit* infinitesimal regularization, not on the optimizer's own trajectory, and at non-vanishing $\lambda>0$ the penalized minimizer is generally *not* the max-margin solution.

## Baselines

**Minimum-norm bias of GD for squared loss.** On an underdetermined least-squares problem, gradient descent from the origin stays in the row space of the data and converges to the minimum-Euclidean-norm interpolating solution. Core idea: the gradient lives in $\mathrm{span}\{\mathbf{x}_n\}$, so $\mathbf{w}(t)-\mathbf{w}(0)$ does too, and the unique interpolant in that span is the min-norm one. *Gap:* this requires a finite, attainable minimizer. For the logistic loss there is no finite minimizer — the norm diverges — so the "min-norm interpolation" statement has no content; a direction-level statement is needed instead.

**Boosting as a regularized path (Rosset, Zhu, Hastie 2004).** Slow/$\varepsilon$-boosting approximately follows the $L_1$-regularized path of its loss; on separable data the normalized path converges to the maximum-$L_1$-margin separator. *Math:* the path limit theorem above, plus the conjecture that $\varepsilon$-boosting tracks $\hat{\mathbf{w}}_\lambda$. *Gap:* it analyzes a *path of explicit-penalty optima* (and a coordinate-descent-like procedure), not gradient descent's iterates, and only approximately; and the relevant geometry there is $L_1$, not $L_2$.

**Coordinate descent / AdaBoost on the exponential loss (Telgarsky 2013).** AdaBoost is coordinate descent on the exponential loss of a linear model; with step sizes scaled by a fixed small constant ("shrinkage"), the iterate path approximates the constrained-optimum path and converges to a maximum-$L_1$-margin classifier, for exponential and similar (tight-exponential-tail) losses. *Math:* relates the descent path to the $L_1$-constrained optima and bounds the margin gap. *Gap:* this is the bias of *coordinate* descent (an $L_1$-geometry algorithm), giving an $L_1$ max-margin limit. The behavior of plain *gradient* descent — which moves in the $L_2$ geometry — is a different question and is not addressed; nor is the exact limiting iterate or its rate for GD.

**Implicit bias of GD on matrix factorization (Gunasekar, Woodworth, Bhojanapalli, Neyshabur, Srebro 2017).** For an underdetermined least-squares objective over a matrix $X$, optimized by gradient descent on a factorization $X=UU^\top$ (a depth-2 linear net), the limit appears to be the minimum-nuclear-norm solution. *Math:* squared loss, gradient flow on the factors. *Gap:* it is a *conjecture*, proven only in restricted cases, and crucially it holds only with *infinitesimal* initialization and step size — the bias depends on the starting point and learning rate, which is exactly the fragility one would like to remove. It is also a regression (squared-loss) story.

## Evaluation settings

A small *synthetic* 2-D linearly separable dataset whose hard-margin SVM solution $\hat{\mathbf{w}}$ and support vectors can be written down exactly (e.g. a handful of points placed on a known margin plus extra interior points) lets one track, as a function of iteration $t$: the weight norm $\lVert\mathbf{w}(t)\rVert$, the training loss $\mathcal{L}(\mathbf{w}(t))$, the angle between $\mathbf{w}(t)$ and $\hat{\mathbf{w}}$, and the margin gap $1/\lVert\hat{\mathbf{w}}\rVert-\min_n\mathbf{x}_n^\top\mathbf{w}(t)/\lVert\mathbf{w}(t)\rVert$. Gradient descent is run with a fixed step size $\eta=1/\sigma_{\max}^2(\mathbf{X})$ from initialization at the origin; momentum and stochastic variants, and an adaptive method (Adam), are natural points of comparison for whether the bias is specific to GD. A larger but qualitative probe is a standard image-classification convolutional network (e.g. CIFAR-10) trained with unregularized SGD, softmax output and cross-entropy, monitoring the last-layer weight norm, training loss, validation loss, and validation error over many epochs past zero training error. Metrics: weight norm growth, training-loss decay, direction/angle convergence, margin, and the validation *loss* and validation *error* tracked separately.

## Code framework

The pieces that already exist: a data pipeline producing a separable design matrix $\mathbf{X}$ and labels, a smooth monotone loss with its derivative, a full-batch gradient-descent loop, and a closed-form SVM solver to provide the reference $\hat{\mathbf{w}}$ for measurement. What does *not* yet exist is any characterization of the limiting direction of the trajectory — that is the empty slot.

```python
import numpy as np

def logistic_loss(u):          # ell(u) = log(1 + e^{-u}); exp-tailed, smooth, decreasing
    return np.log1p(np.exp(-u))

def logistic_grad(u):          # ell'(u) = -1 / (1 + e^{u})  -> -e^{-u} as u -> infty
    return -1.0 / (1.0 + np.exp(u))

def empirical_loss(w, X):      # L(w) = sum_n ell(w^T x_n), labels folded so y_n = 1
    return logistic_loss(X @ w).sum()

def gradient(w, X):            # grad L(w) = sum_n ell'(w^T x_n) x_n
    return X.T @ logistic_grad(X @ w)

def gradient_descent(X, eta, T, w0):
    w = w0.copy()
    traj = []
    for t in range(T):
        w = w - eta * gradient(w, X)     # fixed-step GD; eta < 2/(beta sigma_max^2(X))
        traj.append(w.copy())
    return np.array(traj)

def hard_margin_svm(X):
    # argmin ||w||^2 s.t. w^T x_n >= 1, for reference / measurement only
    raise NotImplementedError

def characterize_limit_direction(traj, X):
    # TODO: what does w(t)/||w(t)|| converge to, how fast, and why?
    #   - the limiting direction (which separating hyperplane?)
    #   - how ||w(t)|| grows with t
    #   - the rate at which the direction / margin converge, vs. the loss
    pass
```
