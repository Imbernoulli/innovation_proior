# The implicit bias of gradient descent on separable data

## Problem

Train an unregularized linear classifier on a linearly separable dataset by minimizing the logistic loss (or any exponential-tailed monotone loss, or multiclass cross-entropy) with gradient descent, run all the way to zero training loss. There is no finite minimizer — to drive the loss to zero the weight norm $\lVert\mathbf{w}(t)\rVert\to\infty$ — and there are infinitely many zero-error separating directions. **Which one does gradient descent select, how fast, and why?** This is the question of the optimizer's *implicit regularization*: the bias that explains generalization with no explicit penalty in the objective.

## Key idea

The exponential tail of the loss makes the gradient $-\nabla\mathcal{L}(\mathbf{w})=\sum_n(-\ell'(\mathbf{w}^\top\mathbf{x}_n))\mathbf{x}_n$ asymptotically dominated by the smallest-margin points (their exponents decay slowest). Those are the support vectors. Re-normalized to unit minimal margin, the limiting direction is forced to satisfy the **KKT conditions of the hard-margin SVM**, so gradient descent converges in direction to the $L_2$ **maximum-margin** separator — independent of initialization and step size. The convergence is logarithmic in $t$, exponentially slower than the loss, which is why optimizing past zero training error keeps helping.

## Setup and assumptions

Data $\{\mathbf{x}_n\}_{n=1}^N$, relabeled so separability means $\exists\mathbf{w}_*:\forall n\ \mathbf{w}_*^\top\mathbf{x}_n>0$. Loss $\mathcal{L}(\mathbf{w})=\sum_n\ell(\mathbf{w}^\top\mathbf{x}_n)$, with $\ell$ positive, $\beta$-smooth, monotone strictly decreasing to $0$, and $-\ell'(u)$ having a **tight exponential tail**: constants $\mu_\pm>0$ with
$$(1-e^{-\mu_- u})e^{-u}\ \le\ -\ell'(u)\ \le\ (1+e^{-\mu_+ u})e^{-u}\quad(\text{large }u),$$
after absorbing leading constants by rescaling $\mathbf{x}_n,\eta$. Logistic $\ell(u)=\log(1+e^{-u})$, exponential $\ell(u)=e^{-u}$, and probit all qualify. Gradient descent: $\mathbf{w}(t+1)=\mathbf{w}(t)-\eta\nabla\mathcal{L}(\mathbf{w}(t))$ with $\eta<2\beta^{-1}\sigma_{\max}^{-2}(\mathbf{X})$, any $\mathbf{w}(0)$.

## Main theorem

$$\boxed{\ \mathbf{w}(t)=\hat{\mathbf{w}}\,\log t+\boldsymbol{\rho}(t),\qquad \hat{\mathbf{w}}=\arg\min_{\mathbf{w}\in\mathbb{R}^d}\lVert\mathbf{w}\rVert^2\ \ \text{s.t.}\ \ \forall n:\ \mathbf{w}^\top\mathbf{x}_n\ge1\ }$$

where $\boldsymbol{\rho}(t)$ is bounded for almost every dataset and grows at most as $O(\log\log t)$ in measure-zero degenerate cases. Consequently
$$\lim_{t\to\infty}\frac{\mathbf{w}(t)}{\lVert\mathbf{w}(t)\rVert}=\frac{\hat{\mathbf{w}}}{\lVert\hat{\mathbf{w}}\rVert}.$$

The vector $\hat{\mathbf{w}}$ is the hard-margin SVM solution, supported on the support vectors $\mathcal{S}=\{n:\hat{\mathbf{w}}^\top\mathbf{x}_n=1\}$ via the KKT conditions
$$\hat{\mathbf{w}}=\sum_n\alpha_n\mathbf{x}_n,\quad \big(\alpha_n\ge0,\ \hat{\mathbf{w}}^\top\mathbf{x}_n=1\big)\ \text{OR}\ \big(\alpha_n=0,\ \hat{\mathbf{w}}^\top\mathbf{x}_n>1\big).$$

## Rates

$$\Big\lVert\frac{\mathbf{w}(t)}{\lVert\mathbf{w}(t)\rVert}-\frac{\hat{\mathbf{w}}}{\lVert\hat{\mathbf{w}}\rVert}\Big\rVert=O\!\Big(\tfrac{1}{\log t}\Big)\ \text{almost everywhere},\quad O\!\Big(\tfrac{\log\log t}{\log t}\Big)\ \text{for all datasets},$$
$$1-\frac{\mathbf{w}(t)^\top\hat{\mathbf{w}}}{\lVert\mathbf{w}(t)\rVert\,\lVert\hat{\mathbf{w}}\rVert}=O\!\Big(\tfrac{1}{\log^2 t}\Big)\ \text{almost everywhere},\quad O\!\Big((\tfrac{\log\log t}{\log t})^2\Big)\ \text{for all datasets},$$
$$\tfrac{1}{\lVert\hat{\mathbf{w}}\rVert}-\frac{\min_n\mathbf{x}_n^\top\mathbf{w}(t)}{\lVert\mathbf{w}(t)\rVert}=O\!\Big(\tfrac{1}{\log t}\Big),$$
$$\mathcal{L}(\mathbf{w}(t))=O\!\Big(\tfrac{1}{t}\Big),\qquad \mathcal{L}_{\text{val}}(\mathbf{w}(t))=\Omega(\log t)\ \text{ if some validation point has }\hat{\mathbf{w}}^\top\mathbf{x}<0.$$
Degenerate datasets pick up a $\log\log t$ factor in the direction/angle rates only; margin and loss rates are unchanged. These bounds are tight except possibly for those $\log\log t$ factors: for $\mathbf{x}=(1,0)$, exp-loss gradient flow integrates exactly to $w_1(t)=\log(t+e^{w_1(0)})$, $w_2(t)=w_2(0)$.

## Proof sketch

1. **GD reaches zero loss with diverging, all-positive margins.** No finite critical point exists ($\mathbf{w}_*^\top\nabla\mathcal{L}=\sum_n\ell'(\mathbf{w}^\top\mathbf{x}_n)(\mathbf{w}_*^\top\mathbf{x}_n)<0$ strictly). The descent lemma for $\beta$-smooth $\mathcal{L}$ gives $\sum_t\lVert\nabla\mathcal{L}\rVert^2<\infty$, so $\nabla\mathcal{L}\to0$; with no finite critical point this forces $\lVert\mathbf{w}(t)\rVert\to\infty$ and $\forall n:\mathbf{w}(t)^\top\mathbf{x}_n\to\infty$.

2. **Why the SVM direction.** With $\ell=e^{-u}$, $-\nabla\mathcal{L}=\sum_n e^{-\mathbf{w}^\top\mathbf{x}_n}\mathbf{x}_n$. Writing $\mathbf{w}=g(t)\mathbf{w}_\infty+\boldsymbol{\rho}$, $g\to\infty$, the exponents $-g\,\mathbf{w}_\infty^\top\mathbf{x}_n$ make all but the **smallest-margin** points exponentially negligible, so $-\nabla\mathcal{L}$ becomes a non-negative combination of support vectors; hence $\mathbf{w}_\infty$ (scaled to unit minimal margin, $\hat{\mathbf{w}}$) satisfies the SVM KKT conditions above.

3. **Why $\log t$ and the residual bound.** Self-consistency $\dot g\sim e^{-g}\Rightarrow g=\log t$. Define $\mathbf{r}(t)=\mathbf{w}(t)-\hat{\mathbf{w}}\log t-\tilde{\mathbf{w}}$ with the offset $\tilde{\mathbf{w}}$ chosen so $\eta e^{-\tilde{\mathbf{w}}^\top\mathbf{x}_n}=\alpha_n$ ($n\in\mathcal{S}$), making $\eta\sum_{\mathcal{S}}e^{-\tilde{\mathbf{w}}^\top\mathbf{x}_n}\mathbf{x}_n=\hat{\mathbf{w}}$ so the gradient's leading $\tfrac1t\hat{\mathbf{w}}$ cancels $\tfrac{d}{dt}(\hat{\mathbf{w}}\log t)$. Continuous-time exp-loss: $\tfrac12\tfrac{d}{dt}\lVert\mathbf{r}\rVert^2=\frac1t\sum_{\mathcal{S}}e^{-\tilde{\mathbf{w}}^\top\mathbf{x}_n}(e^{-\mathbf{x}_n^\top\mathbf{r}}-1)(\mathbf{x}_n^\top\mathbf{r})+(\text{non-}\mathcal{S})\le C\,t^{-\theta}$, using $z(e^{-z}-1)\le0$ for the support terms and $\theta=\min_{n\notin\mathcal{S}}\hat{\mathbf{w}}^\top\mathbf{x}_n>1$ for the interior terms; $\int t^{-\theta}<\infty$ bounds $\mathbf{r}$. Discrete logistic case: expand $\lVert\mathbf{r}(t+1)\rVert^2$; the squared step is $\le\eta^2\lVert\nabla\mathcal{L}\rVert^2+\lVert\hat{\mathbf{w}}\rVert^2 t^{-2}$ (summable), and the cross term $(\mathbf{r}(t+1)-\mathbf{r}(t))^\top\mathbf{r}(t)\le C_1 t^{-\min(\theta,\,1+1.5\mu_+,\,1+0.5\mu_-)}$ (all exponents $>1$, summable) via a case analysis on the sign/size of $\mathbf{x}_n^\top\mathbf{r}$ using the two-sided tail. So $\lVert\mathbf{r}(t)\rVert$ is bounded in the non-degenerate case where the finite offset exists.

4. **Genericity.** $\boldsymbol{\alpha}_{\mathcal{S}}=(\mathbf{X}_{\mathcal{S}}^\top\mathbf{X}_{\mathcal{S}})^{-1}\mathbf{1}$ is a rational function of the data; $\alpha_n=0$ only on a measure-zero polynomial root set, so generically all $\alpha_n>0$, $\tilde{\mathbf{w}}$ exists, and $\boldsymbol{\rho}$ is bounded. If the support vectors span the data, an improved bound $(\mathbf{r}(t+1)-\mathbf{r}(t))^\top\mathbf{r}(t)\le-C_2 t^{-1}$ when $\lVert\mathbf{P}_1\mathbf{r}\rVert\ge\epsilon_1$ forces $\lVert\mathbf{r}\rVert\to0$, i.e. $\boldsymbol{\rho}(t)\to\tilde{\mathbf{w}}$.

5. **Rates.** Normalize and expand $1/\sqrt{1+x}=1-\tfrac{x}{2}+\tfrac{3x^2}{8}+O(x^3)$: the bounded residual contributes $(\mathbf{I}-\hat{\mathbf{w}}\hat{\mathbf{w}}^\top/\lVert\hat{\mathbf{w}}\rVert^2)\boldsymbol{\rho}(t)/(\lVert\hat{\mathbf{w}}\rVert\log t)+O(\log^{-2}t)$, giving the generic $1/\log t$ direction rate; the loss is $\tfrac1t\sum_{\mathcal{S}}e^{-\boldsymbol{\rho}^\top\mathbf{x}_n}+O(t^{-\min(\theta,1+\mu_+)})=O(1/t)$; a misclassified validation point gives $\mathcal{L}_{\text{val}}\ge-\log t\,\hat{\mathbf{w}}^\top\mathbf{x}_k-\boldsymbol{\rho}^\top\mathbf{x}_k=\Omega(\log t)$.

## Extensions

- **Multiclass cross-entropy:** $\mathbf{w}_k(t)=\hat{\mathbf{w}}_k\log t+\boldsymbol{\rho}_k$, $\hat{\mathbf{w}}_k$ solving the $K$-class SVM $\min\sum_k\lVert\mathbf{w}_k\rVert^2$ s.t. $\mathbf{w}_{y_n}^\top\mathbf{x}_n\ge\mathbf{w}_k^\top\mathbf{x}_n+1$, under the assumption that the offset equations $\eta e^{-\mathbf{x}_n^\top(\tilde{\mathbf{w}}_{y_n}-\tilde{\mathbf{w}}_k)}=\alpha_{n,k}$ have a solution.
- **Deep nets, single layer with frozen activation pattern:** the output is linear in that layer's weights ($u_n=\tilde{\mathbf{x}}_{l,n}^\top\mathbf{w}_l$), so the theorem applies to the effective features — the layer's direction converges to the $L_2$ max-margin separator (non-convex except at the last layer).
- **Algorithm dependence:** momentum and stochastic gradients preserve the same bias under the analyzed conditions; coordinate descent (AdaBoost on exp loss) yields the $L_1$ max margin instead (Telgarsky 2013); adaptive methods (AdaGrad/Adam) distort the geometry and need not reach the $L_2$ max margin.

## Takeaway for practice

Do not early-stop on a plateauing training loss or a *rising validation loss* — both are artifacts of $\lVert\mathbf{w}\rVert\to\infty$, not overfitting; the margin (and often the validation 0–1 error) is still improving at the $1/\log t$ rate. Monitor the validation **classification error**, not the validation loss.

```python
import numpy as np
# Exact asymptotics on x=(1,0): exp-loss gradient flow -> w1(t)=log(t+e^{w1(0)}), w2 frozen.
w0 = np.array([0.3, 0.7])
t  = np.logspace(0, 8, 9)
W  = np.stack([np.log(t + np.exp(w0[0])), np.full_like(t, w0[1])])
direction = W / np.linalg.norm(W, axis=0)        # -> (1,0) = max-margin direction, at rate 1/log t
print(np.round(direction[:, ::4], 4))            # t = 1, 1e4, 1e8
```
