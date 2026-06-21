I propose the canonical name *implicit L2 max-margin regularization of unregularized gradient descent on separable linear classification*, because the method is not a new optimizer but rather a characterization of what ordinary fixed-step gradient descent does when it is applied to an unregularized, separable classification loss. The result says that even though no explicit penalty is placed on the weight norm, the trajectory implicitly selects the hard-margin L2 support-vector-machine direction.

The setting is a homogeneous linear classifier trained on vectors `x_1, ..., x_N` in `R^d`, where the labels have already been folded into the examples so that separability means there exists some `w_*` with `w_*^T x_n > 0` for every `n`. The training loss is a smooth classification loss such as `L(w) = sum_n ell(w^T x_n)`, where `ell` is positive, strictly decreasing, and approaches zero only as its argument goes to infinity. We run fixed-step gradient descent `w(t+1) = w(t) - eta grad L(w(t))` with a step size small enough for the standard smooth-descent inequality to hold. The loss can be driven arbitrarily close to zero, but there is no finite minimizer because scaling any separating vector by a larger positive constant keeps decreasing the loss. This means the norm of `w(t)` must diverge, and the only meaningful object that can converge is the normalized direction `w(t) / ||w(t)||`.

The theorem is that, under tight exponential-tail assumptions on `-ell'(u)`, the iterates satisfy `w(t) = what log t + rho(t)`, where `what` is the solution to the hard-margin L2 SVM problem, namely `argmin_w ||w||^2` subject to `w^T x_n >= 1` for all `n`. For almost every dataset the residual `rho(t)` is bounded, and for all separable datasets the residual grows at most like `log log t` after possibly adding smaller iterated-log corrections. Consequently, the normalized iterate converges to the L2 max-margin direction: `w(t) / ||w(t)|| -> what / ||what||`.

The rates that come out of this expansion are striking. For almost every dataset the direction error satisfies `|| w(t)/||w(t)|| - what/||what|| || = O(1/log t)`, and for all datasets the slightly weaker bound `O(log log t / log t)` holds. The normalized margin gap is also `O(1/log t)`, while the training loss decays much faster as `L(w(t)) = O(1/t)`. This separation of rates is the central phenomenon: the loss can look essentially finished while the classifier direction is still slowly sliding toward the large-margin separator. It tells us that continuing to optimize past zero training error is not merely numerical polishing; it keeps changing the normalized predictor and can improve margins and classification error.

The mechanism behind the bias is the exponential tail of the loss. For exponential loss the negative gradient is `-grad L(w) = sum_n exp(-w^T x_n) x_n`. If we write the growing iterate as `w(t) = g(t) w_infty + rho(t)` with `g(t)` large, then each example is weighted by `exp(-g(t) w_infty^T x_n) exp(-rho(t)^T x_n)`. The points with the smallest value of `w_infty^T x_n` have the least negative exponent, so as `g(t)` grows they dominate the gradient and all larger-margin points are exponentially suppressed. The late gradient therefore concentrates on the support vectors, the points closest to the decision boundary. The limiting direction, rescaled so that its smallest margin equals one, must be a nonnegative combination of these support vectors, and it must satisfy `what^T x_n = 1` on the support vectors and `what^T x_n > 1` off the support. These are exactly the Karush-Kuhn-Tucker conditions for the hard-margin L2 SVM. The exponential tail makes the support-vector set visible, and the Euclidean update geometry of full gradient descent selects the L2 maximum-margin separator.

To make the argument self-consistent one also has to determine how fast the leading coefficient grows. On a support vector we have `what^T x_n = 1`, so the support-vector contribution to the gradient scales like `exp(-g(t))`. Matching this to the leading motion `g'(t)` gives the differential equation `g'(t) ~= exp(-g(t))`, which integrates to `g(t) ~= log t`. That logarithmic growth is why the leading term is not linear in time or a power of time; the gradient shrinks exponentially in the margin, and the fixed point of that feedback is logarithmic growth.

The proof then defines a residual `r(t) = w(t) - what log t - wtilde`, where the offset `wtilde` is chosen so that the leading support-vector gradient exactly reconstructs the SVM vector. Specifically, for support vectors `n` in the set `S`, the offset satisfies `eta exp(-x_n^T wtilde) = alpha_n`, where `alpha_n` are the SVM dual coefficients. This makes `eta sum_{n in S} exp(-x_n^T wtilde) x_n = what`, so the dangerous `what/t` term coming from the derivative of `what log t` cancels the `1/t` support-vector part of the gradient. The derivative of `||r||^2/2` then splits into a support-vector part and a non-support-vector part. The support-vector part has the sign `z(exp(-z)-1)`, which is always nonpositive, so support vectors damp the residual rather than pushing it outward. The non-support vectors decay like `t^{-theta}` where `theta = min_{n not in S} what^T x_n > 1`, which is summable. Logistic-tail corrections and discrete-step errors are also summable, so the residual stays bounded. In degenerate cases where a support-vector dual coefficient is zero, one recurses on the remaining subspace and picks up smaller iterated-log terms such as `log log t`, but the normalized direction still converges to the SVM direction.

A practical consequence is that validation loss can behave in a way that looks like overfitting but is not. If the limiting training separator misclassifies a validation point, the growing norm makes that point's logistic loss grow like `log t`, even while margins and classification error can improve. So the right observables for this asymptotic story are classification error and margin behavior, not loss alone. Training loss can be nearly zero long before the classifier direction is close to its limiting large-margin separator.

The code below illustrates the phenomenon on a small two-dimensional separable dataset. It constructs a hard-margin SVM reference direction with a simple quadratic-program solver, then runs fixed-step gradient descent on an exponential loss and compares the normalized iterate to the SVM direction. The numerical direction error should shrink slowly as training continues, while the training loss decays much faster.

```python
import numpy as np
from scipy.optimize import minimize

# Generate a small separable 2D dataset
np.random.seed(0)
N = 40
d = 2
X_pos = np.random.randn(N // 2, d) + np.array([1.5, 1.5])
X_neg = np.random.randn(N // 2, d) + np.array([-1.5, -1.5])
X = np.vstack([X_pos, X_neg])
y = np.hstack([np.ones(N // 2), -np.ones(N // 2)])
X = X * y[:, None]  # fold labels into examples

# Hard-margin L2 SVM: minimize ||w||^2 subject to w^T x_n >= 1
w0 = np.zeros(d)

def objective(w):
    return np.dot(w, w)

def jac(w):
    return 2 * w

cons = [{'type': 'ineq', 'fun': lambda w, xn=xn: np.dot(w, xn) - 1.0}
        for xn in X]
res = minimize(objective, w0, jac=jac, method='SLSQP', constraints=cons,
               options={'ftol': 1e-12, 'maxiter': 1000})
w_svm = res.x
w_svm_dir = w_svm / np.linalg.norm(w_svm)

# Fixed-step gradient descent on exponential loss
w = np.zeros(d, dtype=float)
eta = 0.05
T = 200000
direction_errors = []
losses = []
for t in range(1, T + 1):
    margins = X @ w
    grad = -np.sum(np.exp(-margins)[:, None] * X, axis=0)
    w -= eta * grad
    if t % 5000 == 0 or t == 1:
        w_dir = w / (np.linalg.norm(w) + 1e-12)
        err = np.linalg.norm(w_dir - w_svm_dir)
        loss = np.sum(np.exp(-margins))
        direction_errors.append((t, err))
        losses.append((t, loss))

print("SVM direction:", w_svm_dir)
print("Final GD direction:", w / (np.linalg.norm(w) + 1e-12))
print("Final direction error:", direction_errors[-1][1])
print("Final exponential loss:", losses[-1][1])

# A simple assertion for a sanity check
assert direction_errors[-1][1] < 0.1, "GD direction should approach SVM direction"
```

In summary, the result establishes that unregularized fixed-step gradient descent on separable linear classification implicitly solves the hard-margin L2 SVM problem. The divergence of the weights is structured: the leading term grows as `what log t`, where `what` is the maximum-margin separator, and the residual stays bounded or grows only in smaller iterated-log terms. The reason is the concentration of the late gradient on support vectors combined with the Euclidean geometry of gradient descent. The insight is not just that optimization affects generalization, but that the asymptotic gradient has an invariant support-vector geometry that exposes the optimizer's hidden regularizer.
