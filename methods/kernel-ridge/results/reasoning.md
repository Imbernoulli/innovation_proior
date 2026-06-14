Let me start from what actually goes wrong when I try to fit these data. I have training pairs `(x_1, y_1), ..., (x_T, y_T)`, inputs in `R^n`, real targets, and I want a function that predicts well on new inputs under squared error. The cheapest thing I know is a straight line through the cloud: pick `w` minimizing `sum_t (y_t - w . x_t)^2`, predict `w . x`. The closed form is `w = (X'X)^{-1} X'y`. But two things bite immediately. The response I care about is curved, not flat, so a function that is linear in `x` simply cannot bend the way I need. And even the linear fit is fragile: if I write `X = U D V'`, the estimator's variance scales like `sum_j d_j^{-2} v_j v_j'`, so any direction with a tiny singular value `d_j` gets enormous, sign-unstable coefficients, and if two inputs are collinear or I have more inputs than examples then some `d_j = 0`, `X'X` is singular, and the estimator does not even exist. So I have a rigidity problem and a stability problem at once.

The stability problem has a known cure, and I should take it first because it is clean. Add a penalty on the size of `w`: minimize `a ||w||^2 + sum_t (y_t - w . x_t)^2` for a fixed positive `a`. Setting the gradient to zero gives `(X'X + a I) w = X'y`, so `w = (X'X + a I)^{-1} X'y`. Now `X'X` is positive semidefinite and `a I` is strictly positive definite, so their sum is positive definite, hence always invertible — the singular case is gone. The penalty shrinks the wild small-`d_j` directions (in the eigenbasis each coordinate is multiplied by `d_j^2/(d_j^2 + a)`, which is near 1 for big `d_j` and near 0 for the tiny ones), trading a little bias for a large drop in variance. Take `a = 0` and I am back to least squares; take `a` huge and `w` shrinks to zero. Good. The rigidity problem, though, is untouched — this is still linear in `x`.

The textbook route to nonlinearity is to stop feeding the raw `x` and instead feed a richer set of derived features `phi(x)`: products, powers, basis bumps. A model linear in `phi(x)` is nonlinear in `x`. So I would just run ridge on `phi` instead of `x`: `w = (Phi'Phi + a I)^{-1} Phi'y`, predict `w . phi(x)`. And here is the wall. If `phi` maps an `n`-dimensional input to degree-`d` monomials, its dimension is on the order of `n^d` — for anything nontrivial that is thousands to millions of coordinates. The matrix `Phi'Phi + a I` lives in (feature-dimension x feature-dimension); forming it, storing it, inverting it is hopeless. And if I reach for the kernel I really want — the Gaussian bump `exp(-gamma ||x - y||^2)`, whose feature map is *infinite*-dimensional — then `Phi'Phi` is an infinite matrix and the primal solution is not even a finite object. So the direct feature-space ridge is dead exactly where I need it most. I need the *same* regularized fit but expressed so that its cost depends on the *number of examples* `T`, not on the dimension of `phi`.

Let me stare at the ridge solution and look for that. `w = (Phi'Phi + a I)^{-1} Phi'y`. The thing I cannot afford is the `(feature x feature)` inverse. Is there an identity that moves the inverse to the `(example x example)` side without changing the solution? Take rectangular matrices `P` and `Q` and a positive scalar `a`, with identity matrices of the appropriate sizes. The middle product associates the same way on both sides:

  `(P Q + a I) P = P Q P + a P = P (Q P + a I)`.

If both shifted matrices are invertible, multiply this equality on the left by `(P Q + a I)^{-1}` and on the right by `(Q P + a I)^{-1}`. What remains is

  `(P Q + a I)^{-1} P = P (Q P + a I)^{-1}`.

Now set `P = Phi'` and `Q = Phi`. Then `(Phi'Phi + a I)^{-1} Phi' = Phi' (Phi Phi' + a I)^{-1}`. So

  `w = Phi' (Phi Phi' + a I)^{-1} y`.

Look at what just happened to the inverse: `Phi Phi'` is `(T x T)` — its `(s, t)` entry is `phi(x_s) . phi(x_t)`, an inner product of feature vectors, one number per pair of *training examples*. The inverse I have to compute is now `T x T`, no matter how huge or infinite the feature space is. And `w` itself is `Phi'` times a `T`-vector, i.e. `w = sum_t alpha_t phi(x_t)` where `alpha = (Phi Phi' + a I)^{-1} y`. So the optimal weight vector is automatically a *linear combination of the training feature vectors* — it lives in their span. That is not an assumption I imposed; it fell out of the algebra. It also makes sense: any component of `w` orthogonal to every `phi(x_t)` contributes nothing to any prediction `w . phi(x_t)` on training data but adds to `||w||^2`, so the penalty kills it. The fit is finite-dimensional in disguise.

Now the predictions never need `w` or `phi` explicitly. Predict at a new `x`: `w . phi(x) = (sum_t alpha_t phi(x_t)) . phi(x) = sum_t alpha_t (phi(x_t) . phi(x))`. Both places `phi` shows up — inside `Phi Phi'` and inside this prediction — it appears *only* as an inner product `phi(u) . phi(v)`. So I never have to know `phi` at all; I only need a function that returns that number. Call it `K(u, v) = phi(u) . phi(v)`. Then with `K` the `(T x T)` matrix `K_{s,t} = K(x_s, x_t)` and `k(x)` the `T`-vector `k_t = K(x_t, x)`,

  `alpha = (K + a I)^{-1} y`,    `prediction(x) = sum_t alpha_t K(x_t, x) = k(x)' alpha = y' (K + a I)^{-1} k(x)`.

This is the whole method, and it is computable: the only objects are the `T x T` kernel matrix and one regularized linear solve. The curse of dimensionality is gone because the feature dimension never enters the cost — only `T` does.

I want to re-derive this a second way, from the constrained-optimization side, because I do not fully trust an identity-pushing argument until I have seen the same answer come out of the optimality conditions, and because that route tells me *what the coefficients `alpha` mean*. Write the regularized problem in the feature space but introduce the residuals as explicit variables: minimize `a ||w||^2 + sum_t xi_t^2` subject to `y_t - w . phi(x_t) = xi_t`. That is the same objective — `xi_t` is forced to equal the residual — but now it is a constrained problem I can attack with Lagrange multipliers. Introduce one multiplier `alpha_t` per constraint and form

  `L = a ||w||^2 + sum_t xi_t^2 + sum_t alpha_t (y_t - w . phi(x_t) - xi_t)`.

By the Kuhn-Tucker conditions there are multiplier values at which the saddle point of `L` is the solution of the original constrained problem; I minimize in `w` and `xi` and maximize in `alpha`. Differentiate in `w`: `2 a w - sum_t alpha_t phi(x_t) = 0`, so

  `w = (1 / 2a) sum_t alpha_t phi(x_t)`.

There is the representer form again, now with a meaning attached — `w` is a weighted sum of the training feature vectors, each weighted by its multiplier, and the multiplier is "how hard that constraint is pulling." Differentiate in `xi_t`: `2 xi_t - alpha_t = 0`, so `xi_t = alpha_t / 2` — the multiplier on example `t` is just twice its residual, which is exactly the intuition that the points the fit cannot satisfy get the large coefficients. Now substitute both back to get a problem purely in `alpha`. Plug `w = (1/2a) sum_s alpha_s phi(x_s)` and `xi_t = alpha_t/2` into `L`. The `a ||w||^2` term becomes `a (1/2a)^2 sum_{s,t} alpha_s alpha_t phi(x_s).phi(x_t) = (1/4a) sum_{s,t} alpha_s alpha_t K_{s,t}`. The `sum xi_t^2` term becomes `sum_t alpha_t^2 / 4`. The constraint term `sum_t alpha_t (y_t - w.phi(x_t) - xi_t)` becomes `sum_t alpha_t y_t - (1/2a) sum_{s,t} alpha_t alpha_s K_{s,t} - sum_t alpha_t^2/2`. Collect the quadratic-in-`alpha` pieces: `(1/4a) alpha'K alpha - (1/2a) alpha'K alpha = -(1/4a) alpha'K alpha`, and `(1/4) alpha'alpha - (1/2) alpha'alpha = -(1/4) alpha'alpha`. So the dual objective is `-(1/4a) alpha'K alpha - (1/4) alpha'alpha + alpha'y`. Maximize: differentiate in `alpha` and set to zero, `-(1/2a) K alpha - (1/2) alpha + y = 0`. Multiply through by `2a`: `-K alpha - a alpha + 2a y = 0`, i.e. `(K + a I) alpha = 2a y`, so

  `alpha = 2a (K + a I)^{-1} y`.

These multipliers carry the factor `2a` because of how I scaled `w`. Push them into the prediction and the factor cancels: `prediction(x) = w . phi(x) = (1/2a) sum_t alpha_t K(x_t, x) = (1/2a) k(x)' alpha = (1/2a) k(x)' (2a (K + a I)^{-1} y) = k(x)' (K + a I)^{-1} y = y' (K + a I)^{-1} k(x)`. Identical to the identity-pushing answer. The two routes meet, which is what I wanted: the same closed-form predictor `y' (K + a I)^{-1} k(x)`, and the coefficient vector I will actually store is `(K + a I)^{-1} y` (the `2a` lives only in the intermediate multipliers and never reaches the prediction). I will keep the clean coefficient `c = (K + a I)^{-1} y` and predict `k(x)' c`.

Let me also do this purely in function space, both as a check and because it is the cleanest statement of *what* is being minimized. Forget `phi`; minimize directly over functions `f` in the Hilbert space the kernel induces: `sum_t (y_t - f(x_t))^2 + a ||f||^2`. The reproducing property says `f(x_t) = <f, K(., x_t)>` and `<K(., x_s), K(., x_t)> = K(x_s, x_t)`. Split any candidate `f = f_par + f_perp`, with `f_par` in `span{K(., x_t)}` and `f_perp` orthogonal to that span. Then `f(x_t) = <f, K(., x_t)> = <f_par, K(., x_t)> + <f_perp, K(., x_t)> = f_par(x_t)` because the second term is zero by orthogonality — so the data term sees only `f_par`. Meanwhile `||f||^2 = ||f_par||^2 + ||f_perp||^2 >= ||f_par||^2`, so the penalty strictly prefers `f_perp = 0`. The minimizer is in the span: `f = sum_t c_t K(., x_t)`, i.e. `f(x) = sum_t c_t K(x_t, x)`. That is the representer theorem doing exactly the same work the push-through identity did — it certifies the finite form before I have committed to a loss.

Now plug this `f` into the squared-loss objective. `f(x_i) = sum_j c_j K(x_j, x_i) = (K c)_i`, and `||f||^2 = sum_{i,j} c_i c_j K(x_i, x_j) = c' K c`. So the objective is `||y - K c||^2 + a c' K c = c'K'K c - 2 y'K c + y'y + a c'K c`, and since `K` is symmetric `K'K = K^2`, this is `c'K^2 c - 2 y'K c + y'y + a c'K c`. Differentiate in `c`: `2 K^2 c - 2 K y + 2 a K c = 0`, i.e. `K ((K + a I)c - y) = 0`. I should not silently cancel `K`, because a Gram matrix can be singular. Diagonalize it as `K = U diag(d_i) U'`, write `c = U z` and `y = U r`, and each coordinate of the objective is `(r_i - d_i z_i)^2 + a d_i z_i^2`. When `d_i > 0`, stationarity gives `(d_i + a) z_i = r_i`. When `d_i = 0`, that coordinate never changes the fitted values or the RKHS norm, so the coefficient is redundant. The always-defined shifted solve chooses one representative, because `K + a I` is strictly positive definite for `a > 0`. The canonical finite representation is therefore

  `(K + a I)c = y`.

It gives the same unique function and the same predictions as before, with no stray `2a`. Three independent derivations, one predictor. I am now confident in `(K + a I)c = y`, `prediction(x) = sum_t c_t K(x_t, x)`.

Why squared loss, and not something cleverer? Because squaring is exactly what makes the optimality condition *linear* in the unknowns: the gradient of a sum of squares is linear, so the dual condition is a linear system `(K + a I) c = y` with a one-shot closed-form solution. If I instead used a margin-type or epsilon-insensitive loss — ignore residuals smaller than some tube width — the problem becomes a quadratic program with inequality constraints; its solution is *sparse* (most `c_t` come out zero, only the points outside the tube carry weight), which is lovely at prediction time, but there is no closed form and I must run an iterative QP solver. For a medium number of examples I would much rather pay a dense coefficient vector and get a single, exact, fast linear solve than chase a sparse one through a QP. So squared loss is a deliberate choice: it buys analytic tractability at the cost of a non-sparse solution. (Same model form `f = sum c_t K(x_t, .)` either way; only the loss, and therefore the solver and the sparsity, differ.)

Now the regularizer `a I` deserves a hard look, because `a` is the one knob that is not forced. `K` is a Gram matrix of inner products, so it is positive semidefinite — but it can be singular or merely ill-conditioned: if two training inputs are identical or nearly so, two rows of `K` coincide or nearly coincide and `K` has a zero or tiny eigenvalue. With `a = 0` I would be solving `K c = y`, i.e. demanding `f(x_t) = y_t` exactly — pure interpolation through every noisy training point, with coefficients blowing up wherever `K` is near-singular and predictions oscillating wildly between data points. Adding `a I` lifts every eigenvalue of `K` by `a`, making `K + a I` strictly positive definite (so it is always invertible and a stable Cholesky factorization exists), and it shrinks the coefficients — exactly the bias-for-variance trade ridge gave me in the primal, now in the dual. So `a` controls the interpolation-to-smoothing dial: `a -> 0` interpolates and overfits, while large `a` shrinks the fitted function back toward the zero/prior-mean function. I want it positive enough to keep `K + a I` well conditioned and damp the noise, but not so large that I erase the signal; the exact value is a noise-scale or validation choice.

There is a reading of `a` that pins it down conceptually, and it comes from approaching the same predictor from the Bayesian side. Put a zero-mean Gaussian prior on the weight vector, `w ~ N(0, (1/2a) I)`, and assume the observations are the clean values plus independent Gaussian noise of variance `1/2`. Then `cov(y_s, y_t) = (1/(2a)) K_{s,t} + (1/2) delta_{s,t} = (1/(2a))(K + a I)_{s,t}`, while `cov(y_t, w . phi(x)) = (1/(2a)) k_t(x)`. Gaussian conditioning gives posterior mean `((1/(2a)) k(x))' ((1/(2a))(K + a I))^{-1} y = k(x)' (K + a I)^{-1} y`, exactly the same formula. (This is the predictor geostatisticians call Kriging.) So `a` is, up to the choice of units, the ratio of the noise variance to the prior variance: it is how much I believe the data is noisy relative to how much I believe the function is large. A noisier world, or a stronger prior that the function is small, means a bigger `a` and a smoother fit. That gives me a principled handle: `a` is not an arbitrary fudge factor, it is the noise-to-prior ratio, and the closed form I derived from penalized least squares is the Gaussian-process posterior mean. The frequentist regularized fit and the Bayesian posterior mean are the same object seen from two sides.

Now the kernel itself — which `K` to use. I want something that gives me genuine nonlinearity, is guaranteed to be a valid inner product (symmetric and positive semidefinite, so that `K + a I` is positive definite and every derivation above holds), and has as few shape parameters as possible. The polynomial `K(x, y) = (x . y + 1)^d` is valid and corresponds to all monomials up to degree `d`, but it has a discrete degree to pick and grows stiff at high degree. The one I keep coming back to is the Gaussian radial basis function

  `K(x, y) = exp(-gamma ||x - y||^2)`.

It is symmetric and positive definite for any `gamma > 0`, so it is always a legal kernel; its feature map is infinite-dimensional, so it can in principle approximate any smooth function (it is universal); and it has exactly one shape knob, `gamma`. Think about what `gamma` does. `K(x, y)` is 1 when `x = y` and decays toward 0 as `x` and `y` separate, with `gamma` setting how fast: large `gamma` makes the bump narrow, so each training point only influences predictions in a small neighborhood — a flexible, wiggly fit that can overfit; small `gamma` makes the bump broad, so every point influences everywhere — a smooth, almost-global fit that can underfit. So `gamma` is the bandwidth, the locality dial, and together with `a` it sets the bias-variance behavior of the whole regressor.

How should `gamma` default? Look at the exponent's argument, `gamma ||x - y||^2`. If my inputs have been standardized so each feature has roughly unit spread, then `||x - y||^2` is a sum over the `n` features of per-feature squared differences, each of order 1, so `||x - y||^2` grows roughly like `n`. If I left `gamma` fixed, then in high dimension the argument `gamma . n` would be large for *every* pair of distinct points and the kernel would collapse to near 0 off the diagonal — `K` would be essentially the identity and the fit would learn nothing. To keep the typical exponent order 1 regardless of how many features I have, I should scale `gamma` down with dimension: `gamma = 1 / n_features`. That makes `gamma ||x - y||^2` of order 1 for typical pairs, so the kernel is neither saturated to 0 nor stuck at 1 by default, and the regressor sees real graded similarities. So the default `gamma = 1 / n_features` is not arbitrary either — it is the value that normalizes the distance argument across dimensions.

The RBF kernel uses Euclidean distance, which means it is scale-sensitive: a feature measured in millions and a feature measured in fractions would let the big-magnitude feature dominate `||x - y||^2` entirely, drowning out the others. So before kernelizing I must put the numeric features on comparable footing — usually by centering and scaling them — or the distance, and hence every kernel value, is dictated by whichever feature happens to have the largest raw units. If categorical information is present, it has to become numeric before the kernel sees it. None of this changes the algorithm; it is just making the input representation one on which a distance-based kernel is meaningful.

Let me now assemble the computation exactly as I would run it. Fit: build the numeric feature matrix, form the `T x T` kernel matrix `K` with the chosen kernel, add `a` to its diagonal, and solve the symmetric positive-definite system `(K + a I) c = y` for the coefficient vector `c`. Because `K + a I` is positive definite I can use a Cholesky-based solver, which is the fast and numerically stable choice for a symmetric PD system; if the matrix is numerically singular despite the lift I fall back to a least-squares solve so the fit never simply crashes. Store `c` and the training inputs (I need them to evaluate the kernel against new points). Predict: for new inputs build the `(m x T)` cross-kernel `K_test` with `K_test[i, t] = K(x_new_i, x_t)`, and return `K_test c`. That is the entire method — adding `a` to the diagonal is literally the only place the regularizer enters, and one linear solve produces the whole fit.

```python
import numpy as np
from scipy import linalg


def rbf_kernel(A, B=None, gamma=None):
    # K(x, y) = exp(-gamma * ||x - y||^2), the Gaussian/RBF kernel.
    if B is None:
        B = A
    if gamma is None:
        gamma = 1.0 / A.shape[1]
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b, vectorized over all pairs of rows.
    sq_dist = (np.sum(A * A, axis=1)[:, None]
               + np.sum(B * B, axis=1)[None, :]
               - 2.0 * A @ B.T)
    return np.exp(-gamma * np.clip(sq_dist, 0.0, None))


class KernelRidge:
    """Kernel ridge regression: f(x) = sum_t c_t K(x_t, x), with the dual
    coefficients c solving (K + a I) c = y."""

    def __init__(self, alpha=1.0, kernel="rbf", gamma=None):
        self.alpha = alpha            # a: regularization = noise/prior ratio; lifts K's spectrum
        self.kernel = kernel
        self.gamma = gamma            # for RBF, None means 1/n_features in the kernel routine

    def _get_kernel(self, X, Y=None):
        if self.kernel != "rbf":
            raise ValueError("This compact version implements the RBF path.")
        return rbf_kernel(X, Y, gamma=self.gamma)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        ravel = False
        if y.ndim == 1:
            y = y.reshape(-1, 1)
            ravel = True

        K = self._get_kernel(X)                           # T x T Gram matrix
        K[np.diag_indices_from(K)] += self.alpha          # (K + alpha I): the ONLY place alpha enters
        try:
            dual_coef = linalg.solve(K, y, assume_a="pos", overwrite_a=False)
        except np.linalg.LinAlgError:
            dual_coef = linalg.lstsq(K, y)[0]             # fallback if numerically singular
        self.dual_coef_ = dual_coef.ravel() if ravel else dual_coef
        self.X_fit_ = X                                   # predict needs K(x_new, x_t)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        K_test = self._get_kernel(X, self.X_fit_)         # m x T cross-kernel K(x_new_i, x_t)
        return K_test @ self.dual_coef_                   # f(x) = sum_t c_t K(x_t, x)
```

Let me walk the causal chain back to be sure nothing was assumed. I started rigid and unstable: a linear least-squares fit cannot bend, and it explodes when inputs are collinear or outnumber examples. Ridge's `a I` fixed the stability by making the system positive definite and shrinking the unstable directions, but it was still linear in `x`. Going nonlinear by mapping into a rich feature space `phi` reintroduced the fatal cost — the primal inverse lives in feature-dimension, which is huge or infinite. The push-through identity `(Phi'Phi + a I)^{-1} Phi' = Phi'(Phi Phi' + a I)^{-1}` moved the inverse to the `(example x example)` side, revealing that the optimal `w` is automatically a combination of training feature vectors, so predictions depend on `phi` only through inner products `phi(u).phi(v)`. Replacing those by a kernel `K` removed `phi` entirely and left a predictor — `y'(K + a I)^{-1} k(x)` — whose only objects are the `T x T` kernel matrix and one regularized solve, independent of feature dimension. The Lagrangian/KKT derivation reproduced the same predictor and gave the coefficients their meaning (each multiplier is twice its residual, `w` is their kernel-weighted sum), and the direct RKHS argument via the representer theorem produced the clean `(K + a I) c = y` a third time, with null directions in a singular Gram matrix affecting coefficients but not the fitted function. Squared loss was the choice that keeps the optimality condition linear, hence the closed form, at the price of a dense coefficient vector. The regularizer `a` is the interpolation-to-smoothing dial, equal to the noise-to-prior ratio in the Gaussian view, with the same closed form as the Gaussian-process posterior mean. The Gaussian kernel supplied universal nonlinearity with one bandwidth knob `gamma`, defaulting to `1/n_features` to keep the distance argument order 1 across dimensions, after the input coordinates are put on comparable numerical scales. The fit is one Cholesky solve of `(K + a I) c = y`; prediction is one cross-kernel times `c`.
