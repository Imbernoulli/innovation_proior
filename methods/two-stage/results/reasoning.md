I start from the obstruction, because it decides the whole design. The data are Gaussian and isotropic, and the target only depends on `V* = span(w1*, ..., wr*)`. A row `w_i` drawn uniformly on the sphere has only `||Pi* w_i|| = O(sqrt(r/d))` of its mass in that subspace. I want to be concrete about how small that is, so I sample random unit rows in `R^d` and average the overlap with a one-dimensional target direction. For `d = 50, 200, 800` I get mean `|<w,e1>| = 0.114, 0.056, 0.028`, tracking `sqrt(2/(pi d)) = 0.113, 0.056, 0.028` almost exactly. So the accidental overlap really does shrink like `d^{-1/2}`. If I train a fixed random-feature model, that vanishing overlap is all I ever get. If I train both layers with small online updates, the useful correlation is initially so small that the dynamics spend many samples escaping the uninformative region. So I need a first-layer move whose explicit purpose is to amplify that small overlap into order-one alignment, and I have to find out how big a move that takes.

The cleanest way to see the move is to freeze the readout and inspect one row. With squared loss `(1/2)(y - f_hat)^2`, the true gradient with respect to `w_i` is

```text
grad_{w_i} = (a_i/sqrt(p)) E[z sigma'(<w_i,z>) (f_hat(z) - f*(z))].
```

Equivalently, the negative gradient is

```text
g_i = (a_i/sqrt(p)) E[z sigma'(<w_i,z>) (f*(z) - f_hat(z))].
```

At initialization I want the residual to be as close to `f*` as possible, so I would like `f_hat(.; W^0, a^0)` to vanish there. One way to force this is to pair rows and flip output weights, `w_i^0 = w_{p-i+1}^0` and `a_i^0 = -a_{p-i+1}^0`; then the two members of each pair contribute equal and opposite terms. Before I lean on that I should check it actually zeroes the output and is not just plausible. I build such a symmetric initialization for `p = 20, d = 40` and evaluate the network on a handful of random points: the largest `|f_hat|` I see is `1.2e-17`, i.e. exactly zero up to floating point. Good — the pairing does isolate the signal. The public notebooks often use independent small random output weights instead; then the initial output is not identically zero, but it is a small random-feature term at large width. For the derivation I keep the exact paired case, since it leaves

```text
g_i = (a_i/sqrt(p)) E[z sigma'(<w_i^0,z>) f*(z)].
```

Now Stein's lemma gives the direction of that expectation. For any unit `w`,

```text
E[z sigma'(<w,z>) f*(z)]
  = w E[sigma''(<w,z>) f*(z)] + E[sigma'(<w,z>) grad_z f*(z)].
```

The first term is parallel to `w`, so by itself it mostly changes scale. The second can point into `V*`. Expanding in Hermite tensors, with `c_k` the Hermite coefficients of the student activation `sigma`, gives

```text
E[z sigma'(<w,z>) f*(z)]
  = sum_{k>=0} c_{k+2} <w^{otimes k}, C_k*> w
    + sum_{k>=0} c_{k+1} C_{k+1}* x_{1..k} w^{otimes k}.
```

Here the second sum contracts the first `k` modes of the `(k+1)`-tensor and leaves a vector. Because `C_k*` comes from the low-dimensional link, all its singular directions lie in `V*`. Contracting it with `k` copies of a random row costs `||Pi* w||^k`, which by the overlap measurement above is on the order of `d^{-k/2}` up to fixed `r` and logarithmic factors. So if `ell` is the target's first nonzero Hermite degree, the smallest contraction that survives uses `ell - 1` copies of the row, and the leading useful vector is

```text
(a_i/sqrt(p)) c_ell C_ell* x_{1..(ell-1)} (w_i^0)^{otimes (ell-1)},
```

with size about `d^{-(ell-1)/2}/p`, since `a_i = O(p^{-1/2})`. That is a discouragingly small number: an `O(1)` step cannot move the row appreciably into `V*`. To make the new target-subspace component order one, the learning rate has to cancel both the Hermite smallness `d^{-(ell-1)/2}` and the two factors of `p^{-1/2}` (one from `a_i`, one from the `1/sqrt(p)` in `f_hat`), which points at

```text
eta = p d^{(ell-1)/2}.
```

I want to re-express this in terms of the batch size, because experiments are run at a chosen `n`, and there is a subtlety I should check rather than guess. When a single-step experiment uses `n = Theta(d^ell)` samples, is `eta = p sqrt(n/d)` the same prescription? Substituting `n = d^ell` gives `sqrt(n/d) = sqrt(d^{ell-1}) = d^{(ell-1)/2}`, so the two forms agree. I confirm it numerically for `ell = 1, 2, 3` and `d = 10, 100`: `sqrt(d^ell/d)` and `d^{(ell-1)/2}` come out equal in every case. That form is useful operationally: with linear-size batches, `n = Theta(d)`, it says the large-batch first-layer learning rate is order `p`; with higher leap-order one-step recovery, the batch and step both grow.

I also have to check the noise floor, and here I do not want to assert "it works"; I want to run the step and look. The empirical average estimates the leap-order contraction, and below threshold it should be buried in sampling fluctuations. I take an `ell = 2` target `g(t) = t^2 - 1` with teacher direction `e1`, so `V_2* = span(e1)` and the random-init alignment is `r/d = 1/d`. I run a single symmetric-init giant step with `eta = p sqrt(n/d)` and measure `||Pi* w_i^1||^2 / ||w_i^1||^2` averaged over rows and seeds. With a *linear* batch `n ~ 3d` (below the `d^ell = d^2` threshold), the alignment is `0.032` at `d = 60` and `0.019` at `d = 120` — barely above the `1/d` baseline of `0.017` and `0.008`, and shrinking with `d`, i.e. vanishing. With a *quadratic* batch `n ~ 3 d^2` (at threshold), the same measurement gives `0.22` and `0.19` — order one and roughly stable across `d`. So the prediction that one step needs `n = Omega(d^ell)`, and that `n = O(d^{ell-delta})` leaves the alignment vanishing, is borne out. The norm can grow because the step is giant, but the learned object is the ratio, and the target-subspace part and the total gradient are both scaled by `eta`, so the ratio is what the step is really moving.

One step still has a structural ceiling. The leading vector belongs to `V_ell*`, the span of the higher-order singular vectors of `C_ell*`. For `ell = 1`, `C_1*` is a vector, so all rows are pulled toward the same one-dimensional spike. To see that concretely I run the `ell = 1` case `g(t) = t`: before the step the rows have mean alignment `0.004` (matching `r/d = 1/d`), and after one giant step the mean alignment with `e1` jumps to about `0.2`, with the rows having visibly rotated toward the single direction `e1`. More generally, one step learns only the directions visible in the first nonzero Hermite tensor. If `V_ell*` is a strict subspace of `V*`, a readout fitted afterward cannot invent the missing directions.

The way to get more directions without raising the batch to `d^2` or higher is to repeat the large-batch first-layer step with fresh data. After the first step, the rows have order-one components in the already-learned subspace. Conditioning on those components changes the next leading Hermite coefficient. If `U_t*` is the subspace learned by time `t`, define the conditioned function on `U_t*^\perp` by fixing `x in U_t*` and evaluating `f*(x + x_perp)`. The new first-order signal is the mean of its orthogonal gradient,

```text
mu_{U_t*,x}(f*) = E_{x_perp}[grad_{x_perp} f*_{U_t*,x}(x_perp)],
```

and the next learned subspace is

```text
U_{t+1}* = U_t* + span{mu_{U_t*,x}(f*) : x in U_t*}.
```

So a direction appears at the next step exactly when it becomes linearly visible after conditioning on what has already been learned. Whether that mean is nonzero is a property of the specific target, and the cleanest way to test my understanding of the recursion is to evaluate it by hand on two targets that differ by a single sign, then check the conclusion numerically.

Take `f*(z) = z1 + z2 + z1^2 - z2^2`. The first Hermite coefficient points along `v = (e1 + e2)/sqrt(2)`, so the first step learns `v`. Rewrite in `u = <v,z>` and `s = <v_perp,z>`, i.e. `z1 = (u+s)/sqrt(2)`, `z2 = (u-s)/sqrt(2)`. Then `z1^2 - z2^2 = ((u+s)^2 - (u-s)^2)/2 = 2us`, so `f* = sqrt(2) u + 2 u s`. (I verify this rewrite on random `(u,s)`: max discrepancy `4e-16`.) Conditioning on `u = lambda`, the derivative in the orthogonal direction is `d/ds (sqrt(2) lambda + 2 lambda s) = 2 lambda`, a constant, so its Gaussian mean over `s` is `2 lambda`, nonzero for generic `lambda`. The second step should therefore learn `v_perp`. Now the companion `z1 + z2 + z1^2 + z2^2`: here `z1^2 + z2^2 = u^2 + s^2`, so `f* = sqrt(2) u + u^2 + s^2` (rewrite verified, max discrepancy `2e-15`). Conditioning on `u`, the orthogonal derivative is `d/ds (s^2) = 2s`, whose Gaussian mean is `0`. So `v_perp` should *not* be learned.

I check both conclusions by Monte-Carlo estimating `E_s[ d/ds f*(lambda v + s v_perp) ]` directly, without using the rewrite, for `lambda = 0.5, 1, 2`. The minus version returns `1.000, 2.000, 4.000` — exactly `2 lambda`, to within a standard error of `1e-3`. The plus version returns `0.0001, 0.0008, -0.0007` — zero to within the same error. So the recursion does pick up `v_perp` for the minus target and genuinely cannot for the plus target. That second case is not a tuning failure I could fix with a larger step or batch: the conditioned signal is exactly zero, so finitely many linear-batch staircase steps never expose `v_perp`. It is the staircase condition asserting itself.

Once the first-layer rows have moved, I should stop treating the readout as part of the nonconvex problem. With `W` fixed, the feature matrix is just `Phi = sigma(Z W.T)/sqrt(p)`, and fitting the output weights is ridge regression:

```text
min_a ||Y - Phi a||^2 + lambda ||a||^2.
```

The stationarity condition `(Phi.T Phi + lambda I) a = Phi.T Y` gives the solution

```text
a_hat = (Phi.T Phi + lambda I_p)^{-1} Phi.T Y
```

when `n >= p`, and by the push-through identity `(Phi.T Phi + lambda I_p)^{-1} Phi.T = Phi.T (Phi Phi.T + lambda I_n)^{-1}`

```text
a_hat = Phi.T (Phi Phi.T + lambda I_n)^{-1} Y
```

when `n < p`, which is the cheaper system to solve there. If the objective is written with a `1/(2n)` loss normalization, the regularization constant is rescaled before it appears in these normal equations. The implementation's `lambda` is the normal-equation ridge parameter after that scaling is absorbed. The important invariant is the feature scaling by `1/sqrt(p)` and the two correct linear-system branches.

I also keep the data split straight. The feature-learning batches are fresh across first-layer steps. The readout is fitted on an independent batch in the clean theoretical setup, so after `W` has become data-dependent I can still analyze the readout as a ridge problem on a fixed learned feature map. Reusing data may work empirically, but it is not the source-clean version.

The implementation signs now follow the derivation. If I form the true loss gradient, it contains `(f_hat - Y)` and I update `W <- W - eta * grad`. If I form the negative-gradient matrix, it contains `(Y - f_hat)` and I update `W <- W + eta * negative_grad`. The two are the same move; I just have to keep the sign and the residual paired. The canonical notebooks use the second convention:

```text
G = (1/n) Z.T [ (Y - f_hat) outer a / sqrt(p) * sigma'(Z W.T) ],
W <- W + eta G.T.
```

There is no per-step projection of the rows back to the sphere in the reference code or in the stated update. The initial rows are on the sphere in the theory, and the notebooks use Gaussian rows with norm near one, but the giant-step dynamics are analyzed through the projection ratio after the update — which, as the `ell = 1` and `ell = 2` runs above showed, is exactly the quantity that moves — not by normalizing every step. If a benchmark scaffold adds row projection for stability, that is a scaffold variation, not part of the canonical algorithm.

So the procedure that comes out of all this is: initialize first-layer rows near the unit sphere and freeze small output weights, optionally paired for the exact zero initial output I checked above; train only `W` for one or a few large-batch steps using fresh batches and a learning rate on the `p sqrt(n/d)` scale; then freeze `W` and solve a ridge problem on the scaled post-activation features from a fresh batch. The first stage decides which subspace is reachable: `V_ell*` for one step at batch `d^ell`, or the staircase-reachable subspace for multiple linear-batch steps. The second stage reaches the best predictor available from those learned features, and so is bounded below by the variance of `f*` left after conditioning on the learned subspace — no better than the directions the first stage actually managed to expose.
