I start from the obstruction, because it decides the whole design. The data are Gaussian and isotropic, and the target only depends on `V* = span(w1*, ..., wr*)`. A row `w_i` drawn uniformly on the sphere has only `||Pi* w_i|| = O(sqrt(r/d))` of its mass in that subspace. If I train a fixed random-feature model, that tiny accidental overlap is all I ever get. If I train both layers with small online updates, the useful correlation is initially so small that the dynamics spend many samples escaping the uninformative region. So I need a first-layer move whose purpose is explicitly to amplify that small overlap into order-one alignment.

The cleanest way to see the move is to freeze the readout and inspect one row. With squared loss `(1/2)(y - f_hat)^2`, the true gradient with respect to `w_i` is

```text
grad_{w_i} = (a_i/sqrt(p)) E[z sigma'(<w_i,z>) (f_hat(z) - f*(z))].
```

Equivalently, the negative gradient is

```text
g_i = (a_i/sqrt(p)) E[z sigma'(<w_i,z>) (f*(z) - f_hat(z))].
```

At initialization I want the residual to be as close to `f*` as possible. The exact theory can impose paired rows and opposite output weights, `w_i^0 = w_{p-i+1}^0` and `a_i^0 = -a_{p-i+1}^0`, so `f_hat(.; W^0, a^0) = 0`. The public notebooks often use independent small random output weights instead; then the initial output is not identically zero, but it is a small random-feature term at large width. For the derivation I use the exact paired case, because it isolates the signal:

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

Here the second sum contracts the first `k` modes of the `(k+1)`-tensor and leaves a vector. Because `C_k*` comes from the low-dimensional link, all its singular directions lie in `V*`. Contracting it with `k` copies of a random row costs `||Pi* w||^k`, so it is on the order of `d^{-k/2}` up to fixed `r` and logarithmic factors. If `ell` is the target's first nonzero Hermite degree, the leading useful vector is therefore

```text
(a_i/sqrt(p)) c_ell C_ell* x_{1..(ell-1)} (w_i^0)^{otimes (ell-1)},
```

with size about `d^{-(ell-1)/2}/p`, since `a_i = O(p^{-1/2})`. This is the key scale. An `O(1)` step cannot move the row appreciably into `V*`. To make the new target-subspace component order one, I need the learning rate to cancel both the Hermite smallness and the two factors of `p^{-1/2}`:

```text
eta = p d^{(ell-1)/2}.
```

When a single-step experiment uses `n = Theta(d^ell)` samples, the same scaling is written as `eta = Theta(p sqrt(n/d))`. That form is useful operationally: with linear-size batches, `n = Theta(d)`, it says the large-batch first-layer learning rate is order `p`; with higher leap-order one-step recovery, the batch and step both grow.

I also have to check the noise floor. The empirical average estimates the leap-order contraction. If `n = O(d^{ell-delta})`, the target-subspace signal is buried in sampling fluctuations and the post-step alignment remains vanishing, on the order of a polylog factor divided by a power of `d`. Once `n = Omega(d^ell)` and the student activation has the matching nonzero Hermite coefficient, the large step makes `||Pi* w_i^1||^2 / ||w_i^1||^2` bounded away from zero for each row with high probability. The norm can grow because the step is giant, but the learned object is the ratio, and the target-subspace part and the total gradient are scaled by the same `eta`.

One step still has a structural ceiling. The leading vector belongs to `V_ell*`, the span of the higher-order singular vectors of `C_ell*`. For `ell = 1`, `C_1*` is a vector, so all rows are pulled toward the same one-dimensional spike. More generally, one step learns only the directions visible in the first nonzero Hermite tensor. If `V_ell*` is a strict subspace of `V*`, a readout fitted afterward cannot invent the missing directions.

The way to get more directions without raising the batch to `d^2` or higher is to repeat the large-batch first-layer step with fresh data. After the first step, the rows have order-one components in the already-learned subspace. Conditioning on those components changes the next leading Hermite coefficient. If `U_t*` is the subspace learned by time `t`, define the conditioned function on `U_t*^\perp` by fixing `x in U_t*` and evaluating `f*(x + x_perp)`. The new first-order signal is

```text
mu_{U_t*,x}(f*) = E_{x_perp}[grad_{x_perp} f*_{U_t*,x}(x_perp)].
```

The next learned subspace is

```text
U_{t+1}* = U_t* + span{mu_{U_t*,x}(f*) : x in U_t*}.
```

So a direction appears at the next step exactly when it becomes linearly visible after conditioning on what has already been learned. For `f*(z)=z1+z2+z1^2-z2^2`, the first Hermite coefficient points along `v=(e1+e2)/sqrt(2)`. Rewriting with `u=<v,z>` and `s=<v_perp,z>` gives `sqrt(2)u + 2us`; after conditioning on `u=lambda`, the derivative in the orthogonal direction has mean `2 lambda`, so the second step learns `v_perp`. For `z1+z2+z1^2+z2^2`, the rewrite is `sqrt(2)u + u^2 + s^2`; conditioning leaves derivative `2s`, whose Gaussian mean is zero, so `v_perp` is not learned by finite many linear-batch staircase steps. That is not a tuning failure; it is the staircase condition.

Once the first-layer rows have moved, I should stop treating the readout as part of the nonconvex problem. With `W` fixed, the feature matrix is just `Phi = sigma(Z W.T)/sqrt(p)`, and fitting the output weights is ridge regression:

```text
min_a ||Y - Phi a||^2 + lambda ||a||^2.
```

The solution is

```text
a_hat = (Phi.T Phi + lambda I_p)^{-1} Phi.T Y
```

when `n >= p`, and by the push-through identity

```text
a_hat = Phi.T (Phi Phi.T + lambda I_n)^{-1} Y
```

when `n < p`. If the objective is written with a `1/(2n)` loss normalization, the regularization constant is rescaled before it appears in these normal equations. The implementation's `lambda` is the normal-equation ridge parameter after that scaling is absorbed. The important invariant is the feature scaling by `1/sqrt(p)` and the two correct linear-system branches.

I also keep the data split straight. The feature-learning batches are fresh across first-layer steps. The readout is fitted on an independent batch in the clean theoretical setup, so after `W` has become data-dependent I can still analyze the readout as a ridge problem on a fixed learned feature map. Reusing data may work empirically, but it is not the source-clean version.

The implementation signs now follow the derivation. If I form the true loss gradient, it contains `(f_hat - Y)` and I update `W <- W - eta * grad`. If I form the negative-gradient matrix, it contains `(Y - f_hat)` and I update `W <- W + eta * negative_grad`. The canonical notebooks use the second convention:

```text
G = (1/n) Z.T [ (Y - f_hat) outer a / sqrt(p) * sigma'(Z W.T) ],
W <- W + eta G.T.
```

There is no per-step projection of the rows back to the sphere in the reference code or in the stated update. The initial rows are on the sphere in the theory, and the notebooks use Gaussian rows with norm near one, but the giant-step dynamics are analyzed through the projection ratio after the update, not by normalizing every step. If a benchmark scaffold adds row projection for stability, that is a scaffold variation, not part of the canonical algorithm.

So the final procedure is: initialize first-layer rows near the unit sphere and freeze small output weights, optionally paired for exact zero initial output; train only `W` for one or a few large-batch steps using fresh batches and a learning rate on the `p sqrt(n/d)` scale; then freeze `W` and solve a ridge problem on the scaled post-activation features from a fresh batch. The first stage decides which subspace is reachable: `V_ell*` for one step at batch `d^ell`, or the staircase-reachable subspace for multiple linear-batch steps. The second stage reaches the best predictor available from those learned features and is bounded below by the variance of `f*` left after conditioning on the learned subspace.
