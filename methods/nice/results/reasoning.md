I want the maximum-likelihood route, but I do not want the usual costs that come with it. If I use an undirected model, I immediately inherit a partition function and MCMC. If I use a variational directed model, I get a good sampling story but train a lower bound with an approximate stochastic encoder. If I use an autoregressive model, I get exact likelihood but sampling becomes a long sequential chain through the coordinates. If I use an adversarial generator, I get samples but lose likelihood and exact inference. The combination I am after is exact likelihood, exact inference, and easy sampling in one object, and none of the four standard families gives me all three.

So let me ask what would. If the encoder were itself a bijection `h = f(x)` between spaces of the same dimension, with a simple prior on `h`, then inference is just `f(x)`, sampling is draw `h` from the prior and apply `f^{-1}`, and the density is forced by conservation of mass:

```text
p_X(x) = p_H(f(x)) |det df(x)/dx|.
```

Taking logs gives

```text
log p_X(x) = log p_H(f(x)) + log |det df(x)/dx|.
```

If I choose a factorial prior, the first term becomes `sum_i log p_{H_i}(f_i(x))`, a coordinate-wise sum I am free to optimize. So this single object does deliver all three properties at once, which is more than any of the four families managed.

Before I get attached to it, I want to be sure the determinant term has the sign that keeps the model honest. The worry is that `f` could cheat by contracting the data into a tiny region where the prior happens to be tall, inflating `p_H(f(x))` for free. Let me see what the determinant does in that case. If `f` contracts a data neighborhood, `|det df/dx|` is small, its log is negative, and the likelihood pays a log-volume price exactly where the prior term was inflated; if `f` expands around high-density data, the determinant term is positive. The two terms pull against each other, so shrinking the data before scoring it is penalized. Good — the change of variables is self-policing, which is precisely what a lower bound never guarantees.

Now the obstruction is visible. For a general neural network in `D` dimensions, the Jacobian determinant costs `O(D^3)` and is numerically fragile. I need a map that is expressive like a neural network but whose determinant is readable without forming a dense matrix. A triangular Jacobian would solve the determinant part, because a triangular matrix's determinant is the product of its diagonal entries. But I cannot force every weight matrix in the network to be triangular — that would cripple the conditioner. I need a triangular Jacobian without a triangular network.

Here is a structure that might do it. Split the coordinates into two blocks, `I_1` and `I_2`. Leave the first block alone and transform the second block using a function of the first:

```text
y_{I_1} = x_{I_1}
y_{I_2} = g(x_{I_2}; m(x_{I_1})).
```

Here `g` only needs to be invertible in its first argument once the second argument is fixed. The function `m` can be a deep neural network, because I never need to invert it. Differentiating, the Jacobian has block form

```text
[ I                         0 ]
[ dy_{I_2}/dx_{I_1}   dy_{I_2}/dx_{I_2} ],
```

so the claim is that the determinant collapses to `det(dy_{I_2}/dx_{I_2})` and the whole derivative of the conditioner `m`, sitting in the off-diagonal block, drops out. I want to actually see this happen rather than trust the block formula, because the whole design rests on it. Take `D = 4`, blocks `I_1 = {0,1}`, `I_2 = {2,3}`, an arbitrary nonlinear conditioner `m(x_1) = (sin(x_1[0]) + x_1[1]^2, x_1[0] x_1[1])`, the additive law `y_2 = x_2 + m(x_1)`, and evaluate at `x = (0.3, -0.7, 1.1, 0.5)`. The numerical Jacobian comes out as

```text
[  1       0      0   0 ]
[  0       1      0   0 ]
[  0.9553 -1.4    1   0 ]
[ -0.7     0.3    0   1 ].
```

The off-diagonal block is full of the conditioner's derivatives, exactly as feared, but the matrix is lower-triangular with ones on the diagonal, and `det = 1.0` numerically, matching `det` of the bottom-right `2x2` block, which is also `1.0`. So the conditioner's capacity really does land entirely off the diagonal and leaves the determinant untouched. The inverse is just as direct:

```text
x_{I_1} = y_{I_1}
x_{I_2} = g^{-1}(y_{I_2}; m(y_{I_1})).
```

This gives me arbitrary capacity in the conditioner, a triangular Jacobian, and no inversion of the conditioner.

Now I choose the coupling law. The additive law is the most stable option:

```text
y_{I_2} = x_{I_2} + m(x_{I_1}),
x_{I_2} = y_{I_2} - m(y_{I_1}).
```

The bottom-right Jacobian block is the identity, so the determinant of this layer is exactly `1` and its log-determinant is `0` — which is what the `D=4` check above already showed. Multiplicative and affine versions live in the same framework. If `y_{I_2} = x_{I_2} * b` with all entries of `b` nonzero, the bottom-right block is `diag(b)` and the log-determinant is `sum_j log |b_j|`; if `y_{I_2} = x_{I_2} * b_1 + b_2`, the additive shift does not touch the diagonal and the log-determinant is `sum_j log |(b_1)_j|`, again requiring nonzero scale entries. Those versions expose local volume changes inside the layer, but learned multiplicative scales can blow up or collapse and are numerically awkward to keep nonzero. With a rectified conditioner, the additive layer stays piecewise linear and stable, so I take the additive case first.

A consequence I have to face immediately: one additive layer copies half the coordinates straight through. A second layer with the roles exchanged modifies the other half, but I cannot just assume that two layers let every coordinate influence every other. Let me trace the dependency actually. I track a boolean matrix where entry `(a, b)` is true if the current value of coordinate `a` depends on original input coordinate `b`; an additive layer that updates a block conditioned on the other block ORs the conditioning block's dependencies into the updated rows. Starting from the identity and alternating `(update I_2 | cond I_1)`, `(update I_1 | cond I_2)`, ... for `D = 4`:

```text
after layer 1:        after layer 2:        after layer 3:
[1 0 0 0]             [1 1 1 1]             [1 1 1 1]
[0 1 0 0]             [1 1 1 1]             [1 1 1 1]
[1 1 1 0]             [1 1 1 0]             [1 1 1 1]
[1 1 0 1]             [1 1 0 1]             [1 1 1 1]
```

After one layer only `I_2` sees `I_1`; after two layers `I_1` is now fully mixed but `I_2` still has a zero, because the layer that just updated `I_1` could not feed back into `I_2` yet; only after the third layer is the dependency matrix all-ones. So influence crosses fully in both directions at three alternating layers, not two — a fourth adds nothing to reachability but is a conservative practical choice for capacity. Each individual layer still has log-determinant zero, so the whole stack of additive layers remains volume-preserving.

That last fact creates the next problem. A volume-preserving transformation can bend and rearrange space, but density modeling needs local contraction and expansion — if the total Jacobian determinant is pinned to `1`, the model can never concentrate probability mass. I do not want to sacrifice the stable additive layers, so I put the volume change in a separate top layer: a diagonal positive scaling. If the hidden state after the additive stack is `u`, set

```text
h_i = exp(s_i) u_i.
```

This is invertible by multiplying by `exp(-s_i)`, and its Jacobian is `diag(exp(s_i))`, whose log-determinant should be `sum_i s_i`. Checking once on `s = (0.3, -1.1, 0.7, 2.0, -0.2, 0.05)`: the numerical `log|det diag(exp s)|` is `1.75`, and `sum_i s_i = 1.75`. They agree. Therefore the full likelihood for the additive stack plus top scaling is

```text
log p_X(x) = sum_i log p_{H_i}(f_i(x)) + sum_i s_i.
```

The sign matters. In the encoder direction the diagonal map multiplies volume by `prod_i exp(s_i)`, so the log-determinant enters with `+ sum_i s_i`. The prior term and the determinant term then oppose each other: the prior prefers codes near high-density regions of the prior, which would push every scale toward zero to crush the codes inward, while the `+ sum_i s_i` term diverges to `-infinity` as any `s_i -> -infinity` and so forbids that collapse. The inverse scales `sigma_i = S_ii^{-1}` read as a nonlinear analogue of a component spectrum: large `sigma_i` means the data varies along that latent coordinate, small `sigma_i` means the model has suppressed it.

For the prior I only need independent coordinates from a standard family. A standard Gaussian gives

```text
log p(h_i) = -0.5 * (h_i^2 + log(2*pi)).
```

A standard logistic gives

```text
log p(h_i) = -log(1 + exp(h_i)) - log(1 + exp(-h_i)).
```

I claimed the logistic is gentler to optimize because its score is bounded; let me confirm the bound rather than wave at it. The derivative of the logistic log-density is `sigmoid(-h_i) - sigmoid(h_i)`. Sampling `h` over `[-50, 50]`, this score stays in `[-1.0, 1.0]` at the extremes, whereas the Gaussian score `-h_i` is `-50` at `h_i = 50` and grows without bound. So the logistic gives a saturating gradient on outliers while the Gaussian gives an ever-steeper one — a real reason to prefer the logistic when the optimizer occasionally pushes codes far out.

I would like the construction to contain familiar special cases, both as a sanity check on the likelihood and to understand what it generalizes. Take an affine map `z = Lx + b` with lower-triangular `L` and a standard Gaussian prior. Then `log p_X(x) = -0.5 ||Lx + b||^2 - (D/2) log(2 pi) + sum_i log |L_ii|`. This should be nothing but maximum-likelihood Gaussian fitting in a triangular parameterization, with `L^T L` playing the role of the inverse covariance. To be sure I am not fooling myself, I test it numerically: draw `200000` samples from `N(mu, Sigma)` in `D = 3` with a non-diagonal `Sigma`, take the Cholesky factor of the empirical precision so that `L L^T = Sigma_emp^{-1}`, and apply `z = L^T (x - mu_emp)`. The covariance of `z` comes back as the identity to three decimals, and the log-determinant term computed two ways agrees: `sum_i log |L_ii| = 0.00641` equals `-0.5 log det Sigma_emp = 0.00641`. So whitening really is the linear special case of this likelihood, and the change-of-variables objective reduces to Gaussian MLE exactly when the map is affine — the nonlinear coupling stack is the generalization of whitening to a learned curved transform.

There is a second special case worth working through, the variational autoencoder, because if the same invertible-likelihood machinery can reproduce SGVB I will understand exactly what the deterministic model gives up and gains. Take the reparameterized recognition model `z = g_phi(epsilon; x)` with `epsilon ~ N(0, I)`, define the standardized residual

```text
xi = (x - f_theta(z)) / sigma,
```

and put a standard Gaussian prior on `(z, xi)`. The change-of-variables log-density on the pair `(x, epsilon)` is

```text
log p(x, epsilon) = log p_H(z, xi) - D_X log sigma
                    + log |det dg_phi(epsilon; x)/d epsilon|.
```

Now I do not want to assert that this matches SGVB; I want to derive the matching term by term. The standard reparameterization identity is `log q_phi(z|x) = log p_epsilon(epsilon) - log |det dg_phi/d epsilon|`, which just says the recognition Jacobian converts the noise density into the density of `z`. Subtract `log p_epsilon(epsilon)` from both sides of the change-of-variables expression. The `+ log |det dg_phi/d epsilon|` term combines with the subtracted `- log p_epsilon(epsilon)` to give exactly `- log q_phi(z|x)`. The factorial prior splits as `log p_H(z, xi) = log p_Z(z) + log p(xi)`, and since `xi` is the standardized Gaussian residual, `log p(xi) - D_X log sigma` is precisely `log p_{X|Z}(x|z)` — the residual scaling `-D_X log sigma` is the Jacobian of the `x -> xi` standardization that turns the unit-variance residual density back into the density of `x` itself. Collecting the survivors:

```text
log p(x, epsilon) - log p_epsilon(epsilon)
  = log p_X|Z(x|z) + log p_Z(z) - log q_phi(z|x),
```

which is the sampled variational objective. So the SGVB estimator is what this likelihood becomes on an augmented `(x, epsilon)` space with a stochastic encoder; the deterministic NICE-style model is the same change-of-variables principle but with `f` exact and invertible, trading the freedom to make `f` non-invertible for an exact rather than approximate inference.

For images I also have to prevent a continuous density from exploiting discrete pixels. A continuous model can place arbitrarily narrow spikes on the finite grid and drive likelihood upward without learning anything, so if a normalized 8-bit value is `k/255`, I replace it by `(k + u)/256` with `u` uniform on `[0, 1]`. For data stored in `[-1, 1]`, the bin width is `2/256 = 1/128`. This turns the grid into continuous observations and gives a meaningful likelihood target. When I report bits per dimension, I add the `D log 256` discrete-level constant to the negative log-likelihood and divide by `D log 2`; if a fixed preprocessing such as whitening has a constant log-determinant, that constant belongs in evaluation even if it is omitted from the training loss.

The final construction is therefore: dequantize the data, optionally apply fixed linear preprocessing, reorder or split the coordinates into two equal halves, stack additive coupling layers that alternate which half is updated, append the exponentiated diagonal scale, and train by exact maximum likelihood. Encoding is the forward transform and its accumulated log-determinant; sampling draws from the factorial prior and applies the inverse layers in reverse order. The hard nonlinear functions appear only as conditioners, which is what kept the determinant and inverse trivial in every check above while leaving the transformation free to be highly nonlinear. To close the loop I verify forward-then-inverse on a `D = 6` stack with a nonlinear conditioner and a random input: the round-trip reconstruction error is `5.6e-17`, machine precision, so the map really is the exact bijection the whole derivation assumed.
