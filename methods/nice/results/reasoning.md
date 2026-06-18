I want the maximum-likelihood route, but I do not want the usual costs that come with it. If I use an undirected model, I immediately inherit a partition function and MCMC. If I use a variational directed model, I get a good sampling story but train a lower bound with an approximate stochastic encoder. If I use an autoregressive model, I get exact likelihood but sampling becomes a long sequential chain through the coordinates. If I use an adversarial generator, I get samples but lose likelihood and exact inference. The combination I am after is exact likelihood, exact inference, and easy sampling in one object.

The only way I see to get all three at once is to make the encoder itself the change of variables. Let `h = f(x)` be a bijection between spaces of the same dimension, and put a simple prior on `h`. Then inference is just `f(x)`, sampling is draw `h` from the prior and apply `f^{-1}`, and the density is forced by conservation of mass:

```text
p_X(x) = p_H(f(x)) |det df(x)/dx|.
```

Taking logs gives

```text
log p_X(x) = log p_H(f(x)) + log |det df(x)/dx|.
```

If I choose a factorial prior, the first term becomes `sum_i log p_{H_i}(f_i(x))`. This is exactly the kind of representation objective I wanted: twist the data into coordinates that look independent under a simple prior. The determinant term has the right sign in the data-to-latent direction. If `f` contracts a data neighborhood, `|det df/dx|` is small and the likelihood pays a negative log-volume price; if it expands around high-density data, the determinant term is positive. So the model cannot win merely by shrinking the data before scoring it under the prior.

Now the obstruction is visible. For a general neural network in `D` dimensions, the Jacobian determinant costs `O(D^3)` and is numerically fragile. I need a map that is expressive like a neural network but whose determinant is readable without forming a dense matrix. Triangular matrices solve the determinant part because their determinant is the product of the diagonal entries. The challenge is to get a triangular Jacobian without forcing every weight matrix in the network to be triangular, because triangular weights would be too restrictive.

I split the coordinates into two blocks, `I_1` and `I_2`. I leave the first block alone and transform the second block using a function of the first:

```text
y_{I_1} = x_{I_1}
y_{I_2} = g(x_{I_2}; m(x_{I_1})).
```

Here `g` only needs to be invertible in its first argument once the second argument is fixed. The function `m` can be a deep neural network, because I never need to invert it. The Jacobian has block form

```text
[ I                         0 ]
[ dy_{I_2}/dx_{I_1}   dy_{I_2}/dx_{I_2} ],
```

so the determinant is only `det(dy_{I_2}/dx_{I_2})`. The entire derivative of the conditioner `m` is in the off-diagonal block and drops out of the determinant. The inverse is also direct:

```text
x_{I_1} = y_{I_1}
x_{I_2} = g^{-1}(y_{I_2}; m(y_{I_1})).
```

That is the crucial design shape: arbitrary capacity in the conditioner, a triangular Jacobian, and no inversion of the conditioner.

Now I choose the coupling law. The additive law is the most stable option:

```text
y_{I_2} = x_{I_2} + m(x_{I_1}),
x_{I_2} = y_{I_2} - m(y_{I_1}).
```

The bottom-right Jacobian block is the identity, so the determinant of this layer is exactly `1` and its log-determinant is `0`. Multiplicative and affine versions are possible in the same framework. If `y_{I_2} = x_{I_2} * b` with all entries of `b` nonzero, then the log-determinant is `sum_j log |b_j|`. If `y_{I_2} = x_{I_2} * b_1 + b_2`, the log-determinant is `sum_j log |(b_1)_j|`, again requiring nonzero scale entries. Those versions expose local volume changes inside the layer, but learned multiplicative scales can be numerically awkward. With a rectified conditioner, the additive layer stays piecewise linear and stable, so I take the additive case first.

One additive layer leaves half of the coordinates copied through. A second layer with the roles exchanged modifies the other half, but I should check whether all coordinates can influence all others. The dependency graph needs at least three alternating layers before influence can pass fully across the two blocks in both directions. Using four layers is a conservative practical choice. Each individual layer still has log-determinant zero, so the whole stack of additive layers remains volume-preserving.

That creates the next problem. A volume-preserving transformation can bend and rearrange space, but density modeling needs local contraction and expansion. I do not want to sacrifice the stable additive layers, so I put volume change in a separate top layer: a diagonal positive scaling. If the hidden state after the additive stack is `u`, set

```text
h_i = exp(s_i) u_i.
```

This layer is invertible by multiplying by `exp(-s_i)`, and its log-determinant is exactly `sum_i s_i`. Therefore the full likelihood for the additive stack plus top scaling is

```text
log p_X(x) = sum_i log p_{H_i}(f_i(x)) + sum_i s_i.
```

The sign is important. In the encoder direction the diagonal map multiplies volume by `prod_i exp(s_i)`, so the log-determinant enters with `+ sum_i s_i`. The prior term and the determinant term oppose each other: the prior prefers codes near high-density regions of the prior, while the determinant term prevents all scales from collapsing to zero. The inverse scales `sigma_i = S_ii^{-1}` can be read as a nonlinear analogue of a component spectrum: large `sigma_i` means the data varies along that latent coordinate, small `sigma_i` means the model has suppressed it.

For the prior I only need independent coordinates from a standard family. A standard Gaussian gives

```text
log p(h_i) = -0.5 * (h_i^2 + log(2*pi)).
```

A standard logistic gives

```text
log p(h_i) = -log(1 + exp(h_i)) - log(1 + exp(-h_i)).
```

The logistic score is bounded between `-1` and `1`, because the derivative is `sigmoid(-h_i) - sigmoid(h_i)`. That is easier to optimize than the Gaussian score `-h_i`, which grows without bound.

I also want the construction to contain familiar special cases. If I use an affine map `z = Lx + b` with lower-triangular `L` and a standard Gaussian prior, the log-determinant is `sum_i log |L_ii|`. Maximizing the same change-of-variables likelihood is just maximum-likelihood Gaussian fitting in a triangular parameterization, so whitening appears as a linear special case.

The variational-autoencoder connection is another consistency check. Take the reparameterized recognition model `z = g_phi(epsilon; x)` with `epsilon ~ N(0, I)`, define the standardized residual

```text
xi = (x - f_theta(z)) / sigma,
```

and put a standard Gaussian prior on `(z, xi)`. The change-of-variables log-density on the pair `(x, epsilon)` is

```text
log p(x, epsilon) = log p_H(z, xi) - D_X log sigma
                    + log |det dg_phi(epsilon; x)/d epsilon|.
```

Subtracting `log p_epsilon(epsilon)` gives the usual sampled variational objective because `log q_phi(z|x) = log p_epsilon(epsilon) - log |det dg_phi/d epsilon|`. Thus

```text
log p(x, epsilon) - log p_epsilon(epsilon)
  = log p_X|Z(x|z) + log p_Z(z) - log q_phi(z|x).
```

The signs line up: the residual scaling contributes `-D_X log sigma`, and the recognition Jacobian becomes the `-log q` term after subtracting the auxiliary noise density. This tells me the same invertible-likelihood machinery can reproduce the SGVB estimator on an augmented space, while the direct deterministic model avoids a stochastic approximate encoder for its own inference.

For images I also have to prevent a continuous density from exploiting discrete pixels. If a normalized 8-bit value is `k/255`, I replace it by `(k + u)/256` with `u` uniform on `[0, 1]`. For data stored in `[-1, 1]`, the bin width is `2/256 = 1/128`. This turns the grid into continuous observations and gives a meaningful likelihood target. When I report bits per dimension, I add the `D log 256` discrete-level constant to the negative log-likelihood and divide by `D log 2`; if a fixed preprocessing such as whitening has a constant log-determinant, that constant belongs in evaluation even if it is omitted from the training loss.

The final construction is therefore: dequantize the data, optionally apply fixed linear preprocessing, reorder or split the coordinates into two equal halves, stack additive coupling layers that alternate which half is updated, append the exponentiated diagonal scale, and train by exact maximum likelihood. Encoding is the forward transform and its accumulated log-determinant; sampling draws from the factorial prior and applies the inverse layers in reverse order. The whole point is that the hard nonlinear functions appear only as conditioners, so the determinant and inverse remain trivial while the transformation can still be highly nonlinear.
