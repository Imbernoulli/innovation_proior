# Neural Tangent Kernel

For a network realization map `F(theta)=f_theta`, define the tangent kernel

```text
Theta(theta) = sum_p partial_{theta_p} F(theta) tensor partial_{theta_p} F(theta).
```

Parameter gradient descent on a functional cost `C(f)` induces function-space motion

```text
partial_t f_theta(t) = -nabla_{Theta(theta(t))} C |_{f_theta(t)}.
```

For fully connected depth-`L` networks with

```text
alpha^(0)(x) = x
alpha_tilde^(l+1)(x) = n_l^(-1/2) W^(l) alpha^(l)(x) + beta b^(l),
alpha^(l)(x) = sigma(alpha_tilde^(l)(x))
f_theta(x) = alpha_tilde^(L)(x),
```

iid `N(0,1)` parameters, fixed input/output dimensions, fixed depth, and hidden widths `n_1,...,n_(L-1)` tending to infinity sequentially, the initialization limit is

```text
Theta^(L) -> Theta_infty^(L) tensor Id_{n_L}.
```

The scalar kernel is explicit:

```text
Sigma^(1)(x,x') = (1/n_0) x^T x' + beta^2
Sigma^(l+1)(x,x') =
  E_{f ~ GP(0,Sigma^(l))}[sigma(f(x)) sigma(f(x'))] + beta^2
dotSigma^(l+1)(x,x') =
  E_{f ~ GP(0,Sigma^(l))}[sigma'(f(x)) sigma'(f(x'))]

Theta_infty^(1)(x,x') = Sigma^(1)(x,x')
Theta_infty^(l+1)(x,x') =
  Theta_infty^(l)(x,x') dotSigma^(l+1)(x,x') + Sigma^(l+1)(x,x').
```

The training-time theorem requires `sigma` to be Lipschitz and twice differentiable with bounded second derivative, and requires the integrated training direction `int_0^T ||d_t||_{p_in} dt` to stay stochastically bounded. Under those conditions, uniformly on finite time intervals,

```text
Theta^(L)(t) -> Theta_infty^(L) tensor Id_{n_L}.
```

For ordinary gradient descent, `d_t = -d|_{f_theta(t)}`, so infinite-width training is kernel gradient descent with the limiting tangent kernel. For least squares, this is equivalently

```text
partial_t f_t = Phi_K(<f_star - f_t, .>_{p_in})
f_t = f_star + exp(-t Pi)(f_0 - f_star),
```

where `Pi(f)=Phi_K(<f,.>_{p_in})`. On a finite dataset,

```text
Pi(f)_k(x) = (1/N) sum_i sum_k' f_k'(x_i) K_kk'(x_i,x).
```

At convergence with an invertible Gram matrix, the mean predictor is ridgeless kernel regression with `Theta_infty`, with a centered Gaussian residual from initialization whose variance is zero on the training points. Convergence is fastest along the largest kernel principal components.

The proof depends on the scaling: hidden activations and individual last-layer weight columns move by `O(width^(-1/2))` on finite time intervals, while the aggregate Jacobian Gram matrix remains order one. The contribution beyond earlier neural-network kernels is that the kernel is not merely a static covariance of random networks; it is the tangent kernel that generates the gradient-descent trajectory.

The canonical Neural Tangents implementation matches this object in its `parameterization='ntk'` branch. `stax.Dense` initializes weights as standard Gaussians, initializes biases as standard Gaussians when `b_std` is provided, applies the affine map with a fan-in `1/sqrt(width)` factor, and updates layer kernels by

```text
nngp <- W_std^2 nngp + b_std^2   if a bias is present
nngp <- W_std^2 nngp             otherwise
ntk  <- nngp + W_std^2 ntk
```

after the dense affine step. Its `parameterization='standard'` branch uses a different finite-width scaling and is not the paper's NTK parameterization. The empirical API computes the finite-network NTK as `J(X1) J(X2)^T`, with `J` the Jacobian with respect to parameters; for exact finite-network dynamics, no output axes should be traced or diagonalized unless the corresponding independence/diagonal assumptions are intended.
