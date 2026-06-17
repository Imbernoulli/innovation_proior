# Neural Tangent Kernel

For a network realization map `F(theta)=f_theta`, define the tangent kernel

```text
Theta(theta) = sum_p partial_{theta_p} F(theta) tensor partial_{theta_p} F(theta).
```

Parameter gradient descent on any functional cost `C(f)` induces function-space motion

```text
partial_t f_theta(t) = -nabla_{Theta(theta(t))} C |_{f_theta(t)}.
```

For fully connected depth-`L` networks with

```text
alpha_tilde^(l+1)(x) = n_l^(-1/2) W^(l) alpha^(l)(x) + beta b^(l),
```

iid Gaussian initialization, fixed depth, and hidden widths tending to infinity, Jacot, Gabriel, and Hongler prove that the tangent kernel converges at initialization and remains constant during training:

```text
Theta^(L)(t) -> Theta_infty^(L) tensor Id.
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

Thus infinite-width training is kernel gradient flow with this limiting tangent kernel. For least squares,

```text
f_t = f_star + exp(-t Pi)(f_0 - f_star),
```

where `Pi` is the kernel operator on the training distribution. At convergence, the mean predictor is ridgeless kernel regression with `Theta_infty`, and convergence is fastest along the largest kernel principal components.

The proof depends on the scaling: each hidden activation and individual parameter contribution moves only `O(width^(-1/2))`, while the aggregate Jacobian Gram matrix remains order one. That is the lazy or linearized infinite-width regime. The contribution beyond earlier neural-network kernels is that the kernel is not merely a static covariance of random networks; it is the tangent kernel that actually generates the gradient-descent trajectory.

