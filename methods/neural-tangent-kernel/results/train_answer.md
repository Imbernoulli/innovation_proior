I propose the canonical name Neural Tangent Kernel for the object and training regime described here. The central question is how to understand gradient descent on a wide neural network not as a search through parameter space, but as a deterministic evolution in function space. The parameter vector theta is a poor coordinate system for the loss: many parameter values implement the same function, and a convex cost on predictions becomes non-convex once composed with the network realization map F: theta -> f_theta. What matters is the current function f_theta, not the particular point theta that represents it.

A small perturbation of the parameters changes the function through the Jacobian of the realization map. Each parameter direction therefore defines a feature function, the partial derivative of f_theta with respect to that parameter. These tangent features collect into a Gram kernel, the neural tangent kernel Theta_theta(x,x') = sum_p partial_theta_p f_theta(x) partial_theta_p f_theta(x'). When I run gradient descent on a cost C(f), the chain rule turns parameter updates into function-space motion governed by exactly this kernel: the prediction at an input x moves according to the inner product, under Theta_theta, between the current functional gradient and the tangent feature at x. In other words, parameter gradient descent is already kernel gradient descent, except that the kernel is attached to the current parameters and could in principle keep changing.

The first difficulty is that a moving kernel is no simpler than the original non-convex training problem. I need a reason for the tangent Gram matrix to stabilize. The clue is the NTK parameterization and the infinite-width limit. Scale every affine layer by 1/sqrt(width), and initialize the weights as independent standard Gaussians. With this scaling a single weight has a vanishing influence on any hidden representation, of order 1/sqrt(width), so one might worry that lower layers cannot learn. But the network has many such tiny directions. A single hidden preactivation moves by order 1/sqrt(width), while the sum of the corresponding tangent-feature contributions can remain order one. Learning does not come from individual parameters moving far; it comes from the aggregate tangent space driving an order-one change in the output.

At initialization this aggregate converges to a deterministic limit. For a depth-L fully connected network the limit is built from two companion recursions. The first is the neural-network Gaussian process covariance Sigma. With input dimension n_0 and bias scale beta, the base layer covariance is Sigma^(1)(x,x') = (1/n_0) x^T x' + beta^2. Each subsequent layer draws a Gaussian process with covariance Sigma^(l) and passes it through the nonlinearity, so Sigma^(l+1)(x,x') = E_{f ~ GP(0,Sigma^(l))}[sigma(f(x)) sigma(f(x'))] + beta^2. The second recursion is the derivative covariance dotSigma^(l+1)(x,x') = E_{f ~ GP(0,Sigma^(l))}[sigma'(f(x)) sigma'(f(x')]. The tangent kernel then satisfies Theta_infty^(1) = Sigma^(1) and Theta_infty^(l+1) = Theta_infty^(l) dotSigma^(l+1) + Sigma^(l+1). This is strictly richer than the static GP kernel because it remembers how changes in lower-layer parameters affect the output.

The theorem that makes the object useful is that the kernel does not drift during training. Under a Lipschitz, twice-differentiable activation with bounded second derivative, and under a stochastic boundedness condition on the integrated training direction, the tangent kernel at time t converges in probability to the same initialization limit, uniformly on finite time intervals. The proof controls how much each preactivation and each normalized weight matrix can move on a finite interval: they shift by O(width^{-1/2}), and a Gronwall argument prevents these small shifts from accumulating into a large representation change. Individual tangent features are nearly frozen, but their collective Gram matrix remains order one and drives real learning. Hence, in the infinite-width limit, gradient descent is exactly kernel gradient descent with the fixed positive-definite tangent kernel Theta_infty.

For squared loss this collapse gives an explicit linear dynamics. On a finite dataset the kernel operator Pi maps a function f to (1/N) sum_i sum_k' f_k'(x_i) K_kk'(x_i, x). The error along each eigenfunction of Pi decays exponentially with rate equal to the corresponding eigenvalue. Large-eigenvalue data directions are fitted quickly, while small-eigenvalue directions are learned only slowly. At convergence, the mean predictor is ridgeless kernel regression with Theta_infty. The random initialization contributes a centered Gaussian residual whose variance is zero on the training points, so the training set is interpolated. This explains both the inductive bias of wide networks and why gradient descent can converge globally in this regime.

The positive-definiteness result supplies the final bridge. If the limiting tangent kernel is strictly positive definite on the data, kernel gradient descent on a convex functional cannot stall away from the global minimum. For non-polynomial activations, the Hermite coefficients are rich enough that the limiting kernel is strictly positive definite on distinct inputs. Therefore the theory gives not only a descriptive infinite-width limit but also a convergence guarantee for the actual gradient-descent trajectory.

The object that must actually be computed is the closed-form recursion for the covariance and its derivative, propagated layer by layer until it produces the tangent kernel:

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

This is the same pair of companion recursions described above, made explicit layer by layer: propagate Sigma forward as an ordinary Gaussian-process covariance, propagate dotSigma forward as its derivative analogue, and combine the two into Theta_infty at each depth. The boundary condition Sigma^(1) is fixed only by the input Gram matrix and the bias scale, and every later layer is generated purely from the two preceding covariances — no reference to the actual random weights survives past this recursion, which is exactly the deterministic infinite-width limit that the training-time theorem shows is preserved at initialization.

The canonical reference implementation is the Neural Tangents library with parameterization='ntk', and it realizes this same recursion as a running update rather than a closed form. Its stax.Dense layer applies fan-in 1/sqrt(width) scaling, initializes weights as standard Gaussians, and after each dense affine step updates two stored matrices:

```text
nngp <- W_std^2 nngp + b_std^2   if a bias is present
nngp <- W_std^2 nngp             otherwise
ntk  <- nngp + W_std^2 ntk
```

nngp is the running Sigma, ntk is the running Theta, and this update is the layerwise recursion above written for one layer at a time. Its parameterization='standard' branch uses a different finite-width scaling and is not this object. The empirical API computes the finite-network NTK directly as the Jacobian outer product J(X1) J(X2)^T, which is the exact finite analog of the tangent kernel and converges to the closed-form Theta_infty above as the hidden widths grow, before training even begins — the same limit that the training-time theorem shows the actual gradient-descent trajectory continues to respect once training starts moving the parameters.
