I propose the canonical name Neural Tangent Kernel for the object and training regime described here. The central question is how to understand gradient descent on a wide neural network not as a search through parameter space, but as a deterministic evolution in function space. The parameter vector theta is a poor coordinate system for the loss: many parameter values implement the same function, and a convex cost on predictions becomes non-convex once composed with the network realization map F: theta -> f_theta. What matters is the current function f_theta, not the particular point theta that represents it.

A small perturbation of the parameters changes the function through the Jacobian of the realization map. Each parameter direction therefore defines a feature function, the partial derivative of f_theta with respect to that parameter. These tangent features collect into a Gram kernel, the neural tangent kernel Theta_theta(x,x') = sum_p partial_theta_p f_theta(x) partial_theta_p f_theta(x'). When I run gradient descent on a cost C(f), the chain rule turns parameter updates into function-space motion governed by exactly this kernel: the prediction at an input x moves according to the inner product, under Theta_theta, between the current functional gradient and the tangent feature at x. In other words, parameter gradient descent is already kernel gradient descent, except that the kernel is attached to the current parameters and could in principle keep changing.

The first difficulty is that a moving kernel is no simpler than the original non-convex training problem. I need a reason for the tangent Gram matrix to stabilize. The clue is the NTK parameterization and the infinite-width limit. Scale every affine layer by 1/sqrt(width), and initialize the weights as independent standard Gaussians. With this scaling a single weight has a vanishing influence on any hidden representation, of order 1/sqrt(width), so one might worry that lower layers cannot learn. But the network has many such tiny directions. A single hidden preactivation moves by order 1/sqrt(width), while the sum of the corresponding tangent-feature contributions can remain order one. Learning does not come from individual parameters moving far; it comes from the aggregate tangent space driving an order-one change in the output.

At initialization this aggregate converges to a deterministic limit. For a depth-L fully connected network the limit is built from two companion recursions. The first is the neural-network Gaussian process covariance Sigma. With input dimension n_0 and bias scale beta, the base layer covariance is Sigma^(1)(x,x') = (1/n_0) x^T x' + beta^2. Each subsequent layer draws a Gaussian process with covariance Sigma^(l) and passes it through the nonlinearity, so Sigma^(l+1)(x,x') = E_{f ~ GP(0,Sigma^(l))}[sigma(f(x)) sigma(f(x'))] + beta^2. The second recursion is the derivative covariance dotSigma^(l+1)(x,x') = E_{f ~ GP(0,Sigma^(l))}[sigma'(f(x)) sigma'(f(x')]. The tangent kernel then satisfies Theta_infty^(1) = Sigma^(1) and Theta_infty^(l+1) = Theta_infty^(l) dotSigma^(l+1) + Sigma^(l+1). This is strictly richer than the static GP kernel because it remembers how changes in lower-layer parameters affect the output.

The theorem that makes the object useful is that the kernel does not drift during training. Under a Lipschitz, twice-differentiable activation with bounded second derivative, and under a stochastic boundedness condition on the integrated training direction, the tangent kernel at time t converges in probability to the same initialization limit, uniformly on finite time intervals. The proof controls how much each preactivation and each normalized weight matrix can move on a finite interval: they shift by O(width^{-1/2}), and a Gronwall argument prevents these small shifts from accumulating into a large representation change. Individual tangent features are nearly frozen, but their collective Gram matrix remains order one and drives real learning. Hence, in the infinite-width limit, gradient descent is exactly kernel gradient descent with the fixed positive-definite tangent kernel Theta_infty.

For squared loss this collapse gives an explicit linear dynamics. On a finite dataset the kernel operator Pi maps a function f to (1/N) sum_i sum_k' f_k'(x_i) K_kk'(x_i, x). The error along each eigenfunction of Pi decays exponentially with rate equal to the corresponding eigenvalue. Large-eigenvalue data directions are fitted quickly, while small-eigenvalue directions are learned only slowly. At convergence, the mean predictor is ridgeless kernel regression with Theta_infty. The random initialization contributes a centered Gaussian residual whose variance is zero on the training points, so the training set is interpolated. This explains both the inductive bias of wide networks and why gradient descent can converge globally in this regime.

The positive-definiteness result supplies the final bridge. If the limiting tangent kernel is strictly positive definite on the data, kernel gradient descent on a convex functional cannot stall away from the global minimum. For non-polynomial activations, the Hermite coefficients are rich enough that the limiting kernel is strictly positive definite on distinct inputs. Therefore the theory gives not only a descriptive infinite-width limit but also a convergence guarantee for the actual gradient-descent trajectory.

The canonical reference implementation is the Neural Tangents library with parameterization='ntk'. Its stax.Dense layer applies fan-in 1/sqrt(width) scaling, initializes weights as standard Gaussians, and updates the stored kernels by nngp <- W_std^2 nngp + b_std^2 after the affine step and ntk <- nngp + W_std^2 ntk. This matches the recursion above. The empirical API computes the finite-width Jacobian outer product, which is the direct finite analog of the tangent kernel and can be compared against the analytic infinite-width prediction as width grows.

The script below verifies the two-layer scalar recursion empirically. It builds a finite two-layer ReLU network in NTK scaling, computes the exact parameter-space tangent kernel for a batch of inputs, and compares it with the Monte-Carlo infinite-width prediction obtained from the Sigma and dotSigma formulas. As width grows, the two estimates agree, illustrating how the aggregate Jacobian Gram matrix stabilizes while individual parameters barely move.

```python
import numpy as np

np.random.seed(0)

d, m, n = 10, 4000, 4          # input dim, hidden width, number of inputs
beta = 0.2                     # bias scale

# Inputs normalized so that E[||x||^2] / d = 1
X = np.random.randn(n, d)
X = X / np.linalg.norm(X, axis=1, keepdims=True) * np.sqrt(d)
Z = np.random.randn(n, d)
Z = Z / np.linalg.norm(Z, axis=1, keepdims=True) * np.sqrt(d)

# NTK-parameterized two-layer ReLU network:
# h(x) = relu(W x / sqrt(d) + beta b0)
# f(x) = a^T h(x) / sqrt(m) + beta b1
W = np.random.randn(m, d)
b0 = np.random.randn(m)
a = np.random.randn(m)
b1 = np.random.randn()

Hx = np.maximum(0.0, X @ W.T / np.sqrt(d) + beta * b0)  # (n, m)
Hz = np.maximum(0.0, Z @ W.T / np.sqrt(d) + beta * b0)

# Empirical tangent kernel = sum_p df(x) df(z) over all parameters.
Sigma1 = X @ Z.T / d + beta ** 2                         # (n, n)

# Contribution from output weights a and output bias b1
K_a = Hx @ Hz.T / m
K_bias = beta ** 2

# Contribution from hidden weights W and hidden bias b0
Gx = (Hx > 0).astype(float).T                            # (m, n), relu' indicator
Gz = (Hz > 0).astype(float).T
scaled_Gx = a[:, None] * Gx                              # (m, n)
scaled_Gz = a[:, None] * Gz
S = scaled_Gx.T @ scaled_Gz / m                          # (n, n), converges to dotSigma2
K_W_b0 = Sigma1 * S

Theta_emp = K_a + K_W_b0 + K_bias

# Infinite-width analytic prediction for ReLU.
c11 = (np.linalg.norm(X, axis=1) ** 2 / d + beta ** 2)[:, None]   # (n, 1)
c22 = (np.linalg.norm(Z, axis=1) ** 2 / d + beta ** 2)[None, :]   # (1, n)
rho = Sigma1 / np.sqrt(c11 * c22)
rho = np.clip(rho, -1.0, 1.0)
theta = np.arccos(rho)

dotSigma2 = (np.pi - theta) / (2.0 * np.pi)
Sigma2 = np.sqrt(c11 * c22) / (2.0 * np.pi) * (
    np.sin(theta) + (np.pi - theta) * np.cos(theta)
) + beta ** 2

Theta_ntk = Sigma2 + Sigma1 * dotSigma2

print("Empirical finite-width NTK:\n", Theta_emp)
print("Analytic infinite-width NTK:\n", Theta_ntk)
rel = np.linalg.norm(Theta_emp - Theta_ntk) / np.linalg.norm(Theta_ntk)
print(f"Relative Frobenius error: {rel:.4f}")
```
