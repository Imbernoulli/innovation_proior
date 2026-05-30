# The Neural Tangent Kernel

## Problem

Gradient descent on a neural network's parameters minimizes a highly non-convex
parameter-space loss, yet wide networks reliably reach zero training loss and
generalize. The Neural Tangent Kernel (NTK) explains this by describing what
gradient descent does to the *function* the network computes, in function space
where the cost is convex.

## Key idea

Track the network function f_theta instead of the weights. Under gradient flow
theta_dot = -grad_theta (C ∘ F), the chain rule makes the function evolve as
kernel gradient descent,

    partial_t f_theta = - nabla_Theta C,

against the tangent kernel

    Theta(x, x') = sum_p partial_{theta_p} f_theta(x) ⊗ partial_{theta_p} f_theta(x')
                 = < grad_theta f_theta(x), grad_theta f_theta(x') >.

For squared-error loss this is the linear-on-data ODE f_dot = -Theta (f - f*).
In the NTK parametrization (pre-activation = (1/sqrt(n_in)) W a + beta b, with
W, b iid N(0,1)), two facts hold as the hidden widths go to infinity:

1. **Deterministic at initialization.** Theta converges in probability to an
   explicit kernel Theta^(L)_inf ⊗ Id, given by a layer-wise recursion alongside
   the NNGP covariance Sigma:

        Sigma^(1)(x,x')      = x^T x' / n0 + beta^2
        Sigma^(L+1)(x,x')    = E_{f~N(0,Sigma^(L))}[ sigma(f(x)) sigma(f(x')) ] + beta^2
        Sigma_dot^(L+1)(x,x')= E_{f~N(0,Sigma^(L))}[ sigma'(f(x)) sigma'(f(x')) ]
        Theta^(1)_inf        = Sigma^(1)
        Theta^(L+1)_inf      = Theta^(L)_inf * Sigma_dot^(L+1) + Sigma^(L+1).

   It depends only on depth, nonlinearity, and initialization variance.

2. **Constant during training.** Each weight and activation moves at rate
   O(1/sqrt(width)) (a coupled Grönwall bound), so Theta(t) -> Theta^(L)_inf
   uniformly on [0, T]. Per-neuron motion vanishes, but the collective effect of
   the width-many neurons keeps the lower layers learning (the
   Theta^(L)_inf * Sigma_dot term).

## Consequences

- Training becomes a linear ODE in function space against a fixed kernel:
  f_t = f* + e^{-t Pi}(f_0 - f*), with Pi the kernel integral operator.
- For non-polynomial Lipschitz sigma and L >= 2, Theta^(L)_inf is positive
  definite on the sphere (via Daniely's Hermite dual + the Schoenberg/Gneiting
  criterion), so training converges to the global optimum of the function-space
  cost.
- The function relaxes along kernel principal components at rate e^{-lambda_i t};
  large eigenvalues (top components) converge first, motivating early stopping.
- At t -> infinity the mean predictor is ridgeless kernel regression with
  Theta^(L)_inf (equivalently the GP-MAP estimate): a wide trained network
  generalizes exactly as a kernel machine.

## Code

```python
import numpy as np
import torch


def relu_dual(cov_xx, cov_xpxp, cov_xxp):
    """Gaussian expectations for ReLU (arc-cosine kernels, Cho & Saul 2009):
    E[relu(X)relu(X')] = Sigma (minus bias) and E[relu'(X)relu'(X')] = Sigma_dot."""
    denom = np.sqrt(cov_xx * cov_xpxp)
    rho = np.clip(cov_xxp / np.maximum(denom, 1e-12), -1.0, 1.0)
    theta = np.arccos(rho)
    nngp = denom / (2 * np.pi) * (np.sin(theta) + (np.pi - theta) * np.cos(theta))
    nngp_dot = (np.pi - theta) / (2 * np.pi)
    return nngp, nngp_dot


def infinite_ntk(X, Xp, depth, beta=0.1):
    """Analytic infinite-width NTK Theta_inf^(L) and NNGP Sigma^(L) for ReLU."""
    n0 = X.shape[1]
    sig = X @ Xp.T / n0 + beta ** 2
    sig_xx = (X * X).sum(1) / n0 + beta ** 2
    sig_pp = (Xp * Xp).sum(1) / n0 + beta ** 2
    theta = sig.copy()
    for _ in range(depth - 1):
        nngp, nngp_dot = relu_dual(sig_xx[:, None], sig_pp[None, :], sig)
        sig_xx, _ = relu_dual(sig_xx, sig_xx, sig_xx); sig_xx += beta ** 2
        sig_pp, _ = relu_dual(sig_pp, sig_pp, sig_pp); sig_pp += beta ** 2
        theta = theta * nngp_dot + (nngp + beta ** 2)   # Theta^(L+1) recursion
        sig = nngp + beta ** 2
    return theta, sig


class WideMLP(torch.nn.Module):
    """Finite-width net in the NTK parametrization."""
    def __init__(self, n0, width, depth, beta=0.1):
        super().__init__()
        self.beta = beta
        sizes = [n0] + [width] * (depth - 1) + [1]
        self.Ws = torch.nn.ParameterList(torch.nn.Parameter(torch.randn(o, i))
                                         for i, o in zip(sizes[:-1], sizes[1:]))
        self.bs = torch.nn.ParameterList(torch.nn.Parameter(torch.randn(o))
                                         for o in sizes[1:])
        self.scales = [1.0 / np.sqrt(i) for i in sizes[:-1]]

    def forward(self, x):
        a, last = x, len(self.Ws) - 1
        for i, (W, b) in enumerate(zip(self.Ws, self.bs)):
            a = self.scales[i] * (a @ W.T) + self.beta * b
            if i != last:
                a = torch.relu(a)
        return a.squeeze(-1)


def empirical_ntk(net, X):
    """Finite-width NTK Theta(x,x') = sum_p d_theta f(x) . d_theta f(x')."""
    params = list(net.parameters())
    J = []
    for xi in X:
        out = net(xi.unsqueeze(0)).squeeze()
        g = torch.autograd.grad(out, params, retain_graph=True)
        J.append(torch.cat([gi.reshape(-1) for gi in g]))
    J = torch.stack(J)
    return (J @ J.T).detach().numpy()


def kernel_regression(K_train, K_test, y, ridge=0.0):
    """t->infinity mean predictor = ridgeless kernel regression."""
    A = K_train + ridge * np.eye(K_train.shape[0])
    return K_test @ np.linalg.solve(A, y)


if __name__ == "__main__":
    np.random.seed(0); torch.manual_seed(0)
    X = np.random.randn(8, 4)
    theta_inf, _ = infinite_ntk(X, X, depth=3)
    acc = sum(empirical_ntk(WideMLP(4, 4000, 3).double(),
                            torch.tensor(X)) for _ in range(20)) / 20
    print(np.linalg.norm(acc - theta_inf) / np.linalg.norm(theta_inf))  # ~0.01
```

The analytic infinite-width NTK and the empirical NTK of a wide network (averaged
over initializations) agree to about 1% relative error, confirming the recursion
and the parametrization.
