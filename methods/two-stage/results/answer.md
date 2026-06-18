# Two-stage large-batch training for multi-index targets

## Method

Train the two layers separately.

1. **Feature step(s):** freeze the output weights `a` and update only the first-layer rows with large-batch gradient descent on squared loss. The exact-theory initialization may pair hidden units so the initial network output is zero; the public notebooks use small random output weights, where the same correlation form is approximate at large width.
2. **Readout step:** freeze the learned `W` and solve ridge regression on the scaled conjugate-kernel features `sigma(Z W.T) / sqrt(p)` using an independent batch.

The first-layer update can be written with either sign convention:

```text
grad_i = (a_i/sqrt(p)) (1/n) sum_n z_n sigma'(<w_i,z_n>) (f_hat(z_n) - y_n)
w_i <- w_i - eta grad_i
```

or equivalently

```text
neg_grad_i = (a_i/sqrt(p)) (1/n) sum_n z_n sigma'(<w_i,z_n>) (y_n - f_hat(z_n))
w_i <- w_i + eta neg_grad_i.
```

At initialization, Stein's lemma and the Hermite expansion give

```text
E[neg_grad_i]
 = (a_i/sqrt(p)) [sum_k c_{k+2}<w_i^{otimes k}, C_k*> w_i
   + sum_k c_{k+1} C_{k+1}* x_{1..k} w_i^{otimes k}],
```

where `c_k` are Hermite coefficients of `sigma`. If `ell` is the first nonzero Hermite degree of `f*`, the leading useful target-subspace term is

```text
(a_i/sqrt(p)) c_ell C_ell* x_{1..(ell-1)} (w_i^0)^{otimes (ell-1)},
```

with size `d^{-(ell-1)/2}/p`. A one-step feature-learning rate therefore scales as

```text
eta = Theta(p d^{(ell-1)/2}),
```

or `Theta(p sqrt(n/d))` when the one-step batch is `n = Theta(d^ell)`. One step recovers only `V_ell*`, the span of the singular directions of `C_ell*`. Multiple fresh-batch steps with `n = Theta(d)` grow the learned subspace by the conditioning recursion

```text
U_0* = {0},
U_{t+1}* = U_t* + span{mu_{U_t*,x}(f*) : x in U_t*}.
```

## Reference Code

This NumPy version follows the public `GiantStep` notebooks' sign and ridge conventions: form a negative-gradient matrix with `(Y - f_hat)`, update `W` by adding it, scale features by `1/sqrt(p)` before ridge, and do not project rows after each step.

```python
import numpy as np


def sigma(x):
    return np.maximum(x, 0.0)


def sigma_prime(x):
    return (x > 0).astype(float)


def net(Z, W, a):
    p = W.shape[0]
    return sigma(Z @ W.T) @ a / np.sqrt(p)


def sample_data(n, d, teacher, link):
    Z = np.random.randn(n, d)
    Y = link(Z @ teacher.T)
    return Z, Y


def init_weights(p, d, symmetric=True):
    """Sphere init; symmetric=True gives exact zero initial output."""
    if symmetric:
        if p % 2:
            raise ValueError("symmetric initialization requires even p")
        half = p // 2
        W_half = np.random.randn(half, d)
        W_half /= np.linalg.norm(W_half, axis=1, keepdims=True)
        a_half = np.random.uniform(-1.0, 1.0, size=half) / np.sqrt(p)
        W = np.vstack([W_half, W_half])
        a = np.concatenate([a_half, -a_half])
        return W, a

    W = np.random.randn(p, d)
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    a = np.random.uniform(-1.0, 1.0, size=p) / np.sqrt(p)
    return W, a


def negative_first_layer_gradient(Z, Y, W, a):
    """Return G with shape (d, p), so the update is W <- W + eta * G.T."""
    n, p = Z.shape[0], W.shape[0]
    residual = Y - net(Z, W, a)
    gates = sigma_prime(Z @ W.T)
    weighted = (residual[:, None] * a[None, :] / np.sqrt(p)) * gates
    return (Z.T @ weighted) / n


def ridge_estimator(Phi, y, lam):
    """Solve min_b ||y - Phi b||^2 + lam ||b||^2."""
    n, p = Phi.shape
    if n >= p:
        return np.linalg.solve(Phi.T @ Phi + lam * np.eye(p), Phi.T @ y)
    return Phi.T @ np.linalg.solve(Phi @ Phi.T + lam * np.eye(n), y)


def two_stage(d, p, teacher, link, n_feature, n_readout, steps, eta_scale=1.0, lam=1.0):
    W, a0 = init_weights(p, d, symmetric=True)
    eta = eta_scale * p * np.sqrt(n_feature / d)

    for _ in range(steps):
        Z, Y = sample_data(n_feature, d, teacher, link)
        G = negative_first_layer_gradient(Z, Y, W, a0)
        W = W + eta * G.T

    Z2, Y2 = sample_data(n_readout, d, teacher, link)
    Phi = sigma(Z2 @ W.T) / np.sqrt(p)
    a_hat = ridge_estimator(Phi, Y2, lam)

    def predict(Ztest):
        return sigma(Ztest @ W.T) @ a_hat / np.sqrt(p)

    return W, a_hat, predict
```

## Guarantees And Limits

- If `n = O(d^{ell-delta})` for one step, the learned target-subspace fraction remains vanishing.
- If `n = Omega(d^ell)` and the matching student Hermite coefficient is nonzero, one giant step gives nontrivial alignment with `V_ell*`; for `ell=1`, this is a one-dimensional spike.
- With multiple `n = Theta(d)` fresh-batch steps, only staircase-connected directions become visible in finite time.
- With `W` frozen and bounded width, the readout cannot beat the best predictor that depends on the learned subspace `U`:

```text
E[(f*(z) - f_hat(z))^2] >= E[Var(f*(z) | P_U z)] - o(1).
```

The final predictor is therefore only as good as the subspace recovered by the first stage.
