# Context: feature learning in shallow networks on multi-index targets

## Research Question

Inputs are high-dimensional isotropic Gaussians, `z ~ N(0, I_d)`, while the label depends only on a fixed low-dimensional projection:

```text
y = f*(z) = g*(<w1*, z>, ..., <wr*, z>),
```

with orthonormal teacher directions `w1*, ..., wr*` spanning `V*` and `r << d`. The distribution of `z` itself gives no preferred axes; all structure is in the unknown subspace used by the target. A two-layer network

```text
f_hat(z; W, a) = (1/sqrt(p)) sum_i a_i sigma(<w_i, z>)
```

has the right expressive form: the first-layer rows could rotate toward `V*`, and then the readout would only need to fit a function of `r` coordinates. The question is whether gradient-based training makes that rotation happen, and what controls the sample complexity relative to the known kernel and online-SGD baselines.

The observable feature-learning quantity is the projection of each row onto the target subspace, for example `||Pi* w_i||^2 / ||w_i||^2`, where `Pi*` projects onto `V*`.

## Background

Under Gaussian inputs, Hermite tensors are the natural coordinates. A square-integrible target has

```text
f(z) = sum_k <C_k(f), H_k(z)>,
```

and inner products decompose degree by degree. For a multi-index target `f*(z)=g*(W* z)` with orthonormal teacher rows, the Hermite coefficients satisfy

```text
C_k(f*) = C_k(g*) . (W*, ..., W*),
```

so every singular direction of every nonzero `C_k(f*)` lies in `V*`. The first nonzero Hermite degree, the leap index `ell`, measures how hidden the relevant directions are from a gradient at random initialization. In a single-index model this is the information exponent.

The first-layer gradient contains expectations of the form

```text
E[z sigma'(<w,z>) f*(z)].
```

Stein's lemma rewrites this as a term along `w` plus a term involving `grad f*`; after Hermite expansion, contractions against powers of the small random overlap `Pi* w` determine which target directions are visible. Since a random unit vector has `||Pi* w|| = O(sqrt(r/d))`, each additional Hermite order costs a factor near `d^{-1/2}`.

## Baselines

**Fixed random features / conjugate-kernel ridge.** If `W` is frozen at initialization and only `a` is fitted, the model is a kernel method with features `sigma(W z)`. In the proportional high-dimensional regime it has a degree barrier: to learn a degree-`k` part of the target it needs `n` and `p` on the order of `d^k`.

**Small first-layer updates.** A small step from initialization remains close to the lazy or linearized regime and can alter constants and learn a linear component.

**Online one-sample SGD.** For single-index Gaussian problems, the information exponent controls the time to escape the uninformative region. Starting from random overlap `O(d^{-1/2})`, online SGD needs polynomially many samples and updates, `d log d` for exponent `2` and roughly `d^{ell-1}` for larger `ell`, before it reaches nontrivial correlation.

**One-step representation learning with preprocessing.** Earlier multi-index results use one gradient step after estimating and subtracting lower-degree Hermite components of the labels, exposing several directions at once using a large batch.

**Staircase analyses.** Boolean and Gaussian staircase results explain when directions become learnable sequentially: already-learned coordinates can make new ones linearly visible after conditioning.

## Evaluation Settings

The pre-method yardsticks are fixed before any algorithmic choice:

- **Data model:** Gaussian `z`, fixed orthonormal teacher subspace `V*`, and polynomial or Hermite-link targets with a controlled leap index.
- **Feature metrics:** row alignment `||Pi* w_i||^2 / ||w_i||^2` and subspace error between the learned row span and `V*`.
- **Prediction metric:** population mean-squared error on fresh Gaussian test data.
- **Resource axes:** batch size, number of first-layer updates, width `p`, and whether data used to train features is reused for the readout.
- **Comparisons:** random-feature or conjugate-kernel ridge at the same `(n,p)`, small-step/lazy training, one-pass SGD, and one-step large-batch representation learning with preprocessing.

These are settings and baselines only; no algorithm outcomes are assumed.

## Code Framework

The available substrate is a two-layer network, Gaussian data sampler, activation derivative, and ridge solver. The open design slots are initialization, the first-layer update rule, and the final readout fit.

```python
import numpy as np

def sigma(x):
    return np.maximum(x, 0.0)

def sigma_prime(x):
    return (x > 0).astype(float)

def net(Z, W, a):
    return (sigma(Z @ W.T) @ a) / np.sqrt(W.shape[0])

def sample_data(n, d, teacher, link):
    Z = np.random.randn(n, d)
    Y = link(Z @ teacher.T)
    return Z, Y

def ridge_estimator(Phi, y, lam):
    """Solve min_b ||y - Phi b||^2 + lam ||b||^2."""
    n, p = Phi.shape
    if n >= p:
        return np.linalg.solve(Phi.T @ Phi + lam * np.eye(p), Phi.T @ y)
    return Phi.T @ np.linalg.solve(Phi @ Phi.T + lam * np.eye(n), y)

def init_weights(p, d):
    # TODO: choose first-layer scale/orientation and frozen readout initialization.
    raise NotImplementedError

def update_first_layer(W, a, batch, eta):
    # TODO: compute the first-layer update from the current residual.
    raise NotImplementedError

def fit_readout(W, batch, lam):
    # TODO: freeze W and solve the readout in closed form.
    raise NotImplementedError
```
