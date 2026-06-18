# Two-stage (layer-wise) training of two-layer networks for multi-index models

## Problem

Gaussian inputs `z ~ N(0, I_d)`, target `y = f*(z) = g*(<w1*,z>, ..., <wr*,z>)` with
orthonormal teacher directions spanning a low-dimensional subspace `V* = span(w1*, ...,
wr*)`, `r << d`. Goal: drive the first-layer rows of a two-layer network

```
f_hat(z; W, a) = (1/sqrt(p)) * a^T sigma(W z)
```

to align with `V*` (recover the relevant subspace) and turn the resulting features into a
predictor of `f*`, with sample/iteration cost that beats both fixed-kernel/random-features
ridge (which never learns `V*` and pays the degree-`k` barrier `n = Theta(d^k)`) and
one-pass SGD (which pays the information-exponent rate `n = Theta(d^{ell-1})`).

## Key idea

Split the training of the two layers.

- **Stage 1 (feature learning):** freeze the second layer `a`, train the first layer `W`
  by gradient descent on the squared loss with a *fresh batch each step* and a *giant*
  learning rate. A symmetric `+/-` second-layer init makes the network output zero at the
  start, so the first gradient is a clean correlation between the activation derivative and
  the labels. Stein's lemma plus the Hermite expansion show its useful part is the
  *leap-index* term of size `d^{-(ell-1)/2)}` (with `ell` the lowest nonzero Hermite degree
  of the target), so the step must be large — `eta = O(p sqrt(n/d))`, equivalently
  `eta = p d^{(ell-1)/2}` for one step — to make the alignment `Theta(1)`. One step recovers
  only the leap-order directions `V_ell*` (a single direction if `ell = 1`); iterating giant
  steps climbs a *conditioning staircase*: each learned direction acts as a ladder that
  exposes the directions linearly connected to it, recovering the staircase-reachable part
  of `V*` with `n = O(d)`.

- **Stage 2 (readout):** freeze `W` and fit `a` in closed form by ridge regression on the
  frozen conjugate-kernel features `phi(z) = sigma(W z)`, using a fresh batch
  (independent of the stage-1 data). This is convex with a closed-form solution.

The reach of the final predictor equals the subspace stage 1 recovered: on the learned
subspace `U` the features act like finite-dimensional random features and can fit any
function of `U`; orthogonal to `U`, with bounded width `p`,
`E[(f* - f_hat)^2] >= E[Var(f*(z) | P_U z)] - o(1)`.

## Algorithm

```
Init:  rows w_i^0 ~ Unif(S^{d-1}); second layer a^0 in symmetric +/- pairs (so f_hat = 0).
Fix lr scaling eta ~ p sqrt(n/d), ridge strength lambda, steps T.

Stage 1 (for t = 0 ... T-1):
    draw a fresh batch (Z, Y),  Y = f*(Z)
    G_i  = (a_i / sqrt(p)) * (1/n) sum_nu z^nu sigma'(<w_i^t, z^nu>) (f_hat(z^nu) - f*(z^nu))
    W^{t+1} = W^t - eta G        # giant step; rows re-normalized to S^{d-1}

Stage 2:
    draw a fresh batch (Z, Y)
    X = sigma(W^T Z)             # conjugate-kernel features, X in R^{n x p}
    a_hat = X^T (X X^T + lambda I_n)^{-1} Y     if n < p
          = (X^T X + lambda I_p)^{-1} X^T Y     if n >= p
Predict:  f_hat(z) = (1/sqrt(p)) a_hat^T sigma(W^T z)
```

The first-layer gradient comes from Stein's lemma:
`E[z sigma'(<w,z>) f*(z)] = w E[sigma''(<w,z>) f*(z)] + E[sigma'(<w,z>) grad_z f*(z)]`,
whose Hermite expansion `sum_k c_{k+2} <w^{⊗k},C_k*> w + sum_k c_{k+1} C_{k+1}* x_{1..k}
w^{⊗k}` is dominated, at a random init, by the leap-order term
`C_ell* x_{1..(ell-1)} (w^0)^{⊗(ell-1)}` of size `d^{-(ell-1)/2}` — hence the giant lr.
Below `n = Theta(d^ell)` this term is buried in noise and nothing is learned.

## Reference implementation

```python
import numpy as np

def sigma(x):                 # ReLU student activation
    return np.maximum(x, 0.0)

def sigma_prime(x):
    return (x > 0).astype(float)

def net(Z, W, a, p):          # two-layer net with 1/sqrt(p) prefactor
    return (1.0 / np.sqrt(p)) * sigma(Z @ W.T) @ a

def sample_data(n, d, teacher, link):
    Z = np.random.randn(n, d)
    Y = link(Z @ teacher.T)               # y = g*(U z)
    return Z, Y

def ridge_estimator(X, y, lam):
    # min_a ||y - X a||^2 + lam ||a||^2 ; cheaper / better-conditioned branch
    m, p = X.shape
    if m >= p:
        return np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ y)
    else:                                  # push-through identity, n < p
        return X.T @ np.linalg.solve(X @ X.T + lam * np.eye(m), y)

def two_stage(d, p, r, teacher, link, n, T, eta_scale=10.0, lam=1.0):
    # init: rows on the sphere; second layer in +/- pairs (zero output at start)
    W = np.random.randn(p, d); W /= np.linalg.norm(W, axis=1, keepdims=True)
    a = np.sign(np.random.randn(p)) / np.sqrt(p)        # frozen during stage 1

    # Stage 1: giant gradient steps on the first layer, fresh batch each step
    eta = eta_scale * p * np.sqrt(n / d)
    for _ in range(T):
        Z, Y = sample_data(n, d, teacher, link)
        resid = Y - net(Z, W, a, p)
        G = (1.0 / n) * Z.T @ ((1.0 / np.sqrt(p)) *
              np.outer(resid, a) * sigma_prime(Z @ W.T))   # (d, p)
        W = W + eta * G.T
        W /= np.linalg.norm(W, axis=1, keepdims=True)      # keep rows on S^{d-1}

    # Stage 2: freeze W, closed-form ridge on conjugate-kernel features (fresh batch)
    Z2, Y2 = sample_data(n, d, teacher, link)
    X = sigma(Z2 @ W.T)
    a_hat = ridge_estimator(X / np.sqrt(p), Y2, lam)
    predict = lambda Zte: (1.0 / np.sqrt(p)) * sigma(Zte @ W.T) @ a_hat
    return W, a_hat, predict
```

PyTorch variant for a fixed `Linear(d, p) -> ReLU -> Linear(p, 1)` student (the
first-layer rows projected to the unit sphere each step, the output layer frozen during
stage 1, then refit once by ridge on the post-ReLU features):

```python
import torch

def stage1_step(model, opt, Zb, Yb):
    # gradient step on the first layer only, then re-normalize its rows
    opt.zero_grad(set_to_none=True)
    loss = ((model(Zb).view(-1) - Yb) ** 2).mean()
    loss.backward()
    out = model[2]                                   # freeze output layer
    if out.weight.grad is not None: out.weight.grad.zero_()
    if out.bias is not None and out.bias.grad is not None: out.bias.grad.zero_()
    opt.step()
    with torch.no_grad():                            # rows back to the unit sphere
        W = model[0].weight
        W.div_(W.norm(dim=1, keepdim=True).clamp(min=1e-8))
    return float(loss)

def stage2_ridge(model, Z, Y, lam):
    # freeze first layer, ridge-fit output weights on post-ReLU features (+ bias col)
    with torch.no_grad():
        feats = torch.relu(model[0](Z))
        feats = torch.cat([feats, torch.ones_like(feats[:, :1])], dim=1)
        gram = feats.T @ feats + lam * torch.eye(feats.shape[1])
        sol = torch.linalg.solve(gram, feats.T @ Y)
        model[2].weight.copy_(sol[:-1].unsqueeze(0))
        if model[2].bias is not None: model[2].bias.copy_(sol[-1:])
```

Both branches implement the same method: giant first-layer steps for feature learning,
then a single closed-form ridge readout on the frozen conjugate-kernel features.
