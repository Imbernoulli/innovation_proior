# Sparse parity learning with SGD, distilled

The default sparse-parity learning protocol trains a generic neural network — with off-the-shelf
initialization and an off-the-shelf optimizer, no sparsity prior — on the `(n,k)`-sparse parity
task, and shows two things: ordinary SGD solves it in a number of iterations near the
`n^{O(k)}` computational (statistical-query) lower bound, and the long flat plateau in the loss
hides *steady* progress — the relevant features are amplified continuously via a **Fourier gap**
in the population gradient, made visible by a weight-based **progress probe**.

## Problem it solves

Recover a hidden size-`k` subset `S` from labelled examples `(x, y)`, `x ~ Unif({±1}^n)`,
`y = chi_S(x) = prod_{i in S} x_i` (equivalently `y = (sum_{i in S} x_i) mod 2`). Statistically
trivial (`Theta(k log n)` samples) but believed computationally hard (`n^{Omega(k)}`). The
question is diagnostic: does plain gradient training succeed, how fast, and by what mechanism —
given that the loss curve sits at chance and then jumps discontinuously, and that a standard
argument says the gradient carries no signal about `S`.

## Key idea

1. **The "no signal" pessimism is over the wrong family.** The gradient-variance bound
   `Var(H,F,w) <= G(w)^2/|H|` (orthogonal targets, square/Lipschitz loss, any architecture) gives
   exponentially small variance `G^2/2^n` over *all* `2^n` parities. The sparse problem fixes
   `|S| = k`, so the family has only `C(n,k) ≈ n^k` members and the same bound reads `G^2/n^k` — a
   *polynomial* residual signal of scale `~ n^{-k/2}`, liftable by a batch of size `~ n^k` (and the
   SQ floor ties them: `B · t ~ n^k`).

2. **The signal is a Fourier gap in the population gradient.** For a ReLU neuron `f = (w·x)_+` at
   all-ones or random-sign init with correlation loss, coordinate `j` of `E[grad]` is a Fourier coefficient of the
   threshold derivative `1[w·x>=0] = (1+Maj(w⊙x))/2`:
   - `j in S` -> degree-`(k-1)` coefficient (large), since `y x_j = chi_{S\{j}}`;
   - `j not in S` -> degree-`(k+1)` coefficient (small), since `y x_j = chi_{S∪{j}}`.

   Majority's spectrum obeys `|xi_{k-1}| = ((n-k)/(k-1))|xi_{k+1}|`, so the lower degree dominates;
   the **Fourier gap** is `gamma = |xi_{k-1}| - |xi_{k+1}| >= 0.03 (n-1)^{-(k-1)/2} = Theta(n^{-(k-1)/2})`.

3. **Fourier gap => recoverability.** If `g(w)` estimates the population gradient to sup-norm
   `< gamma/2`, the `k` largest-magnitude coordinates of `g(w)` are exactly `S` — recoverable from
   `O(log n / gamma^2) = O~(n^{k-1})` samples (one big-batch step), at the SQ floor.

4. **Plateau progress (small batches).** Each relevant weight is a biased random walk: drift
   `~ gamma · t` (the population gradient) plus diffusion `~ sigma sqrt(t)` (minibatch noise).
   Drift outruns noise, so relevant weights climb monotonically. The classifier threshold does not flip until they
   overtake the irrelevant ones — hence the flat loss then sharp jump (plateau = amplification, not
   search). The **progress probe** `rho(w_{0:t}) = ||w_t - w_0||_inf` (a linear estimate of
   the initial population gradient) rises steadily through the plateau.

5. **Not search, not lazy.** Black-box signatures rule out memoryless search: no successes near
   `t=0` (search is `Geom`, mode at 0), `k`-adaptive `n^{O(k)}` scaling, concentration *conditioned
   on init*, sub-linear returns to width. The NTK lower bound (`D R^2 < eps^2 C(n,k)` leaves some
   `S` unfittable, so fixed features need `D = Omega(n^k)`) rules out the lazy regime at small/unit
   width — success requires genuine feature learning (weights leave init).

## End-to-end guarantees

**Two-layer ReLU MLP, large batch (SGD learns sparse parities).** With even `k`, large `n`,
symmetric `±1` init (so `f(x;theta_0)=0`, hinge `ell'(0,y)=-y`), width `r = Omega(2^k k log(k/eps))`,
batch `B = Omega(n^k log(n/eps))`:
- First step (`eta_0 = 1/(k|xi_{k-1}|)`, `lambda_0 = 1`): lucky neurons (constant `w` on `S`, with
  staggered biases `b_i = -1/2 + (i+1)/k`) become ridge functions `h_i ≈ sigma((1/2k) sum_{j in S} x_j + b_i)`;
  irrelevant-coordinate leakage is `O(sqrt(log/n))` by the Fourier gap.
- A fixed second layer `u*` (`||u*||_inf <= 8k`) combines them into `2 chi_S` exactly.
- Freeze layer 1, train the convex second layer: `min_t F(u^{(t)}) <= F(u*) + M rho/sqrt(T)`.

Result: `E[min_t ell(f(x;theta_t), y)] <= eps` in `O(k^3 r^2 n / eps^2)` iterations.

**Disjoint-PolyNet (real-output tree parity machine), exact trajectory.**
`f(x; w_{1:k}) = prod_{i=1}^k <w_i, x_{P_i}>` on a `k`-block partition with one relevant index per
block. Population gradient `g_i = -(prod_{j != i} w_{j,1}) e_1` (nonzero only at relevant indices —
no Fourier gap needed). Gradient flow on the relevant weights `v_i`: `dot{v}_i = prod_{j != i} v_j`,
`d(v_i^2)/dt = 2 prod_j |v_j|` (lockstep). With `q = v_i^2 - v_i(0)^2`, `dot{q} = 2 prod(q+v_i(0)^2)^{1/2}`,
sandwiched by Maclaurin's inequality between geometric and arithmetic means -> finite-time blow-up.
Error via Berry-Esseen:
`err >= 1/2 - (1/2) prod_i (erf(|v_i|/(||u_i||_2 sqrt2)) + c||u_i||_inf/||u_i||_2^3)`.
**Phase transition (`k >= 3`):** `T(1/2 - gamma)/T(0) >= 1 - O((n')^{1-k/2} gamma^{2/k-1})` — a
`1-o(1)` fraction of training spent above `49%` error. **SGD at any batch** converges in
`O~((n')^{2k-1} log(1/eps))` steps (Azuma-Hoeffding on the gradient-noise martingale `|s_{i,j}| <= 1/2`,
relevant weight grows `~ (1/(3n'-2))^{k-1} sqrt(t)`).

## Defaults and why

Fixed two-layer ReLU MLP (`W = 512`), standard Xavier/Glorot uniform init (zero bias), online
single-pass fresh batches, batch size 128, stock AdamW (`lr = 1e-3`, `wd = 1e-2`,
`betas = (0.9, 0.999)`, `eps = 1e-8`), BCE loss.

- **ReLU:** its derivative is a threshold -> majority spectrum -> the Fourier gap is largest with
  non-smooth activations. (Smooth `z^k` works too; `z^{k'<k}` cannot represent parity by
  orthogonality.)
- **Standard init, no sparsity prior:** the mechanism is robust to the init scheme; the point is
  that *standard* training suffices. (Symmetric `±1` is only a proof convenience to zero the output.)
- **Online / single-pass:** every step is an unbiased estimate of the *same* population gradient ->
  clean drift, no overfitting confound. (Multi-pass on a small dataset instead surfaces grokking;
  weight decay accelerates that delayed generalization.)
- **Batch size:** large `B` (`~ n^k`) resolves the `gamma/2` tolerance in one step; small `B` (even
  1) works via drift>noise but slower, with `B · t ~ n^k`.
- **Optimizer/loss:** the mechanism is optimizer- and loss-agnostic (it lives in the population
  gradient), so stock defaults are exactly the right baseline.

## Working code

Grounded in the standard supervised-training harness: fixed MLP, online fresh batches, AdamW + BCE,
plus the weight-state probe.

```python
import torch
from torch import nn


def build_model(n_features: int, width: int = 512) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(n_features, width),
        nn.ReLU(),
        nn.Linear(width, 1),
        nn.Sigmoid(),
    )


def init_model(model: nn.Sequential) -> None:
    # Standard, data-independent init (must not see the secret).
    for layer in model:
        if isinstance(layer, nn.Linear):
            gain = nn.init.calculate_gain("relu") if layer is model[0] else 1.0
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)


def parity_labels(x: torch.Tensor, secret) -> torch.Tensor:
    idx = torch.tensor(secret, dtype=torch.long, device=x.device)
    return x.index_select(1, idx).sum(dim=1).remainder(2).to(torch.float32)


def make_online_batch(secret, n_features, batch_size, generator):
    # Fresh i.i.d. uniform binary batch -> unbiased sample of the SAME population gradient.
    x = torch.randint(0, 2, (batch_size, n_features), generator=generator).float()
    return x, parity_labels(x, secret)


def make_test_set(secret, n_features, test_size, generator):
    x = torch.randint(0, 2, (test_size, n_features), generator=generator).float()
    return x, parity_labels(x, secret)


def get_optimizer_config() -> dict[str, float]:
    return {"lr": 1e-3, "wd": 1e-2, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8}


def state_probe(model: nn.Sequential, init_state) -> float:
    # ||W_t - W_0||_inf on the first layer: rises with the drift while loss stays flat.
    W0 = init_state["0.weight"]
    W = model[0].weight.detach()
    return (W - W0).abs().max().item()


def train_parity(secret, n_features, width=512, batch_size=128, steps=100_000,
                 test_size=16_384, seed=0, device="cpu"):
    model = build_model(n_features, width).to(device)
    init_model(model)
    init_state = {name: p.detach().clone() for name, p in model.named_parameters()}
    opt_cfg = get_optimizer_config()
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg["lr"],
        betas=(opt_cfg["beta1"], opt_cfg["beta2"]),
        eps=opt_cfg["eps"],
        weight_decay=opt_cfg["wd"],
    )
    criterion = nn.BCELoss()
    train_gen = torch.Generator().manual_seed(seed)
    test_gen = torch.Generator().manual_seed(seed + 1)
    test_x, test_y = make_test_set(secret, n_features, test_size, test_gen)

    for t in range(1, steps + 1):
        xb, yb = make_online_batch(secret, n_features, batch_size, train_gen)
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(set_to_none=True)
        loss = criterion(model(xb).view(-1), yb)
        loss.backward()                       # noisy estimate of the population gradient
        opt.step()                            # relevant coordinates drift up by ~gamma
        if t % 250 == 0:
            print(f"step {t}  loss {loss.item():.4f}  rho {state_probe(model, init_state):.4f}")
    with torch.no_grad():
        preds = model(test_x.to(device)).view(-1)
        acc = ((preds >= 0.5) == (test_y.to(device) >= 0.5)).float().mean().item()
    return model, acc
```
