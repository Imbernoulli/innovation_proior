# GraN-DAG, distilled

GraN-DAG (Gradient-based Neural DAG learning) recovers a directed acyclic causal graph from
purely observational data with *nonlinear* mechanisms. It models each variable's conditional with
its own neural network, reads a single nonnegative `d × d` "weighted adjacency" matrix off those
networks' weights via a path-product argument, and enforces acyclicity with the smooth
matrix-exponential constraint `tr e^{A_phi} - d = 0`, solving the resulting constrained maximum-
likelihood problem with an augmented Lagrangian. It is the nonlinear extension of the
continuous-constraint paradigm: it keeps NOTEARS's smooth acyclicity but replaces the linear SEM
with per-variable neural networks.

## Problem it solves

Learn the fully *directed* DAG `G` from observational samples of `X = (X_1, ..., X_d)` when the
data-generating mechanisms are nonlinear. Two obstacles: the space of DAGs is super-exponential in
`d` (forcing greedy combinatorial search), and direction is not identifiable from `P_X` without
assumptions. GraN-DAG targets the identifiable nonlinear additive-noise regime and avoids
combinatorial search by continuous optimization.

## Key idea

Give variable `j` a fully-connected network `NN_j` taking `X_{-j}` (its own input masked to zero)
and outputting the parameters `theta_(j)` of `X_j`'s conditional:

```
theta_(j) = W^{(L+1)}_(j) g( ... g(W^{(2)}_(j) g(W^{(1)}_(j) X_{-j})) ... )
```

The contribution: turn the weights of all `d` networks into a single nonnegative `d × d` matrix
`A_phi` such that `(A_phi)_{ij} = 0` certifies that `theta_(j)` does not depend on `X_i`, then use
it inside NOTEARS's acyclicity constraint.

- **NN paths and path products.** Information flows from input `i` to output `k` only along
  computation paths `(W^{(1)}_{h_1 i}, ..., W^{(L+1)}_{k h_L})`. A path is inactive iff some weight
  on it is zero. Its *path product* `prod_l |W^{(l)}|` is nonnegative, zero iff inactive. Output `k`
  is independent of input `i` if *all* paths from `i` to `k` are inactive, i.e. the sum of their
  path products is zero.
- **Connectivity matrix.** Summing path products over routes is matrix multiplication of the
  absolute-value weight matrices: `C = |W^{(L+1)}| ... |W^{(2)}| |W^{(1)}| ∈ R^{m×d}_{>=0}`, and
  `C_{ki}` is the total strength from input `i` to output `k`. So `C_{ki} = 0` is sufficient for
  output `k` to be independent of input `i`.
- **Weighted adjacency.** Sum over the `m` output components to make all of `theta_(j)` independent
  of `X_i`:

```
(A_phi)_{ij} = sum_{k=1}^m (C_(j))_{ki}   for j != i,   (A_phi)_{ii} = 0
```

`A_phi >= 0` entrywise, and `(A_phi)_{ij} = 0  =>  theta_(j) independent of X_i` (no edge i -> j).

- **Acyclicity.** Because `A_phi` is already nonnegative (no Hadamard square needed), apply the
  matrix-exponential constraint directly:

```
h(phi) = tr e^{A_phi} - d = 0      (<=>  G_phi is acyclic)
```

This runs the NOTEARS closed-walk argument twice: once at the NN-path level (to build `A_phi`) and
once at the graph-path level (the `tr e^{A_phi} - d` constraint).

## Objective and guarantee

Constrained maximum likelihood (each `sum_j log p_j` is a valid joint log-likelihood only when
`h = 0`):

```
max_phi  E_{X ~ P_X} sum_{j=1}^d log p_j(X_j | X_{pi_j^phi}; phi_(j))    s.t.   tr e^{A_phi} - d = 0
```

**Identifiability transfer (Proposition).** If the true CGM `(P_X, G)` lies in the representable
class `C`, and `C` equals the set `M(A)` of models satisfying an identifiability assumption set `A`
under which `G` is identifiable, then the optimal solution recovers `G_{phi*} = G`. *Proof:* the
population log-likelihood is maximal iff `P_phi = P_X`; this is achievable since `(P_X, G) ∈ C`; by
identifiability no other model in `C` with a different graph reproduces `P_X`; hence the optimum's
graph is `G`. (Population / exact-optimization statement; finite-sample, non-convex practice gives
a stationary point.)

## Optimization (augmented Lagrangian)

Solve a sequence of subproblems, each an approximate maximizer of

```
L(phi, lambda_t, mu_t) = E[ sum_j log p_j ] - lambda_t h(phi) - (mu_t / 2) h(phi)^2
```

with updates after each subproblem converges:

```
lambda_{t+1} = lambda_t + mu_t h(phi_t*)
mu_{t+1}     = eta * mu_t   if h(phi_t*) > gamma * h(phi_{t-1}*),   else mu_t
```

Defaults: `lambda_0 = 0`, `mu_0 = 1e-3`, `eta = 10`, `gamma = 0.9`; stop when `h <= 1e-8`. Each
subproblem is solved by minibatch RMSprop (implicit regularization; scales to NN parameters), with
*early stopping* when a held-out estimate of `L` stops improving. Per-network architecture: 2
hidden layers of 10 units, leaky-ReLU, Xavier init; minibatch 64. The experimental setting can use
`1e-2` for the first subproblem and `1e-4` after restarts; the compact implementation below uses
`lr = 1e-3` and reuses it unless a restart learning rate is supplied.

## Overfitting control and DAG extraction

Maximum likelihood never penalizes adding an edge, so it is regularized externally:

- **Edge clamping during training:** when `(A_phi)_{ij} < epsilon = 1e-4`, permanently mask input
  `i` off network `j` (the mask is an extra non-learned factor in the `C` product).
- **Preliminary neighbor selection (PNS)** for `d >= 50` (as in CAM): fit an ExtraTrees regressor
  of each variable on the others; keep candidate parents with importance `> 0.75 * mean`.
- **Pruning** (as in CAM): fit a generalized additive model of each node on its parents; drop
  parents with covariate p-value `> 0.001`.
- **Jacobian thresholding for the final DAG:** `(A_phi)_{ij} = 0` is *sufficient but not
  necessary* for independence (path products can cancel; neurons can saturate). The realized
  sensitivity is the expected absolute Jacobian `J = E_X |∂L/∂X|^T`, where `L_i` is the conditional
  likelihood in the mathematical description; the implementation uses the per-variable
  log-likelihood score for this Jacobian. Remove edges from smallest `J_{ij}` upward, stopping at
  the smallest threshold whose remaining graph is acyclic (test: `tr(A^k) = 0` for `k = 1..d`).

## Relation to prior methods

- **NOTEARS** supplies the smooth matrix-exponential constraint. With a single linear layer, the
  path construction collapses to a nonnegative edge-strength matrix from the coefficients; NOTEARS
  uses `W ∘ W` for the same nonnegative-input acyclicity argument. GraN-DAG keeps the smooth
  constraint and generalizes the edge-strength construction to nonlinear networks.
- **CAM** assumes additive mechanisms `sum_i f_{ij}(x_i)`; GraN-DAG drops the additivity but reuses
  CAM's PNS and pruning.
- **DAG-GNN** shares parameters across the `f_j` via a GNN; GraN-DAG keeps the mechanisms
  independent (one network each).
- **MADE** fixes the input mask (hence the variable ordering) a priori to enforce the
  autoregressive property (= acyclicity for a fixed order); GraN-DAG *learns* the mask/ordering from
  data through the continuous constraint.

## Working code

For the nonlinear-Gaussian ANM the network outputs the conditional mean and the noise std is a
learned, parent-independent parameter. The function returns `B` with `B[i, j] != 0` meaning
`j -> i`.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions


def run_causal_discovery(X: np.ndarray) -> np.ndarray:
    """GraN-DAG: learn a DAG from observational nonlinear (Gauss-ANM) data.

    B[i, j] != 0 means j -> i.
    """
    import os
    seed = int(os.environ.get("SEED", "42"))
    torch.manual_seed(seed); np.random.seed(seed)
    n, d = X.shape
    DT = torch.float64

    class GranDagModel(nn.Module):
        """One MLP per variable (NonlinGaussANM): 2 hidden layers x 10 units,
        leaky-ReLU. Outputs the mean of each variable's Gaussian conditional."""
        def __init__(self):
            super().__init__()
            # binary input mask = current candidate adjacency (not a Parameter)
            self.adjacency = torch.ones(d, d, dtype=DT) - torch.eye(d, dtype=DT)
            layers = [d, 10, 10, 1]                      # [in, hid1, hid2, mean]
            self.W = nn.ParameterList(); self.b = nn.ParameterList()
            for k in range(len(layers) - 1):
                self.W.append(nn.Parameter(torch.zeros(d, layers[k+1], layers[k], dtype=DT)))
                self.b.append(nn.Parameter(torch.zeros(d, layers[k+1], dtype=DT)))
            self.log_std = nn.ParameterList(             # parent-independent N_j std
                [nn.Parameter(torch.zeros(1, dtype=DT)) for _ in range(d)])
            g = nn.init.calculate_gain('leaky_relu')     # Xavier init per network
            with torch.no_grad():
                for j in range(d):
                    for w in self.W:
                        nn.init.xavier_uniform_(w[j], gain=g)

        def _forward(self, x):
            for k in range(3):
                if k == 0:                               # first layer: mask inputs
                    x = torch.einsum("tij,ljt,bj->bti", self.W[k],
                                     self.adjacency.unsqueeze(0), x) + self.b[k]
                else:
                    x = torch.einsum("tij,btj->bti", self.W[k], x) + self.b[k]
                if k < 2:
                    x = F.leaky_relu(x)
            return torch.unbind(x, 1)                    # d tensors of (batch, 1)

        def log_lik(self, x, detach_target=False):
            mus = self._forward(x); parts = []
            for i in range(d):
                mu = mus[i].squeeze(1); sig = torch.exp(self.log_std[i])
                xi = x[:, i].detach() if detach_target else x[:, i]
                parts.append(distributions.Normal(mu, sig).log_prob(xi).unsqueeze(1))
            return torch.cat(parts, 1)                   # (batch, d)

        def w_adj(self):
            """A_phi: path-normalised product of |W|, summed over outputs.
            (A_phi)[i,j]=0  =>  theta_j independent of X_i."""
            prod = torch.eye(d, dtype=DT); pn = torch.eye(d, dtype=DT)
            off = (1.0 - torch.eye(d, dtype=DT)).unsqueeze(0)
            for i, w in enumerate(self.W):
                aw = torch.abs(w)
                if i == 0:
                    prod = torch.einsum("tij,ljt,jk->tik", aw, self.adjacency.unsqueeze(0), prod)
                    pn   = torch.einsum("tij,ljt,jk->tik", torch.ones_like(aw), off, pn)
                else:
                    prod = torch.einsum("tij,tjk->tik", aw, prod)
                    pn   = torch.einsum("tij,tjk->tik", torch.ones_like(aw), pn)
            prod = prod.sum(1); pn = pn.sum(1)
            return (prod / (pn + torch.eye(d, dtype=DT))).t()

    model = GranDagModel()

    tn = int(n * 0.8)                                    # train / held-out split
    Xtr = torch.as_tensor(X[:tn], dtype=DT); Xte = torch.as_tensor(X[tn:], dtype=DT)
    rtr = np.random.RandomState(seed); rte = np.random.RandomState(seed + 1)
    def sample(data, rng, bs):
        idx = rng.choice(data.shape[0], size=int(bs), replace=False)
        return data[torch.as_tensor(idx).long()]

    mu, lamb = 1e-3, 0.0                                 # mu_0, lambda_0
    opt = torch.optim.RMSprop(model.parameters(), lr=1e-3)
    a_val, not_nll, h_hist = [], [], []
    BS, ITER, WIN = min(64, tn), 30000, 100

    for it in range(ITER):
        model.train()
        loss = -model.log_lik(sample(Xtr, rtr, BS)).mean()
        model.eval()
        wa = model.w_adj()
        h = torch.trace(torch.matrix_exp(wa)) - d        # tr e^{A_phi} - d
        al = loss + 0.5 * mu * h ** 2 + lamb * h          # augmented Lagrangian
        opt.zero_grad(); al.backward(); opt.step()

        with torch.no_grad():                             # edge clamping
            model.adjacency *= (wa > 1e-4).to(DT)

        not_nll.append(0.5 * mu * h.item() ** 2 + lamb * h.item())
        if it % WIN == 0:                                 # held-out criterion
            with torch.no_grad():
                vl = -model.log_lik(sample(Xte, rte, Xte.shape[0])).mean()
                a_val.append([it, vl.item() + not_nll[-1]])

        delta = -np.inf
        if it >= 2 * WIN and it % (2 * WIN) == 0:
            t0, th, t1 = a_val[-3][1], a_val[-2][1], a_val[-1][1]
            delta = (t1 - t0) / WIN if (min(t0, t1) < th < max(t0, t1)) else -np.inf

        if h.item() > 1e-8:
            if abs(delta) < 1e-4 or delta > 0:           # subproblem converged
                lamb += mu * h.item()                     # dual ascent
                h_hist.append(h.item())
                if len(h_hist) >= 2 and h_hist[-1] > h_hist[-2] * 0.9:
                    mu *= 10                              # escalate penalty
                a_val[-1][1] += (0.5 * mu * h.item() ** 2 + lamb * h.item() - not_nll[-1])
                opt = torch.optim.RMSprop(model.parameters(), lr=1e-3)
        else:
            with torch.no_grad():
                model.adjacency *= (wa > 0).to(DT)        # h ~ 0: converged
            break

    # final DAG via expected-Jacobian sensitivity (realized dependence)
    model.eval()
    xj = Xtr.clone().requires_grad_(True)
    lps = torch.unbind(model.log_lik(xj, detach_target=True), 1)
    jac = torch.zeros(d, d, dtype=DT)
    for i in range(d):
        gi = torch.autograd.grad(lps[i], xj, retain_graph=True,
                                 grad_outputs=torch.ones(Xtr.shape[0], dtype=DT))[0]
        jac[i] = gi.abs().mean(0)
    A = jac.t().detach().numpy()

    with torch.no_grad():                                 # remove weakest until acyclic
        for thr in np.unique(A):
            keep = torch.tensor(A > thr + 1e-8, dtype=DT)
            na = model.adjacency * keep
            prod = torch.eye(d, dtype=DT); ok = True
            for _ in range(d):
                prod = na @ prod
                if prod.trace() != 0:
                    ok = False; break
            if ok:
                model.adjacency = na; break

    # adjacency[j, t]=1 means j->t; B[i, j]=1 means j->i  =>  B = adjacency.T
    return model.adjacency.t().detach().numpy()
```
