**Problem.** DirectLiNGAM failed because it is linear on nonlinear data (dense, wrong-direction graphs;
worst on Gaussian noise). The next rung must model the mechanisms as genuinely nonlinear functions,
while avoiding both the super-exponential DAG search and direction non-identifiability.

**Key idea.** GraN-DAG keeps NOTEARS's continuous-constraint paradigm but replaces the linear SEM with
one neural network per variable (the ANM conditional `X_j = f_j(parents) + N_j`). The obstacle — a stack
of nets has no coefficient matrix — is solved by running NOTEARS's closed-walk argument one level
deeper: input `i` reaches an output only along network paths, a path is dead iff any weight on it is
zero, so the sum of absolute path-products is the matrix product `|W^{(L)}|⋯|W^{(1)}|`; its
output-summed, column form is a nonnegative `d×d` adjacency `A` with `A_{ij}=0 ⇒ X_j ⊥ X_i`. Acyclicity
is then `tr e^{A} - d = 0`, solved by an augmented Lagrangian; the DAG is extracted by removing weakest
edges by *realized Jacobian sensitivity* (not by `A`, which only upper-bounds dependence).

**Why it should help — and where it stays weak.** Nonlinear ANMs are identifiable even with Gaussian
noise *because* the mechanisms are nonlinear, so GraN-DAG should lift ER20-Gauss off DirectLiNGAM's
floor. But it has *no explicit sparsity penalty and no significance pruning*, and the order is only
implicit in the constraint, so it tends to converge to over-connected acyclic graphs: low precision,
high SHD, large seed variance — especially on the 20-node graphs and the 150-sample low-data scenario.

**Same-named vs. paper.** Faithful gcastle-style GraN-DAG (NonlinearGaussANM, 2×10 leaky-ReLU, RMSprop
lr 1e-3, path-normalized weight-product adjacency, convergence-based augmented Lagrangian, Jacobian DAG
enforcement) — but with two task-specific deviations: edge clamping is applied **every 500 iterations at
a stricter 1e-3 threshold** (not every step at 1e-4) for cross-seed stability, and there is **no PNS and
no GAM pruning** (the only sparsity pressure is the constraint + clamp + Jacobian threshold).

**Hyperparameters.** `μ_0=1e-3`, `λ_0=0`, escalate `μ×10`, RMSprop lr `1e-3`, 30000 iters, window 100,
`h_tol=1e-8`, batch `min(64, 0.8n)`, 80/20 split, clamp threshold `1e-3` every 500 iters; seed from
`SEED`.

```python
def run_causal_discovery(X: np.ndarray) -> np.ndarray:
    """GraN-DAG (Lachapelle et al., ICLR 2020).

    B[i,j] != 0 means j -> i (causal-learn convention).
    """
    import os
    import torch, torch.nn as nn, torch.nn.functional as F
    from torch import distributions

    seed = int(os.environ.get("SEED", "42"))
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    np.random.seed(seed)

    n, d = X.shape
    DT = torch.float64

    # ================================================================== #
    # Per-variable MLP model (NonlinearGaussANM, 2x10, leaky-relu)       #
    # ================================================================== #
    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            # Explicit adjacency mask (not a Parameter -- updated by clamping only)
            self.adjacency = torch.ones(d, d, dtype=DT) - torch.eye(d, dtype=DT)
            layers = [d, 10, 10, 1]  # [input, hidden1, hidden2, output_dim]
            self.wt = nn.ParameterList()
            self.bi = nn.ParameterList()
            for k in range(len(layers) - 1):
                self.wt.append(nn.Parameter(
                    torch.zeros(d, layers[k + 1], layers[k], dtype=DT)))
                self.bi.append(nn.Parameter(
                    torch.zeros(d, layers[k + 1], dtype=DT)))
            # Per-variable learnable noise log-std (ANM model)
            self.log_std = nn.ParameterList(
                [nn.Parameter(torch.zeros(1, dtype=DT)) for _ in range(d)])
            # Xavier init matching gcastle's reset_params() order
            g = nn.init.calculate_gain('leaky_relu')
            with torch.no_grad():
                for nd in range(d):
                    for w in self.wt:
                        nn.init.xavier_uniform_(w[nd], gain=g)
                    for b in self.bi:
                        b[nd].zero_()

        def _fwd(self, x):
            """Per-variable forward pass with adjacency masking on first layer."""
            for k in range(3):
                if k == 0:
                    x = torch.einsum("tij,ljt,bj->bti", self.wt[k],
                                     self.adjacency.unsqueeze(0), x) \
                        + self.bi[k]
                else:
                    x = torch.einsum("tij,btj->bti", self.wt[k], x) \
                        + self.bi[k]
                if k < 2:
                    x = F.leaky_relu(x)
            return torch.unbind(x, 1)  # d tensors of (batch, 1)

        def log_lik(self, x, detach_target=False):
            """(batch, d) per-variable Gaussian log-likelihoods."""
            preds = self._fwd(x)
            parts = []
            for i in range(d):
                mu = preds[i].squeeze(1)
                sig = torch.exp(self.log_std[i])
                xi = x[:, i].detach() if detach_target else x[:, i]
                parts.append(
                    distributions.Normal(mu, sig).log_prob(xi).unsqueeze(1))
            return torch.cat(parts, 1)

        def w_adj(self):
            """Weighted adjacency via product of |weights|, path-normalised."""
            prod = torch.eye(d, dtype=DT)
            pn = torch.eye(d, dtype=DT)
            off = (1.0 - torch.eye(d, dtype=DT)).unsqueeze(0)
            for i, w in enumerate(self.wt):
                wa = torch.abs(w)
                if i == 0:
                    prod = torch.einsum("tij,ljt,jk->tik",
                                        wa, self.adjacency.unsqueeze(0), prod)
                    pn = torch.einsum("tij,ljt,jk->tik",
                                      torch.ones_like(wa), off, pn)
                else:
                    prod = torch.einsum("tij,tjk->tik", wa, prod)
                    pn = torch.einsum("tij,tjk->tik",
                                      torch.ones_like(wa), pn)
            prod = prod.sum(1)
            pn = pn.sum(1)
            return (prod / (pn + torch.eye(d, dtype=DT))).t()

    mdl = _M()

    # ================================================================== #
    # Data split (80/20, no shuffle, no normalise -- gcastle defaults)    #
    # ================================================================== #
    tn = int(n * 0.8)
    Xtr = torch.as_tensor(X[:tn], dtype=DT)
    Xte = torch.as_tensor(X[tn:], dtype=DT)
    rng_tr = np.random.RandomState(seed)
    rng_te = np.random.RandomState(seed + 1)

    def _samp(data, rng, bs):
        idx = rng.choice(data.shape[0], size=int(bs), replace=False)
        return data[torch.as_tensor(idx).long()]

    # ================================================================== #
    # Augmented-Lagrangian training (convergence-based mu/lambda update)  #
    # ================================================================== #
    mu, lamb = 0.001, 0.0        # penalty & dual variable
    opt = torch.optim.RMSprop(mdl.parameters(), lr=0.001)
    a_val, nns, hh = [], [], []  # validation AL, not-nll, constraint history
    BS, ITER, WIN = min(64, tn), 30000, 100

    for it in range(ITER):
        mdl.train()
        xb = _samp(Xtr, rng_tr, BS)
        loss = -mdl.log_lik(xb).mean()
        mdl.eval()

        wa = mdl.w_adj()
        h = torch.trace(torch.matrix_exp(wa)) - d
        al = loss + 0.5 * mu * h ** 2 + lamb * h

        opt.zero_grad()
        al.backward()
        opt.step()

        # Edge clamping -- only apply periodically to avoid premature
        # irreversible edge removal that causes instability across runs.
        # gcastle default threshold is 1e-4, but applying every step is
        # too aggressive; applying every 500 iterations with a stricter
        # weight threshold (1e-3) is more stable.
        if it > 0 and it % 500 == 0:
            with torch.no_grad():
                mdl.adjacency *= (wa > 1e-3).to(DT)

        nns.append(0.5 * mu * h.item() ** 2 + lamb * h.item())

        # Validation every WIN iterations
        if it % WIN == 0:
            with torch.no_grad():
                vl = -mdl.log_lik(
                    _samp(Xte, rng_te, Xte.shape[0])).mean()
                a_val.append([it, vl.item() + nns[-1]])

        # Convergence delta (checked every 2*WIN)
        dl = -np.inf
        if it >= 2 * WIN and it % (2 * WIN) == 0:
            t0, th, t1 = a_val[-3][1], a_val[-2][1], a_val[-1][1]
            if not (min(t0, t1) < th < max(t0, t1)):
                dl = -np.inf
            else:
                dl = (t1 - t0) / WIN

        # Lambda / mu update
        if h.item() > 1e-8:
            if abs(dl) < 1e-3 or dl > 0:
                lamb += mu * h.item()
                hh.append(h.item())
                if len(hh) >= 2 and hh[-1] > hh[-2] * 0.9:
                    mu *= 10
                # Adjust moving-average validation to account for new mu/lambda
                gap = (0.5 * mu * h.item() ** 2
                       + lamb * h.item() - nns[-1])
                a_val[-1][1] += gap
                opt = torch.optim.RMSprop(mdl.parameters(), lr=0.001)
        else:
            # Converged -- final clamping of zero-weight edges
            with torch.no_grad():
                mdl.adjacency *= (wa > 0).to(DT)
            break

    # ================================================================== #
    # DAG enforcement: Jacobian threshold + weakest-edge removal          #
    # ================================================================== #
    mdl.eval()
    xj = Xtr.clone().requires_grad_(True)
    ll = mdl.log_lik(xj, detach_target=True)       # (tn, d)
    lps = torch.unbind(ll, 1)                       # d tensors of (tn,)
    jac = torch.zeros(d, d, dtype=DT)
    for i in range(d):
        g = torch.autograd.grad(
            lps[i], xj, retain_graph=True,
            grad_outputs=torch.ones(Xtr.shape[0], dtype=DT))[0]
        jac[i] = g.abs().mean(0)
    A = jac.t().detach().numpy()

    # Find smallest threshold that produces an acyclic graph
    with torch.no_grad():
        for thr in np.unique(A):
            keep = torch.tensor(A > thr + 1e-8, dtype=DT)
            na = mdl.adjacency * keep
            # Acyclicity check via matrix-power trace
            prod = torch.eye(d, dtype=DT)
            ok = True
            for _ in range(d):
                prod = na @ prod
                if prod.trace() != 0:
                    ok = False
                    break
            if ok:
                mdl.adjacency = na
                break

    # adj[j,t]=1 means j->t; B[i,j]=1 means j->i  =>  B = adj.T
    return mdl.adjacency.t().detach().numpy()
```
