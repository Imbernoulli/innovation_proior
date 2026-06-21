DirectLiNGAM gave me the floor, and its numbers say exactly where to push: F1 of 0.319 on SF20-GP, 0.245 on ER20-Gauss, 0.188 on ER12-LowSample, with brutal SHD (60 and 96 on the two 20-node graphs) and precision around 0.18–0.24. It is laying down many arrows and most are wrong — the dense, wrong-direction graph a *linear* residual produces on nonlinearly-generated data, because $x_i - \beta x_j$ stays $x_j$-dependent in both directions and the exogeneity test can no longer separate source from non-source. The sharpest confirmation is ER20-Gauss being worst on both F1 and SHD: nonlinear mechanisms *and* Gaussian noise, so the Darmois–Skitovitch converse the whole method rests on simply fails. The diagnosis is not "tune DirectLiNGAM" but "it is solving the wrong problem" — the leverage on this data is the nonlinearity. So the next rung must model the mechanisms as genuinely nonlinear functions, without inheriting the two diseases of the field: the super-exponential combinatorial DAG search and direction non-identifiability.

Identifiability first, because it says the problem is solvable. The additive-noise model $X_j = f_j(X_{\mathrm{pa}(j)}) + N_j$ with independent noise and *nonlinear* $f_j$ is identifiable from the distribution — and crucially this holds even when the noise is *Gaussian*, as long as $f$ is nonlinear. That is precisely why DirectLiNGAM died on ER20-Gauss and a nonlinear method should not: there the nonlinearity, not the noise shape, carries the direction signal. But the model class I fit has to be rich enough to *represent* the nonlinear mechanism, or I cannot see the fingerprint at all.

I propose **GraN-DAG** (Lachapelle et al., 2020). It keeps NOTEARS's continuous-constraint paradigm — which avoids searching the discrete DAG space — but replaces the linear SEM with one neural network per variable. To set up the constraint I reproduce NOTEARS's acyclicity argument, because I will run it twice. For a nonnegative matrix $B$, $(B^k)_{jj}$ counts closed walks of length $k$ through node $j$, so $\mathrm{tr}(B^k)$ counts length-$k$ cycles, and $B$ is acyclic iff $\mathrm{tr}(B^k) = 0$ for every $k$. The finite powers overflow; the matrix exponential reweights the length-$k$ counts by $1/k!$, taming the explosion and giving the clean statement $\mathrm{tr}\,e^{B} = d$ iff $B$ is a DAG. For real weights one replaces $B$ by its Hadamard square, $h(W) = \mathrm{tr}\,e^{W\circ W} - d = 0$. Solve a smooth score subject to $h = 0$ with an augmented Lagrangian and the whole graph updates at once.

NOTEARS itself is not enough, because its $W$ *is* the coefficient matrix of a *linear* SEM — the contribution of $i$ to $j$ is the single scalar $W_{ij}$, leaving no room for nonlinear dependence, so on this data it would mis-score directions just as DirectLiNGAM did. I want the continuous constraint but with a flexible, *independent* nonlinear model per variable: give each variable $j$ its own network taking the others as input and outputting the mean of $X_j$'s Gaussian conditional, with a learned parent-independent noise std — that is the ANM written exactly. Mask the $j$-th input so a variable cannot be its own parent.

The wall, and the heart of the method, is that with a stack of nets there is no single coefficient telling me whether $X_j$ depends on $X_i$ — variable $i$ enters $\mathrm{NN}_j$ through a tangle of weights across every layer. So before I can write $h$ I must manufacture, out of the weights, a single nonnegative $A_{ij}$ that is zero exactly when the output does not depend on input $i$. Information flows from $i$ to an output only along computation *paths* through the hidden units, and a path is dead iff any weight on it is zero. So output $k$ is independent of input $i$ iff *every* path from $i$ to $k$ is dead. Quantify a path by the product of absolute weights along it — nonnegative, zero iff some link is zero — and "every path dead" becomes "the sum of all path products is zero," since a sum of nonnegatives vanishes iff every term does. Summing path products over all intermediate indices *is* matrix multiplication of the absolute-weight matrices: $C = |W^{(L)}| \cdots |W^{(1)}|$, with $C_{ki}$ the total path strength from input $i$ to output $k$. Sum over outputs, zero the diagonal, and $A$ is the nonnegative $d\times d$ matrix $h$ wanted — born nonnegative, so I need no Hadamard square. The constraint is $h = \mathrm{tr}\,e^{A} - d = 0$. I have run the walk-counting argument twice: once over neural-network paths to build $A$, once over graph paths to constrain it.

In implementation each per-variable model is an MLP with **two hidden layers of ten leaky-ReLU units**, Xavier-initialized, weights stacked as a $(d, \text{out}, \text{in})$ tensor per layer so all $d$ networks evaluate in parallel via one `einsum`; the first layer is masked by an `adjacency` matrix (initialized to all-ones-minus-identity). The score is the per-variable Gaussian log-likelihood with a learned per-variable `log_std` (the ANM noise). The adjacency $A$ is the **path-normalized** product of absolute weight matrices — the raw path product divided by the product of all-ones matrices through the same mask, keeping entries on a comparable scale across depth — column-summed over outputs and transposed into the $i\to j$ reading. The optimizer is **RMSprop at lr $10^{-3}$** with the convergence-based augmented-Lagrangian schedule ($\mu_0 = 10^{-3}$, $\lambda_0 = 0$, $\lambda \mathbin{+}= \mu h$, escalate $\mu \times 10$ when the constraint stops shrinking, reset the optimizer state on each update), an 80/20 split with the held-out augmented Lagrangian as the convergence signal, up to 30000 iterations and $h_{\text{tol}} = 10^{-8}$.

Two harness-specific choices shape the numbers. First, **edge clamping**: the gcastle default masks any input whose $A$ entry falls below $10^{-4}$ *every step*, but I apply the clamp only every 500 iterations at a stricter $10^{-3}$ threshold, because per-step clamping irreversibly kills edges too aggressively and destabilizes runs across seeds — a stability patch, not the canonical schedule, so this fill is less prone to prematurely killing a true edge but slower to sparsify. Second, **final DAG enforcement**: rather than thresholding $A$ (which only *upper-bounds* true dependence, since path products can cancel in the realized function or hidden units can saturate), I compute the **realized sensitivity** — the expected absolute Jacobian of each conditional's log-likelihood with respect to each input — and remove the weakest edges in increasing order of that Jacobian until the graph is acyclic, checked by the $\mathrm{tr}(A^k)=0$ closed-walk fact on the thresholded binary graph. What this fill omits relative to the fullest procedure is the CAM-style preliminary neighbor selection and the GAM significance pruning: there is **no PNS and no post-hoc significance test**, so the only sparsity pressure is the $\ell$-free augmented Lagrangian, the clamping, and the Jacobian threshold.

That last point is where I expect GraN-DAG to stay weak, and it is the same place DirectLiNGAM was weak: precision on dense graphs. With no explicit sparsity penalty, a per-variable net free to fit spurious dependence, and the order learned only implicitly through the constraint, the augmented Lagrangian can converge to a graph that is *acyclic but over-connected* — low precision, high recall, high SHD — especially on the 20-node graphs, with large per-seed variance because RMSprop on a non-convex objective with reinitialized state lands in different basins per seed. So the falsifiable claim is narrow: GraN-DAG should prove the nonlinearity is the leverage by lifting ER20-Gauss off DirectLiNGAM's floor, while its lack of disciplined edge selection leaves it precision-starved and SHD-heavy — the gap a later rung must close.

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
