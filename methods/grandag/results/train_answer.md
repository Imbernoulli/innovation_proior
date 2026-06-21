We are given observational samples of a random vector X = (X_1, ..., X_d) and the goal is to recover the directed acyclic graph G that encodes which variable directly causes which. Two difficulties block the obvious approaches. First, the space of DAGs grows super-exponentially in d, and acyclicity is a global discrete constraint, so standard methods fall back on greedy edge-addition or ordering searches. Second, direction is not identifiable from the joint distribution P_X alone: many DAGs share the same conditional independences, so observational data can only recover a Markov equivalence class unless we add parametric assumptions that break the cause-effect symmetry.

Existing ideas each leave a gap. Constraint-based algorithms such as PC return a CPDAG with unoriented edges. Linear score-based methods such as NOTEARS cast structure learning as continuous optimization, but encode the graph as a single coefficient matrix and therefore cannot represent nonlinear mechanisms. CAM decouples order search from sparse additive regression, but it assumes additive mechanisms with no parent interactions and still relies on a greedy order search. RESIT directly exploits additive-noise identifiability by residual independence testing, yet its combinatorial search barely scales past twenty nodes. What is needed is a method that keeps the continuous-constraint freedom of NOTEARS while fitting a genuinely nonlinear, per-variable conditional model.

I propose GraN-DAG, short for Gradient-based Neural DAG learning. The method assigns each variable j its own neural network NN_j that takes the other variables as input and outputs the parameters of X_j's conditional distribution. For the standard nonlinear-Gaussian additive noise model, this means NN_j predicts the mean of X_j given its candidate parents, while a learned parent-independent noise standard deviation captures the independent noise term N_j. This gives enough capacity to represent arbitrary nonlinear mechanisms, but it introduces a representation problem: NOTEARS needs a single d × d weighted adjacency matrix to enforce acyclicity, while a stack of neural networks has no such coefficient.

The central construction solves this by reading a nonnegative weighted adjacency A_phi directly from the network weights. Information flows from input i to output k only along computation paths through the layers. A path is dead whenever any weight on it is zero, so the absolute product of weights along a path is a nonnegative number that is zero exactly when the path is inactive. Summing these path products over all routes is the same as multiplying the absolute-value weight matrices: C = |W^{(L+1)}| ... |W^{(2)}| |W^{(1)}|. Then C_{ki} is the total strength from input i to output k, and C_{ki} = 0 is sufficient for output k to be independent of input i. Summing over the m output components and transposing into i → j convention gives the weighted adjacency (A_phi)_{ij}; if it is zero, variable j's conditional does not depend on variable i.

Because A_phi is entrywise nonnegative by construction, the smooth acyclicity constraint from NOTEARS applies directly: h(phi) = tr e^{A_phi} - d = 0 if and only if the implied graph is acyclic. The learning problem is therefore a constrained maximum-likelihood program, maximizing the sum of per-variable conditional log-likelihoods subject to h(phi) = 0. In the population limit, if the true nonlinear additive-noise model lies inside the representable class, identifiability ensures that the constrained optimum recovers the true DAG. In practice the problem is solved by an augmented Lagrangian: a sequence of unconstrained subproblems that penalize h(phi)^2 and pull it toward zero through a dual variable, solved with minibatch RMSprop. Small edge weights are periodically clamped to zero during training, and because A_phi only upper-bounds true dependence, the final DAG is extracted by ranking edges by the realized expected absolute Jacobian of each conditional log-likelihood and removing the weakest edges until acyclicity holds.

```python
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions


def run_causal_discovery(X: np.ndarray) -> np.ndarray:
    """GraN-DAG: learn a DAG from observational nonlinear (Gauss-ANM) data.

    Returns B with B[i, j] != 0 meaning j -> i.
    """
    seed = int(os.environ.get("SEED", "42"))
    torch.manual_seed(seed)
    np.random.seed(seed)
    n, d = X.shape
    DT = torch.float64

    class GranDagModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.adjacency = torch.ones(d, d, dtype=DT) - torch.eye(d, dtype=DT)
            layers = [d, 10, 10, 1]
            self.W = nn.ParameterList()
            self.b = nn.ParameterList()
            for k in range(len(layers) - 1):
                self.W.append(nn.Parameter(torch.zeros(d, layers[k + 1], layers[k], dtype=DT)))
                self.b.append(nn.Parameter(torch.zeros(d, layers[k + 1], dtype=DT)))
            self.log_std = nn.ParameterList(
                [nn.Parameter(torch.zeros(1, dtype=DT)) for _ in range(d)])
            g = nn.init.calculate_gain('leaky_relu')
            with torch.no_grad():
                for j in range(d):
                    for w in self.W:
                        nn.init.xavier_uniform_(w[j], gain=g)

        def _forward(self, x):
            for k in range(3):
                if k == 0:
                    x = torch.einsum("tij,ljt,bj->bti", self.W[k],
                                     self.adjacency.unsqueeze(0), x) + self.b[k]
                else:
                    x = torch.einsum("tij,btj->bti", self.W[k], x) + self.b[k]
                if k < 2:
                    x = F.leaky_relu(x)
            return torch.unbind(x, 1)

        def log_lik(self, x, detach_target=False):
            mus = self._forward(x)
            parts = []
            for i in range(d):
                mu = mus[i].squeeze(1)
                sig = torch.exp(self.log_std[i])
                xi = x[:, i].detach() if detach_target else x[:, i]
                parts.append(distributions.Normal(mu, sig).log_prob(xi).unsqueeze(1))
            return torch.cat(parts, 1)

        def w_adj(self):
            prod = torch.eye(d, dtype=DT)
            pn = torch.eye(d, dtype=DT)
            off = (1.0 - torch.eye(d, dtype=DT)).unsqueeze(0)
            for i, w in enumerate(self.W):
                aw = torch.abs(w)
                if i == 0:
                    prod = torch.einsum("tij,ljt,jk->tik", aw,
                                        self.adjacency.unsqueeze(0), prod)
                    pn = torch.einsum("tij,ljt,jk->tik",
                                      torch.ones_like(aw), off, pn)
                else:
                    prod = torch.einsum("tij,tjk->tik", aw, prod)
                    pn = torch.einsum("tij,tjk->tik", torch.ones_like(aw), pn)
            prod = prod.sum(1)
            pn = pn.sum(1)
            return (prod / (pn + torch.eye(d, dtype=DT))).t()

    model = GranDagModel()

    tn = int(n * 0.8)
    Xtr = torch.as_tensor(X[:tn], dtype=DT)
    Xte = torch.as_tensor(X[tn:], dtype=DT)
    rtr = np.random.RandomState(seed)
    rte = np.random.RandomState(seed + 1)

    def sample(data, rng, bs):
        idx = rng.choice(data.shape[0], size=int(bs), replace=False)
        return data[torch.as_tensor(idx).long()]

    mu, lamb = 1e-3, 0.0
    opt = torch.optim.RMSprop(model.parameters(), lr=1e-3)
    a_val, not_nll, h_hist = [], [], []
    BS, ITER, WIN = min(64, tn), 30000, 100

    for it in range(ITER):
        model.train()
        loss = -model.log_lik(sample(Xtr, rtr, BS)).mean()
        model.eval()
        wa = model.w_adj()
        h = torch.trace(torch.matrix_exp(wa)) - d
        al = loss + 0.5 * mu * h ** 2 + lamb * h
        opt.zero_grad()
        al.backward()
        opt.step()

        if it > 0 and it % 500 == 0:
            with torch.no_grad():
                model.adjacency *= (wa > 1e-3).to(DT)

        not_nll.append(0.5 * mu * h.item() ** 2 + lamb * h.item())
        if it % WIN == 0:
            with torch.no_grad():
                vl = -model.log_lik(sample(Xte, rte, Xte.shape[0])).mean()
                a_val.append([it, vl.item() + not_nll[-1]])

        delta = -np.inf
        if it >= 2 * WIN and it % (2 * WIN) == 0:
            t0, th, t1 = a_val[-3][1], a_val[-2][1], a_val[-1][1]
            if min(t0, t1) < th < max(t0, t1):
                delta = (t1 - t0) / WIN

        if h.item() > 1e-8:
            if abs(delta) < 1e-3 or delta > 0:
                lamb += mu * h.item()
                h_hist.append(h.item())
                if len(h_hist) >= 2 and h_hist[-1] > h_hist[-2] * 0.9:
                    mu *= 10
                a_val[-1][1] += (0.5 * mu * h.item() ** 2
                                 + lamb * h.item() - not_nll[-1])
                opt = torch.optim.RMSprop(model.parameters(), lr=1e-3)
        else:
            with torch.no_grad():
                model.adjacency *= (wa > 0).to(DT)
            break

    model.eval()
    xj = Xtr.clone().requires_grad_(True)
    lps = torch.unbind(model.log_lik(xj, detach_target=True), 1)
    jac = torch.zeros(d, d, dtype=DT)
    for i in range(d):
        gi = torch.autograd.grad(lps[i], xj, retain_graph=True,
                                 grad_outputs=torch.ones(Xtr.shape[0], dtype=DT))[0]
        jac[i] = gi.abs().mean(0)
    A = jac.t().detach().numpy()

    with torch.no_grad():
        for thr in np.unique(A):
            keep = torch.tensor(A > thr + 1e-8, dtype=DT)
            na = model.adjacency * keep
            prod = torch.eye(d, dtype=DT)
            ok = True
            for _ in range(d):
                prod = na @ prod
                if prod.trace() != 0:
                    ok = False
                    break
            if ok:
                model.adjacency = na
                break

    return model.adjacency.t().detach().numpy()
```
