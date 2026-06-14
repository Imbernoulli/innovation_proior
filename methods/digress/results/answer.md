# DiGress, distilled

DiGress is a **discrete** denoising diffusion model for generating graphs with categorical
node and edge attributes. It corrupts a graph by editing node/edge *categories* through Markov
transition matrices applied independently to each node and each edge, keeping the graph a genuine
sparse discrete graph at every step, and trains a graph transformer to predict the *clean* graph
from the noisy one. Because the clean target of each node and edge is a class label, distribution
learning over graphs collapses into a pile of independent node/edge **classification** tasks
trained with cross-entropy — no graph matching, permutation invariant. Sampling iterates a reverse
step that folds the network's clean-graph prediction through the analytic Bayes posterior.

## Problem it solves

One-shot generation of graphs (categorical nodes/edges) that match a data distribution, with the
two structural constraints honored: permutation equivariance with a permutation-invariant loss
(no arbitrary node order, no n!-permutation likelihood), and preserved sparsity/structure (unlike
continuous/Gaussian graph diffusion, which densifies the adjacency mid-process and erases edges,
connectivity, and cycle counts).

## Key ideas

1. **Discrete, per-element diffusion.** For a categorical variable with K classes (one-hot row
   vector `x`), the forward step is a transition matrix: `q(z^t|z^{t-1}) = z^{t-1} Q^t`,
   `[Q^t]_{ij}=P(i→j)`. Diffuse independently on each node type (matrix `Q_X`, size a×a) and edge
   type (`Q_E`, size b×b, "no edge" being one type), never on the exponential graph state:
   `q(G^t|G) = (X Q̄_X^t, E Q̄_E^t)` with `Q̄^t = Q^1…Q^t`. Undirected: noise the upper triangle of
   `E` and symmetrize. Three required diffusion properties all hold: closed-form marginal `Q̄^t`,
   closed-form posterior, x-independent limit.
2. **Closed-form Bayes posterior.** `q(z^{t-1}|z^t,x) ∝ z^t (Q^t)′ ⊙ x Q̄^{t-1}`
   (normalizer `x Q̄^t (z^t)′`). Derived by Bayes + Markov: `z^t(Q^t)′` is the likelihood
   vector over candidate previous classes, and `xQ̄^{t-1}` is the prior marginal at step `t-1`.
3. **Predict the clean graph (x₀-parameterization) ⇒ classification.** The denoiser `φ_θ(G^t)`
   outputs per-node and per-edge distributions over *clean* classes `(p̂^X, p̂^E)`; train with
   `l = Σ_i CE(x_i, p̂^X_i) + λ Σ_{ij} CE(e_{ij}, p̂^E_{ij})` (default `λ=5` on edges). This is
   permutation invariant exactly because it is a sum of identical per-node / per-edge terms.
4. **Marginal transitions (the key practical improvement).** Uniform noise drifts sparse graphs
   into dense junk, far from data, so the model wastes denoising steps re-sparsifying. The prior
   must be permutation-invariant, so the usable independent family has one node distribution `u`
   and one edge distribution `v`: `∏_i u × ∏_{ij} v`. **Optimal-prior L₂ projection, stated on the
   statistics this prior can represent:** the independent invariant prior cannot encode
   correlations, so project the collection of one-site marginals onto the shared-node/shared-edge
   family. Proof: minimize `J_X(u)=Σ_i ||u-p_i^X||²`, where `p_i^X` is the node-type marginal at
   slot `i`; expanding yields `J_X(u)=n||u-(1/n)Σ_i p_i^X||²+const`, so `u=(1/n)Σ_i p_i^X=m_X`.
   Similarly, `J_E(v)=Σ_ij ||v-p_ij^E||²` gives `v=(1/n²)Σ_ij p_ij^E=m_E`. Achieve this limit with
   `Q_X^t = α^t I + β^t 1 m_X′`, `Q_E^t = α^t I + β^t 1 m_E′` (`β^t = 1−α^t`); since `(1m′)²=1m′`,
   `Q̄^t = ᾱ^t I + β̄^t 1 m′` with `ᾱ^t=∏α^τ`, `β̄^t=1−ᾱ^t`, rows → `m`. Cosine schedule
   `ᾱ^t = cos²(½π (t/T+s)/(1+s)) / cos²(½π s/(1+s))`, `s≈0.008`.
5. **Structural & spectral feature augmentation.** MPNNs/graph transformers are ≤1-WL and cannot
   count cycles. Since discrete noisy graphs stay sparse and well-defined, compute descriptors at
   each step and feed them in: cycle counts (closed-form trace/Frobenius formulas, e.g.
   `X₃=diag(A³)/2`, up to 5-cycles per node and 6-cycles per graph), Laplacian spectral features
   (# components = mult. of eigenvalue 0, first nonzero eigenvalues/eigenvectors), and molecular
   features (valency, weight). This augments the denoiser; it is not required to define the
   diffusion process itself.
6. **Equivariance / exchangeability.** Equivariant network + invariant product limit + equivariant
   reverse map ⇒ exchangeable generated distribution ⇒ likelihood tractable on one representative
   (an ELBO from the per-step posterior KLs gives an NLL for comparison, though training uses CE).

## Reverse (sampling) step

Marginalize the analytic posterior over the network's clean-graph belief:
`p_θ(x_i^{t-1}|G^t) = Σ_x q(x_i^{t-1}|x_i=x, x_i^t) p̂^X_i(x)` (similarly for edges), model the
joint reverse as a product over nodes/edges, sample a discrete `G^{t-1}`. Loop `t=T→1` from
`G^T ∼ q_X × q_E` (the marginal limit), node count drawn from the empirical distribution.

## Conditional generation (discrete classifier guidance)

Train a regressor `g_η(G^t)=ŷ`. Conditional reverse: `q̇(G^{t-1}|G^t,y) ∝ q(G^{t-1}|G^t) q̇(y|G^{t-1})`.
First-order expansion makes the (non-factorized) guidance term split over nodes/edges; with
`q̇(y|G^t)=N(g(G^t),σ_y I)`, `p_η(ŷ|G^{t-1}) ∝ exp(−λ⟨∇_{G^t}‖ŷ−y‖², G^{t-1}⟩)`, multiplied into
the reverse step. Subgraph conditioning: mask-and-overwrite kept nodes/edges each step.

## Algorithms

```
Training DiGress (one step):
  sample t ~ U(1..T)
  sample G^t ~ (X Q̄_X^t, E Q̄_E^t)            # discrete noisy graph
  z   = f(G^t, t)                              # structural + spectral features
  p̂^X, p̂^E = φ_θ(G^t, z)                       # predict clean classes
  step on  l_CE(p̂^X, X) + λ l_CE(p̂^E, E)       # cross-entropy

Sampling from DiGress:
  sample n ~ data;  G^T ~ q_X(n) × q_E(n)      # marginal (or uniform) limit
  for t = T .. 1:
    z = f(G^t, t);  p̂^X, p̂^E = φ_θ(G^t, z)
    p_θ(x_i^{t-1}|G^t) = Σ_x q(x_i^{t-1}|x_i=x, x_i^t) p̂^X_i(x)   # and edges
    G^{t-1} ~ ∏_i p_θ(x_i^{t-1}|G^t) ∏_{ij} p_θ(e_{ij}^{t-1}|G^t)
  return G^0
```

## Working code

A self-contained implementation faithful to the canonical structure: marginal transition
matrices, cosine schedule, a graph-transformer denoiser predicting clean node/edge classes,
cross-entropy training, and reverse sampling that marginalizes the analytic posterior over the
prediction. (For a plain structural dataset: node type trivial `a=1`, edge type `b=2` =
{no-edge, edge}.)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math


def cosine_alpha_bar(T, s=0.008):
    steps = torch.arange(T + 1, dtype=torch.float64)
    abar = torch.cos(0.5 * math.pi * (steps / T + s) / (1 + s)) ** 2
    abar = abar / abar[0]
    return torch.clamp(abar, 1e-6, 1.0)


class GraphTransformerLayer(nn.Module):
    """Node, edge and graph-level streams with edge-aware FiLM attention."""
    def __init__(self, dim, n_heads=8, ff_dim=None):
        super().__init__()
        ff_dim = ff_dim or 4 * dim
        self.n_heads, self.head_dim = n_heads, dim // n_heads
        self.q = nn.Linear(dim, dim); self.k = nn.Linear(dim, dim); self.v = nn.Linear(dim, dim)
        self.e_mul = nn.Linear(dim, dim); self.e_add = nn.Linear(dim, dim)
        self.y_x_mul = nn.Linear(dim, dim); self.y_x_add = nn.Linear(dim, dim)
        self.y_e_mul = nn.Linear(dim, dim); self.y_e_add = nn.Linear(dim, dim)
        self.x_out = nn.Linear(dim, dim); self.e_out = nn.Linear(dim, dim)
        self.y_out = nn.Sequential(nn.Linear(9 * dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.normX1 = nn.LayerNorm(dim); self.normX2 = nn.LayerNorm(dim)
        self.normE1 = nn.LayerNorm(dim); self.normE2 = nn.LayerNorm(dim)
        self.normY1 = nn.LayerNorm(dim); self.normY2 = nn.LayerNorm(dim)
        self.ffX = nn.Sequential(nn.Linear(dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, dim))
        self.ffE = nn.Sequential(nn.Linear(dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, dim))
        self.ffY = nn.Sequential(nn.Linear(dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, dim))

    def forward(self, X, E, y, node_mask):
        B, N, C = X.shape
        x_mask = node_mask.unsqueeze(-1).float()
        e_mask = (node_mask.unsqueeze(1) & node_mask.unsqueeze(2)).unsqueeze(-1).float()
        q = self.q(X).view(B, N, self.n_heads, self.head_dim)
        k = self.k(X).view(B, N, self.n_heads, self.head_dim)
        v = self.v(X).view(B, N, self.n_heads, self.head_dim)
        scores = (q[:, :, None] * k[:, None, :]) / math.sqrt(self.head_dim)
        e_mul = self.e_mul(E).view(B, N, N, self.n_heads, self.head_dim)
        e_add = self.e_add(E).view(B, N, N, self.n_heads, self.head_dim)
        scores = scores * (e_mul + 1) + e_add
        logits = scores.sum(-1).masked_fill(e_mask.squeeze(-1).unsqueeze(-1) == 0, -1e9)
        newE = scores.flatten(start_dim=3)
        newE = self.y_e_add(y).view(B, 1, 1, C) + (self.y_e_mul(y).view(B, 1, 1, C) + 1) * newE
        E = self.normE1(E + self.e_out(newE) * e_mask)
        E = self.normE2(E + self.ffE(E))
        attn = F.softmax(logits, dim=2)
        out = torch.einsum('bijh,bjhd->bihd', attn, v).reshape(B, N, C)
        out = self.y_x_add(y).view(B, 1, C) + (self.y_x_mul(y).view(B, 1, C) + 1) * out
        X = self.normX1(X + self.x_out(out) * x_mask)
        X = self.normX2(X + self.ffX(X))
        x_count = x_mask.sum(1).clamp_min(1)
        e_count = e_mask.sum((1, 2)).clamp_min(1)
        x_mean = (X * x_mask).sum(1) / x_count
        e_mean = (E * e_mask).sum((1, 2)) / e_count
        x_min = X.masked_fill(x_mask == 0, float("inf")).min(1).values
        x_max = X.masked_fill(x_mask == 0, -float("inf")).max(1).values
        e_min = E.masked_fill(e_mask == 0, float("inf")).amin((1, 2))
        e_max = E.masked_fill(e_mask == 0, -float("inf")).amax((1, 2))
        x_std = torch.sqrt((((X - x_mean[:, None]) * x_mask) ** 2).sum(1) / x_count)
        e_std = torch.sqrt((((E - e_mean[:, None, None]) * e_mask) ** 2).sum((1, 2)) / e_count)
        x_pool = torch.cat([x_mean, x_min, x_max, x_std], dim=-1)
        e_pool = torch.cat([e_mean, e_min, e_max, e_std], dim=-1)
        y = self.normY1(y + self.y_out(torch.cat([y, x_pool, e_pool], dim=-1)))
        y = self.normY2(y + self.ffY(y))
        return X * x_mask, E * e_mask, y


class Denoiser(nn.Module):
    def __init__(self, a, b, extra_x=0, extra_e=0, extra_y=0, dim=256, n_layers=5, n_heads=8):
        super().__init__()
        self.a, self.b = a, b
        self.node_in = nn.Sequential(nn.Linear(a + extra_x, dim), nn.ReLU(), nn.Linear(dim, dim), nn.ReLU())
        self.edge_in = nn.Sequential(nn.Linear(b + extra_e, dim), nn.ReLU(), nn.Linear(dim, dim), nn.ReLU())
        self.y_in = nn.Sequential(nn.Linear(1 + extra_y, dim), nn.ReLU(), nn.Linear(dim, dim), nn.ReLU())
        self.layers = nn.ModuleList([GraphTransformerLayer(dim, n_heads) for _ in range(n_layers)])
        self.node_out = nn.Linear(dim, a); self.edge_out = nn.Linear(dim, b)

    def forward(self, Xt, Et, t_frac, node_mask, extraX=None, extraE=None, extraY=None):
        B, N, _ = Xt.shape
        if extraX is not None:
            Xt = torch.cat([Xt, extraX], dim=-1)
        if extraE is not None:
            Et = torch.cat([Et, extraE], dim=-1)
        y_in = t_frac.view(B, 1).float()
        if extraY is not None:
            y_in = torch.cat([y_in, extraY], dim=-1)
        X = self.node_in(Xt); E = self.edge_in(Et); y = self.y_in(y_in)
        for layer in self.layers:
            X, E, y = layer(X, E, y, node_mask)
        edge_logits = self.edge_out(E)
        edge_logits = 0.5 * (edge_logits + edge_logits.transpose(1, 2))
        return self.node_out(X), edge_logits


class GraphGenerator(nn.Module):
    def __init__(self, max_nodes, n_node_types=1, n_edge_types=2,
                 extra_x=0, extra_e=0, extra_y=0, dim=256, n_layers=5, n_heads=8,
                 T=500, lr=2e-4, lambda_e=5.0, feature_fn=None, **kwargs):
        super().__init__()
        self.max_nodes, self.a, self.b = max_nodes, n_node_types, n_edge_types
        self.T, self.lambda_e = T, lambda_e
        self.denoiser = Denoiser(self.a, self.b, extra_x, extra_e, extra_y, dim, n_layers, n_heads)
        self.feature_fn = feature_fn
        self.register_buffer('abar', cosine_alpha_bar(T))
        self.register_buffer('mX', torch.ones(self.a) / self.a)
        self.register_buffer('mE', torch.ones(self.b) / self.b)
        self.optimizer = optim.AdamW(self.denoiser.parameters(), lr=lr, weight_decay=1e-12, amsgrad=True)
        self._counts = None

    def set_marginals(self, mX, mE):
        self.mX.copy_(mX); self.mE.copy_(mE)

    def _extra_features(self, Xt, Et, t_frac, node_mask):
        if self.feature_fn is None:
            return None, None, None
        return self.feature_fn(Xt, Et, t_frac, node_mask)

    def _Qbar(self, abar_t, m):
        K = m.numel(); I = torch.eye(K, device=m.device)
        one_m = torch.ones(K, 1, device=m.device) @ m.view(1, K)
        return abar_t * I + (1 - abar_t) * one_m

    def _Qt(self, abar_t, abar_s, m):
        K = m.numel(); I = torch.eye(K, device=m.device)
        one_m = torch.ones(K, 1, device=m.device) @ m.view(1, K)
        alpha_t = (abar_t / abar_s).clamp(0, 1)
        return alpha_t * I + (1 - alpha_t) * one_m

    def _apply_noise(self, Xoh, Eoh, node_mask, t_idx):
        B = Xoh.shape[0]
        Xt = torch.empty_like(Xoh); Et = torch.empty_like(Eoh)
        N = Xoh.shape[1]
        edge_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        edge_mask = edge_mask & ~torch.eye(N, device=Xoh.device, dtype=torch.bool).unsqueeze(0)
        for bb in range(B):
            at = self.abar[t_idx[bb]]
            pX = Xoh[bb] @ self._Qbar(at, self.mX)
            pE = Eoh[bb] @ self._Qbar(at, self.mE)
            Xt[bb] = F.one_hot(torch.multinomial(pX, 1).squeeze(-1), self.a).float()
            e = torch.multinomial(pE.reshape(-1, self.b), 1).reshape(pE.shape[:-1])
            e = torch.triu(e, 1); e = e + e.transpose(0, 1)
            Et[bb] = F.one_hot(e, self.b).float()
        Xt[~node_mask] = F.one_hot(torch.zeros((), dtype=torch.long, device=Xoh.device), self.a).float()
        Et[~edge_mask] = F.one_hot(torch.zeros((), dtype=torch.long, device=Eoh.device), self.b).float()
        return Xt, Et

    def train_step(self, Xoh, Eoh, y=None, node_mask=None, extraX=None, extraE=None, extraY=None):
        self.train(); self.optimizer.zero_grad()
        B, N, _ = Xoh.shape; device = Xoh.device
        if node_mask is None:
            node_mask = torch.ones(B, N, dtype=torch.bool, device=device)
        edge_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        edge_mask = edge_mask & ~torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
        t_idx = torch.randint(1, self.T + 1, (B,), device=device)
        Xt, Et = self._apply_noise(Xoh, Eoh, node_mask, t_idx)
        t_frac = t_idx.float() / self.T
        autoX, autoE, autoY = self._extra_features(Xt, Et, t_frac, node_mask)
        extraX = extraX if extraX is not None else autoX
        extraE = extraE if extraE is not None else autoE
        extraY = extraY if extraY is not None else autoY
        if y is not None and y.numel() > 0:
            extraY = y if extraY is None else torch.cat([y, extraY], dim=-1)
        node_logits, edge_logits = self.denoiser(Xt, Et, t_frac, node_mask, extraX, extraE, extraY)
        node_ce = F.cross_entropy(node_logits.reshape(-1, self.a), Xoh.argmax(-1).reshape(-1), reduction='none')
        edge_ce = F.cross_entropy(edge_logits.reshape(-1, self.b), Eoh.argmax(-1).reshape(-1), reduction='none')
        node_loss = (node_ce * node_mask.reshape(-1).float()).sum() / node_mask.sum().clamp_min(1)
        edge_loss = (edge_ce * edge_mask.reshape(-1).float()).sum() / edge_mask.sum().clamp_min(1)
        loss = node_loss + self.lambda_e * edge_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.denoiser.parameters(), 1.0)
        self.optimizer.step()
        return {"loss": loss.item(), "edge_loss": edge_loss.item()}

    def _posterior_over_x0(self, zt, Qt, Qsb, Qtb):
        left = (zt @ Qt.transpose(-1, -2)).unsqueeze(-2)
        num = left * Qsb
        denom = (zt @ Qtb.transpose(-1, -2)).unsqueeze(-1)
        return num / denom.clamp_min(1e-6)

    @torch.no_grad()
    def sample(self, n_samples, device):
        self.eval(); N = self.max_nodes
        if self._counts is not None:
            nc = torch.multinomial(self._counts.to(device), n_samples, replacement=True).clamp_min(2)
        else:
            nc = torch.full((n_samples,), N, device=device)
        node_mask = torch.arange(N, device=device).unsqueeze(0) < nc.unsqueeze(1)
        edge_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        edge_mask = edge_mask & ~torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
        e0 = torch.multinomial(self.mE.to(device), n_samples * N * N, replacement=True).view(n_samples, N, N)
        e0 = torch.triu(e0, 1); e0 = e0 + e0.transpose(1, 2)
        Et = F.one_hot(e0, self.b).float()
        Xt = F.one_hot(torch.multinomial(self.mX.to(device), n_samples * N, replacement=True).view(n_samples, N), self.a).float()
        Xt[~node_mask] = F.one_hot(torch.zeros((), dtype=torch.long, device=device), self.a).float()
        Et[~edge_mask] = F.one_hot(torch.zeros((), dtype=torch.long, device=device), self.b).float()
        for t in range(self.T - 1, -1, -1):
            abar_t, abar_s = self.abar[t + 1], self.abar[t]
            t_frac = torch.full((n_samples,), (t + 1) / self.T, device=device)
            extraX, extraE, extraY = self._extra_features(Xt, Et, t_frac, node_mask)
            node_logits, edge_logits = self.denoiser(Xt, Et, t_frac, node_mask, extraX, extraE, extraY)
            pX0 = F.softmax(node_logits, -1); pE0 = F.softmax(edge_logits, -1)
            postX = self._posterior_over_x0(Xt, self._Qt(abar_t, abar_s, self.mX),
                                            self._Qbar(abar_s, self.mX), self._Qbar(abar_t, self.mX))
            postE = self._posterior_over_x0(Et, self._Qt(abar_t, abar_s, self.mE),
                                            self._Qbar(abar_s, self.mE), self._Qbar(abar_t, self.mE))
            probX = (pX0.unsqueeze(-1) * postX).sum(-2).clamp_min(1e-6)
            probE = (pE0.unsqueeze(-1) * postE).sum(-2).clamp_min(1e-6)
            probX = probX / probX.sum(-1, keepdim=True)
            probE = probE / probE.sum(-1, keepdim=True)
            Xt = F.one_hot(torch.multinomial(probX.reshape(-1, self.a), 1).view(n_samples, N), self.a).float()
            e = torch.multinomial(probE.reshape(-1, self.b), 1).view(n_samples, N, N)
            e = torch.triu(e, 1); e = e + e.transpose(1, 2)
            Et = F.one_hot(e, self.b).float()
            Xt[~node_mask] = F.one_hot(torch.zeros((), dtype=torch.long, device=device), self.a).float()
            Et[~edge_mask] = F.one_hot(torch.zeros((), dtype=torch.long, device=device), self.b).float()
        return Xt, Et, nc.long()
```

## Relation to prior methods

- **Continuous/score graph diffusion (GDSS, EDP-GNN):** same diffusion idea but in continuous
  space with Gaussian noise; DiGress keeps the graph discrete throughout, preserving sparsity and
  structure and enabling structural-feature augmentation.
- **D3PM (discrete diffusion for text/images):** supplies the discrete machinery — transition
  matrices, closed-form `Q̄^t`, the Bayes posterior, the x₀-parameterization, uniform/cosine. DiGress
  lifts it to graphs (per node/edge, equivariant), adds the *marginal* transition and the
  optimal-prior theorem, the graph-transformer denoiser, and structural/spectral features; it
  trains on plain cross-entropy rather than D3PM's hybrid ELBO+CE objective.
- **VAE / autoregressive graph generators:** DiGress is one-shot and permutation-invariant (no
  ordering, no graph matching), via the per-element classification loss and exchangeability.
