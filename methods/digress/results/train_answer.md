The problem is one-shot generation of graphs whose node and edge attributes are categorical, with "no edge" treated as just another edge category. Graphs are unordered, so the same graph can be represented by any of n! adjacency matrices; a model that treats the matrix as a fixed array wastes capacity memorizing arbitrary orderings, and the permutation-sum likelihood is intractable. Graphs are also sparse and discrete: only O(n) of the O(n^2) possible edges are present, and the structure we care about, degrees, cycles, connected components, and small substructures, is defined only on a genuine graph. A good generator must therefore be permutation equivariant with a permutation-invariant loss, and it must keep the object discrete and structured throughout generation.

Existing approaches each miss at least one of these requirements. Autoregressive models such as GraphRNN and GRAN impose an arbitrary node ordering and generate sequentially, so they are not permutation invariant. VAE generators decode a fixed-size probabilistic adjacency tensor and then need an expensive graph-matching step to align the output with the target. Normalizing-flow generators give exact likelihoods but are constrained by invertibility requirements that limit the dependencies they can model. Continuous diffusion models such as EDP-GNN and GDSS embed the graph in a real-valued space and add Gaussian noise, but halfway through the forward process the adjacency becomes a dense, blurry matrix of continuous values. Degrees, cycles, and connected components are no longer defined, so the denoiser cannot be fed structural descriptors that would help it reproduce the data statistics. What is missing is a diffusion model that treats the graph as a graph at every step.

The method is DiGress, a discrete denoising diffusion model for graph generation. Instead of adding Gaussian noise to continuous tensors, DiGress corrupts the graph by editing node and edge categories through Markov transition matrices applied independently to each node and each edge. For a single categorical variable represented as a one-hot row vector x, the t-step marginal is q(z^t | x) = x Qbar^t, where Qbar^t = Q^1 ... Q^t is the product of per-step transition matrices. Applied to a graph, this means two small matrices, Q_X for node types and Q_E for edge types, with "no edge" as one of the edge categories. The upper triangle of the edge tensor is noised and then symmetrized, so undirectedness is preserved by construction. Because the noise acts independently on each node and edge, the forward marginal has a closed form, the Bayes posterior q(z^{t-1} | z^t, x) has a closed form, and the limit is independent of the clean data. These are exactly the three properties that make continuous diffusion efficient, now satisfied in a fully discrete setting.

The posterior is derived by Bayes rule: q(z^{t-1} | z^t, x) is proportional to z^t (Q^t)^T elementwise multiplied by x Qbar^{t-1}, normalized by x Qbar^t (z^t)^T. The first factor is the likelihood of each candidate previous class producing the observed noisy class in one step, and the second factor is the marginal distribution after t-1 steps from the clean data. This closed form lets the model use the clean graph as the regression target, avoiding the high-variance z^{t-1} target that would depend on the sampled trajectory.

A uniform transition matrix, which jumps to any class with equal probability, satisfies the three diffusion properties, but its stationary distribution is uniform and therefore dense. For sparse graphs this is a poor endpoint: the reverse process would waste many early steps simply making the graph sparse again. The prior must be permutation invariant, and the most expressive invariant independent prior has the product form product_i u times product_{ij} v, where u is shared across nodes and v is shared across edges. Projecting the one-site marginals of the data onto this family gives u equal to the empirical node-type marginal and v equal to the empirical edge-type marginal. This leads to marginal transitions Q^t = alpha^t I + beta^t 1 m^T, where m is the data marginal and beta^t = 1 - alpha^t. Because 1 m^T is idempotent, the cumulative transition remains closed form: Qbar^t = abar^t I + (1 - abar^t) 1 m^T, with abar^t the product of the alpha values. The schedule is the cosine schedule abar^t = cos^2(0.5 pi (t/T + s)/(1+s)) / cos^2(0.5 pi s/(1+s)) with s around 0.008. With this choice the noisy graphs along the chain keep the right edge density, so the denoiser can focus on learning real structure.

The denoiser is a graph transformer that predicts, for every node and every edge, a distribution over the clean categories. Since the targets are class labels, the entire training objective reduces to a sum of independent cross-entropy terms over nodes and edges. This loss is permutation invariant precisely because it is a sum of identical per-node and per-edge terms; the predicted and target tensors permute together, and the sum simply reindexes. Equivariant architecture plus invariant loss plus invariant product limit gives an exchangeable generated distribution, so likelihood can be evaluated on a single representative without summing over permutations.

Because the noisy graph is still a real sparse graph at every step, the denoiser can be augmented with structural and spectral features that plain message-passing networks cannot compute for themselves, such as cycle counts up to length five or six and Laplacian eigenvalues and eigenvectors. The transformer maintains three coupled streams: node features, edge features, and a graph-level feature that also encodes the normalized timestep. Edge features modulate attention scores through feature-wise linear modulation, and a PNA-style pooling over max, min, mean, and standard deviation keeps the global stream permutation invariant.

Sampling starts from the marginal limit, draws a node count from the empirical training distribution, and iterates the reverse step from t = T down to 1. At each step the network predicts the clean graph, and the analytic posterior is marginalized over that prediction to obtain p_theta(x_i^{t-1} | G^t) = sum_x q(x_i^{t-1} | x_i = x, x_i^t) p_hat_i^X(x), and similarly for edges. The joint reverse step is modeled as a product over all nodes and edges, and a discrete G^{t-1} is sampled. Conditional generation on graph-level properties can be added with a first-order classifier-guidance term without retraining the unconditional model.

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
