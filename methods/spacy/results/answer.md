# SPACY: Spatiotemporal Causal Discovery

**SPACY** (SPAtiotemporal Causal discoverY) is a variational-inference framework that discovers, end-to-end, a small set of latent time series and the temporal causal graph among them from high-dimensional gridded spatiotemporal data. It sidesteps two obstacles to causal discovery on grids — scale and spatial redundancy that masks long-range causal links — by performing discovery in a low-dimensional latent space, and by mapping latents to the grid with kernel-parametrized **spatial factors** that aggregate spatially proximate points into coherent, interpretable regions.

## Problem

Given `N` samples of `V`-variate, `L`-point gridded time series of length `T` (`L = ∏ L_k`, here `K = 2` so `L = L_1 × L_2`), infer (1) the `D << L` latent series `Z ∈ R^{D×T}` driving the data and (2) the temporal causal DAG `G ∈ {0,1}^{(τ+1)×D×D}` over them (one instantaneous slice `G^0`, `τ` lagged slices), unsupervised.

## Generative model

Latent dynamics follow an additive-noise temporal SCM with both lagged and instantaneous parents:

  `Z_d^t = f_d(Pa_G^d(<t), Pa_G^d(t)) + η_d^t`,

with `f_d` shared across nodes via per-node embeddings and an edge-gated aggregate,

  `f_d = ξ_f( Σ_{k=0}^{τ} Σ_{j=1}^{D} G^{(k)}_{j,d} · λ_f([Z_j^{t−k}, e_j^k], e_d^0) )`,

and history-dependent noise modeled by a conditional spline normalizing flow. (A linear variant, SPACY-L, replaces `f_d` with `Σ_k Σ_{d'} (G ∘ W)^k_{d',d} Z_{d'}^{t−k}` and Gaussian noise.)

Spatial factors `F ∈ R^{L×D}` are RBF kernels with center `ρ_d` and scale; the observation is a grid-pointwise nonlinearity of the factor–latent product plus Gaussian noise:

  `X_ℓ^t = g_ℓ([F Z]_ℓ^t) + ε_ℓ^t`,  `ε_ℓ^t ∼ N(0, σ_ℓ²)`,  `g_ℓ(x) = Ξ([x, E_ℓ])`,
  `F_{ℓd} = exp(−‖x_ℓ − ρ_d‖² / exp(γ_d))` in the isotropic form. The implementation's anisotropic form samples `P_d = A_d A_dᵀ + diag(exp(B_d))` and uses it directly as a precision-like matrix: `F_{ℓd}=exp(−½(x_ℓ−ρ_d)ᵀP_d(x_ℓ−ρ_d))`.

## Objective (ELBO)

  `log p_θ(X) ≥ Σ_n E_{q(Z^n|X^n) q(G) q(F)} [ log p_θ(X^n | Z^n, F) + ( log p_θ(Z^n | G) − log q(Z^n | X^n) ) ] − KL(q(G) ‖ p(G)) − KL(q(F) ‖ p(F))`.

- `log p(X|Z,F)`: Gaussian likelihood = `−Σ_ℓ ‖X_ℓ − g_ℓ(FZ)_ℓ‖²`.
- `log p(Z|G)`: spline-flow log-likelihood of the SCM residual `Z^t − f(Pa)` (MSE residual for SPACY-L).
- `q(G) = ∏ Bernoulli`, instantaneous slice as a 3-way categorical (`i→j` / `j→i` / none); hard Gumbel-softmax samples. `p(G) ∝ exp(−α‖G‖² − σ h(G^0))` with `h(G^0) = tr(e^{G^0}) − D` (acyclicity, augmented Lagrangian).
- `q(F)`: Gaussians on `ρ` (sigmoid into `[0,1]^K`) and on the scale params, reparameterized; KL/entropies closed-form.
- `q(Z|X)`: MLP encoder producing Gaussian mean/log-variance; reparameterization trick. A β factor (`β = D/4`) weights the latent negative log-likelihood plus `E_q log q(Z|X)` term in the minimized negative ELBO.

Training: an augmented-Lagrangian schedule drives `h(G^0)→0`; the latent SCM and graph are frozen for the first ~200 epochs so the spatial factors and encoder settle first. Distance metric is Euclidean for Cartesian grids, Haversine for global (spherical) data.

## Identifiability

Generalizing the grid to a continuous domain `(0,1)^K` (Spatial Factor Process), if two models induce the same observational distribution everywhere, then the latents coincide up to permutation and scaling and the factor families match, without assuming no-instantaneous-effects, sparsity, or sufficient variability. Key steps: (i) a denoising lemma removes additive noise via characteristic functions (`e^{is x̄}φ_ε = e^{is x̃}φ_ε`, with `φ_ε ≠ 0` a.e., gives `x̄ = x̃`); (ii) linear independence of the factor functions handles the linear case directly; (iii) for diffeomorphic `g_ℓ` and real-analytic factors, the `L>>D` overdetermination forces the inter-model map `h_ℓ` to be affine (its `log|det|`-Jacobian derivative is `ℓ`-constant, so `h''_ℓ ≡ 0`); under the needed no-gauge condition that a nonconstant common spatial multiplier cannot keep a full-rank tuple inside the admissible kernel family, the affine ambiguity reduces to permutation plus coordinate scaling. The causal graph over the recovered latents is then identifiable under the latent SCM's own conditions (stationarity, Markov, minimality, sufficiency, well-defined density, conditions on `f`/`g`).

## Code

```python
import torch, torch.nn as nn, torch.nn.functional as F, math
import pyro.distributions as distrib
from pyro.distributions.transforms.spline import ConditionalSpline

def reparameterize(mean, logvar):
    return mean + torch.exp(0.5 * logvar) * torch.randn_like(logvar)

class SpatialFactors(nn.Module):
    """q(F): RBF kernels with learned center rho and anisotropic precision-like P = A A^T + diag(exp(B))."""
    def __init__(self, num_variates, num_nodes, nx, ny, spherical=False, simple=False):
        super().__init__()
        self.spherical, self.simple = spherical, simple
        z = lambda d: nn.Parameter(torch.zeros(num_variates, num_nodes, 1, d))
        self.rho_mu, self.rho_logvar     = z(2), z(2)
        gamma_dim = 1 if (spherical or simple) else 6         # 6 = 4 -> A (2x2), 2 -> B
        self.gamma_mu, self.gamma_logvar = z(gamma_dim), z(gamma_dim)
        self.grid_coords = create_grid(nx, ny)[None, None].expand(num_variates, num_nodes, -1, -1)
        self.sigmoid = nn.Sigmoid()

    def get_spatial_factors(self):
        centers = self.sigmoid(reparameterize(self.rho_mu, self.rho_logvar))
        scale   = reparameterize(self.gamma_mu, self.gamma_logvar)
        if self.spherical or self.simple:
            scale = torch.exp(scale)
        mode = 'Haversine' if self.spherical else 'Euclidean'
        grid_dist = calculate_distance(self.grid_coords, centers, distance_mode=mode)
        if self.simple:
            exponent = torch.sum(-torch.square(self.grid_coords - centers) / scale.expand(-1, -1, -1, 2), dim=-1)
            return torch.exp(exponent)
        if self.spherical:
            P = scale
        else:
            A = scale[..., :4].view(*scale.shape[:-1], 2, 2)
            B = scale[..., 4:]
            P = A @ A.transpose(-1, -2) + torch.diag_embed(torch.exp(B))
        exponent = -0.5 * torch.einsum('...ik,...kl,...il->...i', grid_dist, P, grid_dist)
        return torch.exp(exponent)                            # (V, D, nx*ny)

    def calculate_entropy(self):
        def ent(lv): return 0.5 * (lv.sum() + lv.shape[-1] * (1 + math.log(2*math.pi)))
        return ent(self.rho_logvar) + ent(self.gamma_logvar)

class TemporalAdjacencyMatrix(ThreeWayGraphDist):
    """q(G): 3-way categorical for instantaneous slice, Bernoulli for lagged; Gumbel-softmax hard samples."""
    def __init__(self, input_dim, lag, tau_gumbel=1.0):
        super().__init__(input_dim=input_dim, tau_gumbel=tau_gumbel)
        self.lag = lag
        self.logits_lag = nn.Parameter(torch.zeros(2, lag, input_dim, input_dim))

    def sample_graph(self):
        adj = torch.zeros(self.lag + 1, self.input_dim, self.input_dim, device=self.device)
        adj[0]  = self._triangular_vec_to_matrix(
                     F.gumbel_softmax(self.logits, tau=self.tau_gumbel, hard=True, dim=0))
        adj[1:] = F.gumbel_softmax(self.logits_lag, tau=self.tau_gumbel, hard=True, dim=0)[1]
        return adj

    def calculate_sparsity(self, G):          return torch.sum(G)
    def calculate_dagness_penalty(self, G0):  return torch.trace(torch.matrix_exp(G0)) - G0.shape[-1]

class RhinoSCM(nn.Module):
    """Latent SCM: edge-gated aggregation of embedded parents -> predicted Z^t."""
    def __init__(self, embedding_dim, lag, num_nodes):
        super().__init__()
        self.embeddings = nn.Parameter(torch.randn(lag+1, num_nodes, embedding_dim) * 0.01)
        self.f = MLP(2*embedding_dim, 1, embedding_dim, num_layers=2)
        self.g = MLP(embedding_dim+1, embedding_dim, embedding_dim, num_layers=2)

    def forward(self, Z, A):                                  # Z: (b, lag+1, num_nodes)
        E = self.embeddings.expand(Z.shape[0], -1, -1, -1)
        X_enc = self.g(torch.cat((Z.unsqueeze(-1), E), dim=-1))
        X_sum = torch.einsum("lij,blio->bjo", A.flip([0]), X_enc)
        X_sum = torch.cat([X_sum, E[:, 0]], dim=-1)
        return self.f(X_sum).squeeze(-1)                      # f(Pa) = predicted Z^t

class TemporalConditionalSplineFlow(nn.Module):
    """History-dependent residual likelihood used by Rhino."""
    def __init__(self, hypernet):
        super().__init__()
        self.hypernet = hypernet                               # TemporalHyperNet
        self.num_bins, self.order = hypernet.num_bins, hypernet.order

    def log_prob(self, residual, X_history, expanded_G):
        num_nodes = residual.shape[-1]
        transform = nn.ModuleList([
            ConditionalSpline(self.hypernet, input_dim=num_nodes,
                              count_bins=self.num_bins, order=self.order, bound=5.0)
        ])
        base = distrib.Normal(torch.zeros(num_nodes, device=residual.device),
                              torch.ones(num_nodes, device=residual.device))
        ctx = {"X": X_history.unsqueeze(-1), "A": expanded_G, "embeddings": None}
        return distrib.ConditionalTransformedDistribution(base, transform).condition(ctx).log_prob(residual)

    def calculate_likelihood(self, X_true, X_predict, X_history, expanded_G=None, mean=False):
        residual = (X_true - X_predict).view(X_true.shape[0], -1)
        log_prob = self.log_prob(residual, X_history, expanded_G)
        log_prob = log_prob.mean(-1) if mean else log_prob.sum(-1)
        return -torch.mean(log_prob)

class SpatialDecoderNN(nn.Module):
    """g_l: shared MLP on the factor-latent product with a per-grid-point embedding."""
    def __init__(self, nx, ny, num_variates, embedding_dim, lag, num_nodes):
        super().__init__()
        self.embeddings = nn.Parameter(torch.randn(num_variates, nx*ny, embedding_dim) * 1e-1)
        self.g = MLP(embedding_dim + 1, 1, hidden_dim=64, num_layers=2)
    def forward(self, Z, Fac):
        X_hat = torch.einsum("bd,vdl->bvl", Z, Fac)          # [F Z]_l
        E = self.embeddings.expand(X_hat.shape[0], -1, -1, -1)
        return self.g(torch.cat((X_hat.unsqueeze(-1), E), dim=-1)).squeeze(-1)

class SPACY(nn.Module):
    def __init__(self, lag, num_nodes, nx, ny, num_variates, scm_model, spatial_factors,
                 graph_sparsity_factor):
        super().__init__()
        self.lag, self.num_nodes = lag, num_nodes
        self.f_tilde = MLP(nx*ny, 2*num_nodes, hidden_dim=64, num_layers=2)   # q(Z|X)
        self.spatial_factors = spatial_factors                               # q(F)
        self.scm_model = scm_model                                           # p(Z|G)
        self.temporal_graph_dist = TemporalAdjacencyMatrix(num_nodes, lag)   # q(G)
        self.spatial_decoder = SpatialDecoderNN(nx, ny, num_variates, 32, lag, num_nodes)
        self.graph_sparsity_factor = graph_sparsity_factor

    def forward(self, X):
        X_lag = convert_data_to_timelagged(X, self.lag)
        b = X_lag.shape[0]
        Zp = self.f_tilde(X_lag).view(b, self.lag+1, 2*self.num_nodes)
        Z_mean, Z_logvar = Zp[..., :self.num_nodes], Zp[..., self.num_nodes:]
        Z = reparameterize(Z_mean, Z_logvar)
        G = self.temporal_graph_dist.sample_graph()
        Z_hat = self.scm_model(Z, G)
        Fac = self.spatial_factors.get_spatial_factors()
        X_hat = self.spatial_decoder(Z[:, -1], Fac)
        return X_lag, X_hat, Z_mean, Z_logvar, Z_hat, Z, G, Fac

    def compute_loss_terms(self, X_lag, X_hat, Z_mean, Z_logvar, Z_hat, Z, G, total_fragments):
        b = X_lag.shape[0]
        X_true = X_lag[:, :, -1]
        recon = torch.sum((X_hat - X_true) ** 2) / b                         # -log p(X|Z,F)
        cd = self.scm_model.calculate_likelihood(                            # -log p(Z|G)
                X_true=Z[:, -1], X_pred=Z_hat, X_history=Z[:, :-1],
                expanded_G=G.unsqueeze(0).repeat(b, 1, 1, 1), mean=True)
        D = Z_logvar.shape[-1]
        z_entropy = -0.5 * (Z_logvar.sum() + D * (1 + math.log(2*math.pi))) / (b * self.num_nodes)
        kl_term = (self.num_nodes / 4) * (cd + z_entropy)                    # beta-weighted negative-ELBO term
        sparsity = self.graph_sparsity_factor * self.temporal_graph_dist.calculate_sparsity(G)
        dag   = self.temporal_graph_dist.calculate_dagness_penalty(G[0]) / total_fragments
        g_ent = -self.temporal_graph_dist.entropy() / total_fragments
        f_ent = self.spatial_factors.calculate_entropy() / total_fragments
        total = recon + sparsity / total_fragments + kl_term + g_ent + f_ent
        return total, dag    # `dag` is handled by an augmented-Lagrangian wrapper
```

Training drives `dag` to zero via the augmented Lagrangian; the SCM and graph modules are frozen for the first ~200 epochs (only the encoder and spatial factors learn), then jointly optimized. For multivariate data, per-variate spatial factors `F^{(v)}` and encoders are used and the latents concatenated before causal discovery.
