We have gridded time series — climate, neuroscience, epidemiology — where at every one of $L$ points on a 2-D grid we observe a length-$T$ series, possibly across several variates $V$, repeated over $N$ samples. The phenomena that matter are causal and spatiotemporal: an oscillation over the mid-Pacific nudging Southeast-Asian rainfall a month later, a teleconnection between regions thousands of kilometres apart. What we want is the temporal causal graph — same-time and time-lagged directed edges — describing how the driving processes push on one another. The naive attack, treat the $L$ grid points as $L$ variables and run temporal causal discovery on them, dies for two reasons, and the way it dies dictates what to build. First, scale: the mature constraint-based methods (the PC family extended to time series) run conditional-independence tests whose number and conditioning-set size explode with the number of variables, and a $100\times100$ grid is $10^4$ variables, a global grid tens of thousands. Second, and more insidious, spatial redundancy: two grid points a few kilometres apart carry near-duplicate series, so when a CI test conditions on the neighbours of a genuine long-range cause it conditions on near-copies of that cause, soaking up exactly the statistical signal it was trying to attribute to the distant link — the test loses power and the teleconnection vanishes or gets mangled. Both pathologies point the same way: do not do discovery on the grid. Do it on a small set of $D \ll L$ latent driving factors, each a single series, with the causal graph living among the $D$ of them.

The tempting but quietly wrong way to get those latents is two-stage: reduce the grid with PCA or Varimax-rotated PCA to $D$ components, then run causal discovery on the components. This is what is done in climate practice, and the reduction has no idea about causality — PCA maximizes variance, Varimax rotates for "simplicity", and neither knows whether a component is a causally coherent entity. The reduction can smear two distinct drivers together or split one across several, and nothing downstream can recover what was destroyed upstream because the two stages optimize unaligned objectives. There is even a concrete symptom: PCA/Varimax modes have weight vectors nonzero essentially everywhere, so a "mode" is a global, diffuse pattern that cannot be read as a region, and projecting its causal effect back to the grid wires up half the planet. Latent identifiability theory for temporal processes does exist, but it buys recovery only under assumptions I explicitly do not want — no instantaneous effects, sparsity of the latent graph, or sufficient variability across regimes — all hard to verify and none built for gridded space. The single-parent-decoding route (each grid point driven by exactly one latent) earns identifiability cheaply but forbids overlapping factors, and real grid points over the tropical Pacific are jointly driven by overlapping atmospheric and oceanic modes. So the requirement crystallizes: latent extraction and causal discovery must be *the same optimization*, end-to-end, with factors that respect space, allow overlap, and still come out identifiable.

I propose SPACY — SPAtiotemporal Causal discoverY — a variational framework that jointly infers the $D$ latent series, their temporal causal graph, and a spatial map relating latents to the grid, in one objective. The design follows from writing down how the data is *made* and then inverting it. I posit latent series $Z \in \mathbb{R}^{D\times T}$ evolving under causal dynamics, each spread over the grid by a spatial pattern, with the grid observation being the (possibly nonlinear, noisy) superposition. The latent dynamics are an additive-noise temporal SCM with both lagged and instantaneous parents,
$$ Z_d^t = f_d\big(\mathrm{Pa}_G^d(<t),\, \mathrm{Pa}_G^d(t)\big) + \eta_d^t, $$
where $f_d$ is made differentiable in the graph and shared across nodes by keeping a trainable embedding per node-and-lag and forming the prediction as an edge-gated sum over potential parents passed through shared networks,
$$ f_d = \xi_f\Big( \sum_{k=0}^{\tau}\sum_{j=1}^{D} G^{(k)}_{j,d}\cdot \lambda_f\big([Z_j^{t-k}, e_j^k],\, e_d^0\big)\Big). $$
Because each edge variable $G^{(k)}_{j,d}$ multiplies its parent's contribution, the gradient flows into $G$, and the same inner $\lambda_f$ and outer $\xi_f$ serve every node so adding nodes adds no functions. The noise is not a fixed Gaussian but a conditional spline normalizing flow whose parameters are predicted from history, letting the noise distribution bend with the past — which Granger-style models cannot do, since they ignore instantaneous effects and history-dependent noise entirely. A linear special case, SPACY-L, replaces $f_d$ with $\sum_k \sum_{d'} (G\circ W)^k_{d',d} Z_{d'}^{t-k}$ under Gaussian noise, the same skeleton with weights instead of MLPs, handy for sanity.

The load-bearing design choice is the spatial map $F \in \mathbb{R}^{L\times D}$. A free $L\times D$ matrix would be $L\cdot D$ parameters with no notion of space — its columns free to be scattered, everywhere-nonzero smears, exactly the uninterpretable mess I am escaping. What I actually believe is that a driving mode occupies a coherent region whose influence decays smoothly from a centre — locality plus smoothness — and the object that expresses that is a radial basis function. Each factor becomes a kernel, $F_{\ell d} = \exp(-\lVert x_\ell - \rho_d\rVert^2 / \mathrm{exp}(\gamma_d))$ in the isotropic form, described by a centre $\rho_d$ and a scale instead of $L$ free weights: parameter-cheap, forced-local, smooth, and interpretable (point at the map: "this factor is here, this big"). I deliberately let kernels overlap rather than impose single-parent decoding, taking the harder modelling problem to keep multi-driver structure. The grid observation is then a per-point nonlinearity of the factor–latent product plus Gaussian noise,
$$ X_\ell^t = g_\ell\big([FZ]_\ell^t\big) + \varepsilon_\ell^t,\qquad \varepsilon_\ell^t \sim \mathcal{N}(0,\sigma_\ell^2),\qquad g_\ell(x)=\Xi([x, E_\ell]), $$
with a per-grid-point embedding $E_\ell$ so the pointwise nonlinearity can vary across the grid.

Fitting maximizes $\log p_\theta(X)$, but $p(X)=\int p(X|Z,F)\,p(Z|G)\,p(F)\,p(G)\,dZ\,dG\,dF$ integrates out three latent objects and is intractable, so I introduce a factorized variational posterior $q(Z|X)\,q(G)\,q(F)$ and, by Jensen, lower-bound the evidence. Factoring the joint by the generative assumptions and grouping the log-ratio by which variable each piece owns gives
$$ \log p_\theta(X) \ge \sum_n \mathbb{E}_{q(Z^n|X^n)q(G)q(F)}\!\Big[\log p_\theta(X^n|Z^n,F) + \big(\log p_\theta(Z^n|G) - \log q(Z^n|X^n)\big)\Big] - \mathrm{KL}\big(q(G)\,\Vert\,p(G)\big) - \mathrm{KL}\big(q(F)\,\Vert\,p(F)\big). $$
Every term has a clean job: the first is reconstruction, which under Gaussian observation noise is a negative squared error $-\sum_\ell \lVert X_\ell - g_\ell(FZ)_\ell\rVert^2$; the bracket is the latent-causal fit $\log p(Z|G)$ — the spline-flow log-likelihood of the SCM residual $Z^t - f(\mathrm{Pa})$, or the Gaussian residual for SPACY-L — plus the encoder entropy $-\log q(Z|X)$; and the two KLs keep the graph and factor posteriors near their priors.

Each $q$ forces a small decision. The encoder $q(Z|X)$ is amortized: an MLP reads the grid window and outputs Gaussian mean and log-variance, sampled with the reparameterization trick so gradients flow. The factor posterior $q(F)$ puts Gaussians on the centre $\rho_d$ and the scale and reparameterizes both; the centre must live on the grid, so I sample it unconstrained and squash with a sigmoid into $[0,1]^K$, and because a scalar scale gives only circular blobs while real modes are elongated, the scale parameters build a positive-definite matrix $P_d = A_d A_d^\top + \mathrm{diag}(\exp B_d)$ used directly in the exponent, $F_{\ell d}=\exp\!\big(-\tfrac{1}{2}(x_\ell-\rho_d)^\top P_d (x_\ell-\rho_d)\big)$, giving anisotropic ellipses with $P$ as a learned precision. The graph posterior $q(G)$ models each edge as a Bernoulli, but two wrinkles need care. To backprop through a discrete graph sample I use the Gumbel-softmax with hard, straight-through samples — REINFORCE would be unbiased but brutally high-variance — getting a genuine binary $G$ forward and a usable gradient backward. And the instantaneous slice must be a DAG: sampling each ordered pair independently invites immediate two-cycles, so for each unordered instantaneous pair I draw a single three-way categorical ($i\to j$, $j\to i$, or none) and never instantiate both directions, while lagged edges (which point forward in time and may self-loop) stay plain Bernoullis. The three-way trick kills two-cycles but not longer ones, so I still enforce acyclicity with the smooth trace-of-matrix-exponential penalty $h(G^0) = \mathrm{tr}(e^{G^0}) - D$, which is exactly zero iff $G^0$ is acyclic and can be driven to zero by gradient descent. I fold it together with a sparsity term into the graph prior $p(G)\propto \exp(-\alpha\lVert G\rVert^2 - \sigma\,h(G^0))$ and drive the constraint home with an augmented-Lagrangian schedule rather than a fixed weight. Two training pragmatics matter: a half-formed early graph corrupts the factors and latents, so I freeze the SCM and graph modules for the first $\sim200$ epochs and let the encoder and spatial factors settle before learning structure on stable latents; and a $\beta$ factor (with $\beta = D/4$) balances the latent term against reconstruction in the minimized negative ELBO.

What makes the kernels more than a nice prior is that they buy the whole identifiability story — without the no-instantaneous / sparsity / sufficient-variability crutches — paid for entirely by the overdetermination $L \gg D$. Push the grid to a continuous domain $(0,1)^K$ so each factor is a *function* $F_{\psi_d}(\ell)$ and the model at each point is $X^t(\ell)=g_\ell(F_\ell^\top Z^t)+\varepsilon$, a spatial factor process. Suppose two models induce the same observational distribution at every location and time. The noise is additive and additive noise convolves densities, so $\delta_{\bar x}*p_\varepsilon = \delta_{\tilde x}*p_\varepsilon$ with $\bar x = g_\ell(F_\ell^\top Z)$ and $\tilde x = \hat g_\ell(\hat F_\ell^\top \hat Z)$; transforming to Fourier where convolution is product gives $e^{is\bar x}\varphi_\varepsilon(s) = e^{is\tilde x}\varphi_\varepsilon(s)$, and provided the noise characteristic function $\varphi_\varepsilon$ is nonzero almost everywhere it cancels, forcing $\bar x = \tilde x$ — the noise is gone and the clean signals coincide. In the linear case $g=\hat g=\mathrm{Id}$ this is a vanishing linear combination of factor functions over the whole domain, and linear independence of the family forces the index sets to match and $Z = \hat Z$ up to permutation. For diffeomorphic $g_\ell$, inverting gives $F_\ell^\top Z = h_\ell(\hat F_\ell^\top \hat Z)$ with $h_\ell = g_\ell^{-1}\circ\hat g_\ell$ — a linear readout of $Z$ equal to a nonlinear $h_\ell$ of a linear readout of $\hat Z$ at every point. Choosing $D$ locations where both factor matrices $M_F, M_{\hat F}$ are full rank (possible on a full-measure open set, since their determinants are real-analytic and not identically zero) and stacking gives an invertible inter-model map $\Theta$; because any valid tuple yields the *same* $Z$, $\Theta$ is independent of the evaluation points, and differentiating $\log|\det J_\Theta|$ in $\hat Z_i$ while perturbing one evaluation point shows that $[h''_\ell/h'_\ell](\cdot)\,F_{\hat\psi_i}(\ell)$ is constant in $\ell$. Linear independence of the analytic factor functions (which the identity theorem extends to any open set) then forces those quantities to vanish, hence $h''_\ell \equiv 0$: each $h_\ell$ is affine. Under the needed no-gauge condition — that no nonconstant common spatial multiplier can carry a full-rank tuple of learned kernels back into the admissible family on an open set — the affine slope is spatially constant, and a final differentiation in $\hat Z_i$ shows each Jacobian column relating $Z$ and $\hat Z$ has exactly one nonzero entry, so $Z = PS\hat Z$ and the factor families coincide. Identifiability up to the unavoidable permutation and scaling, with linear independence and real analyticity — precisely what RBFs supply — doing the work. On a finite grid the measure-zero pathologies do not literally vanish, but with $L \gg D$ they are probabilistically negligible. Identifying the latents and factors this way, the causal graph over them is then identifiable under the temporal SCM's own conditions: causal stationarity, the Markov property, minimality, sufficiency, a well-defined density, and the smoothness/non-invertibility conditions on $f$ and $g$.

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
