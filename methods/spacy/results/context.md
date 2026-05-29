# Context: discovering causal structure from high-dimensional gridded time series

## Research question

Across climate science, neuroscience, and epidemiology, sensors produce **spatiotemporal gridded time series**: at each of `L` points laid out on a 2-D grid we observe a length-`T` time series, often for several physical variates `V`, repeated across `N` samples. The phenomena that matter — atmospheric oscillations, teleconnections between regions thousands of kilometres apart, brain-region interactions — are *causal* and evolve over both space and time.

The problem is: **infer the causal relationships that drive such data**, so that one can predict future states, reason about interventions, and uncover mechanisms. Concretely we want a temporal causal graph (with both same-time and time-lagged directed edges) describing how the system's driving factors influence one another.

Two structural difficulties make a direct attack hopeless and define what any solution must achieve:

1. **Dimensionality.** `L` is enormous — a `100×100` grid is `10^4` "variables", a global climate grid is `~28,000` per variate. Methods that reason about causal relations among the raw grid points do not scale; in particular conditional-independence-test–based discovery needs a number of tests that explodes with the number of variables.
2. **Spatial redundancy that masks causality.** Spatially adjacent grid points carry near-duplicate, highly correlated series. When a discovery method conditions on a nearby correlated point, it absorbs the statistical signal of a genuine but *distant* causal link, sapping the power to detect long-range (teleconnection) relations and producing wrong edges.

So a solution must (i) operate on a small number `D << L` of driving factors rather than the raw grid, (ii) make those factors correspond to genuinely causally-relevant entities, (iii) tolerate same-time (instantaneous) edges and complex noise, and (iv) ideally come with a guarantee that the recovered factors are *the* true ones rather than an arbitrary entangled mixture.

## Background

**Temporal causal discovery.** The dominant frameworks for finding causal structure in time series are:

- *Granger causality* and its neural extensions — `X` Granger-causes `Y` if past `X` improves prediction of `Y`. It captures only predictive relations and, by construction, ignores same-time (instantaneous) effects, hidden confounders, and noise whose distribution depends on history.
- *Structural Causal Model (SCM)* approaches that write each variable as a function of its parents plus noise, `X_i^t = f_i(Pa_i) + noise`. Linear instances (vector-autoregressive LiNGAM; dynamic-Bayesian-network adaptations of continuous-optimization DAG learning) recover lagged/instantaneous structure but only for *linear* relations.
- *Constraint-based* methods that extend the PC algorithm to time series via conditional-independence testing, handling instantaneous effects but inheriting the scaling and spatial-correlation problems above.

**Differentiable SCM learning over a graph.** A modern line makes the causal graph a *learnable, differentiable* object. The graph is treated as a random binary adjacency tensor `G ∈ {0,1}^{(τ+1)×D×D}` (one slice per lag plus an instantaneous slice); a variational posterior over edges is optimized jointly with the structural functions. Two ingredients recur:
- *A smooth acyclicity penalty.* For the instantaneous slice `G^0` to define a DAG one uses the trace-of-matrix-exponential characterization `h(G^0) = tr(e^{G^0}) − D`, which is zero exactly when `G^0` is acyclic, is smooth, and so can be driven to zero by an **augmented Lagrangian** during gradient training (the continuous-optimization-for-DAGs idea).
- *Discrete-but-differentiable graph samples.* To take ELBO gradients through a sampled graph one uses the **Gumbel-softmax** straight-through estimator, drawing hard binary edges while keeping a usable gradient.

A representative such model writes each factor's dynamics as an **additive-noise model with history-dependent noise**, `Z_i^t = f_i(Pa(<t), Pa(t)) + g_i(Pa(<t), ε_i^t)`, implementing `f` with per-node trainable embeddings and an adjacency-weighted sum over (lagged and instantaneous) neighbours fed through a shared network, and implementing the noise with a **conditional spline normalizing flow** so the noise distribution can flex with history. It handles nonlinear functions, instantaneous edges, and history-dependent noise simultaneously, and comes with identifiability conditions (causal stationarity, the causal Markov property, causal minimality, causal sufficiency, a well-defined density, and smoothness/non-invertibility conditions on `f` and `g`). Its limitation here is that it is defined over the *observed* variables — applied to `L` grid points it neither scales nor exploits spatial structure.

**Latent-variable / representation approaches and their identifiability.** Because working on the grid is hopeless, one extracts a few latent series and works on them. Identifiability theory for latent temporal processes (recovering the latents up to permutation and scaling from observations under a nonlinear invertible mixing) has been established, but **only under assumptions that are hard to verify in practice**: the *absence of instantaneous effects*, *sparsity* of the latent causal graph, or *sufficient variability* of the latent process (enough change across regimes/segments). A separate identifiability technique, drawn from the variational-autoencoder / nonlinear-ICA literature, "denoises" the observation: because additive noise acts as a *convolution* on densities, equality of two observational densities becomes, after a Fourier transform, equality of the characteristic-function-weighted signals — and provided the noise characteristic function is nonzero almost everywhere, the noiseless signals must coincide. This denoising step is the standard entry point to identifiability proofs for additive-noise latent models.

**Building latents that respect space.** A latent factor on a grid should aggregate *spatially proximate* points. Topographic / neural factor-analysis work models such factors with **radial basis function (RBF) kernels** — each factor is `exp(−‖x − ρ‖²/scale)`, a smooth, local, parameter-efficient bump centred at `ρ`. This both enforces locality and is, mathematically, a member of a **linearly independent, real-analytic** family of functions — a property (linear independence; analyticity) that will turn out to matter for identifiability.

**Motivating empirical observations about existing systems.**
- Conditional-independence-based discovery does not scale to grids, and conditioning on spatially redundant neighbours masks true distant causal links — repeatedly observed in climate causal-discovery practice.
- Dimensionality-reduction modes from PCA / Varimax-rotated PCA have weight vectors that are **nonzero essentially everywhere**, so the resulting "modes" are spatially diffuse and hard to attribute to a physical region; mapping causal effects from such a mode back to the grid connects many grid points spuriously.

## Baselines

- **Two-stage: rotated-PCA + constraint-based discovery (Mapped-PCMCI).** First reduce the grid to a handful of components with Varimax-rotated PCA; then run PCMCI⁺ (the PC-algorithm extension that handles lagged + instantaneous links via conditional-independence tests, here with a partial-correlation test) on the components; finally project edges back to the grid. *Gap:* the reduction is performed **independently of any causal objective**, so the components need not be causally meaningful; the linear PCA modes are spatially diffuse; and the CI tests assume largely linear relations and remain costly.
- **Two-stage: correlation/proximity modes + linear response (Linear-Response).** Build regional modes from correlation and spatial proximity, then apply linear-response theory to infer lagged causal dependencies. *Gap:* again the modes are formed independently of causal structure, the method is linear, and it scales poorly because it computes correlations across many series.
- **Identifiable latent temporal representation learning (LEAP, TDRL).** Learn latent time series and their temporal relations from sequence data with identifiability guarantees. *Gap:* they assume away instantaneous effects and/or need sufficient variability or sparsity, and they incorporate **no spatial prior**, so they are ill-suited to gridded spatial data and degrade as the number of factors grows.
- **Single-parent latent decoding (CDSD).** Jointly learn latents and a causal graph over them from observed time series, under the **single-parent decoding** assumption: *each observed variable is influenced by exactly one latent*. This sparsity yields identifiability of both the latents and the graph (via a denoising argument). *Gap:* the single-parent constraint forbids **overlapping factors** — but real grid points are jointly driven by several interacting modes (e.g. atmosphere and ocean) — and in nonlinear regimes it is prone to mode collapse and scattered factors.

## Evaluation settings

- **Synthetic gridded data with known ground truth.** Sample a ground-truth temporal DAG (Erdős–Rényi, with on the order of `4D` instantaneous and `2D` lagged edges, regenerated until the instantaneous part is acyclic); generate latent series via either a *linear* SCM (random weights, additive Gaussian noise) or a *nonlinear* SCM (random MLP structural functions with history-dependent conditional-spline noise); place `D` RBF spatial factors with random centres/scales on the grid; map factors×latents to the grid with either an *identity* (linear) map or a random-MLP (nonlinear) map plus Gaussian grid noise. Typical sizes: `N=100` samples, `T=100` timesteps, grid `100×100` (`L=10^4`), with `D ∈ {10,20,30}`; a multivariate variant with `V=2`. Scalability is probed by sweeping the grid from `30×30` to `250×250`.
- **Mixed real–simulated global climate data.** Monthly global temperature and precipitation on a `145×192` grid over 24-month windows; de-seasonalized by subtracting per-month means; distances measured with the great-circle (Haversine) metric to respect the sphere.
- **Metrics.** Orientation **F1 score** of the inferred temporal causal graph against ground truth, and **mean correlation coefficient (MCC)** between inferred and true latent series. Because latents are recovered only up to permutation, the inferred nodes are matched to ground truth with the Hungarian algorithm before scoring; for real data, recovered factors and edges are assessed qualitatively against known phenomena.

## Code framework

The available machinery is a grid-tensor data pipeline, a standard deep-learning stack with automatic differentiation, the reparameterization trick, the Gumbel-softmax straight-through estimator, the trace-exponential acyclicity function, and an augmented-Lagrangian optimizer loop. A generic harness can reserve a single latent-model slot connecting grid observations to a temporal graph.

```python
import torch
import torch.nn as nn

# ---- reusable primitives -------------------------------------------------

def reparameterize(mean, logvar):                     # VAE trick
    std = torch.exp(0.5 * logvar)
    return mean + std * torch.randn_like(std)

def gumbel_softmax_hard(logits, tau=1.0, dim=0):      # discrete + differentiable sample
    return torch.nn.functional.gumbel_softmax(logits, tau=tau, hard=True, dim=dim)

def acyclicity(G0):                                   # smooth exact DAG penalty
    return torch.trace(torch.matrix_exp(G0)) - G0.shape[-1]

class MLP(nn.Module):                                 # generic MLP
    def __init__(self, in_dim, out_dim, hidden, layers): ...
    def forward(self, x): ...

class SpatialMap(nn.Module):
    """Map a small number D of latent series onto the L-point grid, while
    aggregating spatially proximate grid points. The form of this map is the
    open question."""
    def forward(self):
        raise NotImplementedError  # TODO: how to parametrize the grid<->latent map

class GraphPosterior(nn.Module):
    """A learnable distribution over the temporal causal graph G among the D
    latents (lagged + instantaneous edges), giving differentiable samples,
    a sparsity term, the acyclicity penalty, and an entropy."""
    def sample_graph(self):     raise NotImplementedError  # TODO
    def sparsity(self, G):      raise NotImplementedError  # TODO
    def entropy(self):          raise NotImplementedError  # TODO

class LatentDynamics(nn.Module):
    """A differentiable temporal causal model over the D latents that can
    score how likely a latent series is given the graph (the latent prior
    p(Z|G)): nonlinearity, instantaneous edges, flexible noise."""
    def forward(self, Z, G):                 raise NotImplementedError  # TODO
    def log_prob(self, Z_true, Z_pred, ...): raise NotImplementedError  # TODO

class Encoder(nn.Module):
    """Amortized q(Z|X): from grid observations to latent mean/log-variance."""
    def forward(self, X):  raise NotImplementedError  # TODO

class Model(nn.Module):
    def __init__(self, L, D, lag):
        self.encoder        = Encoder(...)
        self.spatial_map    = SpatialMap(...)
        self.graph          = GraphPosterior(...)
        self.dynamics       = LatentDynamics(...)
        # plus a decoder taking (latent, spatial map) -> grid reconstruction
        self.decoder        = None  # TODO

    def forward(self, X):
        # TODO: encode X -> Z; sample G and the spatial map; reconstruct X
        raise NotImplementedError

    def compute_loss_terms(self, X, model_outputs, total_fragments):
        # TODO: assemble reconstruction, latent-dynamics score, encoder
        #       density term, graph prior/entropy, spatial-map entropy,
        #       and the acyclicity constraint handled by the outer optimizer
        raise NotImplementedError

# augmented-Lagrangian training loop that drives acyclicity(G0) -> 0
```
