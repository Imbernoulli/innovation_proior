# Synthesis — SPACY (Spatiotemporal Causal Discovery)

## The pain point / first-principles object

**Goal:** Given high-dimensional gridded time series `X ∈ R^{L×T}` (L huge, e.g. 100×100 = 10^4 grid points, possibly V variates), infer (a) a small number `D << L` of latent time series `Z ∈ R^{D×T}` and (b) the temporal causal DAG `G` (lag + instantaneous edges) among them — jointly, end-to-end, unsupervised.

**Why hard:**
1. **Dimensionality / scalability.** Causal discovery on L variables doesn't scale; conditional-independence (CI) methods (PC/PCMCI) need exponentially many tests.
2. **Spatial correlation masks causality.** Neighboring grid points are redundant/highly correlated. Conditioning on a nearby correlated point in a CI test soaks up the statistical signal of a *distant* true causal link (teleconnection) → reduced power, wrong edges.
3. **So you must work in a latent space** — but how you build the latent space matters.

## Design-decision → why table (with rejected alternatives)

| Decision | Why this | Rejected alternative & its failure |
|---|---|---|
| Discover causality in a **latent** space of size D, not on the L-grid | Scalability + the latent vars are the "causally relevant entities" (climate modes), not raw pixels | CI tests on full grid: exponential, and spatial correlation masks links |
| **Joint / end-to-end** inference of latents + graph (not two-stage) | Two-stage (reduce-then-discover) picks latents *independent of* causal structure → latents need not align with causally relevant entities | Mapped-PCMCI (Varimax→PCMCI), Linear-Response: dim-reduction decoupled from causality |
| **Spatial factors F ∈ R^{L×D}** mapping Z→X, parametrized by **kernels** (not a free matrix) | Need to *aggregate spatially proximate, correlated* grid points under one latent; kernel enforces locality + smoothness + parameter efficiency (each factor = a few params) | Free L×D matrix: O(LD) params, no locality prior, diffuse non-interpretable modes (like PCA/Varimax weight vectors nonzero everywhere) |
| **RBF kernel** F_{ℓd}=exp(−‖x_ℓ−ρ_d‖²/exp(γ_d)) | Locality (decays with distance) + smooth + only center ρ + scale γ per factor; RBFs are a *linearly independent, real-analytic* family → needed for the identifiability proof | Any linearly-independent analytic family works in theory; Matérn ablation shows robustness. A non-analytic/non-LI basis breaks the identity-theorem argument |
| **Allow multiple latent parents per grid point** (overlapping modes) | Real spatiotemporal systems: a grid point is driven by several interacting modes (atmosphere + ocean) | CDSD single-parent decoding: each observed var has exactly ONE latent → too restrictive, mode collapse in nonlinear settings |
| **Grid-pointwise nonlinearity g_ℓ** (shared MLP Ξ with per-grid embedding E_ℓ): X_ℓ = g_ℓ([FZ]_ℓ)+ε_ℓ | Lets observation be a *nonlinear, invertible* function of the latent×factor product; identifiability still holds (general SFP theorem) | Pure linear map: less expressive; identifiability proof handles both |
| **Latent SCM = Rhino** (additive noise model, MLP f_d via embeddings + adjacency-weighted neighbor sum; conditional spline-flow noise) | Need a *differentiable, identifiable* temporal causal model that handles **instantaneous edges + history-dependent noise**; Rhino does all three | Granger (predictive only, no instantaneous/confounders/hist-noise); VARLiNGAM/DYNOTEARS (linear only) |
| **Variational inference / ELBO** over Z, G, F | Marginal likelihood intractable (3 sets of latents); amortized VAE-style encoder for Z, mean-field q(G), q(F) | EM / exact marginalization infeasible |
| **q(G) = product of independent Bernoullis**, sampled with **Gumbel-softmax (hard, straight-through)** | Need a differentiable sample of a discrete graph for the ELBO gradient | REINFORCE: high variance |
| Instantaneous part of G via **3-way categorical** (i→j / j→i / none) | Avoids sampling both directions → fewer 2-cycles, cleaner DAG sampling | Independent Bernoulli on both (i,j) and (j,i): allows immediate 2-cycles |
| **Acyclicity** via NOTEARS h(G⁰)=tr(e^{G⁰})−D as a soft penalty in p(G), enforced by **augmented Lagrangian** | Continuous, differentiable acyclicity; auglag drives the constraint→0 over training | Combinatorial DAG search: not differentiable |
| **Sparsity prior** p(G) ∝ exp(−α‖G‖² − σ·h(G⁰)) | Causal graphs are sparse; prevents dense spurious graphs | No prior: overfit dense graph |
| q(F): center ρ ~ N, **sigmoid(ρ)** to keep center in [0,1]^K; scale γ via a learned 2×2 **A Aᵀ + diag(exp(B))** covariance (Mahalanobis) | Reparameterizable; the AAᵀ+exp(B) makes Σ positive-definite and **anisotropic** (ellipse-shaped modes, not just circles) | Isotropic single scalar scale: can't capture elongated modes; ablation shows isotropic RBF still works but anisotropic is more flexible |
| q(Z|X): MLP encoder → mean & log-var of Gaussian, reparameterization trick | Amortized inference of latents from observations; standard VAE | Per-sample optimization of Z: not amortized, slow |
| **β-scaling** of the latent-SCM KL term (β = D/4 in code) | Balance reconstruction vs. causal-prior fitting (β-VAE style); without it the SCM term can dominate/vanish | β=1: imbalance |
| **Freeze SCM + graph for first 200 epochs**, train only F + encoder | Let spatial factors + latents settle *before* causal structure is imposed; avoids wrong early graph corrupting the representation | Train all jointly from step 0: unstable, bad causal discovery |
| **Metric flexible** (Euclidean for synthetic, Haversine for global climate) | Global data lives on a sphere; great-circle distance respects Earth's curvature | Euclidean on lat/lon: distorts distances near poles / across dateline |

## The identifiability story (the theoretical heart)

**Why identifiability matters:** we want the recovered Z, F to be *the* true ones (up to permutation+scaling), not some entangled rotation. Prior CRL identifiability (LEAP/TDRL/Hälvä/Lachapelle) needs **no instantaneous effects**, **sparsity**, or **sufficient variability** of the latent process — all hard to verify. SPACY drops all of those by exploiting the **spatial overdetermination**: L >> D means many grid equations constrain few latents.

**Spatial Factor Process (SFP):** generalize the grid to a *continuous* domain G=(0,1)^K (infinite resolution). At each ℓ: X^t(ℓ)=g_ℓ(F_ℓ^⊤ Z^t)+ε_ℓ^t, with F_ℓ=[F_{ψ1}(ℓ),…,F_{ψD}(ℓ)] from a **linearly-independent** family of functions.

**Denoising Lemma (from Khemakhem 2020 iVAE, App B.2.2 Step I):** if observational densities match for all ℓ,t, then the *noiseless* signals match: g_ℓ(F_ℓ^⊤Z)=ĝ_ℓ(F̂_ℓ^⊤Ẑ). Proof: p(X(ℓ)=x|Z) = δ_{x̄} * p_ε (convolution of Dirac at x̄=g_ℓ(F_ℓ^⊤Z) with noise). Equal densities ⇒ δ_{x̄}*p_ε = δ_{x̃}*p_ε. Fourier transform: e^{is x̄}φ_ε(s) = e^{is x̃}φ_ε(s). Since char. function φ_ε ≠ 0 a.e. (the assumption), e^{is x̄}=e^{is x̃} a.e. ⇒ x̄=x̃.

**Linear theorem (g=Id):** denoising gives F_ℓ^⊤Z = F̂_ℓ^⊤Ẑ ∀ℓ,t, i.e. Σ_j F_{ψj}(ℓ)Z_j − Σ_j F_{ψ̂j}(ℓ)Ẑ_j = 0 ∀ℓ. Linear independence of the function family ⇒ the two factor-sets must coincide and Z=PẐ (permutation). [Argument: if the index sets ψ and ψ̂ were disjoint, LI forces all Z=0, contradicting non-degeneracy; matching indices forces Z_j=Ẑ_{V(j)}.]

**General theorem (g_ℓ diffeomorphism, F real-analytic):** denoising ⇒ F_ℓ^⊤Z = h_ℓ(F̂_ℓ^⊤Ẑ) with h_ℓ=g_ℓ^{-1}∘ĝ_ℓ.
- *Lemma (overlapping LI analytic):* det of the D×D matrix M_F(ℓ_1..ℓ_D) is analytic; nonzero somewhere (LI) ⇒ its zero set has measure zero (Mityagin) ⇒ the full-rank set Φ_F has measure 1 and is open; Φ_F ∩ Φ_F̂ has measure 1, nonempty, open.
- Pick D points where both matrices are invertible ⇒ Z = M_F^{-⊤} 𝓗(M_F̂^⊤ Ẑ) =: Θ_ℓ(Ẑ), an invertible map. Same map for any ℓ' in the open set.
- Take log|det Jacobian|, differentiate w.r.t. Ẑ_i. Because the map is the *same* across nearby ℓ (perturb one ℓ_1 along any direction u within the open ball), the quantity Γ_i(Ẑ)= [h''_ℓ/h'_ℓ](F̂_ℓ^⊤Ẑ)·F_{ψ̂i}(ℓ) is **constant in ℓ**. Pairing i,j and using LI of F_{ψ̂i},F_{ψ̂j} on the open ball ⇒ Γ_i=Γ_j=0 ⇒ h''_ℓ ≡ 0 ⇒ **h_ℓ is affine**, h'_ℓ = c_0 constant.
- Differentiate F_ℓ^⊤Z = h_ℓ(F̂_ℓ^⊤Ẑ) w.r.t Ẑ_i: Σ_k F_{ψk}(ℓ)∂Z_k/∂Ẑ_i = c_0 F_{ψ̂i}(ℓ). LI of {F_ψk} ⇒ F_{ψ̂i}=F_{ψk} for exactly one k and ∂Z_k/∂Ẑ_i nonzero for exactly that k ⇒ Jacobian is **permutation×scaling** ⇒ Z = PS Ẑ, factor families match.

**Finite-grid caveat:** continuous theory; with L>>D the system is overdetermined and the measure-zero pathologies are improbable. Causal graph recovery is then handed to Rhino's own identifiability (stationarity, Markov, minimality, sufficiency, well-defined density, conditions on f/g).

## Load-bearing ancestors (write-ups)

- **Rhino (Gong et al., 2022/2023).** Differentiable temporal SCM: Z_i^t = f_i(Pa(<t),Pa(t)) + g_i(Pa(<t),ε). f via per-node trainable embeddings + adjacency-weighted sum over (lagged+instantaneous) neighbors through a shared MLP; noise via **conditional spline flow** (history-dependent). Graph posterior = product of Bernoullis (instantaneous as 3-way), Gumbel-softmax samples, ELBO + augmented-Lagrangian NOTEARS acyclicity. Handles instantaneous edges + hist-dep noise + nonlinearity. **Limitation for us:** operates directly on *observed* variables → doesn't scale to L-grid and ignores spatial structure. SPACY plugs Rhino into the *latent* space.
- **CDSD / Single-Parent Decoding (Brouillard et al. 2024; Boussard et al. 2023).** Learns latent Z + graph from observed time series under **single-parent decoding**: each observed var has exactly one latent parent → identifiable via a denoising (characteristic-function) argument. **Limitation:** one parent per grid point — can't model overlapping modes. SPACY relaxes to multiple parents and gets identifiability instead from spatial overdetermination + LI analytic factors.
- **Khemakhem et al. 2020 (iVAE / VAE↔nonlinear ICA).** Source of the denoising-by-characteristic-function technique and the "identifiability up to permutation+scaling via ELBO" framing. Their identifiability needs conditioning side-information / factorized prior; SFP replaces that with the spatial-overdetermination argument.
- **LEAP / TDRL (Yao et al. 2021/2022) & Lachapelle et al.** Identifiable latent temporal causal processes — but require no-instantaneous-effects / sufficient-variability / sparsity. SPACY's contribution is to *drop* those assumptions.
- **NOTEARS (Zheng et al. 2018).** Smooth exact acyclicity h(W)=tr(e^{W∘W})−d; enables continuous DAG optimization via augmented Lagrangian. Used for the instantaneous adjacency.
- **VAE / reparameterization (Kingma & Welling 2014); Gumbel-softmax (Jang et al. 2017); neural spline flows (Durkan et al. 2019).** The estimation machinery.
- **Mapped-PCMCI / Varimax (Tibau et al. 2022) & Linear-Response (Falasca et al. 2024).** The two-stage baselines: reduce dimension (Varimax-rotated PCA / correlation modes) then run PCMCI / linear-response on the modes. Decoupled reduction → modes need not be causally relevant; PCA weight vectors nonzero everywhere → diffuse, hard-to-interpret modes.
- **NTFA / topographic factor analysis (Manning 2014, Sennesh 2020, Farnoosh 2021).** Prior use of RBF spatial factors to model brain-imaging spatial structure — precedent for kernel-parametrized factors.

## Motivating/diagnostic findings (pre-method, allowed in context)
- Deeper reliance on CI tests does not scale to grids (glymour2019review).
- Spatial proximity → redundant correlated series; conditioning on neighbors masks distant (teleconnection) causal links (xavi2022teleconnection).
- PCA/Varimax modes are spatially diffuse (weights nonzero everywhere) → hard to attribute to physical regions.

## ELBO (the objective the code computes)
log p(X) ≥ Σ_n E_{q(Z|X)q(G)q(F)}[ log p(X^n|Z^n,F)  +  (log p(Z^n|G) − log q(Z^n|X^n)) ] − KL(q(G)‖p(G)) − KL(q(F)‖p(F)).
- recon: ‖X − ĝ(FZ)‖² (Gaussian likelihood = MSE).
- p(Z|G): Rhino conditional-spline log-likelihood of the residual Z^t − f(Pa); for SPACY-L it's MSE of the linear-SCM residual.
- q(Z|X) entropy: closed form Gaussian.
- KL(q(G)‖p(G)) = sparsity (α‖G‖²) + acyclicity (σ·h(G⁰)) + graph entropy (Gumbel/Bernoulli+categorical).
- KL(q(F)‖p(F)): Gaussian KL on ρ, γ (prior N(0,I)).

## Code skeleton (final → scaffold correspondence)
- `class SpatialFactors`: q(F) — ρ_mu/ρ_logvar, γ params (A,B), get_spatial_factors() = RBF/Mahalanobis. **Scaffold slot:** `class SpatialMap` stub.
- `class TemporalAdjacencyMatrix`: q(G), Gumbel sample, sparsity, h(G), entropy. **Scaffold slot:** `class GraphPosterior` stub.
- `class RhinoSCM` (+ likelihood): latent SCM f and noise. **Scaffold slot:** `class LatentDynamics` stub.
- `class SpatialDecoderNN`: g_ℓ (einsum FZ then shared MLP w/ per-grid embedding). part of decoder slot.
- encoder MLP `f_tilde`: q(Z|X). part of encoder slot.
- `SPACY.forward` / `compute_loss_terms`: assembles ELBO. **Scaffold slot:** training loop computing recon+KL.
