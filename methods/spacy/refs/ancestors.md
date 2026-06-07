# Load-bearing ancestors (researched online; verified against primary text)

- **Rhino** — Gong et al., "Deep Causal Temporal Relationship Learning with history-dependent noise" (arXiv 2210.14706). Differentiable temporal SCM: additive-noise model with instantaneous + lagged parents, per-node embeddings + edge-gated MLP aggregation, conditional-spline-flow history-dependent noise, Bernoulli graph posterior with Gumbel-softmax, ELBO + augmented-Lagrangian NOTEARS acyclicity. Used as the latent-space causal model. Limitation: defined over observed variables → no scale, no spatial prior.
- **CDSD / Single-Parent Decoding** — Brouillard et al. 2024 / Boussard et al. 2023, "Causal Representation Learning via Single-Parent Decoding" (arXiv 2410.07013). Joint latents + graph from observed series under the single-parent assumption (each observed var → exactly one latent), identifiable via a denoising (characteristic-function) argument. Limitation: forbids overlapping factors; nonlinear mode collapse.
- **iVAE / VAE↔nonlinear ICA** — Khemakhem, Kingma, Monti, Hyvärinen 2020 (arXiv 1907.04809). Source of the denoising-by-characteristic-function technique (additive noise = convolution → Fourier → cancel φ_ε) and the "identifiable up to permutation+scaling via ELBO" framing.
- **NOTEARS** — Zheng et al. 2018 (arXiv 1803.01422). Smooth exact acyclicity h(W)=tr(e^{W})−d enabling continuous DAG optimization via augmented Lagrangian.
- **LEAP / TDRL** — Yao et al. 2021 / 2022. Identifiable latent temporal causal processes, but require no-instantaneous-effects / sufficient-variability / sparsity; no spatial prior.
- **Mapped-PCMCI / Varimax** — Tibau et al. 2022; PCMCI⁺ Runge 2020. Two-stage reduce-then-discover baseline; PCA/Varimax modes are spatially diffuse.
- **Linear-Response** — Falasca et al. 2024. Correlation/proximity modes + linear-response causal inference.
- **Estimation machinery** — VAE/reparameterization (Kingma & Welling 2014), Gumbel-softmax (Jang et al. 2017), neural spline flows (Durkan et al. 2019), topographic RBF factor analysis (Manning 2014; Sennesh 2020; Farnoosh 2021).

Canonical implementation snapshot: `../code/` (official repo, Rose-STL-Lab/SPACY).
