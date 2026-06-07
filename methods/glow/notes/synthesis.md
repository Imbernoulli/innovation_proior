# Glow — synthesis notes (Phase 1.5)

## The pain point at the time (2018)

Generative modeling of high-dim images. Three likelihood-based families:
- **Autoregressive (PixelRNN/PixelCNN, WaveNet):** exact likelihood, very sharp, but **synthesis is serial** — O(D) sequential steps (one pixel/subpixel at a time). Slow for large images. Latents have unknown marginals (no clean usable latent space).
- **VAEs:** parallel train & sample, but only optimize a **lower bound** (ELBO); inference of latents is **approximate**; can be hard to optimize.
- **Flows (NICE, RealNVP):** exact log-likelihood, exact latent inference, parallel train AND sample, constant-memory backprop (RevNets). Underexplored vs GANs/VAEs. But RealNVP had not matched GAN sample quality and its log-likelihood lagged on benchmarks.

Goal: push flow-based models — keep all flow advantages (exact LL, exact invertible inference, parallel sampling) while **improving log-likelihood** and demonstrating they can do **high-resolution realistic synthesis** (the thing GANs were known for).

## Core math: change of variables (the foundation, from NICE)

x = g(z), z = f(x) = g^{-1}(x), bijective. Prior p(z) simple (spherical Gaussian).
Change of variables:
  p_X(x) = p_Z(f(x)) · |det(∂f/∂x)|
  log p(x) = log p_Z(z) + log|det(dz/dx)|.
Compose f = f_1 ∘ ... ∘ f_K, h_0 = x, h_K = z:
  log p(x) = log p_Z(z) + Σ_i log|det(dh_i/dh_{i-1})|.
The whole game: design invertible layers whose **log|det Jacobian|** is cheap. If Jacobian is triangular, det = product of diagonal, so log|det| = sum(log|diag|) — O(D).

Continuous data needs **dequantization**: x is 8-bit ints; modeling a density on a discrete grid puts infinite density on points. Add uniform noise u~U(0,a) (a = 1/256 in [0,1] or 1/n_bins): x̃ = x + u. Then minimizing −log p(x̃) bounds the discrete log-likelihood. Constant c = −M·log a. Loss reported as **bits/dim** = nll / (log 2 · D).

## Baseline ancestors (load-bearing)

### NICE (Dinh et al. 2014)
- **Additive coupling.** Split x into (x_a, x_b). y_a = x_a + m(x_b); y_b = x_b, where m is an arbitrary NN. Inverse: x_a = y_a − m(y_b); x_b = y_b. m need not be invertible.
- Jacobian is lower/upper triangular with **unit diagonal** ⇒ **det = 1, log-det = 0** ("volume preserving"). Cheap, trivially invertible.
- To give the model volume flexibility, NICE adds a final **diagonal scaling** layer (a learned per-dim scale s, log-det = Σ log|s|).
- Permutation between coupling layers: must swap which half is transformed, else x_b never changes. NICE alternates the partition.
- **Gap:** purely additive (no per-coupling scale), simple partition pattern; designed for small data, not conv images at scale.

### RealNVP (Dinh et al. 2016)
- **Affine coupling.** y_a = x_a ⊙ exp(s(x_b)) + t(x_b); y_b = x_b. s,t are NNs of x_b.
  Inverse: x_a = (y_a − t(y_b)) ⊙ exp(−s(y_b)); x_b = y_b. Still no need to invert s,t.
  Jacobian = [[diag(exp s), ∂/∂x_b], [0, I]] — block triangular ⇒ **log-det = Σ s(x_b)** (sum of the log-scales). Gives the flow volume change per layer (additive was a special case s=0).
- **Masking instead of literal split**, two patterns: **checkerboard** (spatial) and **channel** masking. Alternate masks so every variable gets transformed.
- **Squeeze:** reshape h×w×c → (h/2)×(w/2)×4c, trading space for channels so channel masking has channels to work with at deeper levels.
- **Multi-scale architecture:** after some flow steps, **factor out** half the dims as latent z_i (Gaussian) and continue transforming the rest at lower spatial resolution. Distributes the latent across scales, reduces compute/memory, gives a coarse-to-fine latent. log-det unaffected (it's just defining part of z early); the factored-out half gets its own Gaussian log-prob.
- **Batch normalization** inside coupling NNs and as a flow layer (its scale contributes to the Jacobian) to train deep flows.
- **Gap 1 — permutation is fixed:** between couplings RealNVP only *reverses channel order* (or uses fixed checkerboard/channel alternation). A fixed permutation is a hand-chosen, non-learned way to mix dims. Could a *learned* mixing be better?
- **Gap 2 — batchnorm at batch size 1:** BN noise variance ∝ 1/(batch per PU). High-res images force batch size 1 per GPU ⇒ BN is unstable/degrades.
- **Gap 3 — checkerboard masking adds architectural complexity** (two mask types, spatial patterns).

## The method that drops out (Glow)

One **step of flow** = actnorm → invertible 1×1 conv → affine coupling. Stack K steps per level, L levels with squeeze + multi-scale split.

### 1. Actnorm (replaces batchnorm)
- Per-channel affine: y_{i,j} = s ⊙ x_{i,j} + b (s,b are length-c vectors, shared over spatial positions i,j).
- **Data-dependent init (DDI):** initialize s,b so post-actnorm activations have zero mean, unit variance per channel given the *first* minibatch. After that, s,b are ordinary trainable params — **no dependence on batch statistics at train time**, so it works at batch size 1. (This is weight-norm-style data-dependent init, Salimans & Kingma 2016.)
- **Log-det:** for one position it's log|det diag(s)| = Σ log|s|; applied at every one of h·w spatial positions independently ⇒ **h·w·Σ log|s|**.
- WHY over BN: removes the batch-statistics dependence (BN's activation noise ∝ 1/batchsize blows up at batch 1); keeps the normalization benefit for training deep flows.

### 2. Invertible 1×1 convolution (the contribution; replaces fixed permutation)
- A 1×1 conv with c→c channels applies the SAME c×c matrix W to the channel vector at every spatial location: y_{i,j} = W x_{i,j}.
- KEY observation: a **permutation of channels is exactly multiplication by a permutation matrix.** RealNVP's "reverse the channel order" and "fixed random permutation" are both special cases of "multiply channels by a fixed matrix." So generalize: **let W be a learned, invertible c×c matrix** (initialize as random rotation/orthogonal so it starts as a clean invertible mixing with log-det 0). The permutation becomes a learnable linear mixing of channels — strictly more expressive than any fixed permutation, and it learns *how* to route information into the coupling split.
- **Log-det derivation:** the full Jacobian of the layer over the whole tensor is block-diagonal: independently at each of the h·w positions the map is x_{i,j} ↦ W x_{i,j}, contributing log|det W| each ⇒ total **h·w·log|det W|**. (Equivalently the conv weight is reshaped to [1,1,c,c]; the determinant of a 1×1 conv operator over an h×w grid is (det W)^{hw}.)
- Inverse: just W^{-1} as the 1×1 conv weight. Sampling needs W^{-1} once per layer.
- **Cost:** computing det W and W^{-1} is O(c^3); the conv itself is O(h·w·c^2). For typical c these are comparable, so the overhead is small (~7% wallclock measured). But for large c the O(c^3) determinant is the worry.
- **LU parameterization (make log-det O(c)):** parameterize W directly as
    W = P · L · (U + diag(s))
  P = fixed permutation matrix (chosen at init, not trained), L = lower-triangular with unit diagonal, U = strictly upper-triangular (zero diagonal), s = vector. Then det L = 1, det(U+diag s) = Π s_k, det P = ±1, so
    **log|det W| = Σ log|s|** — O(c), no determinant computation at all.
  Init: sample a random rotation W_0, take its LU/PLU decomposition to set P (fixed), and the initial L, U, s (trainable). To keep sign of s fixed and optimize stably, store sign(s) (fixed) and log|s| (trained); U+diag(s) reconstructed as U·(strict-upper-mask) + diag(sign_s · exp(log_s)).
  Paper notes wallclock difference between det-W and LU versions wasn't large at their c, but LU is the principled fix for large c.

### 3. Affine coupling (from RealNVP, simplified)
- Split along **channels only** (drop checkerboard — "we only perform splits along the channel dimension, simplifying the overall architecture"). The 1×1 conv now does the cross-channel mixing that checkerboard masking used to provide, so checkerboard is redundant.
- (log s, t) = NN(x_b); s = exp(log s); y_a = s ⊙ x_a + t; y_b = x_b. Inverse divides. log-det = Σ log|s|. Additive coupling = special case s=1, log-det 0.
- NN = 3 convs: 3×3 (→512 ch) → ReLU → 1×1 (512) → ReLU → 3×3-zero-init (→output). Middle is 1×1 because in/out are wide.
- **Zero init of the last conv** of each NN ⇒ each coupling starts as **identity** ⇒ the whole deep flow starts as identity, stabilizing very deep training. (conv2d_zeros also has a per-channel learned logscale·3 multiplier, init 0.)
- In the OpenAI code the affine scale is `sigmoid(h+2.)` (kept positive, near 1 at init) rather than raw exp — a stability choice; the paper writes s=exp(log s).

### Permutation ablation (motivating/diagnostic, allowed)
Three choices compared for "what mixes channels between couplings": (a) reverse channel order (RealNVP), (b) fixed random permutation, (c) invertible 1×1 conv. The 1×1 conv is the generalization; the comparison is the *experiment* (excluded from reasoning's results, but the framing — these are special cases of a linear map — is the insight that drives the design).

### Multi-scale + squeeze (from RealNVP, kept)
- squeeze 2: h×w×c → h/2×w/2×4c (factor 2). Gives channels for channel-coupling and the 1×1 conv to act on, and downsamples.
- After each level (except last), split2d: factor out half the channels as latent with a learned conditional Gaussian prior (mean,logsd = zero-init conv of the kept half), squeeze the rest, continue. Top level: learned (optionally class-conditional) Gaussian prior.

### Temperature sampling (qualitative, additive case)
Sample z from N(0, T^2 I): p_{θ,T}(x) ∝ p_θ(x)^{T^2}. T≈0.7 trades diversity for quality. Forward-looking only (no results).

## Design-decision → why table

| Decision | Why this | Rejected alternative & failure |
|---|---|---|
| Flow / change-of-variables backbone | exact LL, exact latent inference, parallel sample | AR: serial sampling; VAE: only a bound, approx inference |
| Triangular/structured Jacobians | log-det = sum of log-diagonal, O(D) | general det is O(D^3), intractable |
| Dequantize with uniform noise | density model on discrete data → bound on discrete LL; avoids infinite density | training directly on ints is ill-posed |
| Coupling layers (vs general bijector) | invertible w/o inverting the NN; tractable log-det | autoregressive transforms invert serially (slow sampling) |
| Affine (not just additive) coupling | per-layer volume change ⇒ more expressive; additive is special case s=1 | additive alone is volume-preserving, weaker |
| Replace fixed permutation w/ invertible 1×1 conv | permutation = mult by permutation matrix; generalize to learned invertible W → learns channel routing, lower NLL, faster convergence | fixed reverse/random permutation: not learned, suboptimal mixing |
| Init W as random rotation | orthogonal ⇒ invertible, log|det|=0 at start (no initial volume change) | random non-orthogonal could be near-singular / large det |
| LU parameterization of W | log|det W|=Σlog|s|, O(c) not O(c^3); fixed P, trained L,U,s | direct W needs O(c^3) det each step (matters for large c) |
| store sign(s)+log|s| | keep diagonal sign fixed, optimize log-magnitude stably | optimizing s directly risks sign flips / s→0 (det→0) |
| Actnorm instead of BatchNorm | BN noise var ∝ 1/batch; high-res forces batch 1/PU → BN degrades | BatchNorm unstable at tiny per-PU batch |
| Data-dependent init of actnorm | zero-mean/unit-var start like BN, but then batch-independent params | fixed init: activations badly scaled at start of deep flow |
| Actnorm log-det = h·w·Σlog|s| | same per-channel scale applied at every spatial position | (it's the only correct value; mirrors per-position diag scaling) |
| Channel split only, drop checkerboard | 1×1 conv already mixes channels; simpler architecture | checkerboard adds two mask types & complexity for redundant mixing |
| Zero-init last conv of coupling NN | each step = identity at init ⇒ stable very-deep training | random init: deep flow unstable at start |
| Squeeze 2 before couplings | provides channels for channel-coupling/1×1 conv; downsample | without squeeze, channel coupling has too few channels |
| Multi-scale (factor out half each level) | coarse-to-fine latent, less compute/memory at deep layers | single-scale: heavier, no hierarchical latent |
| NN width 512, 3×3→1×1→3×3 | middle 1×1 since in/out wide; enough capacity for s,t | all-3×3 wider middle = wasteful |
| Adam, bits/dim objective | standard SGD optimizer; LL ↔ compression in bits | — |

## Code grounding
Canonical: openai/glow `model.py` (revnet2d, revnet2d_step, invertible_1x1_conv plain + LU, split2d, codec, prior), `tfops.py` (actnorm_center/scale w/ DDI, squeeze2d/unsqueeze2d, gaussian_diag, conv2d_zeros). Appendix code.py = the plain invconv. Final answer/reasoning code mirrors these (PyTorch-flavored clean reimplementation acceptable but must match structure & log-dets).
</content>
