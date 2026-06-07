# StyleGAN — Synthesis Notes (Phase 1.5)

## The pain point (research question, in-frame)
By late 2018, GANs (esp. ProGAN, BigGAN, SN-GAN) can make high-res, high-FID faces at 1024². But the generator is a **black box**:
- We feed `z ~ N(0,I)` into the first layer and out comes an image. We have **no control** over *what* aspect of the image any part of `z` controls. Pose, identity, hair, lighting all move together when you nudge `z`.
- The **origin of stochastic detail** (exact hair placement, freckles, pores) is unexplained — the network has to *invent* spatial randomness from the deterministic `z` through the conv stack.
- **Latent interpolations** look smooth in demos but there is no quantitative handle on how "entangled" the space is; features can pop in/out mid-interpolation.
- The latent space `Z` is forced to match the data density (sampling probability of factor combinations in `Z` must equal data density), which **forces entanglement** (curved warping) whenever the data manifold is non-uniform.

Goal: redesign the **generator only** (don't touch D or loss — orthogonal to the loss-function debate) so that (a) different scales of image structure are independently controllable, (b) stochastic detail has a dedicated cheap source, (c) the latent representation becomes more disentangled/linear, and (d) FID does not regress (ideally improves). Plus: invent metrics to *measure* disentanglement without an encoder.

## Load-bearing ancestors

### 1. GAN (Goodfellow 2014)
Minimax: G maps `z→x`, D scores real vs fake. Non-saturating G loss `softplus(-D(G(z)))`. StyleGAN keeps this untouched. R1 regularization (Mescheder 2018): penalty `(γ/2)·E[||∇_x D(x)||²]` on reals — stabilizes, and FID keeps improving for far longer than WGAN-GP (so they raise training from 12M→25M images).

### 2. Progressive Growing of GANs (Karras 2017) — the literal baseline (config a)
This *is* the starting codebase. StyleGAN inherits D architecture, resolution-dependent minibatch sizes, Adam hyperparams, generator EMA. Key ProGAN tricks reused/relevant:
- **Progressive growing**: train 4²→…→1024², fade in each new resolution with a learned `lerp` (lod). Lets the net learn coarse structure first.
- **Equalized learning rate**: init weights `w ~ N(0,1)`, then at *runtime* multiply by He constant `c = gain/sqrt(fan_in)` (gain = √2 for (leaky)ReLU). So effective weight is `ŵ = c·w`. WHY at runtime not init: Adam/RMSProp normalize each parameter's update by its own gradient RMS, making the *update size scale-invariant per parameter*. If different layers have different weight dynamic ranges (from He init), Adam takes longer to move the large-range ones — learning speed becomes uneven. Equalizing the range so all weights have the same std (1) and pushing the He scaling to runtime makes the *effective* learning rate identical across all params. Implemented in `get_weight`: with `use_wscale`, `init_std=1/lrmul`, `runtime_coef = he_std·lrmul`.
- **Pixelwise feature normalization** (pixel_norm): normalize each pixel's feature vector to unit length, after each conv in G. Prevents magnitude blow-up from G/D escalation. *StyleGAN drops this in G* (instance norm in AdaIN does the job) but keeps it as the `Z`-normalization step at the front of the mapping network.
- **Minibatch stddev**: append a constant feature map = avg over minibatch of per-pixel feature stddev, near the end of D, to fight mode collapse / increase variation. Kept in D.

### 3. AdaIN / Conditional Instance Norm / Instance Norm — the style-transfer lineage (the core borrow)
The whole architecture is borrowed from **arbitrary style transfer** (Huang & Belongie 2017, AdaIN). The chain:
- **Instance Normalization** (Ulyanov): normalize each *sample*, each *channel* over spatial dims:
  `IN(x) = γ (x-μ(x))/σ(x) + β`, where `μ_nc(x)=(1/HW)Σ_hw x_nchw`, `σ_nc(x)=sqrt((1/HW)Σ(x-μ)²+ε)`. (γ,β learned scalars per channel.) Empirically IN >> BN for style transfer.
- **Why IN works = style normalization** (Huang's Fig.1 experiment, the key empirical insight): they train an IN model and a BN model on (a) original, (b) contrast-normalized, (c) **style-normalized** images. IN's advantage over BN *persists* under contrast normalization but **largely vanishes when images are already style-normalized**. Conclusion: **IN performs a form of style normalization** — it removes instance-specific style (encoded in channel-wise mean/variance of feature maps) and lets the rest of the net focus on content. Foundation: conv feature statistics (Gram matrix; channel-wise mean & variance) carry the **style** of an image (Gatys 2016, Li 2017 "Demystifying" showed matching channel-wise mean+var ≈ matching Gram ≈ style).
- **Conditional Instance Norm** (Dumoulin 2016): one network, N styles, by learning a *different* `(γ^s, β^s)` per style s: `CIN(x;s)=γ^s (x-μ(x))/σ(x)+β^s`. Surprising: same convs, just swapping the affine params → completely different output style. So **the affine params of an IN layer can fully control the output style.** But: params scale linearly with #styles (`2FS`), can't do arbitrary new styles.
- **AdaIN** (Huang 2017): drop the *learned* affine; instead compute the affine *adaptively* from a style input `y`'s own statistics:
  `AdaIN(x,y) = σ(y)·(x-μ(x))/σ(x) + μ(y)`.
  i.e. normalize content x to zero-mean/unit-var per channel, then re-scale/re-bias to the style's `(σ(y),μ(y))`. No learnable params; works for arbitrary styles; cheap. The intuition: a channel detecting a feature (brushstroke) → its mean activation = "how much of that feature" = a style knob; the variance carries finer style info.

The leap StyleGAN makes: in style transfer the style `(σ(y),μ(y))` comes from an *example image*; **what if it comes from a learned function of the latent code instead?** Then `z` (via `w`) *is* the style, injected at every layer.

### 4. The entanglement argument (motivating, pre-method, knowable a priori)
Disentanglement goal: latent space = linear subspaces each controlling one factor. But in a traditional GAN the *input* latent `Z` must be sampled from a fixed prior (Gaussian) and the generator must reproduce the data density. If the real data has a "missing combination" (e.g. no images of {feature A absent} ⊗ {feature B}), but `Z` is a nice round Gaussian, then to map a round `Z` onto a data manifold with holes, the mapping `Z→features` must **warp/curve** so that the forbidden region in feature space has no preimage in the sampled `Z`. Curved mapping = entangled (changing one z-coord moves many factors non-linearly). This is *unavoidable* for any fixed input distribution forced to match data density. Escape: insert an **intermediate** space `W` whose density is *not* constrained to be Gaussian / match data — it's induced by a learned mapping `f(z)`. `f` can "unwarp," absorbing the curvature, leaving `W` flatter/more linear. And there's pressure to do so: generating realistic images from a disentangled rep is easier than from an entangled one.

### 5. Truncation trick (Brock 2018 BigGAN, Marchesi, Glow) — pre-method
Low-density regions of the prior are poorly learned → poor samples. Truncating/shrinking the sampling region toward the mean improves average quality at the cost of variation. StyleGAN does it in `W`: center of mass `w̄ = E_z[f(z)]`, then `w' = w̄ + ψ(w-w̄)`, ψ<1. Reliable without loss changes (unlike BigGAN's orthogonal-reg dependence). Can apply only to low resolutions (coarse styles) so detail is untouched.

### 6. Perceptual distance (Zhang 2018 LPIPS) and VGG (Simonyan 2014) — for the metric
LPIPS: weighted L2 between VGG16 deep embeddings, weights fit to human similarity. Quadratic in small perturbations → used to define perceptual path length.

## The method, end to end (what gets built)

Two networks replacing the monolithic generator:
- **Mapping network f: Z→W**, an 8-layer MLP, `z,w ∈ R^512`. `z` is pixel-normalized (unit length) first. Output `w` is *not* normalized.
- **Synthesis network g**: starts from a **learned constant** `4×4×512` tensor (init to 1). 18 layers (2 per resolution 4²→1024²). At each layer:
  1. add **noise**: a single-channel N(0,1) image, broadcast to all channels via **learned per-channel scale** B, added after the conv. (Per-layer fresh noise.)
  2. add bias, leaky-ReLU (α=0.2).
  3. **AdaIN**: instance-normalize, then scale+bias with **style y=(y_s,y_b)** = a learned **affine transform A** of `w`. `dim(y) = 2·(#channels)`. `AdaIN(x_i,y)=y_{s,i}(x_i-μ(x_i))/σ(x_i)+y_{b,i}`.
  - Last layer → RGB via 1×1 conv (toRGB), like ProGAN.

Why each piece:
- **Mapping net → W**: gives the unconstrained intermediate space (the entanglement escape). Deeper (8) helps because it has more capacity to unwarp; deeper mapping → better FID, separability, path length (but learning rate must be cut 100×: `λ'=0.01λ`, else deep mapping destabilizes training at high LR).
- **Const input + style-only conditioning**: surprising empirical finding — once mapping+AdaIN exist (config c), feeding `z` into the first conv stops helping; so they remove the input layer entirely (config d). The image is generated *purely from styles*. This is the strongest statement that "style at every layer" is sufficient.
- **AdaIN localizes styles**: because AdaIN first *normalizes* (wipes the previous layer's per-channel statistics) then re-applies new style, **each style controls exactly one conv before being overwritten** by the next AdaIN. That's the mechanism behind scale-specific control. Style affects the *whole* feature map (spatially invariant) → global effects (pose, identity, lighting). 
- **Noise inputs**: dedicated cheap source of spatial randomness. Added per-pixel after each conv. Because it's spatially varying and per-pixel, it's ideal for *stochastic* detail; if the net tried to use it for global structure (pose) it'd be spatially inconsistent → penalized by D. And because fresh noise is available every layer, there's no incentive to synthesize randomness from earlier activations → noise effect stays **localized** to matching scale. Coarse-layer noise → big curls/background; fine-layer noise → pores, fine hair. Frees capacity that a traditional gen wastes inventing pseudo-random patterns (hence the repetitive artifacts).
- **Style mixing / mixing regularization**: during training, a fraction (style_mixing_prob=0.9) of images use **two** latents `z1,z2 → w1,w2`, switching at a random crossover layer (w1 before, w2 after). Decorrelates adjacent styles → prevents net assuming neighboring styles are correlated → stronger localization. Test-time: mixing coarse styles of B into A transfers pose/face shape; mixing fine styles transfers color/microstructure.
- **Truncation in W**, applied per-layer with a cutoff (low-res only), via `lerp(w̄, w, ψ)`.

### The two disentanglement metrics (must be derived in full)
1. **Perceptual path length (PPL)**: subdivide an interpolation path, sum LPIPS over segments; in the fine limit it's an integral. Approximate with small `ε=1e-4`. In `Z` (spherical interp, since z is on a sphere — normalized): 
   `l_Z = E[ (1/ε²) · d( G(slerp(z1,z2;t)), G(slerp(z1,z2;t+ε)) ) ]`, `z1,z2~P(z), t~U(0,1)`.
   In `W` (linear interp, since w not normalized): `l_W = E[ (1/ε²) d( g(lerp(f(z1),f(z2);t)), g(lerp(...;t+ε)) ) ]`.
   Divide by ε² because LPIPS d is **quadratic** in the perturbation (it's ~ a metric/Mahalanobis form), so the per-segment cost ~ ε²·(local speed)²; dividing by ε² gives the squared-speed density, whose expectation = path-length functional. Shorter total = flatter/more linear/less entangled. Subtlety: full-path measure is slightly biased toward Z, because a flattened W can contain off-manifold regions between on-manifold endpoints (badly reconstructed) inflating l_W; restricting to endpoints t∈{0,1} reduces l_W (not l_Z) — observed. 100k samples.
2. **Linear separability**: if W is disentangled, a single factor → a linear direction → a hyperplane separates that binary attribute. Train 40 auxiliary attribute classifiers (= D architecture, no minibatch-stddev). Generate 200k images, classify, sort by confidence, keep top 100k (the confident half) → labeled latents. Fit a linear **SVM** on the latent (`z` for trad, `w` for style) to predict each attribute; compute conditional entropy `H(Y|X)`, Y=classifier label, X=SVM side. Low H = the linear plane already determines the label = consistent linear direction. Final score `exp(Σ_i H(Y_i|X_i))` over the 40 attributes; exponentiate (like inception score) to map log→linear domain. Lower = more separable = less entangled. W consistently beats Z; deeper mapping improves it; putting a mapping net in front of a *traditional* gen wrecks Z separability but improves W (and FID) — evidence the synthesis net inherently favors a disentangled input.

## Config ladder (the discovery order — motivating, FID is diagnostic of *baselines*, fine to mention as the build path)
a baseline ProGAN → b improved baseline (bilinear up/down with binomial blur filter [1,2,1], longer train, tuned LR, R1 loss, start at 8²) → c +mapping+AdaIN → d remove input layer, use learned const → e +noise → f +mixing regularization. (The *proposed-method* benchmark win numbers themselves are out of scope; the ladder is the construction logic.)

## Key implementation facts (from canonical NVlabs/stylegan TF code) — to ground final code
- `get_weight`: equalized LR via runtime_coef; mapping uses lrmul=0.01.
- `style_mod`: `style = dense(w, 2C); reshape to (N,2,C,1,1); return x*(style[:,0]+1) + style[:,1]`. NOTE the `+1`: the affine output is centered so y_s learns around 1 (bias for y_s init to 1). This is the AdaIN scale+bias.
- `apply_noise`: noise `N(0,1)` shape `[N,1,H,W]`, per-channel learned `weight` (init 0), `x + noise*weight`.
- `instance_norm`: subtract spatial mean, divide by spatial std (per N,C). No learned affine (the affine is the style_mod).
- **Layer epilogue order in code**: `apply_noise → apply_bias → activation(lrelu) → [pixel_norm off] → instance_norm → style_mod`. (Paper Fig.1 draws AdaIN right after conv+noise; code does norm+style after the nonlinearity. Match the code.)
- `pixel_norm` used only as the Z-normalization at front of mapping (normalize_latents=True).
- First block 4×4: learned const (init ones), tile to batch, epilogue(0); then Conv 3×3, epilogue(1). Subsequent res: Conv0_up (upscale+conv+blur) epilogue, Conv1 epilogue.
- Mapping: pixel_norm, then 8× [dense(512, lrmul=.01)+bias+lrelu], broadcast to num_layers.
- G_style wrapper: mapping → EMA of w (dlatent_avg, beta .995) → style mixing (prob .9, random crossover) → truncation (psi .7, cutoff 8 layers) → synthesis.
- Loss untouched: non-saturating logistic + R1 (γ=10) for FFHQ; WGAN-GP for CelebA-HQ.
- Inits: convs/dense/affine N(0,1); const=1; biases & noise scales=0; **y_s biases=1**.

## Design-decision → why (table)
- 8-layer MLP mapping: capacity to unwarp Z→W; deeper monotonically helps FID/sep/PPL. Rejected: feed z directly (entangled, must match data density).
- lrmul 0.01 on mapping: deep mapping + high LR → instability; cut LR 100×.
- pixel-norm z at front: keep z on unit sphere (consistent scale into MLP); matches slerp interpolation choice for the Z-metric.
- learned constant input: styles alone suffice (config c→d empirical); removes the only place data-density-matching could re-enter.
- AdaIN (not WCT/general feature transform Li2017/Siarohin2018): efficiency + compact (just mean/var) vs full whitening-coloring covariance.
- normalize-then-restyle (the IN inside AdaIN): the *reason* styles localize to one layer — normalization erases prior style so each AdaIN's effect is overwritten by the next.
- spatially-invariant style vs per-pixel noise: clean split global(style)/stochastic(noise); D penalizes spatially-inconsistent global use of noise.
- per-layer fresh noise: no incentive to synthesize randomness from activations → localized stochasticity; frees capacity (kills repetitive artifacts).
- per-channel learned noise scale B: lets each channel decide how much stochasticity it wants; init 0 so noise fades in.
- mixing reg (p=0.9): decorrelate adjacent styles → localization; improves robustness to test-time mixing.
- R1 over WGAN-GP (FFHQ): FID keeps decreasing far longer → train 25M imgs.
- bilinear resample w/ binomial blur [1,2,1]: anti-alias (shift-invariance, Zhang 2019) vs nearest-neighbor; part of improved baseline b.
- truncation in W per-layer w/ cutoff: trade variation for quality only at coarse scales; reliable in W without loss change.
- divide PPL by ε²: LPIPS quadratic → squared local speed; gives length density.
- exp(ΣH) separability: log→linear domain like inception score, comparable numbers.
- SVM + keep-confident-half: remove ambiguous labels so the linear-separability signal isn't drowned by classifier noise.

## Code-framework scaffold correspondence
Pre-method skeleton = ProGAN-style harness: equalized-LR `get_weight`, `dense`, `conv2d`, up/downscale w/ blur, `apply_bias`, `leaky_relu`, `pixel_norm`, `minibatch_stddev`, a `Generator` that maps z→image via a conv stack, `Discriminator` (kept), non-saturating+R1 loss, Adam, progressive-growing training loop. Empty slots (the contribution): the mapping network `f`, the per-layer **style injection** module, the **noise injection** module, the synthesis network built from a **learned constant**, **style mixing**, **truncation in the intermediate space**, and the two **disentanglement metrics**. Final code fills exactly these.
