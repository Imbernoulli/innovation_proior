# StyleGAN2 — synthesis notes (design-decision → why)

## Pain point / starting state
- A style-based generator (StyleGAN): mapping net f: Z→W (8 FC layers, 100x lower LR),
  affine A: w→style s, synthesis g with AdaIN at every layer, per-pixel noise, constant 4x4 input.
  Progressive growing (PGGAN) for stability at 1024^2.
- Observed phenomena (pre-method facts, knowable by staring at the trained model):
  1. **Droplet/blob artifacts**: water-droplet blobs appear in ALL feature maps starting ~64x64,
     stronger at higher res. Present even when invisible in final RGB. ~0.1% of images: droplet
     missing → severely broken image. Discriminator does not remove it.
  2. **Phase/texture-sticking artifacts** from progressive growing: teeth/eyes stuck at preferred
     pixel locations, jumping rather than moving smoothly. Hypothesis: each resolution momentarily
     IS the output res → forced to emit max-frequency detail → excessive high-freq in intermediate
     layers → broken shift-invariance.
  3. **PPL correlates with quality**: smoother W→image mapping ↔ higher perceived quality. FID/P&R
     blind to it (texture bias of Inception/VGG features).

## Decision → why (with rejected alternatives)

### 1. Diagnose droplet = AdaIN
- AdaIN normalizes mean+var of EACH feature map separately → destroys relative magnitudes between maps.
- Hypothesis: generator deliberately creates a strong localized spike that dominates the per-map
  statistics; after AdaIN normalizes by that spike, the generator has effectively rescaled everything
  else however it wants — i.e. it smuggles signal-strength info past instance norm. The droplet is
  the spike. Supported by: remove normalization → droplet disappears completely.
- Why not just remove AdaIN? Removing normalization kills the droplet AND improves FID slightly, BUT
  loses scale-specific control / style mixing (styles can amplify a feature map by >10x; without
  per-sample counteraction the next layers can't operate). So we want to keep normalization's effect
  but remove its data-dependence.

### 2. Architecture cleanup (Fig 2c) before redesign
- Move bias + noise OUTSIDE the style block (operate on normalized data). Why: in original StyleGAN
  bias/noise are applied inside the block, so their relative impact is inversely proportional to the
  current style magnitude → unpredictable. Outside → predictable.
- After this, normalization/modulation on **std only** (mean not needed). Why: empirically sufficient
  once bias/noise are moved out.
- Remove bias/noise/norm on the constant 4x4 input. Why: no observable drawback.
- Initialize const c1 ~ N(0,1); single shared noise scaling factor per layer; weights ~N(0,1),
  biases & noise scales = 0, EXCEPT affine (style) biases = 1 (so initial style ≈ 1, a no-op scale).

### 3. Weight demodulation (the core fix) — DERIVED
- Modulation = scale input feature map i by s_i. Fold into weights: w'_{ijk} = s_i · w_{ijk}.
  (i = input map, j = output map, k = kernel footprint.)
- Goal of the instance norm that followed: remove s's effect from output statistics.
  Do it analytically instead of from data.
- Assume inputs i.i.d., unit std. Output map j: o_j = Σ_{i,k} w'_{ijk} x_{ik}.
  Var(o_j) = Σ_{i,k}(w'_{ijk})² Var(x) = Σ_{i,k}(w'_{ijk})²  ⇒  σ_j = sqrt(Σ_{i,k}(w'_{ijk})²).
  (i.e. output scaled by the L2 norm of the corresponding weights.)
- Restore unit std → divide output map j by σ_j. Bake into weights:
  **w''_{ijk} = w'_{ijk} / sqrt( Σ_{i,k}(w'_{ijk})² + ε ).**   ε avoids div-by-zero.
- This is "weaker" than instance norm (statistical assumption vs actual contents) — and that is the
  point: it is deterministic, data-independent, so the generator can't game it with a spike → no
  droplet, while style scaling (hence style mixing) is preserved.
- Lineage: same flavor as He/Glorot initializer variance analysis; same calculation as weight
  normalization (Salimans 2016) used there as reparameterization not as a norm replacement.
- tRGB (output 1x1 conv) layers: modulate but DO NOT demodulate (they produce RGB, not unit-variance
  features for a downstream conv).
- Activation scaling: scale leaky-ReLU so it preserves expected signal variance (so the unit-variance
  assumption holds through the nonlinearity). gain = sqrt(2) for lrelu(0.2).
- Implementation: weights differ per sample → can't use a plain conv. Use **grouped convolution**:
  reshape minibatch of N samples into 1 sample with N groups; reshape back after. No data copy.

### 4. Path-length regularization — DERIVED
- Want: fixed-size step in W ⇒ fixed-magnitude change in image, independent of w and direction
  (well-conditioned mapping → better quality, much easier inversion).
- Local map metric = Jacobian J_w = ∂g/∂w. Want ‖J_w^T y‖ ≈ const for random image-space y.
- Regularizer:  E_{w, y~N(0,I)} ( ‖J_w^T y‖₂ − a )².
- Efficient: J_w^T y = ∇_w (g(w)·y) — one backprop, no explicit Jacobian.
- a = dynamic target = EMA of ‖J_w^T y‖₂ (decay β_pl=0.99). Why dynamic not fixed: a only sets a
  global scale (in theory irrelevant), but a fixed a that mismatches init scale wastes the critical
  early training pushing weight magnitudes around → degrades final model & hurts run-to-run
  consistency. EMA targets whatever scale already exists.
- **Appendix proof**: minimized (in high dim L) when J_w is orthogonal up to a global scale.
  Sketch (re-derived):
  - SVD J_w^T = U Σ̃ V^T; radial symmetry of y and of L2 norm ⇒ U,V drop out ⇒ depends only on
    singular values: L_w = E_{ỹ}( ‖Σ ỹ‖₂ − a )², ỹ~N(0,I) in R^L, Σ diagonal.
  - In high L, N(0,I) mass concentrates on shell r=√L (1D normal in r, mean √L, std 1/√2).
  - Plug shell density; minimized when (r‖Σφ‖ − a)² has its min on the shell for every direction φ,
    i.e. Σ = (a/√L) I — all singular values equal → orthogonal up to scale.
  - Consequence: orthogonal Jacobian preserves curve lengths ⇒ g is (locally) an isometry ⇒
    straight lines in W map to geodesics on image manifold ⇒ smooth interpolations, no detours.
- Weight: γ_pl = ln2 / ( r²(ln r − ln 2) ), r = output res. Verified: pl_weight=2 absorbs the
  1/num_pixels (=1/r²) from y normalization and 1/num_affine_layers (=1/(2log2 r − 2)) from the
  per-layer mean → 2/(r²·2(log2 r −1)) = 1/(r²(log2 r −1)) = ln2/(r²(ln r − ln2)). ✓
- Related: Odena 2018 Jacobian clamping (they use finite differences on J δ; we do J^T y analytically).
  Spectral norm only bounds the LARGEST singular value → does not equalize the rest → not the same.

### 5. Lazy regularization
- Reg terms (R1, path length) need not be computed every step. Compute every k steps in a separate
  pass; share Adam state. R1 at k=16 (D), path length at k=8 (G), no harm.
- Adjust optimizer hyperparams for the extra reg step: c=k/(k+1), λ'=cλ, β1'=β1^c, β2'=β2^c;
  multiply reg term by k to keep gradient magnitude balanced.
- pl_minibatch_shrink=2: compute path reg on half the minibatch to save memory.

### 6. Remove progressive growing → skip-G + residual-D (Fig 7)
- PGGAN stabilizes but causes the phase/sticking artifacts.
- Survey: skip connections (U-Net/MSG-GAN), residual nets, hierarchical. MSG-GAN connects matching
  resolutions of G and D with multiple skips.
- **Skip generator**: each block's tRGB upsampled-and-summed into a running RGB image (no topology
  change, but low-res dominates early then shifts to high-res — preserves PGGAN's coarse-to-fine
  schedule emergently, not enforced).
- **Residual discriminator**: residual blocks; resembles classifiers where resnets help → big FID win.
- 3x3 generator/disc choices from the 9-way ablation: skip-G + residual-D best. Residual-G was
  harmful (except one LSUN Car case).
- Residual blocks multiply the merged path by 1/√2 to cancel the variance doubling from adding two
  paths (in classification resnets BN hides this; here there's no BN, so it's explicit and crucial).
- Bilinear up/downsampling everywhere; residual D uses 1x1 conv skip with down-sampling.

### 7. Resolution usage / capacity
- Skip-G lets us measure per-resolution contribution (std of each tRGB's pixel output, normalized to
  sum 100%). Found high-res layers under-contribute → images look like sharpened 512² not true 1024².
- Fix: double feature maps in 64²–1024² layers (+22% G params, +21% D params) → high-res contributes,
  FID & recall improve. This is config F (the final "StyleGAN2").

### 8. Projection / inversion (uses smoothness)
- Project into unextended W (not per-layer W+) so result is something G could really produce.
- Optimize w and per-layer noise maps; add ramped Gaussian noise to w (explore); LPIPS image loss
  at 256²; noise-map autocorr regularizer (force noise to be signal-free unit Gaussian) at
  multi-scale pyramid, weight α=1e5; renormalize noise to 0-mean unit-var each step.
- Path-length-regularized G is dramatically easier to invert → can attribute a generated image to G.

## Out of scope (proposed-method results): FID/PPL tables, LSUN comparisons, attribution histograms.

## Canonical code anchors (NVlabs/stylegan2)
- training/networks_stylegan2.py: modulated_conv2d_layer (eq1+eq3), G_synthesis_stylegan2
  (skip/resnet), D_stylegan2 (resnet).
- training/loss.py: G_logistic_ns_pathreg (path length + EMA a + γ_pl comment).
- Lazy reg + c-correction: training_loop.py / optimizer; G_reg_interval=8, D_reg_interval=16.
