# Synthesis — Spatial Transformer Networks

## Pain point / research question
CNNs are powerful but not spatially invariant to large transforms. Max-pooling gives only small, local, pre-defined translation invariance (2x2 support) — real invariance to scale/rotation/warp only emerges over a deep hierarchy, and intermediate feature maps are demonstrably NOT invariant to large input transforms (Lenc & Vedaldi 2015; Cohen & Welling 2015). Pooling is a fixed, local, hand-wired receptive field. We want a module that can ACTIVELY warp a feature map, conditioned on the input itself, trained end-to-end with backprop only, no extra supervision, droppable anywhere in a net.

## Load-bearing ancestors (lineage)
- **CNN + local max-pooling (LeCun 1998):** the baseline. Pooling = the only built-in spatial-invariance mechanism, but local & fixed. Gap: no scale/rotation/warp invariance except via depth; intermediate maps not invariant.
- **Data augmentation:** the de facto crutch — train on rotated/scaled/translated copies so the net learns invariance by brute force. Costs capacity + data + compute, never gives a guarantee, doesn't normalize pose at inference.
- **Transforming auto-encoders / capsules (Hinton 1981 canonical frames; Hinton 2011 transforming autoencoders; Tieleman 2014):** model objects as transformed parts, predict 2D affine transforms. But trained with transformation SUPERVISION (transforms given as input/target). Gap: need ground-truth transforms.
- **Invariance/equivariance analysis (Lenc & Vedaldi 2015; Cohen & Welling 2015):** the DIAGNOSTIC — measured that CNN reps are not invariant to large transforms. This is the motivating empirical fact.
- **Filter-bank / group approaches (Gens & Domingos 2014 deep symmetry nets; Sohn & Lee 2012; Kanazawa 2014 scale-invariant CNN; Bruna & Mallat 2013 scattering):** build invariance into the FEATURE EXTRACTOR (transform the filters / tie weights across the group). Gap: fixed set of transforms baked in, grows cost, manipulates extractor not data.
- **Attention/glimpse models (Schmidhuber & Huber 1991 fovea; Ba/Mnih 2015 DRAM; Sermanet 2014; Gregor 2015 DRAW; Xu 2015):** take crops/glimpses. Ba/Mnih, Sermanet use REINFORCE (non-differentiable crop). DRAW uses differentiable Gaussian-kernel attention but a generative read/write, axis-aligned. Girshick 2014 R-CNN region proposals = attention via external algorithm. Gap: RL training (high variance) or limited (axis-aligned Gaussian) attention.
- **Classic image warping / texture mapping (Foley 1994):** inverse warping — for each output pixel, look up source coord, interpolate. The source/target transform is exactly graphics texture-mapping. This is the technical seed for output->input mapping + bilinear interp.

## The method (derived)
Three parts, end-to-end differentiable, conditioned on input:
1. **Localisation network** θ = f_loc(U), U ∈ R^{HxWxC}. Any net (FC or conv) ending in a regression layer. For affine, θ is 6-D.
2. **Grid generator (parameterised sampling grid).** Output pixels on regular grid G={(xᵗ_i,yᵗ_i)}. Affine: (xˢ_i, yˢ_i)ᵀ = A_θ (xᵗ_i, yᵗ_i, 1)ᵀ, A_θ = [[θ11,θ12,θ13],[θ21,θ22,θ23]]. Normalized coords [-1,1]. OUTPUT→INPUT (every output pixel gets exactly one source point — surjective coverage of output, vs forward mapping which leaves holes/collisions). Attention special case A_θ=[[s,0,tx],[0,s,ty]]. General: T_θ = M_θ B, B learnable target grid (superset incl. projective, TPS, piecewise affine). Requirement: differentiable w.r.t. θ.
3. **Differentiable sampler.** Generic: V_i^c = Σ_n Σ_m U^c_{nm} k(xˢ_i−m; Φx) k(yˢ_i−n; Φy). Integer kernel = nearest neighbour (eqn integer), not differentiable in coords. Bilinear:
   V_i^c = Σ_n Σ_m U^c_{nm} max(0,1−|xˢ_i−m|) max(0,1−|yˢ_i−n|).
   Subgradients:
   ∂V_i^c/∂U^c_{nm} = max(0,1−|xˢ_i−m|) max(0,1−|yˢ_i−n|)   [the bilinear weight]
   ∂V_i^c/∂xˢ_i = Σ_n Σ_m U^c_{nm} max(0,1−|yˢ_i−n|) · {0 if |m−xˢ_i|≥1; +1 if |m−xˢ_i|<1 and m≥xˢ_i; −1 if |m−xˢ_i|<1 and m<xˢ_i}
   (deriv of 1−|xˢ−m| w.r.t xˢ is −sign(xˢ−m): +1 when m≥xˢ, −1 when m<xˢ; 0 outside support). Similarly ∂V/∂yˢ.
   Then ∂xˢ/∂θ, ∂yˢ/∂θ trivial from affine (e.g. ∂xˢ/∂θ11 = xᵗ). Sub-gradients because of |·| kinks. GPU-efficient: only sum over the 4-pixel kernel support, not all HW.

## Design decisions → why
- **Output→input (inverse) mapping, not forward:** forward (input pixel → output location) scatters: some output pixels get 0 sources (holes), some get many (collisions). Inverse mapping = exactly one source coord per output pixel → well-defined, complete output. Same reason graphics uses inverse texture mapping.
- **Normalized [-1,1] coords:** decouple transform from resolution; identity A_θ=[[1,0,0],[0,1,0]] regardless of H,W; same θ meaning across feature-map sizes.
- **Bilinear over nearest:** nearest (integer/Kronecker-delta kernel) has zero gradient w.r.t. xˢ almost everywhere → no signal to localisation net. Bilinear is piecewise-linear in xˢ → nonzero subgradient → trainable.
- **Sub-gradient at kinks:** |·| not differentiable at 0; pick a subgradient, fine for SGD.
- **Affine 6 params:** minimal that gives crop+translate+rotate+scale+skew; contraction (|det 2x2|<1) = crop/zoom.
- **Init regression layer to identity** (weights 0, bias = [1,0,0,0,1,0]): start as a no-op so the host net trains like a normal CNN, then deviates. Avoids destroying the input at init.
- **Lower LR for localisation net (1/10, or 1e-4 for Inception):** transform params are high-leverage / sensitive; large steps warp wildly and destabilise.
- **Same warp across channels:** preserves spatial alignment between channels (one geometric object).
- **Droppable anywhere / multiple / parallel:** self-contained, fast (~6% overhead); deeper ST acts on richer features; parallel ST = multiple objects/parts.
- **No extra supervision:** gradient from the task loss flows through sampler→grid→θ→f_loc. The "how to transform" is cached in f_loc weights.

## Reference code: PyTorch tutorial STN (code/spatial_transformer_tutorial.py)
- localization = small conv stack; fc_loc → 6 values → theta.view(-1,2,3)
- fc_loc last layer init to identity (weight.zero_(), bias=[1,0,0,0,1,0])
- grid = F.affine_grid(theta, x.size()); x = F.grid_sample(x, grid)
- affine_grid builds the normalized output grid and applies A_θ (output→input); grid_sample does bilinear sampling. Modern PyTorch defaults to align_corners=False, so result code should set the convention explicitly.
