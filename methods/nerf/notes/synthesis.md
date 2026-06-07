# NeRF synthesis notes (pre-Phase-2)

## Pain point / research question
Novel view synthesis: given a sparse set of RGB images with known camera poses, render
photorealistic images from *new* viewpoints. Long-standing problem. At the time, two
dominant families both hit a wall:
- mesh/surface-based + differentiable rasterizers/pathtracers: gradient-based mesh
  optimization is brittle (local minima, poor conditioning), needs a template mesh of
  fixed topology — unavailable for real scenes.
- discrete volumetric (voxel grids, MPIs, CNN-predicted RGBA volumes): great quality, but
  storage and compute scale as O(resolution^3); high-res imagery is fundamentally
  bottlenecked by the discrete sampling of 3D space. LLFF stores >15GB for one scene.
- neural implicit shape (DeepSDF, occupancy nets): continuous, compact, but need ground
  truth 3D geometry; the ones relaxed to 2D supervision (DVR, SRN) collapse the scene to a
  single opaque surface per ray → oversmoothed, low geometric complexity.

Goal a solution must hit: continuous (no resolution cap), compact (fits in MLP weights),
optimizable from 2D images alone (differentiable rendering), and able to express both
high-frequency geometry/texture AND view-dependent appearance (specularities).

## Core objects to derive
1. Scene as a 5D function F:(x,d)->(c,sigma). MLP. density from x only, color from (x,d).
2. Classical volume rendering integral (Kajiya-Von Herzen 1984; Max 1995 review):
   C(r)=∫ T(t) σ(r(t)) c(r(t),d) dt, T(t)=exp(-∫_{tn}^t σ ds).
   Derivation: absorption+emission optical model. Over [t, t+dt], probability ray is
   occluded by a particle = σ(t)dt. Light reaching the eye from that slab = (radiance it
   emits c) × (prob a particle is there σ dt) × (prob nothing between tn and t blocks it,
   = T(t)). Transmittance ODE: dT/dt = -σ(t) T(t) → T(t)=exp(-∫σ). σ = differential
   probability of ray terminating at a particle.
3. Quadrature (Max 1995 eq). Assume σ, c piecewise constant on [t_i, t_{i+1}], width δ_i.
   - On a bin where σ is constant: T within the bin decays as exp(-σ_i s). Contribution of
     bin i = ∫_0^{δ_i} T_i exp(-σ_i s) σ_i c_i ds = T_i c_i (1 - exp(-σ_i δ_i)).
     [since ∫_0^δ σ exp(-σ s) ds = 1 - exp(-σ δ).]
   - T_i = transmittance up to start of bin i = exp(-Σ_{j<i} σ_j δ_j).
   - So Ĉ = Σ_i T_i (1 - exp(-σ_i δ_i)) c_i. Define α_i = 1-exp(-σ_i δ_i) → alpha
     compositing, and T_i = Π_{j<i} (1-α_j). This is the bridge: continuous integral →
     discrete alpha-compositing, exactly the operation discrete-volume methods already use.
   - weights w_i = T_i α_i. Σ w_i = accumulated opacity (acc_map).
4. Stratified sampling: deterministic quadrature would query MLP at fixed locations →
   effectively discretizes the representation. Instead partition [tn,tf] into N bins, draw
   t_i ~ U[bin i]. Over training the MLP is queried at continuous positions → continuous
   representation. δ_i = t_{i+1}-t_i.
5. Positional encoding γ(p) = (sin(2^0 π p), cos(2^0 π p), ..., sin(2^{L-1} π p),
   cos(2^{L-1} π p)). Applied per-coordinate. L=10 for x, L=4 for d. Motivated by spectral
   bias (Rahaman 2018): ReLU MLPs are biased to low-frequency functions; raw (x,d)->(c,σ)
   gives oversmoothed renders. Lifting input to a high-frequency basis lets the downstream
   MLP fit high-frequency variation. Why this basis: it's a Fourier feature map; high
   frequencies up to 2^{L-1} let two nearby points map to distant points in feature space,
   so the (smooth in feature space) MLP can still produce sharp variation in input space.
   Ablation: L=5 too few, L=15 no better than 10 (benefit saturates once 2^L exceeds the
   max image frequency ~1024 = 2^10).
6. View-direction split. σ = σ(x) only (geometry must be view-independent → multiview
   consistency, density can't change with camera). c = c(x,d) (specular/non-Lambertian).
   Architecture: 8 layers, 256 wide, ReLU, skip connection concatenating γ(x) at layer 5
   (DeepSDF trick); outputs σ + 256-feature; concat γ(d), 1 layer 128 wide → RGB (sigmoid).
   σ rectified by ReLU (nonneg). Ablation: no-VD can't do specularities.
7. Hierarchical sampling. Dense uniform sampling wastes queries on empty/occluded space.
   Two networks: coarse + fine. Coarse: N_c=64 stratified samples → weights w_i. Rewrite
   Ĉ_c = Σ w_i c_i. Normalize ŵ_i = w_i / Σ w_j → piecewise-constant PDF along ray. Sample
   N_f=128 more via inverse-transform sampling of this PDF (biased to high-density regions).
   Evaluate fine net at union (N_c+N_f) samples → Ĉ_f. Unlike importance sampling (treats
   each sample as independent estimate of whole integral), here samples are a nonuniform
   discretization of the whole domain.
8. Loss: L = Σ_r [||Ĉ_c - C||^2 + ||Ĉ_f - C||^2]. Both coarse and fine in loss so coarse
   weights stay meaningful for sampling. Adam lr 5e-4 → 5e-5 exp decay, batch 4096 rays.
9. NDC for real forward-facing scenes (appendix). Real scenes go to infinity → unbounded.
   Map camera-space rays to NDC cube via perspective projection matrix; z becomes linear in
   disparity (inverse depth), so far=∞ maps to a finite bound. Derivation: project ray
   o+td with matrix M, require π(o+td)=o'+t'd'; fix t=0↔t'=0; get o'=π(o),
   t'=1 - o_z/(o_z+t d_z) (→1 as t→∞), and d' components. With far=∞: a_z=1, b_z=2n. Shift
   o to near plane first (t_n = -(n+o_z)/d_z) so sampling t' linearly in [0,1] = linear in
   disparity. For real scenes also add N(0,1) noise to σ pre-ReLU as regularizer.

## Design-decision → why table
- density from x only: enforce multiview consistency; geometry can't depend on camera.
- color from (x,d): non-Lambertian/specular appearance.
- alpha = 1-exp(-σδ): falls out of integrating exp-decaying transmittance over a constant-σ
  bin; reduces to standard alpha compositing → reuses graphics machinery, differentiable.
- stratified (not deterministic) sampling: keeps representation continuous, avoids fixed
  query grid.
- positional encoding: overcome spectral bias / low-frequency bias of MLP.
- L=10 (x), L=4 (d): enough frequencies to hit image detail; saturates past 2^L≈max image
  freq; d needs fewer (appearance varies more smoothly in direction).
- skip connection at layer 5: DeepSDF; reinject input coords so deep layers keep access.
- sigmoid on RGB (bounded [0,1]), ReLU on σ (nonneg density).
- hierarchical coarse+fine: allocate samples to occupied space, sample-efficient.
- both colors in loss: keep coarse weight distribution useful for fine sampling.
- NDC: bound unbounded real scenes; uses disparity so ∞ is finite.

## Canonical code (yenchenlin/nerf-pytorch, mirrors bmild/nerf)
- run_nerf_helpers.py: Embedder/get_embedder (PE), NeRF MLP, sample_pdf (inverse CDF),
  ndc_rays, get_rays.
- run_nerf.py: run_network (embed pts+dirs, batchify), raw2outputs (alpha, cumprod
  transmittance via cumprod(1-alpha) exclusive, weights, rgb_map), render_rays (stratified
  z_vals, coarse pass, sample_pdf for fine, union+sort, fine pass), loss img2mse on rgb +
  rgb0.
- Key code facts: dists scaled by ||rays_d|| (so δ is true metric length). transmittance =
  cumprod(cat([1, 1-alpha]))[:-1] = exclusive cumprod = T_i. weights = alpha*T.
  white_bkgd: rgb += (1-acc). sample_pdf weights use weights[...,1:-1] (drop end bins).
