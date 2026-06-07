# PointNet — synthesis (arXiv 1612.00593, verified — Qi, Su, Mo, Guibas 2016)

## Pain point / goal
Deep nets on 3D geometry. Convolutions need REGULAR formats (image grids, voxel grids) for weight sharing. Point clouds/meshes are irregular -> researchers voxelize or render to multi-view images first. That transformation: (1) makes data unnecessarily voluminous (voxels are mostly empty, cubic blowup), (2) introduces quantization artifacts that obscure natural invariances. Goal: a deep net that DIRECTLY consumes raw point clouds (set of (x,y,z), optionally +features), unified for classification (k class scores) and segmentation (n x m per-point scores).

## Three properties of point sets (drive the three design ideas)
1. **Unordered.** A set of N points -> network must be invariant to N! permutations of feeding order.
2. **Interaction among points.** Points live in a metric space; neighbors form meaningful local structures; model should capture local structure + their interactions.
3. **Invariance under transformations.** Rigid/affine transform of all points together shouldn't change the category or per-point labels.

## Permutation invariance: three strategies, two fail
1. **Sort into canonical order.** FAILS: in high-dim there is no ordering stable w.r.t. point perturbations. Proof by contradiction: such an ordering would be a bijection R^d -> R^1 that preserves spatial proximity as dimension reduces — impossible in general. Empirically, MLP on sorted points performs poorly (slightly > unsorted).
2. **RNN over the sequence + permutation-augmented training.** FAILS to scale: "Order Matters" (Vinyals 2015) shows order can't be fully omitted; RNN robust to ordering only for short sequences (dozens), not thousands of points (typical point-cloud size). Empirically worse.
3. **Symmetric function aggregation.** WIN. A symmetric function takes n vectors -> one vector invariant to order (e.g. + and * are symmetric binary ops).

## Core construction
Approximate a general set function by a symmetric function on transformed elements:
  f({x_1,...,x_n}) ~= g(h(x_1),...,h(x_n))
where f: 2^{R^N} -> R, h: R^N -> R^K (shared per-point), g: (R^K)^n -> R symmetric.
Empirically: h = a shared MLP (same weights on every point, applied independently), g = composition of a single-variable function (gamma, the post-MLP) and MAX POOLING (element-wise max over points). Collection of K such h's -> learn many f's capturing different set properties.

## Universal approximation theorem (Theorem 1) + PROOF (worked in reasoning)
X = {S: S ⊆ [0,1]^m, |S|=n}; f continuous set function w.r.t. Hausdorff distance d_H. Then for all eps>0, exists continuous h and symmetric g = gamma ∘ MAX such that |f(S) - gamma(MAX_{x in S} h(x))| < eps.
Proof (1-D version, generalizes per-dimension):
- Continuity: pick delta_eps s.t. d_H(S,S')<delta_eps => |f(S)-f(S')|<eps.
- K = ceil(1/delta_eps); sigma(x)=floor(Kx)/K maps point to left end of its interval. S~ = {sigma(x): x in S}. d_H(S,S~) < 1/K <= delta_eps => |f(S)-f(S~)| < eps.
- Soft indicator h_k(x) = exp(-d(x, [(k-1)/K, k/K])); h(x)=[h_1;...;h_K] in R^K.
- v_j = max over points of h_j(x_i): occupancy of j-th interval. v = MAX(h(x_1),...,h(x_n)) symmetric, in {0,1}^K.
- tau(v) = {(k-1)/K : v_k>=1} maps occupancy back to a set; tau(v(x_1..x_n)) ≡ S~.
- gamma: R^K -> R continuous with gamma(v) = f(tau(v)) on {0,1}^K. Then |gamma(v(x_1..x_n)) - f(S)| = |f(S~)-f(S)| < eps. And gamma(v) = (gamma ∘ MAX)(h(x_1),...,h(x_n)). QED.
Key intuition: worst case the net learns a VOXEL/occupancy representation by partitioning space into K cells; in practice it learns a smarter probing. K = max-pool dimension = "bottleneck dimension".

## Stability/critical-points theorem (Theorem 2) + PROOF
u(S) = MAX_{x in S} h(x) in R^K; f = gamma ∘ u. Then:
(a) For all S, exist C_S, N_S ⊆ X with f(T)=f(S) for any C_S ⊆ T ⊆ N_S.
(b) |C_S| <= K.
Proof: f determined by u(S). For each output dim j, exists x_j in S with h_j(x_j) = u_j(S) (the argmax). C_S = union of these x_j over j=1..K -> determines u, hence f; |C_S| <= K. Adding any point x with h(x) <= u(S) elementwise doesn't change the elementwise max u, hence f; N_S = C_S union all such points (the "upper-bound shape").
Implications: C_S = "critical point set" — f totally determined by <= K points (the object skeleton). Robust to perturbation (h continuous), to deletion of non-critical points, and to insertion of noise points up to N_S. Analogy to sparsity. Empirically: dropping half the input points -> only 3.7% accuracy drop (vs VoxNet ~40%).

## Local + global aggregation (for segmentation)
Global feature [f_1..f_K] (max-pool output) is a global shape signature — enough for classification (SVM/MLP on it). Segmentation needs LOCAL + GLOBAL: concatenate the global feature back onto each per-point feature, then a second per-point MLP -> per-point scores aware of both local geometry and global semantics. Validated: can predict per-point normals (a purely local quantity) — shows local neighborhood info is captured.

## Joint alignment networks (T-Net) — transformation invariance
Align input to a canonical space before feature extraction. Unlike image spatial transformers (Jaderberg 2015, which need sampling/interpolation -> aliasing), point clouds let us just predict an affine matrix and matrix-multiply the coordinates — no new layer, no alias.
- **Input T-Net:** mini-PointNet (shared MLP(64,128,1024) + max pool + FC(512,256)) regresses a 3x3 matrix, initialized to identity; multiply input points by it.
- **Feature T-Net:** same architecture, outputs 64x64 matrix (init identity), applied to the 64-dim point features.
- Feature transform matrix is high-dim -> hard to optimize. Add ORTHOGONALITY regularizer: L_reg = ||I - A A^T||_F^2 (weight 0.001). Orthogonal transform loses no information; stabilizes optimization, improves performance.

## Architecture (classification, verified from code)
input Bx N x3 -> input T-Net (3x3), matmul -> shared MLP(64,64) [conv [1,3] then [1,1]] -> feature T-Net (64x64), matmul -> shared MLP(64,128,1024) -> MAX POOL over N points -> global 1024 -> FC(512, dropout 0.7), FC(256, dropout 0.7), FC(40). BN + ReLU on all but last. 
Segmentation: concat global(1024) to each per-point 64-feat -> shared MLP -> per-point m scores. (Part seg adds one-hot category, skip links.)
Loss: softmax cross-entropy + 0.001 * ||I - A A^T||_F^2 (feature transform orthogonality).

## Hyperparameters (verified)
Adam, lr 0.001, momentum 0.9, batch 32, lr halved every 20 epochs. BN decay 0.5 -> 0.99. Dropout keep 0.7 on last FC (256) in classification. Reg weight 0.001. 1024 input points typical (performance saturates ~1K). K (bottleneck) 64->1024 gives 2-4% gain.

## Lineage / baselines
- Voxel CNNs: 3DShapeNets (Wu 2015), VoxNet (Maturana & Scherer 2015) — voxelize to 32^3 occupancy, 3D conv. Cubic memory/compute blowup, quantization, sparse.
- Multi-view CNNs: MVCNN (Su 2015), Qi 2016 — render to images, 2D CNN, pool views. Loses 3D structure, needs rendering.
- Spatial Transformer Networks (Jaderberg 2015) — learnable spatial transform for images via sampling. T-Net is the point-cloud analogue (simpler, no interpolation).
- Order Matters (Vinyals 2015) — sets-to-sequence; order matters for RNNs.
- VoxNet baseline numbers used for robustness comparison.

## Code grounding (charlesq34/pointnet TF, verified)
- transform_nets.py: input_transform_net (conv 64,128,1024; maxpool; fc 512,256; weights init 0, bias init identity flatten -> 3x3). feature_transform_net same -> 64x64.
- pointnet_cls.py: input T-Net -> matmul -> conv 64,64 -> feature T-Net -> matmul -> conv 64,128,1024 -> maxpool -> fc 512 dp0.7, 256 dp0.7, 40. get_loss = sparse_softmax_xent + reg_weight*l2_loss(A A^T - I).
- Conv [1,3]/[1,1] = shared per-point MLP; I'll express as Conv1d in PyTorch final code (idiomatic, as in fxia22/pointnet.pytorch).
