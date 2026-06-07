# PointNet++ — synthesis (arXiv 1706.02413, verified — Qi, Yi, Su, Guibas 2017)

## Pain point / goal (builds directly on PointNet)
PointNet: f(x_1..x_n) = gamma(MAX_i h(x_i)), permutation invariant, universal continuous-set-function approximator. BUT a SINGLE max-pool over the whole set means there is NO local structure at intermediate scales — every point processed independently then concatenated with one global feature. CNNs win precisely by capturing local patterns at increasing scales along a multi-resolution hierarchy (small receptive fields low, large high), which gives generalizability. PointNet can't recognize fine-grained patterns or generalize to complex scenes.
Goal: hierarchical net that applies PointNet recursively on a NESTED PARTITIONING of the point set, learning local features at increasing contextual scales. Plus handle NON-UNIFORM SAMPLING DENSITY (real scans vary density via perspective, radial, motion).

## Problem setup
X = (M, d): discrete metric space, metric inherited from Euclidean R^n; M ⊆ R^n. Density may be non-uniform. Learn set functions f (classification: label for X; segmentation: per-point label).

## Two issues to address
1. How to PARTITION the point set into overlapping local regions.
2. How to ABSTRACT a set of points/local features via a local feature learner.
Correlated: partitioning must produce COMMON STRUCTURES across partitions so the local feature learner's weights can be SHARED (like conv). Local learner = PointNet (effective, robust to corruption).

## Set Abstraction (SA) level = Sampling + Grouping + PointNet
Input N x (d+C); output N' x (d+C'). N' subsampled points with new local-context features.
1. **Sampling layer.** Iterative FARTHEST POINT SAMPLING (FPS): choose x_{i_j} = most distant (metric) from the already-chosen set. Better coverage than random for same #centroids; DATA-DEPENDENT receptive fields (unlike CNN's fixed strides scanning agnostic of data). First point random.
2. **Grouping layer.** Input N x (d+C) + N' centroids (N' x d). Output N' x K x (d+C). Use BALL QUERY: all points within radius r of centroid (upper limit K). Alternative kNN: fixed count. Ball query preferred because it GUARANTEES a fixed region SCALE -> local feature more generalizable across space (good for local pattern recognition / semantic labeling). K varies across groups but PointNet handles variable count.
3. **PointNet layer.** Input N' x K x (d+C). First TRANSLATE coords to LOCAL FRAME relative to centroid: x_i^(j) := x_i^(j) - xhat^(j) (j=1..d). Then a mini-PointNet (shared MLP per point in region, then max-pool over the K) -> N' x (d+C'). Relative coords + point features capture point-to-point relations in the local region.
Analogy to CNN: higher layer fewer points + larger receptive field; weights shared across space. Differences: neighborhood by METRIC distance (not array index), kernel is a SET function (not convolution).

## Non-uniform density: density-adaptive layers (the "++")
Problem: features learned on dense data don't generalize to sparse regions; small neighborhood may have too few points (sampling deficiency) to recognize patterns robustly. (Counter to CNN wisdom that smaller kernels help — small here = too few points.) Want: inspect closely in dense regions (finest detail), look at larger scale in sparse regions.
- **Multi-Scale Grouping (MSG).** At one SA level, run grouping at MULTIPLE radii (scales), each with its own PointNet; CONCATENATE features across scales. Network learns to weight scales via RANDOM INPUT DROPOUT during training: per training set pick dropout ratio theta ~ Uniform[0,p] (p=0.95), drop each point with prob theta -> sets of varying sparsity + uniformity. At test keep all points. Expensive: runs local PointNet at large-scale neighborhoods for every centroid, costly at low levels (many centroids).
- **Multi-Resolution Grouping (MRG).** Feature of a region at level L_i = concat of TWO vectors: (left) summary of features from each subregion at lower level L_{i-1} via the SA level; (right) feature from directly processing all RAW points in the region with a single PointNet. When density LOW: left vector less reliable (subregions even sparser) -> weight right higher. When density HIGH: left gives finer detail (recursive higher-resolution inspection). Avoids expensive large-scale feature extraction at lowest levels -> more efficient than MSG.

## Feature Propagation (FP) for segmentation
SA subsamples; segmentation needs features for ALL original points. Hierarchical propagation with distance-based INTERPOLATION + skip links. Propagate from N_l points to N_{l-1} (N_l <= N_{l-1}): interpolate feature values at N_{l-1} coords using INVERSE-DISTANCE-WEIGHTED average over k nearest neighbors (default p=2, k=3):
  f^(j)(x) = sum_i w_i(x) f_i^(j) / sum_i w_i(x),  w_i(x) = 1/d(x,x_i)^p.
Concatenate interpolated features with skip-linked features from the SA level at that resolution; pass through a "unit PointNet" (shared FC + ReLU per point, like 1x1 conv). Repeat up to original points.

## Architecture (SSG classification, verified from code)
SA1: npoint=512, radius=0.2, nsample=32, mlp=[64,64,128]
SA2: npoint=128, radius=0.4, nsample=64, mlp=[128,128,256]
SA3: group_all, mlp=[256,512,1024]  -> global feature 1024
FC: 512 (dropout 0.5), 256 (dropout 0.5), 40. BN+ReLU. Softmax cross-entropy loss.
Segmentation: SA layers down, then FP layers up with skip links, unit pointnet, per-point scores.

## Design decisions -> why
- PointNet as local learner: proven universal set approximator, robust to corruption, handles variable point count -> reusable shared-weight block.
- FPS sampling: data-dependent coverage, better than random/grid for same #centroids.
- Ball query (not kNN): fixed metric scale -> generalizable local features.
- Relative (centroid-subtracted) coordinates in PointNet layer: translation invariance + captures point-to-point relations in local frame.
- MSG/MRG + random input dropout: adaptively combine multi-scale features; robust to non-uniform density.
- IDW interpolation (p=2, k=3) + skip links + unit pointnet for FP: recover per-point features for segmentation efficiently.

## Lineage / baselines
- PointNet (Qi 2016): the building block + the thing being extended (single global max-pool, no hierarchy).
- Deep Sets / generic set networks: treat each point independent + global normalization -> only single-point embedding or global feature; PointNet++ makes the metric a first-class citizen.
- CNNs (VGG, Simonyan & Zisserman 2014 — smaller kernels help): multi-resolution hierarchy of local receptive fields; PointNet++ is the point-cloud analogue but with metric neighborhoods + set-function kernels; counter-evidence on "smaller is better" for point density.
- Volumetric CNNs: fixed-stride scanning agnostic of data distribution; FPS is data-dependent.
- FPS algorithm (Eldar et al.); ball query / kNN range search.

## Code grounding (charlesq34/pointnet2, verified)
- pointnet_util.py: sample_and_group (FPS -> gather centroids; query_ball_point; group_point; subtract centroid xyz for translation normalization; concat grouped_xyz + grouped_points). pointnet_sa_module: shared conv2d [1,1] MLP over (npoint,nsample) then reduce_max over nsample axis. pointnet_sa_module_msg: loop over radius_list/nsample_list/mlp_list, max over each, concat. pointnet_fp_module: three_nn + three_interpolate (IDW), concat skip, unit pointnet.
- pointnet2_cls_ssg.py: the three SA layers above + FC.
