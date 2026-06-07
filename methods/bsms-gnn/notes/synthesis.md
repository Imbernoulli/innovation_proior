# Synthesis — BSMS-GNN (Bi-Stride Multi-Scale GNN)

## Pain point / research question
Learning a one-step transition operator for mesh-based PDE simulation (predict state at t+Δt from state at t) on **large, irregular, complex-geometry** meshes. A flat GNN that stacks many message passings (MPs) to move information across space hits two walls:
1. **Complexity** — both #nodes and #MP-iterations grow with mesh size; the computational graph cost is effectively quadratic in scale, so training RAM/time explode.
2. **Over-smoothing** — graph convolution is a low-pass filter; stacking MPs repeatedly projects signals onto the low-frequency eigenspace and washes out high-frequency detail, making deep stacks hard to train.

Multi-scale (U-Net-like) GNNs fix both by building coarse levels: long-range interaction in few hops, fewer MPs. But **building the coarse levels is the unsolved part.** The desiderata for the pooling/coarsening operation:
- (a) conserve correct connectivity at coarse levels (no artificial partitions),
- (b) do not introduce edges that blur/cross geometry boundaries,
- (c) general for any mesh type (2D/3D, tri/tet, surface),
- (d) fully automatic (no human in the loop).

## The two contributions
1. **Bi-stride pooling** — a deterministic, topology-only pooling that is provably 2nd-order connection conservative (2-CC).
2. **A lightweight architecture enabled by it** — one MP per level + a non-parametric weighted transition (down/up-sampling) between levels.

---

## Background concepts (load-bearing ancestors)

### MeshGraphNets / MGN (Pfaff et al. 2020, "Learning Mesh-Based Simulation with Graph Networks")
- The backbone. **Encode-Process-Decode** on a graph built from the mesh. Encoder MLPs lift nodal/edge inputs to latent; the **Processor** runs K (5–20) rounds of message passing updating edge and node embeddings; decoder MLP reads out the per-node output (typically a time-derivative/acceleration), integrated to produce next state.
- MP: edge update e_ij ← f_E(e_ij, v_i, v_j); node update v_i ← f_V(v_i, Σ_j e_ij), all residual + LayerNorm MLPs.
- **World edges**: besides mesh edges, dynamic proximity edges for contact/collision.
- Trained on single-step L2 loss; **training noise** added to inputs so the model learns to correct its own rollout drift.
- *Limitation*: single-scale ("flat"). To propagate info across a large mesh you need ~diameter-many MPs → cost grows with size, and the deep stack over-smooths. This is exactly what BSMS reacts to.

### Graph U-Nets / gPool-gUnpool (Gao & Ji 2019)
- First to put a U-Net (encoder–pool–...–unpool–decoder) on graphs. **gPool**: learn a projection vector p, score each node y_i = x_i·p/‖p‖, keep top-k by score → smaller graph; gUnpool restores positions with zeros.
- **Connectivity problem**: top-k pooling can isolate nodes / split the graph. Their fix: replace A by **A^2** (2nd graph power) so kept nodes connect via 2 hops. They pick K=2 because higher powers over-connect.
- *Limitations BSMS reacts to*: (1) **no guarantee** top-k is 2-CC for an arbitrary graph — learnable scoring can still drop a whole neighborhood and create partitions even after A^2 (paper's Fig. 1a); (2) original gPool used **dense** A multiplications (designed for ~100-node graphs), not scalable; (3) learnable pooling generalizes poorly to **unseen geometries** — the score module pools unfairly on a mesh it wasn't trained on.

### MS-GNN by spatial proximity (Lino et al. 2021/2022; Liu et al. 2021)
- Build coarse levels by placing coarser nodes (rasterized grid / spatial clustering) and connecting by **Euclidean proximity / radius**. Then learnable down/up transitions between levels, multiple MPs per level.
- *Limitation*: spatial proximity introduces **wrong edges across geometry boundaries** — two nodes close in space but far along the surface/geodesic (e.g. two banks of a thin U-channel, two sides of a contact gap) get linked, corrupting the physics. Paper's diagnostic: a 1-D head-to-tail heat-stick test where a spurious cross-gap edge produces wrong inference. Also needs grid resolution / inflation-rate hyperparameters per case.

### Manual coarse meshes (Liu 2021; Fortunato 2022 "MultiScale MeshGraphNets")
- Human draws coarse meshes in CAE software; two/multi-level MGN. Accurate but **labor-intensive** — ~20k meshes for the paper's datasets. Not automatic.

### Guillard coarsening (Lino 2022 multi) / Multipole (Li et al. 2020)
- Guillard node-nesting: only 2D triangle meshes. Multipole: random pooling + multi-level matrix factorization for kernels — random pooling again risks partitions; matrix factorization is heavy.

### Bipartite / DAG bi-partition (Asratian 1998) — the conceptual seed
- In a DAG after topological sort, take **every other depth layer** → a bipartition where every edge crosses the two parts; keeping one part is trivially 2-CC (any dropped node's neighbors are all in the kept part = 1 hop, so any two kept nodes are ≤2 hops). This is the idea bi-stride ports to a general (cyclic) mesh via BFS depth parity.

---

## Key definitions and the central theoretical object

**Adjacency enhancement** A ← A^K: A(i,j)=1 iff edge; A^K(i,j)=1 (booleanized) iff j reachable from i within K hops.

**K-th-order outlier set** O_K under pooling P with kept indices 𝓘: nodes j s.t. A^K(i,j)=0 for all kept i — i.e. j is not within K hops of *any* kept node. P is **K-CC** (K-th-order connection conservative) iff O_K is empty.

**Why K=2 is the target.** Large K is harmful: as K grows, booleanized A^K → all-ones (fully connected), and a single convolution then averages all node features → indistinguishable (worst over-smoothing). So we want the *smallest* K that still conserves connectivity. K=1 is generally impossible after dropping nodes (a dropped node's neighbor might be dropped too). K=2 is the natural minimum — hence "use A^2."

**Bi-stride = the construction that is 2-CC by design.** Run BFS from a seed, get geodesic (hop) distance to every node, keep all nodes at even depth (or all odd — pick the smaller set). Because BFS layers a graph by hop distance, every edge connects nodes whose depths differ by at most 1; so any dropped node (odd) has all neighbors at even depth = kept, i.e. is 1 hop from a kept node; two kept nodes are then ≤2 hops apart through a dropped node. ⇒ O_2 = ∅, 2-CC. Also pooled and unpooled nodes keep **direct** connections (each dropped node touches a kept node), so a single MP suffices to exchange info between the two sets before changing level.

**Coarse-graph rule:** A'_{l+1} ← A_l A_l (= A_l^2), then stride rows/cols to kept set: A_{l+1} ← A'_{l+1}[𝓘,𝓘]. (Code sets diagonal to 1 before squaring so A^2 includes 1-hop neighbors too, then clears diagonal after.)

**Contact edges** (world edges for collision/self-contact): separate matrix A^C built by spatial proximity *only* for contacts (legit there). Enhancement:
A'^C_{l+1} ← A_l A^C_l A_l, then stride to [𝓘,𝓘]. Geometric meaning: a coarse contact edge (i,j) exists if j reachable from i in 2 hops with at least one of the two hops a contact edge.

**Appendix proof that bi-stride + this rule conserves all contact edges** (boolean, undirected): for any fine contact edge (i,j), there exist kept i',j' with A_l[i,i']=A_l[j,j']=1 and A'^C_{l+1}[i',j']=1. Four cases by whether i,j are pooled:
1. both pooled → i'=i, j'=j, done.
2. only i pooled → j (unpooled, not seed) has ≥1 pooled neighbor j' (bi-stride guarantees each dropped node touches a kept node). Then A^C A [i,j'] ≥ A^C[i,j]·A[j,j']=1, and A(A^C A)[i,j'] ≥ A[i,i]·(A^C A)[i,j']=1 (diagonal=1). i'=i.
3. only j pooled → symmetric, pick pooled i' neighbor of i.
4. neither pooled → pick pooled neighbors i' of i and j' of j; chain both multiplications.

(Verified each inequality: products of boolean entries, monotone, lower-bounded by the specific path = 1. Correct.)

**Seeding heuristics** (one BFS seed per connected cluster):
- **MinAve** — seed = node with minimum average geodesic distance to its cluster (run BFS from every node → O(N²)); used for Cylinder/Airfoil/Plate.
- **CloseCenter** — seed = node nearest the spatial centroid of the cluster; O(N) linear; used for Font (~47k nodes, MinAve too slow). The released code defaults to CloseCenter (`nearest_center_seed`).
- Clusters found by repeated BFS (connected components) so a seed search isn't polluted by other components.
- Sensitivity: model is robust to seeding choice; inconsistent train/test seeding raises rollout RMSE only ~1–2× and stays in the same magnitude.

---

## Architecture (BSMS-GNN)
- Encode-Process-Decode (à la MGN) but **encoder/decoder only at the finest level G_1**. Latent dim 128; MLPs are ReLU 2-hidden-layer, residual, LayerNorm on all outputs except decoder.
- **Edge offset not separately encoded**: Δx_ij = x_i − x_j and ‖Δx_ij‖ are simply prepended to the stacked sender/receiver latents as the edge-MLP input (cheaper than MGN's persistent edge encoder + carried edge latent).
- MP per level (Eq. MP): e^s_{l,ij} ← f^s_l(Δx_{l,ij}, v_{l,i}, v_{l,j}) for each of S edge sets; v'_{l,i} ← v_{l,i} + f^V_l(v_{l,i}, Σ_j e^1, …, Σ_j e^S). **One MP per level** (vs MGN's 15, vs Lino's 2–4).
- U-Net shape: down (MP→transition-down→pool) per level, bottom MP, up (unpool→transition-up→MP→add skip from the down-pass output at that level).

## Transition (non-parametric, weighted) — design decisions
Goal: move latent info between adjacent levels **without** learnable modules (those cost ~70% more time/RAM for marginal accuracy).
- **Dead end 1 — no transition** (just pool/unpool, zeros at unpooled nodes, inherited from gUnpool style): low 1-step RMSE but rollouts grow **stripe/mosaic** patterns aligned with coarse edges — unpooled nodes start at zero, are indistinguishable to the processor, and the pooled/unpooled gap is exaggerated over the rollout.
- **Dead end 2 — single unweighted graph convolution** (resembling regular-grid interpolation): **over-smooths**, smears the fine info near the generator (cylinder) and stops propagating downstream. Cause: ignores mesh irregularity — unlike CNN grids, fine nodes don't sit at cell centers and element sizes vary (smaller near interfaces).
- **Resolution — weighted aggregation + symmetric return** (U-Net interpolation analog, accounting for irregularity via nodal weights):
  - Maintain a nodal weight w (init 1 at finest, aggregated down).
  - Row-normalize A: Â_ij = A_ij/Σ_j A_ij; convolve weight ŵ_ij = w_i Â_ij.
  - Contribution table C_ij = ŵ_ij / Σ_i ŵ_ij (share of receiver j's weight from sender i).
  - **Down**: v_j ← Σ_i v_i C_ij (split weighted info from senders, weighted-average at receivers).
  - **Up**: reuse the *same* C: v_i ← Σ_j v_j C^T_ij (transposed-conv analog), distinguishing receivers on return.
  - Alternative Pos-Kernel (inverse-distance kernel, Liu 2021) is competitive but slightly worse; Learnable (Fortunato) slightly more accurate but ~70% heavier.

## Design-decision → why table
| Decision | Why / what breaks otherwise | Rejected alternative |
|---|---|---|
| Multi-scale U-Net over flat GNN | flat needs ~diameter MPs → quadratic cost + over-smoothing | stack more MPs (MGN) |
| Pool by BFS-depth parity (bi-stride) | provably 2-CC, topology-only, any mesh, automatic | learnable top-k (no 2-CC guarantee, bad generalization); random (partitions) |
| Even-or-odd, keep smaller set | balanced pooling ratio ≈½, both are 2-CC | always keep even (could be the larger/smaller arbitrarily) |
| A ← A^2 enhancement (K=2) | smallest K that conserves connectivity after dropping; higher K → all-ones → over-smooth; K=1 can't conserve | K=1 (loses connectivity); K≥3 (over-connect/smooth) |
| One MP per level | bi-stride keeps direct pooled↔unpooled edges, so one MP exchanges info before level change | multiple MPs per level (Lino 2–4, MGN 15) — wasted compute |
| Encoder/decoder only at top level | latent already carries info across levels; per-level enc/dec is redundant | per-level encode/decode |
| Prepend Δx,‖Δx‖ to latents (no separate edge encoder/carried edge latent) | cheaper; offsets give translation-invariant geometry | MGN persistent edge latent |
| Non-parametric weighted transition | learnable transition ~70% more time/RAM for ~marginal gain | learnable MP transition; no transition (mosaic); unweighted graph-conv (over-smooth) |
| Weight-based contribution table C, reuse C^T on the way up | accounts for irregular element sizes; symmetric down/up like U-Net interp/transposed-conv | inverse-distance Pos-Kernel (slightly worse) |
| Contact A^C enhanced as A A^C A | preserves contacts across pooling (proved 4-case); contacts are the only legit spatial-proximity edges | ignore contacts at coarse levels (lose collisions) |
| Per-cluster seeding; MinAve or CloseCenter | balanced BFS start; CloseCenter O(N) for huge meshes vs MinAve O(N²) | single global seed (cross-cluster pollution) |
| Single-step L2 + training noise | learn to correct rollout drift (inherited from MGN) | multi-step BPTT (expensive) |

## Evaluation settings (for context.md — settings only, no proposed-method outcomes)
- Datasets: Cylinder (2D incompressible flow), Airfoil (2D compressible), Plate (3D hyperelastic deforming, tet, contact), Font (new: 3D inflating elastic surfaces with self-contact, tri surface, up to ~47k nodes). 1000/200/200 train/val/test.
- Backbone hyperparams: latent 128, 2-hidden-layer ReLU MLPs, residual, LayerNorm. Levels 6–9 by case. Single RTX 3090. PyTorch + PyG.
- Metrics: RMSE-1 / RMSE-50 / RMSE-all (rollout horizons); training time/step, infer time/step, training cost (hrs/epochs), training & inference RAM. (Baselines: MGN, Lino MS-GNN, GraphUNet.) These are yardsticks — no proposed-method numbers in context.

## Canonical code map (grounds final code)
- `graph_wrappers/graph_wrapper.py`: Graph (flat_edge↔adj_list↔sparse adj_mat), `bfs_dist`, `find_clusters`.
- `graph_wrappers/bsms_graph_wrapper.py`: `BistrideMultiLayerGraph` — `bstride_selection` (BFS parity pooling + per-cluster seed + A^2 enhancement + diagonal handling + `pool_edge` reindex), `nearest_center_seed` (CloseCenter).
- `ops/basic.py`: `MLP`, `GMP` (one MP block, prepend dir+norm), `WeightedEdgeConv` (cal_ew → C; forward aggregating True/False = down/up), `Unpool`.
- `ops/BSMS.py`: `BSGMP` — the U-Net loop (down: GMP, cal_ew, weighted edge conv, pool; bottom GMP; up: unpool, weighted edge conv return, GMP, add skip).
- `models/model.py`: `BSMS_Simulator` — encode/process(BSGMP)/decode + Normalizer + delta integration.
</content>
