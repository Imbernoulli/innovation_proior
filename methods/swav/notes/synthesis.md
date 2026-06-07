# SwAV synthesis notes

## Pain point / research question
Learn visual features without labels, online, scalable to unlimited data. Contrastive instance-discrimination methods (SimCLR/MoCo/NPID) treat every image as its own class and need to *directly compare* many pairs of features: positives (views of same image) pulled together, many negatives pushed apart. Computing all pairwise comparisons is O(N^2); in practice they approximate by drawing negatives from large batches (SimCLR, 4096) or a memory bank/momentum encoder (NPID, MoCo, ~65k stored). All this machinery exists only to *supply negatives*. Question: can we get the same "consistency between views" signal WITHOUT explicit pairwise feature comparison, and WITHOUT collapse, online?

## Load-bearing ancestors (with the gap each leaves)
- **Instance discrimination (Dosovitskiy 2016; Wu et al. 2018 NPID).** Each image = own class. Dosovitskiy: explicit N-way classifier, intractable as N grows. Wu: replace classifier with a memory bank of past features + NCE. Gap: must store features for whole dataset; needs many negatives.
- **MoCo (He et al. 2019).** Momentum encoder + queue of negatives (~65k). Gap: extra encoder, large queue, still pairwise.
- **SimCLR (Chen et al. 2020).** Negatives = other elements of a large batch; NT-Xent loss with temperature τ; strong augmentations; nonlinear projection head (MLP). Gap: needs very large batches for enough negatives.
- **Contrastive loss / InfoNCE roots: Hadsell 2006; Oord 2018; Gutmann 2010 (NCE).** All pairwise.
- **DeepCluster (Caron et al. 2018).** k-means on all features each epoch → pseudo-labels → train classifier to predict them. Avoids explicit negatives. Gaps: OFFLINE (full pass per epoch to recompute assignments); needs re-assignment tricks + balanced sampling to avoid collapse (empty clusters / all-to-one); classifier head must be reset each reassignment (permutation problem) → disrupts training.
- **SeLa / Asano et al. 2019.** Cast pseudo-label assignment as OPTIMAL TRANSPORT with an equipartition constraint, solved by Sinkhorn-Knopp; gives principled, collapse-free assignment. Gap: still OFFLINE, whole-dataset, and rounds to a HARD assignment.
- **Sinkhorn-Knopp / Cuturi 2013.** Entropy-regularized OT solved by alternating row/column normalization; differentiable, fast on GPU. The tool that makes online equipartition feasible.
- **NAT (Bojanowski et al. 2017).** Align features to fixed random targets via Hungarian; supports "prototypes need not be categorical."
- **Multi-crop roots: PIRL/jigsaw (Misra & van der Maaten 2019; Noroozi 2016).** More views help; small crops cover ~20% of image. FixRes (Touvron 2019): train/test resolution mismatch biases features → use a mix of sizes.

## The method, derived
- Encoder f_theta, projection head (2-layer MLP, BN, output 128-D), L2-normalize z onto unit sphere.
- K trainable prototypes C (D x K), a bias-free linear layer; normalize columns to unit norm.
- For each view: prototype scores z^T C; soft assignment ("code") q from one view, predicted from another view's softmax(z^T C / τ). SWAPPED: predict q_s from z_t and q_t from z_s.
  L(z_t, z_s) = ell(z_t, q_s) + ell(z_s, q_t); ell(z, q) = -Σ_k q^(k) log p^(k), p^(k)=softmax_k(z^T c_k / τ).
- **Codes online via OT.** For a batch Z (D x B), code matrix Q (K x B): max_Q Tr(Q^T C^T Z) + ε H(Q), subject to Q in transportation polytope U = {Q: Q1_B = 1/K 1_K, Q^T 1_K = 1/B 1_B} (equipartition: each prototype gets B/K mass). Solution Q* = Diag(u) exp(C^T Z / ε) Diag(v); u,v by Sinkhorn iterations (alternate row/col normalization). 3 iterations. Keep SOFT Q* (rounding to hard converges too fast → worse).
- ε small (0.05). High ε → uniform collapse. τ=0.1.
- **Collapse prevention** comes from the equipartition constraint in the OT (not from negatives). Stop-gradient: codes computed under no_grad (targets); only the prediction term p gets gradients.
- **Small batches:** when B < K can't equipartition; keep a queue of recent features (~3k) to enlarge Z for the assignment only; loss still on batch codes. Far smaller than MoCo's 65k.
- **Multi-crop:** 2 standard crops (224) + V small crops (96); compute codes ONLY from the 2 full-res crops, predict them from all V+2 views. L = Σ_{i in {1,2}} Σ_{v != i} ell(z_v, q_i). Small crops cover small area → bad codes, so only big crops produce codes. More views, ~no extra memory.
- Freeze prototypes first epoch; LARS, lr warmup + cosine; sync BN.

## Sinkhorn derivation (self-check)
M = C^T Z. Lagrangian for max ⟨Q,M⟩ + εH(Q) s.t. row=1/K, col=1/B:
L = Σ_ij Q_ij M_ij - ε Σ Q_ij log Q_ij + Σ_i α_i(Σ_j Q_ij - 1/K) + Σ_j β_j(Σ_i Q_ij - 1/B).
∂/∂Q_ij = M_ij - ε(log Q_ij + 1) + α_i + β_j = 0 → Q_ij = exp((α_i)/ε - 1/2) · exp(M_ij/ε) · exp((β_j)/ε - 1/2) = u_i exp(M_ij/ε) v_j.
So Q* = Diag(u) exp(M/ε) Diag(v). u,v are the unique scalings making rows/cols hit the marginals → Sinkhorn fixed-point: repeat u ← r / (exp(M/ε) v), v ← c / (exp(M/ε)^T u), with r=1/K 1_K, c=1/B 1_B.

## Design choice -> why
- Prototypes instead of stored features: a small fixed set of "anchors" C contrasts views via assignment, no need to store/compare many negatives. # prototypes barely matters (3k-100k all ~equal); random fixed prototypes nearly as good → they are contrast anchors, not class centroids.
- Soft codes not hard: hard rounding too aggressive, converges fast to worse minimum.
- Equipartition constraint: prevents the trivial all-same-code collapse without negatives or sampling tricks.
- ε small: large ε → uniform/collapsed Q.
- Stop-grad on codes: codes are targets; backprop only through prediction.
- Codes from full-res crops only: small crops have partial info → low-quality assignments.
- τ=0.1 sharpens prediction softmax (from instance-discrimination practice).
- L2-normalize z and prototypes: cosine similarity, stable scale.
