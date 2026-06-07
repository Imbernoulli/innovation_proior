# DiffPool synthesis

arXiv 1806.08804 (verified by title search). Ying, You, Morris, Ren, Hamilton, Leskovec. NeurIPS 2018.

## Pain point
GNNs (message passing, Gilmer 2017) compute node embeddings H^(k)=M(A,H^(k-1);θ). For graph
classification need a single vector per graph. Standard approach: globally pool all node embeddings
(sum/mean/set-NN, Duvenaud 2015, Li 2016, set2set). This is FLAT — ignores hierarchical structure
(atoms→functional groups→molecule; communities). CNNs get hierarchy via spatial pooling on coarser
and coarser images. Graphs have no spatial locality / no "m×m patch", and graphs vary in size, so a
deterministic patch/pooling op is ill-defined. Two-stage (deterministic clustering then GCN) exists
(Defferrard 2015 spectral clustering, Simonovsky 2017, Fey 2018) but the clustering is fixed, not
learned end-to-end, and must generalize across graphs.

## Core object
Want a differentiable pooling module: given GNN output Z and A, produce coarsened graph with m<n
nodes: A'∈R^{m×m}, Z'∈R^{m×d}. Must be (a) learned end-to-end, (b) generalize across graphs of
different size/structure, (c) permutation invariant.

## Key idea — soft cluster assignment matrix
At layer l, learn S^(l) ∈ R^{n_l × n_{l+1}}: row = node at layer l, column = cluster at layer l+1.
Row-wise softmax → soft assignment (each node a distribution over next-layer clusters). n_{l+1} is a
hyperparameter (max #clusters), chosen as a fraction of n_l (ratio ~0.1–0.5, called α; performance
insensitive in that range; they use ~25% in experiments — assign ratio 0.1 in some configs).

Pooling equations (given S^(l), A^(l), Z^(l)):
  X^(l+1) = S^(l)ᵀ Z^(l)          ∈ R^{n_{l+1} × d}    (Eq 4) — aggregate node embeds into clusters
  A^(l+1) = S^(l)ᵀ A^(l) S^(l)     ∈ R^{n_{l+1} × n_{l+1}} (Eq 5) — coarsened weighted adjacency
A^(l+1) is real-valued, fully-connected weighted graph; entry ij = connectivity strength clusters i,j.

## Two-GNN design
S and Z computed by TWO SEPARATE GNNs on the SAME inputs (A^(l), X^(l)), distinct parameters:
  Z^(l) = GNN_{l,embed}(A^(l), X^(l))               — embedding GNN (new cluster-node embeddings)
  S^(l) = softmax( GNN_{l,pool}(A^(l), X^(l)) )      — pooling GNN, row-wise softmax; output dim = n_{l+1}
Base case l=0: inputs are original A, F. Penultimate layer L-1: S^(L-1) = vector of 1's → all nodes
to one cluster → single graph embedding → fed to differentiable classifier (softmax). End-to-end SGD.

Each GNN itself = K iterations (K usually 2–6) of GCN message passing
  H^(k) = ReLU( D̃^{-1/2} Ã D̃^{-1/2} H^(k-1) W^(k-1) ),  Ã = A+I, D̃ degree of Ã.
Abstract as Z = GNN(A,X).

## Permutation invariance (Proposition)
For any permutation matrix P: DiffPool(A,Z) = DiffPool(PAPᵀ, PX) provided component GNN is perm
invariant (GNN(A,X)=GNN(PAPᵀ,X)). Proof: GNN_embed, GNN_pool perm invariant by assumption; P
orthogonal so PᵀP=I, apply to Eq 4 (X'=Sᵀ(PᵀP)Z form) and Eq 5 → invariant. Short, lived in reasoning.

## Why aux losses (the optimization difficulty)
Training pooling GNN from classification gradient alone is hard — non-convex, easy to land in spurious
local minima early. Two auxiliary objectives:
1. Link-prediction / auxiliary objective at each layer:
   L_LP = || A^(l) − S^(l) S^(l)ᵀ ||_F   (Frobenius). Encodes "nearby (connected) nodes pooled
   together." Note A^(l) at deeper layers is itself a function of lower S's and changes during training.
   Resembles low-rank matrix factorization of A (SSᵀ≈A) / WSPD, but soft (differentiable) and
   trained jointly with task → finds task-better local minimum than two-step clustering+GCN.
2. Entropy regularization: L_E = (1/n) Σ_i H(S_i), H entropy of row i. Pushes each row toward one-hot
   → crisp cluster membership.
L_LP and L_E from every layer added to classification cross-entropy. Training slower but better /
more interpretable clusters.

## Dense vs sparse behavior (motivating intuition)
Dense subgraph → adjacency ≈ all ones; 11ᵀ all ones; assignment with one column all-ones, rest zero
≈ minimizer of L_LP → dense subgraph collapses to one hypernode. Justified: GNN message passing is
efficient on dense small-diameter cliques (little structural info lost). Sparse subgraphs (paths,
cycles, trees, high diameter) → higher-entropy assignment, split across clusters to preserve structure.
Explains weaker result on COLLAB (very dense).

## Implementation details (appendix)
- PyTorch, TITAN Xp. Undirected graphs as adjacency matrices. Extra input features: node degree and
  clustering coefficient.
- For GraphSAGE backbone: adjacency NOT normalized, bias terms added to conv layers.
- Hidden dims: 64 (Enzymes, Collab, Proteins), 128 (D&D, Reddit-Multi-12k).
- LR swept 1e-5 to 1e-2. Hidden reps at ALL layers concatenated for final graph rep.
- After each graph conv: ℓ2-normalize embeddings + batch-norm layer (helps overfitting).
- Classifier: 2-layer MLP (ReLU then softmax), hidden = GNN hidden dim. Cross-entropy loss.
- L_LP: Frobenius norm then normalize by number of adjacency entries.
- Adam optimizer; gradient clipped at norm 2.0. 4000 epochs, early stopping on moving-window val loss.
- Asymmetry between S and Z networks (from cut/expanded text, real design rationale): embedding net Z
  uses structural features (degree, clustering coeff) + node features; assignment net removes structural
  features (mainly homophily); Z dimension grows / S dimension shrinks at deeper layers (like CNN channels).

## Canonical code (PyG dense_diff_pool, verified from source)
```
s = torch.softmax(s, dim=-1)
out = torch.matmul(s.transpose(1, 2), x)                       # X' = Sᵀ X
out_adj = torch.matmul(torch.matmul(s.transpose(1,2), adj), s) # A' = Sᵀ A S
link_loss = adj - torch.matmul(s, s.transpose(1, 2))           # A - S Sᵀ
link_loss = torch.norm(link_loss, p=2)
if normalize: link_loss = link_loss / adj.numel()
ent_loss = (-s * torch.log(s + 1e-15)).sum(dim=-1).mean()
return out, out_adj, link_loss, ent_loss
```
RexYing/diffpool encoders.py uses a BCE-style link loss variant
(-adj*log(pred)-(1-adj)*log(1-pred)); PyG uses the Frobenius form matching the paper exactly.
Use the PyG Frobenius form as the grounded canonical implementation.

## Design-decision → why table
- Soft (vs hard) assignment: hard cluster assignment is non-differentiable → can't train end-to-end.
  Softmax gives differentiable probabilistic assignment.
- Sᵀ A S for coarse adjacency: bilinear form aggregates edge weights between clusters; falls out of
  treating S as a (soft) node→cluster indicator and computing cluster-cluster connectivity.
- Two separate GNNs (not one): Z and S serve different purposes (representation vs partition). Sharing
  would couple them; separate params let pooling specialize on homophily while embedding keeps features.
- L_LP needed: pure classification gradient too weak/local-minima-prone for the pooling net.
- Entropy reg: without it soft assignment can stay diffuse → no clear clusters → coarse graph
  uninformative. Push to one-hot.
- n_{l+1} as fraction of n_l: must reduce nodes (pooling) but keep enough granularity; insensitive 0.1–0.5.
- Final S = ones vector: collapse to single graph vector for classification.
- ℓ2 norm + batchnorm per layer: magnitude drift over stacked layers; batchnorm curbs overfitting.
