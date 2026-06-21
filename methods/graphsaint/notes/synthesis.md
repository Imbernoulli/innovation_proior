# GraphSAINT synthesis

arXiv 1907.04931 (verified). Zeng, Zhou, Srivastava, Kannan, Prasanna. ICLR 2020.
GraphSAINT = Graph SAmpling based INductive learning meThod.

## Pain point: neighbor explosion
GCN layer: x_v^(ℓ+1) = σ( Σ_{u∈V} Â_{v,u} (W^(ℓ))ᵀ x_u^(ℓ) ), Â = D^{-1}A (row-normalized).
To compute one output node need its neighbors' layer-ℓ activations; those need THEIR neighbors at
ℓ-1; etc. Support set grows exponentially with depth L — "neighbor explosion." Full-batch training
(original Chebyshev GCN, Defferrard/Kipf) doesn't scale to large graphs or deep nets.

## Baselines (layer sampling) and their gaps
All follow: (1) build complete GCN on full graph, (2) sample nodes/edges PER LAYER for minibatches,
(3) fwd/bwd on sampled GCN.
- GraphSAGE: uniform sample fixed #neighbors (2–50) per node per layer. Bounds per-node fanout but
  fanout still multiplies across layers (budget^L support). Expensive for deep nets.
- VR-GCN / S-GCN (Chen 2018): restrict to 2 support nodes/layer via historical activations. Still
  costly even at fanout 2 for deep nets; bookkeeping of stale activations.
- FastGCN (Chen 2018): sample nodes INDEPENDENTLY per layer (importance sampling, ∝ ‖Â_{:,u}‖²),
  constant sample size all layers (fanout 1). But independent per-layer sampling → minibatch can be
  too SPARSE: a node sampled at layer ℓ+1 may have NO sampled neighbor at layer ℓ → broken
  connectivity → low accuracy. Survival prob of input node through L independent samplers ≈
  (1-(1-p)^d)^{L-1}, shrinks with depth.
- AS-GCN (Huang 2018): adaptive sampler conditioned on next-layer nodes (a learned sampling net) →
  good connectivity/accuracy but heavy sampler overhead + extra params.
- ClusterGCN (Chiang 2019): partition graph into dense clusters, minibatch = random clusters; no
  layer sampling so no neighbor explosion. BUT heuristic, biased estimator of full-batch loss (unequal
  node/edge appearance probabilities not corrected).

## Core idea: sample the GRAPH, not the layers
Flip the order: sample a subgraph G_s(V_s,E_s) from G FIRST (|V_s|≪|V|), then build a COMPLETE GCN on
G_s. Each minibatch GCN is small but complete → no neighbor explosion (support stays inside subgraph,
never expands outside). Connectivity within minibatch never drops with depth: if edge present in
layer ℓ it's present in all layers.

Two sampler requirements (intuition): (1) nodes with high mutual influence should land in same
subgraph (so they support each other's propagation within the batch); (2) every edge has
non-negligible sampling prob (explore full feature/label space → generalization). Define "influence"
from topology (connectivity), avoid feature-dependent samplers (too complex).

Algorithm: pre-process (set sampler params, compute norm coeffs α, λ); each minibatch: sample G_s,
build full GCN, forward (normalize aggregation by α), backward from λ-normalized loss, update weights.

## The bias problem and normalization (the key technical contribution)
A connectivity-preserving sampler samples nodes/edges with NON-uniform probability → biases the
minibatch estimator toward frequently-sampled (high-centrality) nodes. Must remove this bias.

Analyze each layer independently (nonlinearities make full analysis hard; same treatment as
FastGCN/AS-GCN). For sampled node v at layer ℓ+1, the aggregated feature using subgraph edges:
  ζ_v^(ℓ+1) = Σ_{u∈V} (Â_{v,u}/α_{u,v}) x̃_u^(ℓ) 1_{u|v},   x̃_u^(ℓ) = (W^(ℓ))ᵀ x_u^(ℓ)
where 1_{u|v} indicates edge (u,v) present in subgraph GIVEN v sampled. α_{u,v} = AGGREGATOR
NORMALIZATION.
Let p_{u,v} = P(edge (u,v) sampled), p_v = P(node v sampled).
Proposition: ζ_v^(ℓ+1) is UNBIASED estimator of full-layer aggregation Σ_u Â_{v,u} x̃_u^(ℓ) IFF
  α_{u,v} = p_{u,v} / p_v.
Proof: E[1_{u|v}] = P((u,v) sampled | v sampled) = p_{u,v}/p_v (conditional). So
E[ζ_v] = Σ_u (Â_{v,u}/α_{u,v}) x̃_u · (p_{u,v}/p_v); set α = p_{u,v}/p_v → cancels → Σ_u Â_{v,u} x̃_u. ✓
(α is the conditional edge probability given v sampled.)

LOSS NORMALIZATION: minibatch loss L_batch = Σ_{v∈G_s} L_v/λ_v, set λ_v = |V|·p_v. Then
  E[L_batch] = (1/|V|) Σ_{v∈V} L_v.
Check: E[Σ_{v∈V_s} L_v/λ_v] = Σ_{v∈V} (L_v/λ_v)·p_v = Σ_v (L_v/(|V|p_v))·p_v = (1/|V|)Σ_v L_v. ✓
(λ corrects per-node loss for unequal sampling freq; so feature learning doesn't favor frequently
sampled nodes.)

ESTIMATING α, λ: for random node/edge samplers p analytic; general samplers no closed form. So
pre-process: run sampler N times → set of N subgraphs. Counters C_v (#subgraphs containing v),
C_{u,v} (#subgraphs containing edge). Then α_{u,v}=C_{u,v}/C_v and the full-loss normalizer is
λ_v=|V_train|·C_v/N. The paper text later writes λ_v=C_v/N as a probability shorthand, but the
unbiased full-average loss and the public code include the |V_train| factor. Subgraphs reused as
minibatches → small overhead.
(Code confirms: norm_loss_train[v] = N/(C_v·|V_train|) = 1/λ_v; norm_aggr[edge u,v]=C_v/C_{u,v}=1/α_{u,v}.)

## Variance reduction (designing the sampler)
Want sampler that minimizes variance of the estimators. Ignore activations / analyze aggregate.
Let b_e^(ℓ) = Â_{v,u} x̃_u^(ℓ-1) + Â_{u,v} x̃_v^(ℓ-1) for edge e=(u,v). Define
  ζ = Σ_ℓ Σ_{v∈G_s} ζ_v^(ℓ)/p_v = Σ_ℓ Σ_e (b_e^(ℓ)/p_e) 1_e^(ℓ).
The /p_v makes ζ unbiased estimator of Σ_ℓ Σ_{v∈V} E[ζ_v^(ℓ)]. Once edge sampled it's in all layers:
1_e^(ℓ)=1_e ∀ℓ. Restrict to INDEPENDENT EDGE SAMPLING: each e ∈ E independently in G_s with prob p_e,
constrained Σ_e p_e = m (budget m = expected #sampled edges).
Theorem: optimal p_e minimizing sum-of-variances over ζ's dimensions:
  p_e = (m / Σ_{e'} ‖Σ_ℓ b_{e'}^(ℓ)‖) · ‖Σ_ℓ b_e^(ℓ)‖.
Proof (scalar b first): independent edges → Cov(1_{e1}^{ℓ1},1_{e2}^{ℓ2})=0 for e1≠e2;
Cov(1_e^{ℓ1},1_e^{ℓ2})=p_e-p_e² (same edge all layers). Then
  Var(ζ) = Σ_e [ (Σ_ℓ b_e^(ℓ))²/p_e ] − Σ_e (Σ_ℓ b_e^(ℓ))².
  (the cross terms and Var=p_e-p_e² combine: Σ_{e,ℓ}(b/p)²(p-p²)... → collapses to above.)
Second term independent of p. Minimize Σ_e (Σ_ℓ b_e^(ℓ))²/p_e s.t. Σ p_e=m. Cauchy-Schwarz:
  [Σ_e (Σ_ℓ b)²/p_e]·[Σ_e p_e] ≥ (Σ_e |Σ_ℓ b|)², equality when (Σ_ℓ b)/√p_e ∝ √p_e
  i.e. p_e ∝ |Σ_ℓ b_e^(ℓ)|. With Σ p_e=m → p_e = m|Σ_ℓ b_e|/Σ_{e'}|Σ_ℓ b_{e'}|. Multi-dim: replace |·|
  with ‖·‖. ✓
But b_e^(ℓ) depends on activations x̃ (expensive, changes during training). SIMPLIFY: drop activation
dependence, make p_e depend on topology only → p_e ∝ Â_{v,u}+Â_{u,v} = 1/deg(u)+1/deg(v).
Intuition match: connected u,v with FEW neighbors are influential to each other → high p_e. ✓

Remark: applying this edge prob as a LAYER sampler (FastGCN style) gives sparse minibatches for L>1
(survival (1-(1-p)^d)^{L-1}), whereas graph sampling keeps full connectivity across all layers.

## Samplers (Appendix algorithm)
All return NODE-INDUCED subgraph from sampled nodes (induction adds back edges among sampled nodes,
empirically helps convergence). Budgets are targets, not exact counts (repeats → fewer).
- Node sampler: paper distribution samples n nodes (with replacement) ∝ P(v) = ‖Â_{:,v}‖² (from
  FastGCN layer sampler); official code uses the training CSR counts/cumulative distribution for the
  implementation path.
- Edge sampler (approximate, O(m) vs O(|E|)): sample m edges (w/ replacement) ∝
  P((u,v)) = (1/deg(u)+1/deg(v)) / normalizer; V_s = endpoints; induce.
- RW sampler: r roots uniform; each does h-hop walk (uniform neighbor each step); collect nodes; induce.
- MRW (multi-dim random walk / frontier sampler, Ribeiro & Towsley): r-node frontier; iteratively
  pick u from frontier ∝ deg(u), step to random neighbor u', swap into frontier; add u to V_s; induce.

## Extensions (discussion)
Because each minibatch is a COMPLETE GCN, drop-in for architecture variants:
- JK-net (jumping knowledge): skip connections straightforward (complete GCN); layer samplers need
  layer-ℓ ⊆ layer-(ℓ-1) which they don't have.
- GAT attention: apply within subgraph (subgraph as representative of full graph); remove the
  within-neighborhood softmax over attention (as AS-GCN suggests); norms still apply.
- High-order layers / graph classification (DiffPool): replace full A with subgraph A_s.

## Implementation / settings (appendix)
- TensorFlow 1.12 (also a PyTorch version); sampling in Cython, parallel over 40 CPU cores.
- Adam optimizer. Grid: hidden {128,256,512}, dropout {0,0.1,0.2,0.3}, lr {0.1,0.01,0.001,0.0001}.
- Hidden dims: 512 PPI, 256 Flickr, 128 Reddit, 512 Yelp, 512 Amazon.
- sample_coverage: keep sampling subgraphs in pre-proc until total sampled nodes > coverage·|V_train|.
- Save best-val-F1-micro model, eval on test. Eval is full-batch (no sampling); code uses
  norm_loss_test=1/(|V_train|+|V_val|+|V_test|) on labeled splits for reporting loss.

## Design-decision → why
- Sample graph not layers: keeps each minibatch a complete connected GCN → no neighbor explosion,
  full inter-layer connectivity (fixes FastGCN sparsity).
- α=p_{u,v}/p_v aggregator norm: makes per-layer aggregation an unbiased estimator of full aggregation.
- λ=|V|p_v loss norm: makes minibatch loss unbiased estimator of full loss; removes high-centrality bias.
- Estimate α,λ by counters over N pre-sampled subgraphs: general samplers have no closed-form p.
- Independent edge sampling + Cauchy-Schwarz: gives closed-form variance-optimal p_e ∝ ‖Σ_ℓ b_e‖.
- Drop activation dependence → p_e ∝ 1/deg(u)+1/deg(v): keeps sampler cheap, topology-only.
- Node-induced subgraph: induction adds edges among sampled nodes → denser/better-connected → faster
  convergence.
- Reuse pre-sampled subgraphs as minibatches: amortize sampling cost.
