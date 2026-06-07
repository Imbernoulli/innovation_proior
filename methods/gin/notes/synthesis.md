# GIN — synthesis (design-decision → why), written before Phase 2

## The pain point at the time (≈2018)
- A flood of message-passing / neighborhood-aggregation GNNs (GCN, GraphSAGE, GG-NN, MPNN, GAT, columnar nets) all of the shape:
  - a_v^{(k)} = AGGREGATE({h_u^{(k-1)} : u ∈ N(v)});  h_v^{(k)} = COMBINE(h_v^{(k-1)}, a_v^{(k)})
  - graph readout h_G = READOUT({h_v^{(K)} : v ∈ G})
- All designed by heuristic/trial-and-error. No theory: how *powerful* is a message-passing GNN? What graph structures can it tell apart, where's the ceiling, and which design choices matter for that ceiling?

## Load-bearing ancestors (verified against primary text / known)
- **GCN (Kipf & Welling 2017):** h_v^{(k)} = ReLU(W · MEAN{h_u^{(k-1)} : u ∈ N(v)∪{v}}). AGGREGATE+COMBINE merged, MEAN over self+neighbors, single linear layer + ReLU per layer (a 1-layer perceptron). Gap: mean loses multiplicity; one linear layer is not a universal approximator of multiset functions.
- **GraphSAGE (Hamilton et al. 2017):** pooling variant a_v = MAX({ReLU(W·h_u) : u ∈ N(v)}); COMBINE = concat[h_v, a_v] then linear. Also mean and LSTM aggregators. Gap: max-pool sees only the *set* (support), discards multiplicity entirely.
- **WL test (Weisfeiler & Lehman 1968) / 1-WL color refinement:** iteratively l_v^{(k)} = HASH(l_v^{(k-1)}, {{l_u^{(k-1)} : u ∈ N(v)}}) with an *injective* hash; declare non-isomorphic if label multisets differ at some iteration. Distinguishes a broad class of graphs (Babai & Kucera 1979); fails only on corner cases (e.g. regular graphs, Cai-Fürer-Immerman). WL subtree kernel (Shervashidze et al. 2011): counts of WL labels as graph features; a k-th iteration label = a height-k rooted subtree.
- **Deep Sets (Zaheer et al. 2017):** any permutation-invariant *set* function = ρ(Σ_{x∈S} f(x)). GIN extends from sets to **multisets** (the harder case — must preserve multiplicity).
- **MPNN (Gilmer et al. 2017) / "representation" survey (Xu et al. 2018):** unifying message-passing abstraction GIN's framework slots into.
- **Universal approximation (Hornik et al. 1989, 1991):** lets MLPs model the abstract f, φ.
- **JK-Networks (Xu et al. 2018):** aggregate node reps across all layers (jumping knowledge) → motivates concatenating per-layer graph readouts.
- **PointNet (Qi et al. 2017):** max-pool learns the "skeleton"/support of a point cloud — the practical face of "max captures the underlying set."

## The reasoning spine
1. A maximally powerful GNN should map two nodes to the same point only if their rooted subtrees are identical. Subtrees are defined recursively via neighborhoods → reduce to: when does the GNN map two **neighborhood multisets** to the same vector? Max power ⇔ aggregation **injective on multisets**.
2. **Multiset** abstraction (Def): X=(S,m), m: S→ℕ≥1 multiplicity. Neighbor features form a multiset (repeats allowed). Countability assumption on feature space (propagates across layers — Lemma countability).
3. **Lemma (≤ WL upper bound):** any aggregation GNN is at most as powerful as 1-WL. Proof: if WL can't separate G1,G2 then it keeps identical label multisets every iteration; by induction build a map φ with h_v^{(i)}=φ(l_v^{(i)}); identical WL neighborhoods ⇒ identical GNN neighborhood features ⇒ identical h after AGGREGATE/COMBINE ⇒ identical readout ⇒ A(G1)=A(G2). Contrapositive gives the claim.
4. **Theorem (= WL if injective):** if AGGREGATE-COMBINE (f, φ injective) and graph READOUT injective, GNN is as powerful as WL. Proof by induction: ∃ injective φ_k with h_v^{(k)}=φ_k(l_v^{(k)}); using WL hash g, ψ∘g^{-1} is injective. So distinct WL label multisets ⇒ distinct h-multisets ⇒ distinct embeddings.
5. **Lemma countability:** ℕ×ℕ countable via φ(m,n)=2^{m-1}(2n-1); finite Cartesian products of countable sets countable; pad-with-dummy injection from bounded multisets into X'^k ⇒ range of each layer countable. So "injective" is the right notion across layers.
6. **Lemma (sum is universal over multisets):** X countable ⇒ ∃ f: f(x)=N^{-Z(x)} (N bounds |X|, Z:X→ℕ) so Σ_{x∈X} f(x) injective on bounded multisets (it's a base-N positional code with digit-counts = multiplicities). Any multiset function g = φ(Σ f(x)). Extends Deep Sets from sets→multisets.
7. **Why mean/max fail (counterexamples, verified):**
   - All-same-feature graph (a): every node feat a, mean/max of f(a) stay f(a) → no structure captured; sum gives 2f(a) vs 3f(a) (distinguishes). With degree as feature, mean recovers sum but max can't.
   - (b) Max fails: neighbor multisets {g,r} vs {g,r,r} → max = max(h_g,h_r) both → collapse. Sum: h_g+h_r vs h_g+2h_r differ; mean ½(h_g+h_r) vs ⅓(h_g+2h_r) differ → mean OK here.
   - (c) Mean & max both fail: {g,r} vs {g,g,r,r}. mean: ½(h_g+h_r)=¼(2h_g+2h_r) equal; max equal; sum h_g+h_r vs 2h_g+2h_r differ.
   - **Corollary mean:** mean h(X1)=h(X2) iff same *distribution* (X2=(S,k·m)); f(x)=N^{-2Z(x)}. Mean = distributional sum.
   - **Corollary max:** max captures the *underlying set* only; f one-hot into ℝ^∞ (f_i(x)=1 iff i=Z(x)). Power ranking SUM ⊐ MEAN ⊐ MAX.
8. **Lemma (1-layer perceptron insufficient):** ∃ X1={1,1,1,1,1}, X2={2,3} with Σ ReLU(Wx) equal for all W (no bias). ReLU homogeneity + same coordinate-signs (positive inputs) ⇒ Σ ReLU(Wx)=ReLU(W Σx) and Σx equal. So σ∘W degenerates to linear sum → not universal; need MLP (≥2 layers) for the abstract f.
9. **Corollary (GIN update):** ∃ f s.t. h(c,X)=(1+ε)f(c)+Σ_{x∈X} f(x) injective over (c,X) for infinitely many ε incl. all irrationals. Proof: case c'=c reduces to sum-lemma; case c'≠c, rearrange ε(f(c)-f(c'))=rational; LHS irrational, RHS rational → contradiction. The (1+ε) term separates the **center node** from its neighbor-sum so the root isn't confused with its neighborhood (e.g. chain a-b-b vs b-a-b). ε can be **learned** (GIN-ε) or **fixed 0** (GIN-0): with ε=0 it's exactly sum over self∪neighbors = adding a self-loop; GIN-0 is slightly less powerful in contrived cases but simpler and generalizes slightly better.
10. **Model:** h_v^{(k)} = MLP^{(k)}((1+ε^{(k)})·h_v^{(k-1)} + Σ_{u∈N(v)} h_u^{(k-1)}). One MLP per layer models f^{(k+1)}∘φ^{(k)} (composition). First layer: if one-hot inputs, sum alone injective, no pre-sum MLP needed.
11. **Graph readout (GIN):** sum each layer's node features (injective on multisets), then **CONCAT across all layers k=0..K** (JK-style) so coarse+fine subtree info all used; provably generalizes WL + WL subtree kernel.

## Design-decision → why table
| Decision | Why | Rejected alt & failure |
|---|---|---|
| Compare power to 1-WL, not full graph isomorphism | exact iso has no known poly algorithm; WL is the strong, efficient heuristic GNNs structurally mirror | aim for full iso → intractable |
| Abstract neighbor agg as multiset function | repeats matter (multiplicity carries structure); injective-on-multisets ⇔ max power | set abstraction loses multiplicity |
| SUM aggregator | only one of {sum,mean,max} that is injective/universal on multisets (Lemma 5) | mean → only distribution (Cor); max → only support (Cor) |
| MLP (≥2 layers) for f/φ | universal approximator of multiset functions; 1-layer σ∘W degenerates to linear sum (Lemma 8) | 1-layer perceptron underfits, not universal |
| (1+ε)·h_v + Σ neighbors | separate center node from neighbor multiset; keep injectivity over (center, multiset) (Cor 9) | sum self∪neighbors w/o weighting can confuse root vs neighbor in non-symmetric trees |
| ε learnable or fixed 0 | learned = strictly max power; ε=0 = self-loop, simpler, generalizes slightly better (GIN-0) | — |
| sum readout + CONCAT over all layers | sum readout injective on multisets; concat (JK-style) keeps all subtree depths; generalizes WL subtree kernel | last-layer-only readout loses earlier (better-generalizing) features |
| countable feature space assumption | makes "injective" the exact right notion; countability propagates across layers (Lemma 6) | uncountable/continuous needs metric notions — left out |
| BatchNorm + ReLU inside MLP, Adam | standard training stability (implementation detail, not power) | — |
