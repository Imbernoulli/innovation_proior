# BigBird synthesis (pre-Phase-2)

## Pain point
- Full self-attention (Vaswani 2017; BERT, Devlin 2018) computes an n×n score matrix → O(n²) time & memory. On commodity hardware caps usable context ≈512 tokens. Blocks long-context tasks: QA, long-doc classification/summarization, genomics (DNA, long sequences).
- Theory of full attention is strong: Yun et al. 2019 — transformers are universal approximators of continuous seq2seq functions on compact domain; Pérez et al. 2019 — full encoder-decoder transformer is Turing complete (arbitrary precision). Question: can we drop most of the n² inner products AND keep both properties?

## Core reframing
- Generalized attention = directed graph D on vertices [n]. Out-neighbors N(i) = the keys query i attends to. Output: Attn_D(X)_i = x_i + Σ_h σ(Q_h(x_i) K_h(X_{N(i)})^T) V_h(X_{N(i)}). Complete digraph = full attention. Adjacency A∈{0,1}^{n×n}, A=ones ⇒ BERT.
- So "reduce complexity" = "sparsify the graph while keeping the function class".

## The three components (each justified)
1. RANDOM: want short average path length so info propagates in few hops. Erdős–Rényi random graph with Θ̃(n) edges has shortest path O(log n); spectrally approximates complete graph (2nd eigenvalue far from 1st) ⇒ rapid mixing of random walk ⇒ info flows fast between any pair. ⇒ each query attends r random keys. A(i,·)=1 for r random keys.
2. WINDOW (locality): NLP/biology data has locality of reference (Clark 2019: neighboring inner products dominate BERT attention). Graph notion = clustering coefficient (cliques). ER graphs have low clustering; Watts–Strogatz small-world graph (ring lattice + rewire) has BOTH low path length and high clustering. Ring lattice = sliding window. ⇒ query i attends i-w/2 .. i+w/2. A(i, i-w/2:i+w/2)=1. (Keep all local edges; don't actually rewire — hardware friendlier, doesn't hurt.)
3. Sanity check (Table @512): R only 60.1 MLM, W only 58.3, R+W 62.7 — all below BERT 64.2. ⇒ window+random INSUFFICIENT. Need a third piece.
4. GLOBAL: tokens that attend to all and are attended by all. Two flavors:
   - ITC (internal): pick subset G of existing tokens, A(i,:)=1, A(:,i)=1 ∀i∈G.
   - ETC (extended): add g NEW global tokens (CLS-like). New matrix B∈{0,1}^{(N+g)×(N+g)}, B(i,:)=1,B(:,i)=1 for i∈{1..g}, B(g+i,g+j)=A(i,j).
   Global is what RECOVERS the theory (star graph) — derived from the proof.

## Complexity
- Each query attends g + w + r keys (all O(1) in n) ⇒ O(n) edges, O(n·d·(g+w+r)) compute, O(n) memory. Linear. Lets sequence length grow ~8×.

## Theory part 1: Universal approximation
- Function class F_CD: continuous f:[0,1]^{n×d}→R^{n×d}, ℓ_p topology, d_p(f1,f2)=(∫‖f1−f2‖_p^p)^{1/p}.
- Star graph S centered at 0: N(i)={0,i} for i≥1, N(0)={1..n}. THEOREM: any graph D containing S is a universal approximator. (BigBird with one global token contains S.)
- Proof structure (Yun 2019, 3 steps):
  - Step 1: approx f by piecewise-constant f̄ on grid G_δ (granularity δ), d_p(f,f̄)≤ε/3. Add positional embedding E with rows δ^{-(i-1)d} to separate columns into disjoint value buckets: δ^{-(i-1)d} ≤ <u,X_i> ≤ δ^{-id}-δ.
  - Step 2 (the heart): build a CONTEXTUAL MAPPING with sparse attn. Contextual mapping q: G_δ→R^n s.t. (1) entries of q(P) distinct, (2) entries across different P,P' all distinct — a unique code for (X,x_i). Original Yun proof used a SELECTIVE SHIFT operator (shift entries in a range by max−min of the block) needing full attention. BigBird's innovation: a SPARSE selective shift implemented with attention over N(i), amount controlled by D, plus a GLOBAL token at index 0.
    - Selective shift lemma: ψ_u(Z;b1,b2)_i = (max_{j∈N(i)}u^T Z_j − min_{j∈N(i)}u^T Z_j)e_1 if b1≤u^T Z_j≤b2 else 0. Built as difference of two hardmax attentions ψ̃(Z;b) (one returns max, other min) ⇒ ψ̃(·,b_Q)−ψ̃(·,b_Q').
    - Contextual map (Lemma): u=[1,δ^{-1},...,δ^{-d+1},δ^{-nd}], x_0=(0,..,0,1). Run n PHASES, each = (low shift on column i: X←X+δ^{-d}ψ(X,v-δ/2,v+δ/2) for v in [δ^{-id},δ^{-(i+1)d}) — only l_i in range; uses global token as max so f_i=δ^{-d}(f̃_0^{k-1}−l_k)+l_k) THEN (high shift on global token: X←X+δ^{-nd}ψ(...) over [S_{k-1},T_{k-1}); since f_k is now global max and l_{k+1} global min, f̃_0^k=δ^{-nd}(f_k−l_{k+1})+f̃_0^{k-1}). Recursion f̃_0^k=(δ^{-(n+1)d}+1)f̃_0^{k-1}−(δ^{-nd-d}+δ^{-nd})l_k−l_{k+1}, expands to closed form (UP)/(LP) giving invariants S_k<f̃_0^k<T_k, T_{k-1}≤f_k<S_k, ordering l_{k+1}<...<l_n<f_1<...<f_k<f̃_0^k. After n phases f̃_0^n uniquely encodes P. One extra layer makes all entries distinct (full contextual map).
    - The star graph is exactly what's needed: column i needs to see column 0 (global) [edge in S as N(i)∋0] and column 0 needs to see all [N(0)=all]. Window+random alone can't reach all columns in O(1) — global is the load-bearing piece.
  - Step 3: replace hardmax+Φ activations by softmax+ReLU, extra ε/3. ⇒ g∈T_D^{2,1,4} with d_p(f,g)≤ε.
- KEY INSIGHT: the theory is what FORCED the global token. The empirical sanity check (R+W<BERT) and the proof both point at the same missing piece.

## Theory part 2: Turing completeness
- Pérez 2019: full encoder+decoder transformer is Turing complete (arbitrary precision; else finite-state). Their construction (B.4) uses FULL attention at decoder to retrieve, in one step, the symbol last written at the next head cell ℓ(j+1) via argmin |<Q,K>| over all past steps.
- BigBird: replace with sparse. Exploit associativity of min/max: min over {0..t} = nested binary mins. Decoder graph D: for j∈N+, 1≤k≤j+1, edges (j(j+1)/2+k, k(k+1)/2) [random-type] and (j(j+1)/2+k, j(j+1)/2+k) self/local-type. Left-to-right (causal). One TM step now spans ~O(√i) transformer (intermediate) steps that aggregate the running min; g(i)=⌊(−1+√(1+8i))/2⌋ maps transformer step→TM step, h(i)=g(i+1)−g(i) is a compute-vs-intermediate indicator.
- 4 decoder layers per construction: L1 cross-attn to encoder + FFN simulates transition δ(q,s)→(q',v,m); L2 FFN updates head position c^{g(i)+1}=c+m; L3 NEW "switching" layer uses h(i) to either advance state (compute node) or copy previous (intermediate node) — needed because intermediate steps must hold state while aggregating; L4 sparse self-attn does one step of the nested-min to find last-written symbol. Final transform F builds y_{r+1}. ⇒ THEOREM: sparse attn with O(n) inner products is Turing complete.

## Theory part 3: Limitations (no free lunch)
- Task 1: given n unit vectors, output for each j its furthest vector u_{j*}, j*=argmax_k‖u_k−u_j‖². For unit vectors ‖u_k−u_j‖²=2−2<u_k,u_j> ⇒ argmax = argmin<u_k,u_j>.
- Full attention: 1 layer. Embed x_i=[u_i;0]. Q([a;b])=−a, K([a;b])=a, V([a;b])=[0;a]. Then attn picks argmax_j<−u_i,u_j>=argmin<u_i,u_j>=furthest; a_i=[u_i;u_{i*}]. O(1) layers.
- Sparse lower bound via Orthogonal Vectors Conjecture: can't decide if min inner product among n boolean vectors is 0 in O(n^{2-ε}) for d≥c log n. If a sparse net (Õ(n) edges, d=Θ(log²n)) solved Task 1 in l layers, total time Õ(n l d³); checking <u_i,u_{i*}>=0 then solves OV in Õ(n) more. l=O(n^{1-ε}) ⇒ OV in Õ(n^{2-ε}), contradiction. ⇒ need Ω̃(n^{1-o(1)}) layers. So sparsity has a real cost on some tasks.

## Implementation (HF/google-research): block-sparse
- Sparse mat-mul is inefficient on GPU (Gray 2017). Solution: BLOCKIFY. Block size b. Reshape Q,K to ⌈n/b⌉×b×d. 
- Block-diagonal scores: A_{jst}=Σ_u Q'_{jsu}K'_{jtu}, cost O(nbd).
- Window: make w copies of blocked key tensor, roll copy j by j blocks (circular shift), stack ⇒ each query block sees w neighbor key blocks, no gather. 
- Global: always concatenate first g key blocks (fixed).
- Random: r random key blocks per query block via gather (r=3 in experiments, small).
- Final packed key tensor K'': ⌈n/b⌉×(g+w+r)b×d; multiply Q'·K'' cost O(n(g+w+r)bd) — dense, GPU-friendly. Reshape to scatter back to BigBird pattern.
- HF code: forward computes Q,K,V; bigbird_block_sparse_attention splits queries into 5 parts: first block (global, attends all), 2nd block (window+global+random), middle blocks q[2:-2] (sliding band via rolled keys + global first/last + gathered random), 2nd-last, last block. _bigbird_block_rand_mask picks random blocks avoiding the 2 global, the window neighbors, and self.

## Design-decision → why
- graph view of attention → lets us import graph sparsification theory; "drop edges" is exactly the lever.
- random edges → spectral gap / small-world short paths → rapid mixing → few hops for any-to-any info flow with O(n) edges.
- window edges → locality of reference (Clark 2019) + small-world high clustering; keep all local edges (no rewire) for hardware.
- global tokens → demanded by BOTH the empirical sanity check (R+W<BERT) and the universality proof (star graph): the contextual-mapping construction needs one token that sees everyone and is seen by everyone. ITC vs ETC: ITC reuses existing tokens (no size change); ETC adds CLS-like tokens (more capacity, better empirically).
- one global token suffices for theory (star); experiments use 2 global blocks.
- block-sparse (block size b) → GPU can't do fine-grained sparse mat-mul; blockify + roll + gather turns it into dense tensor products.
- r=3 random blocks → small enough that gather cost is negligible, large enough for mixing.
- scale 1/√d → standard dot-product attention numerical stability (Vaswani).
