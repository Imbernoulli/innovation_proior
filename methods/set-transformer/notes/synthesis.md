# Set Transformer — synthesis (arXiv 1810.00825, verified — Lee, Lee, Kim, Kosiorek, Choi, Teh 2018, ICLR2019)

## Pain point / goal
Set-input problems: input is a SET, target is permutation-invariant (multiple-instance learning, 3D shape from points, set operations, amortized clustering, few-shot/meta-learning where input set = a task's dataset). Model must be (1) permutation invariant, (2) handle any set size. Feed-forward nets fail both; RNNs are order-sensitive.
Existing solution = set pooling (Deep Sets, Zaheer 2017; Edwards & Storkey 2017): net({x_i}) = rho(pool({phi(x_i)})), phi per-element, pool = sum/mean/max, rho post. Zaheer proved universal for permutation-invariant functions (sum pool, continuous rho/phi). Encoder-decoder: phi = encoder (acts independently on each element), rho∘pool = decoder.
LIMITATION: phi processes each element INDEPENDENTLY -> interactions between elements are discarded. Some problems become unnecessarily hard. Amortized clustering (map a point set to cluster centers): must assign points to clusters while modeling EXPLAINING-AWAY (clusters shouldn't explain overlapping subsets). A pooling net can only learn to QUANTIZE space, and the quantization CAN'T depend on set contents -> underfits. Want to model pairwise/higher-order interactions during encoding, and a learnable, content-dependent pooling.

## Background: attention
Att(Q,K,V; omega) = omega(Q K^T) V. Q in R^{n x d_q}, K in R^{n_v x d_q}, V in R^{n_v x d_v}. QK^T (n x n_v) = pairwise similarity; omega = activation (scaled softmax, omega = softmax(./sqrt(d))). Output = weighted sum of V.
Multihead (Vaswani 2017): project Q,K,V to h subspaces of dim d_q/h etc, apply Att per head, concat, linear W^O. Params lambda = {W_j^Q, W_j^K, W_j^V}, W^O in R^{h d_v^M x d}. Default d_q=d_v=d, d_q^M=d_v^M=d/h.

## Building blocks (all are parameterized NN blocks, not fixed functions)
- **MAB (Multihead Attention Block)** — adaptation of Transformer encoder block WITHOUT positional encoding and dropout:
  H = LayerNorm(X + Multihead(X, Y, Y; omega))
  MAB(X, Y) = LayerNorm(H + rFF(H))
  rFF = row-wise feedforward (per-instance, identical). LayerNorm (Ba 2016).
- **SAB (Set Attention Block)** = MAB(X, X): self-attention WITHIN the set X -> set of equal size. Output encodes pairwise interactions; stack to encode higher-order. (Reduces to a residual block when Q=K=V=X but learns more due to the linear projections inside heads.) Complexity O(n^2).
- **ISAB (Induced Set Attention Block)** — fixes O(n^2). Add m trainable INDUCING POINTS I in R^{m x d} (params of the block).
  H = MAB(I, X) in R^{m x d}       (inducing points attend to X -> low-dim summary)
  ISAB_m(X) = MAB(X, H) in R^{n x d}  (X attends back to H -> n outputs)
  Analogous to low-rank/autoencoder bottleneck (project X to H of size m, reconstruct). Inducing points learn global structure explaining X (e.g. amortized clustering: grid points on 2D plane; encoder compares query points indirectly via proximity to them). Complexity O(nm), m small hyperparameter. Both attentions are between sets of size m and n.
  Inducing-point idea from sparse GP (Snelson 2005) / Nystrom (Fowlkes 2004); also like m memory cells accessed by attention. ISAB = inversion of differential neural dictionary (Pritzel 2017): here queries I are stored, inputs are key-value.
- **PMA (Pooling by Multihead Attention)** — learnable, attention-based pooling. k trainable SEED vectors S in R^{k x d}:
  PMA_k(Z) = MAB(S, rFF(Z))  -> set of k items.
  k=1 usually; k>1 for problems needing k correlated outputs (e.g. clustering, k cluster centers). 
  After PMA with k>1, apply SAB to model interactions / explaining-away among the k outputs: H = SAB(PMA_k(Z)).
  WHY attention pooling: instance influence on target is not equal — e.g. target = max of a set: recoverable from ONE element (the largest), so finding+attending to it is advantageous. Average/max pooling can't do content-dependent weighting.

## Overall architecture
Encoder: X -> Z in R^{n x d}, stack of SABs (O(l n^2)) or ISABs (O(l n m)):
  Encoder(X) = SAB(SAB(X))   or   ISAB_m(ISAB_m(X)).
Decoder: aggregate Z -> k vectors -> FF:
  Decoder(Z) = rFF(SAB(PMA_k(Z))) in R^{k x d}, with PMA_k(Z) = MAB(S, rFF(Z)).

## Analysis (proofs, worked in reasoning)
- **Permutation equivariance** of SAB, ISAB: SAB(X) and ISAB_m(X) are permutation equivariant (permuting input rows permutes output rows identically). MAB(X,Y) is equivariant in X (Multihead(X,Y,Y) applies row-wise to X's queries; LayerNorm + rFF row-wise). For ISAB: H=MAB(I,X) is permutation INVARIANT in X (X enters only as keys/values, softmax sum over them is order-independent — wait, careful: H depends on X as K,V; permuting X permutes K,V rows but the attention sum over them is invariant, so H is invariant to permutations of X); then ISAB_m(X)=MAB(X,H) is equivariant in X (X is the query). So composition equivariant.
- **Permutation invariance** of Set Transformer: encoder equivariant + PMA invariant (S queries are fixed, attend over Z; permuting Z permutes K,V but output invariant) => whole model invariant.
- **Universal approximation** of permutation-invariant functions:
  Lemma: mean is a special case of softmax dot-product attention (s=0 => softmax(0)=uniform => mean). 
  Lemma: decoder can express M_p(z) = (mean_i z_i^p)^{1/p} (rFF front encodes z->z^p, MAB does the mean, rFF back z->z^{1/p}; h=d heads each 1-D, W projections pick dimensions, W^O=I).
  Lemma: PMA can express SUM pooling (seed s=0, omega = 1 + f with f(0)=0 [identity/sigmoid/relu] => attention weights all 1 => output = sum of values).
  Theorem (Zaheer 2017): rFF(sum(rFF(.))) is a universal approximator of permutation-invariant functions.
  Proposition: Set Transformer is universal. Proof: set W^O=0 in every SAB/ISAB -> ignore all pairwise interaction terms -> encoder reduces to instance-wise rFF, Z=rFF(X); decoder can do rFF(sum(Z)); invoke Zaheer's theorem. (Attention isn't NEEDED for universality but is crucial empirically.)

## Design decisions -> why
- Self-attention encoder (SAB): model pairwise + higher-order interactions during encoding, unlike independent per-element phi.
- Inducing points / ISAB: O(n^2) -> O(nm) for large sets, keep representational power; low-rank bottleneck that learns global structure.
- PMA (attention pooling, learnable): content-dependent weighting (max example); seeds adapt to the problem unlike fixed mean/max.
- k seeds + SAB after PMA: produce k correlated outputs, model explaining-away among them (amortized clustering).
- MAB = Transformer encoder block minus positional encoding (a SET has no order — positions would break permutation invariance) and minus dropout.
- Scaled softmax omega = softmax(./sqrt(d)): stable attention (Vaswani's 1/sqrt(d)).

## Code grounding (juho-lee/set_transformer modules.py, verified)
- MAB(dim_Q,dim_K,dim_V,num_heads,ln): fc_q,fc_k,fc_v Linear; split heads by concat on batch dim; A=softmax(Q_ K_^T / sqrt(dim_V)); O = concat(Q_ + A V_); ln0; O = O + relu(fc_o(O)); ln1. (rFF here = single Linear + ReLU residual.)
- SAB = MAB(X,X). ISAB: I = Parameter(1,num_inds,dim_out) xavier; mab0=MAB(dim_out,dim_in,dim_out); mab1=MAB(dim_in,dim_out,dim_out); H=mab0(I.repeat, X); return mab1(X,H). PMA: S=Parameter(1,num_seeds,dim) xavier; mab; forward = mab(S.repeat, X).
- Example arch (cls): enc = ISAB(dim_in,dim_hidden,heads,num_inds) x2; dec = PMA(dim_hidden,heads,num_outputs) -> SAB -> SAB -> Linear(dim_hidden, dim_out).
