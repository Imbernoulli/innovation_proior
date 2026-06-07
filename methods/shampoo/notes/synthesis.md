# Shampoo — synthesis notes (Phase 1.5)

## Pain point / research question
- Preconditioned gradient methods (Newton, quasi-Newton, full-matrix AdaGrad) are powerful but the preconditioner is a d×d matrix. For a weight *matrix* W ∈ R^{m×n}, flattening gives d=mn, so full-matrix AdaGrad maintains an (mn)×(mn) preconditioner H_t = (Σ_s g_s g_s^T)^{1/2} with g=vec(G). Memory m²n², compute O(m³n³) for the inverse root. Infeasible.
- Diagonal AdaGrad/Adam are cheap (O(d) memory) and dominate practice, but throw away all cross-coordinate (curvature) structure.
- Sketching / sub-sampled Newton: super-linear memory/compute for a fine enough approximation; impractical at scale.
- GOAL: capture more than diagonal curvature, at near-first-order cost, by exploiting the *tensor* structure of parameters (matrices for FC/multiclass layers, order-4 tensors for conv).

## The object being optimized (OCO framing)
- Adaptive Online Mirror Descent: w_{t+1} = argmin_w { η g_t·w + ½‖w−w_t‖²_{H_t} }; on R^d this is w_{t+1}=w_t − η H_t^{-1} g_t.
- Regret bound (Lemma, standard): R_T ≤ (1/2η) Σ (‖w_t−w*‖²_{H_t} − ‖w_{t+1}−w*‖²_{H_t}) + (η/2) Σ (‖g_t‖*_{H_t})².
- Adaptive-regularization lemma (Gupta-Koren-Singer 2017 / FTL-BTL): with M_t=Σ g g^T and H_t=argmin_{H≻0} {M_t•H^{-1}+Φ(H)}, Σ(‖g_t‖*_{H_t})² ≤ Σ(‖g_t‖*_{H_T})² + Φ(H_T)−Φ(H_0).
- Full-matrix AdaGrad: choose Φ(H)=tr(H), gives H_t ∝ (Σ g g^T)^{1/2}, regret ~ tr((Σ g g^T)^{1/2}). This is the gold standard we want to approximate.

## The Kronecker idea (core derivation)
- Maintain L_t = εI_m + Σ G G^T (m×m), R_t = εI_n + Σ G^T G (n×n). Cheap: m²+n² memory, O(m³+n³) compute.
- Want a full preconditioner H_t = L_t^{1/2} ⊗ R_t^{1/2} (mn×mn) but never form it.
- vec identity: (L ⊗ R^T) vec(G) = vec(L G R). So applying H_t^{-1}=(L_t^{1/2}⊗R_t^{1/2})^{-1}=L_t^{-1/2}⊗R_t^{-1/2} to g=vec(G) equals vec(L_t^{-1/2} G R_t^{-1/2}). Wait — careful: the identity uses R^T. R_t symmetric ⇒ R_t^{-1/2} symmetric ⇒ R^T=R. So H_t^{-1} g = vec(L_t^{-1/2} G R_t^{-1/2}).
- BUT the update uses L_t^{-1/4} G R_t^{-1/4}, NOT −1/2 on each side. Why?

### The −1/4 algebra (the load-bearing piece)
- The single equivalent preconditioner is H_t = L_t^{1/4} ⊗ R_t^{1/4} (note: +1/4 powers form H, since H is what plays the role of (Σgg^T)^{1/2}). Apply H_t^{-1} = L_t^{-1/4} ⊗ R_t^{-1/4} to g:
  H_t^{-1} g = (L_t^{-1/4} ⊗ R_t^{-1/4}) vec(G) = vec(L_t^{-1/4} G R_t^{-1/4}). (since R_t^{-1/4} symmetric.) ✓ matches Algorithm 1.
- So the preconditioner matrix is H_t = L_t^{1/4} ⊗ R_t^{1/4}. Why 1/4 and not 1/2?
  - Full-matrix AdaGrad's preconditioner is (Σ_s g_s g_s^T)^{1/2}, i.e. the **1/2 power** of accumulated outer products. We want H_t to lower-bound / track that.
  - Key Lemma (moo-lower): εI_{mn} + (1/r)Σ g g^T ⪯ L_T^{1/2} ⊗ R_T^{1/2}, where L_T=εI+ΣGG^T, R_T=εI+ΣG^TG.
  - So L_T^{1/2} ⊗ R_T^{1/2} is the analogue of Σ g g^T (the *un-rooted* gradient covariance), NOT of its square root. To get a preconditioner ∝ (Σ g g^T)^{1/2}, we take the square root of L_T^{1/2}⊗R_T^{1/2}:
    H_t = (L_t^{1/2} ⊗ R_t^{1/2})^{1/2} = L_t^{1/4} ⊗ R_t^{1/4}.  → the **1/4 falls out as ½·½**.
  - Two factors of ½: one ½ from splitting the gradient covariance per-axis (the L^{1/2}⊗R^{1/2} bound on Σgg^T), the second ½ from the AdaGrad square-root that turns a covariance into a preconditioner.
  - Intuition check (step-size decay): L_t, R_t grow like t (sum of t outer products), so L_t^{1/4} ~ t^{1/4}, R_t^{1/4} ~ t^{1/4}, product step scale ~ t^{-1/2} — the canonical O(1/√t) stochastic decay. ✓

### proof of moo-lower (kron-base lemma)
- For single G rank ≤ r: g=vec(G)=Σ_i σ_i (u_i⊗v_i) (SVD G=Σσ_i u_i v_i^T, using vec(uv^T)=u⊗v).
- (Σ w_i)(Σ w_i)^T ⪯ r Σ w_i w_i^T (Cauchy-Schwarz / convexity of α²).
- ⇒ gg^T ⪯ r Σ σ_i² (u_i u_i^T)⊗(v_i v_i^T). Bound v_i v_i^T ⪯ I_n ⇒ (1/r) gg^T ⪯ (GG^T)⊗I_n. Symmetrically (1/r) gg^T ⪯ I_m⊗(G^TG).
- Sum over t, add ε: εI+（1/r)Σgg^T ⪯ I_m⊗B_n and ⪯ A_m⊗I_n where A_m=εI+ΣGG^T, B_n=εI+ΣG^TG.
- I_m⊗B_n and A_m⊗I_n commute. Geometric-mean operator monotonicity (Ando 2004, commuting case): X^{1/2}Y^{1/2} ⪯ ... gives (I_m⊗B_n)^{1/2}(A_m⊗I_n)^{1/2} = (I_m⊗B_n^{1/2})(A_m^{1/2}⊗I_n) = A_m^{1/2}⊗B_n^{1/2}. Done: εI+(1/r)Σgg^T ⪯ A_m^{1/2}⊗B_n^{1/2}.

### regret proof (matrix case)
- Update ≡ w_{t+1}=w_t − η H_t^{-1} g_t with H_t=L_t^{1/4}⊗R_t^{1/4}.
- H_t monotone increasing (Kron monotone from L,R increasing) ⇒ first term of MD bound telescopes: Σ(w_t−w*)^T(H_t−H_{t-1})(w_t−w*) ≤ D² Σ tr(H_t−H_{t-1}) = D² tr(H_T).
- Second term: define Ĥ_t=(rεI+Σgg^T)^{1/2}. From moo-lower + sqrt monotonicity: Ĥ_t ⪯ √r H_t. Adareg lemma with Φ(H)=tr(H)+rε tr(H^{-1}), minimizer = Ĥ_t (since tr(AX+X^{-1}) min at X=A^{-1/2}); gives Σ(‖g_t‖*_{Ĥ_t})² ≤ 2 tr(Ĥ_T). Chain: Σ(‖g_t‖*_{H_t})² ≤ √r Σ(‖g_t‖*_{Ĥ_t})² ≤ 2√r tr(Ĥ_T) ≤ 2r tr(H_T).
- Plug in, η=D/√(2r): R_T ≤ (D²/2η + ηr) tr(H_T) = √(2r) D tr(H_T) = √(2r) D tr(L_T^{1/4}) tr(R_T^{1/4}) (since tr(A⊗B)=tr A tr B).
- Each tr(L_T^{1/4}) ~ O(T^{1/4}) under Lipschitz ⇒ product O(√T), optimal.

## Tensor generalization (order-k)
- Per axis i: contraction G^{(i)} = mat_i(G) mat_i(G)^T (n_i×n_i). Maintain H^i_t = εI + Σ G^{(i)}.
- Update: precondition mode i by (H^i_t)^{-1/(2k)} via tensor-matrix product ×_i, for all i. W_{t+1}=W_t − η G̃_t.
- Exponent −1/(2k): equivalent single preconditioner H_t = (⊗_i H^i_t)^{1/(2k)}; for k=2 recovers −1/4. Same two-½ story generalized: 1/k from splitting covariance across k axes (moo-tensor: εI+Σgg^T ⪯ r ⊗_i (H^i_T)^{1/k}), times ½ from AdaGrad root ⇒ 1/(2k).
- moo-tensor proof: kron-base per matricization gives (1/r_i)vec(mat_i G)vec(mat_i G)^T ⪯ G^{(i)}⊗I; transpose-tensor lemma lifts to (1/r_i)gg^T ⪯ ⊗_{j<i}I ⊗ G^{(i)} ⊗ ⊗_{j>i}I; sum, the k inequalities commute, geometric mean (Σα_i=1, α_i=1/k) ⇒ result with r=(∏r_i)^{1/k}.
- vec-kron-tensor: (⊗_i M_i) vec(G) = vec(G ×_1 M_1 ... ×_k M_k). Induction on k.

## Inverse p-th root computation
- H^{-1/(2k)} computed by SVD/eig: H=Σλ_i u_i u_i^T (symmetric PSD) ⇒ H^α=Σλ_i^α u_i u_i^T. Original impl: SVD, raise singular values to the power. (Scalable variants use Newton/coupled-iteration but that's later.)

## Implementation details (from paper + canonical code)
- Treat each tensor as separate variable ⇒ block-diagonal preconditioner across tensors; only intra-tensor correlations.
- Heuristics: delayed root recomputation every 20–100 steps (amortize SVD cost); momentum on the gradient (Ḡ_t=αḠ_{t-1}+(1−α)G_t, α=0.9).
- Diagonal Shampoo: replace L,R updates with diag(GG^T), diag(G^TG); O(m+n) memory, O(mn) time. Auto-activated when a dimension exceeds ~1200. Regret bound uses D_∞ (entrywise ℓ∞) in place of D.
- Canonical PyTorch reference (moskomule/shampoo.pytorch): loops dims, precond_{i} += mat_i(g) mat_i(g)^T, inv = precond^{-1/order}, applies via matmul on each mode; momentum buffer; update_freq for inverse recompute. Matches Algorithm 2.
- Google scalable_shampoo: exponent_for_preconditioner = 2*rank ⇒ -1/(2k); statistics via tensordot contraction over all-but-i axes; later additions (grafting, partitioning, block-diagonal) are NOT part of the original method.

## Design-decision → why
- Per-axis Kronecker factors (vs full mn×mn): the only way to get below super-linear memory while keeping a *full* (non-diagonal) matrix per mode. Captures intra-axis correlations diagonal AdaGrad misses.
- L=ΣGG^T, R=ΣG^TG: these are exactly the two matricization contractions of the gradient; they're the per-axis second moments, the natural Kronecker factors of the flattened gg^T (proved by moo-lower).
- −1/4 (matrix), −1/(2k) (tensor): ½ (per-axis covariance split) × ½ (AdaGrad sqrt); also gives O(1/√t) decay.
- ε I init: ridge so roots/inverses well-defined; also gives the εI_{mn} term in the lower bound so small eigenvalues don't vanish.
- block-diagonal across tensors: oblivious to architecture, only needs tensor shapes; contrast K-FAC which needs network structure & Fisher sampling.
- delayed root / momentum: pure runtime & convergence heuristics, not core.
