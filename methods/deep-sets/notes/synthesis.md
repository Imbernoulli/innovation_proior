# Deep Sets — synthesis notes (Phase 1.5)

## Pain point / research question
Standard ML (regression/classification) assumes fixed-dimensional vector inputs of a fixed ordering. Many tasks have inputs that are **sets**: an unordered collection of variable size. Examples: population-statistic estimation (entropy/MI from a sample set), point clouds (LiDAR), set expansion (given {lion,tiger,leopard} find jaguar), anomaly detection in a set, galaxy clusters → red-shift. A correct model must be **permutation invariant** (output unchanged under reordering) for set→scalar, or **permutation equivariant** (output reorders the same way) for set→set / per-element labels. Question: what is the *most general* form a permutation-invariant function can take? And what is the most general permutation-equivariant *linear layer*?

## Key objects
- Input set X = {x_1,...,x_M}, x_m ∈ 𝔛, i.e. domain is power set 2^𝔛.
- Property 1 (invariance): f({x_1..x_M}) = f({x_π(1)..x_π(M)}) ∀ permutation π.
- Equivariance: f(πx) = π f(x).

## MAIN STRUCTURE THEOREM (Thm 2, countable case)
f: 2^𝔛→ℝ on a set from a **countable** universe is permutation-invariant **iff** f(X) = ρ(Σ_{x∈X} φ(x)).
- **Sufficiency**: ρ(Σφ(x)) — the sum is order-independent (addition is commutative/associative), so any f of this form is invariant. Trivial direction.
- **Necessity** (the construction): elements countable ⇒ ∃ injective c: 𝔛→ℕ. Set φ(x) = 4^{-c(x)}. Then Σ_{x∈X} φ(x) is a **unique** code for each set X (think base-4 expansion: each element contributes a distinct negative power of 4; a *set* has each element 0 or 1 times, so digits are 0/1 in base 4 — no carries, sum is injective on subsets). So the mapping X ↦ Σφ(x) is injective; define ρ = f ∘ (inverse of that encoding). Then f(X)=ρ(Σφ(x)). QED.
  - Why base 4 not base 2? With base 2, digits 0/1 are fine for a *set* (each element once), but a **multiset** could have an element twice → digit 2 → carry → collision. Base 4 leaves headroom (digits up to 3 with no carry) so the sum is injective even for small multisets; the key is a radix large enough that summing the chosen codes never carries. For pure sets base>2 works; base 4 is a safe canonical choice.

## Uncountable case (fixed size M) — Thm 7, via Newton–Girard
On [0,1]^M (fixed M), any continuous perm-invariant f = ρ(Σ φ(x_m)) with φ(x)=[1,x,x²,...,x^M] ∈ ℝ^{M+1}.
- Lemma 4: sum-of-power map E_q(X)=Σ x_m^q, q=0..M, is **injective** on the sorted simplex 𝒳={x_1≤...≤x_M}. Proof: E(u)=E(v) ⇒ power sums equal ⇒ by Newton–Girard the elementary symmetric polynomials a_m equal (a_m = (1/m)·det of the m×m lower-Hessenberg matrix of power sums) ⇒ the monic polynomials P_u(x)=Π(x−u_m), P_v(x)=Π(x−v_m) have identical coefficients ⇒ same roots ⇒ u=v (after sorting).
- Lemma 6: E has a **continuous inverse**. Range 𝒵=E(𝒳) is compact (continuous image of compact polytope). Coeffs a are polynomial (hence continuous) functions of power sums z. Curgus–Mawhin (Thm 5): roots depend continuously (homeomorphically) on coefficients. Compose ⇒ E^{-1} continuous.
- Thm 7: E:𝒳→𝒵 is a homeomorphism; set ρ(z)=f(E^{-1}(z)), continuous as composition. Sufficiency as before. φ independent of f.
- Connection: Kolmogorov–Arnold representation theorem gives ρ(Σ λ_m φ(x_m)) for *any* continuous multivariate f; dropping the coordinate-dependence λ_m is exactly what permutation-invariance buys you.
- Thm 9: universal approximation — polynomials dense by Stone–Weierstrass; symmetric polynomials = polynomials of power-sum/elementary-symmetric generators (Fundamental Theorem of Symmetric Functions / Chevalley–Shephard–Todd) ⇒ any continuous symmetric f approximable as ρ(Σφ(x)).

## EQUIVARIANT LINEAR LAYER (Lemma 3)
Standard layer f_Θ(x)=σ(Θx), Θ∈ℝ^{M×M}. It is permutation-equivariant **iff** Θ = λI + γ(11ᵀ), λ,γ∈ℝ (diagonal all equal, off-diagonal all equal).
- Reduction: equivariance σ(Θπx)=πσ(Θx). σ pointwise & bijective ⇒ Θπ = πΘ ∀π∈S_M, i.e. Θ **commutes with all permutation matrices**.
- (⇐) I and 11ᵀ each commute with any π; commutativity is linear ⇒ λI+γ11ᵀ commutes.
- (⇒) Suppose Θπ=πΘ ∀π.
  - Diagonal equal: take transposition π_{k,l}. π_{k,l}Θπ_{l,k}=Θ. Reading (l,l) entry: Θ_{k,k}=Θ_{l,l}. So all diagonal = λ.
  - Off-diagonal equal: for two off-diag positions (i,j),(i',j') with i≠i', j≠j': conjugate Θ by π_{j',j}π_{i,i'}; the (i,j) entry maps to Θ_{i',j'}, equate ⇒ Θ_{i,j}=Θ_{i',j'}. Degenerate cases (i=i' or j=j') handled by routing through a third index. So all off-diag = γ.
- Layer: f(x)=σ(λx + γ(11ᵀ)x) = σ(λx + γ·(Σ_m x_m)·1) — input plus broadcast pool. Sum doesn't depend on order ⇒ equivariant.
- Variation: replace sum-pool with maxpool: f(x)=σ(λIx + γ·maxpool(x)·1). At λ=γ the input to σ is "max-normalized"; performs better in some apps.
- Multi-channel (D→D'): f(X)=σ(XΛ − 11ᵀX Γ), Λ,Γ∈ℝ^{D×D'}. Minus sign is a reparameterization (γ→−γ); subtracting the pooled (mean/max) centers the features. Maxpool version σ(XΛ − 1·maxpool(X)·Γ). Reduced form: f(X)=σ(β + (X − 1·maxpool(X))Γ) — single weight Γ + bias β.
- Stacking equivariant layers stays equivariant; a final commutative pool over the set → invariant.

## Architecture (Deep Sets)
- **Invariant**: each x_m → φ(x_m) (MLP, shared weights); sum (or mean) over m; → ρ (MLP) → output. Optionally condition φ(x_m|z).
- Gradient: ∂_{w_φ} ρ(Σφ) = ρ'(Σφ)·Σ ∂_{w_φ}φ(x') — parameter tying across set members is forced, and the theorem says it's the *only* way.
- **Equivariant**: stack PermEqui layers (Γx − Λ·pool(x)), pointwise nonlinearity (ELU/Tanh), final pool → ρ.

## Canonical code grounding (manzilzaheer/DeepSets)
- DigitSum/text_sum.ipynb (Keras): Embedding → Dense(tanh) [=φ] → Lambda sum over set axis → Dense(1) [=ρ]. Invariant model, MAE loss, Adam.
- PointClouds/classifier.py (PyTorch): PermEqui2_max/mean: Γ=Linear(in,out), Λ=Linear(in,out,bias=False); forward: x = Γ(x) − Λ(pool(x)). PermEqui1: single Linear on (x − pool(x)). D / DTanh: 3 PermEqui layers + ELU/Tanh = φ; pool over set; ρ = Dropout+Linear+act+Dropout+Linear(→40 classes). Equivariant model.

## Baselines / lineage to elaborate (prior art)
- Pooling-over-members already used ad hoc: Shi 2015 (panorama pooling), Su 2015 (multi-view CNN pooling), Lopez-Paz 2016 (pooling for causality on a sample set), Hartford 2016 (payoff-matrix row/col invariance + pooling), Sukhbaatar 2016 (CommNet: a special case of the equivariant layer for multi-agent). None had the *characterization* (necessity) — they used pooling as a heuristic.
- Distribution/set kernels: Poczos 2012/2013, Muandet 2012/2013, Szabo 2016, Oliva 2013 — support distribution machines f(p)=Σ α_i y_i K(p_i,p)+b with estimated kernel K̂(p,q)=(1/MM')Σ k(x_i,y_j). Gap: O(N²) kernel matrix, doesn't scale (matrix inversion infeasible for N>16384); fixed kernel, not learned end-to-end.
- de Finetti / exchangeability (Bayesian sets, Ghahramani 2005): exchangeable models factor p(X)=∫Πp(x_m|θ)p(θ)dθ; with exp-family + conjugate prior the marginal = exp(h(α+Σφ(x_m),...)−h(α,...)) — already the ρ(Σφ) shape.
- Vinyals 2015 "Order matters": treat set as sequence, search for a good ordering — concedes ordering matters for RNNs, the opposite of building invariance in.
- RNN/LSTM/GRU on the sequence: not invariant; generalize poorly to larger sets (digit-sum experiment motivation).
- Group-equivariance line (Cohen & Welling 2016 G-CNN, Gens 2014, Ravanbakhsh 2017): equivariance to general groups via parameter-sharing; permutation group S_M is the special case here.

## Evaluation settings (pre-method, no outcomes)
Population-statistic estimation (entropy/MI of Gaussians, rotation/correlation/rank-1/random covariance; MSE vs #sets); sum-of-digits (text & MNIST8m image, train len ≤10 test up to 100; accuracy = exact match); point-cloud classification (ModelNet40, 40 classes, 100/1000/5000 points; accuracy); red-shift regression (redMaPPer clusters, 17 features, scatter metric); set expansion (LDA top-words, concept-set retrieval / image tagging; recall@k); set anomaly detection (CelebA). Baselines: SDM (RBF), LSTM/GRU, 3DShapeNets/VoxNet/MVCNN, redMaPPer/MLP.

## Design-decision → why
- Sum (not concat/RNN): only commutative+associative reduction that is the *characterized* universal form; concat breaks invariance; RNN is order-dependent.
- φ then sum then ρ (not sum-then-MLP of raw x): raw sum Σx_m loses info; φ lifts each element to a rich code so the sum is injective/separating, ρ decodes. Power-series φ(x)=[1,x,...,x^M] is the constructive witness.
- 4^{-c(x)} base-4 code: injective set-sum without carries; headroom over base 2 guards against multiset collisions.
- Equivariant layer two-param tied form: forced by commuting-with-all-permutations; any other Θ breaks equivariance.
- Subtract pool (X − pool(X)) / minus sign: reparam of λI+γ11ᵀ; centering improves conditioning; max-normalization at λ=γ.
- mean vs sum pool: mean = sum/M handles variable M without scale blowup; sum preserves count info (digit-sum needs additive scale). Both commutative.
- maxpool variation: better in some apps (point clouds) — max is robust to set-size and outliers, "max-normalizes" input to σ.
- Dropout 0.5 in ρ, grad clipping: regularization for large variable-size sets; standard.
- ELU/Tanh: smooth pointwise nonlinearity preserving equivariance (must be pointwise).
