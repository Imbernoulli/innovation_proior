Let me start from where the time actually goes. I have a tall matrix A ∈ R^{n×d}, n ≫ d, rank r, and I want least squares: x' with ‖Ax'−b‖₂ ≤ (1+ε) min_x ‖Ax−b‖₂. The whole randomized toolkit reduces this to one move — pick a random linear map S : R^n → R^t with t ≪ n, form the sketch SA and Sb, and solve the tiny problem argmin_x ‖SAx − Sb‖₂. That works as soon as S is a *subspace embedding* for the column space of [A | b]: ‖SAx − Sb‖ = (1±ε)‖Ax − b‖ for every x simultaneously, equivalently ‖SUz‖ = (1±ε)‖z‖ for all z when U is an orthonormal basis of the relevant column space. Same story for low rank, leverage scores, ℓ_p. So the entire game is: build a good subspace embedding fast.

And here is the thing that's been bugging me. With a dense Gaussian S, computing SA costs Θ(ndt). With a Fast-JL or a subsampled Hadamard transform — the clever structured sketches — it drops to Θ(nd log n). That is essentially optimal *for a dense A*. But look at the matrices people actually feed in: document-term, adjacency, recommender, web. They are wildly sparse — nnz(A), the number of nonzeros, is often O(n), not nd. And every one of these sketches reads all nd entries of A, even though almost all of them are zero. I'm paying nd to process O(n) numbers. That's the waste. The dream is a leading term of O(nnz(A)): touch each nonzero a constant number of times, plus a lower-order additive overhead in d, ε that doesn't grow polynomially with n.

So what does it take for S·A to cost O(nnz(A))? Each nonzero A_{ij} contributes to SA only through column j of S. If column j of S has c_j nonzeros, then A_{ij} fans out to c_j entries of the sketch. Total work is Σ_j c_j · (nonzeros of A in column j) — and if every column of S had a constant number of nonzeros, this would be O(nnz(A)). The extreme version: make S have *exactly one nonzero per column*. Then applying S is nothing but "for each nonzero A_{ij}, add it (with a sign) to a single bucket." That is the cheapest sketch imaginable. Let me write it concretely. Pick a hash h : [n] → [t], and put a single ±1 in column i at row h(i). Formally S = ΦD, where Φ ∈ {0,1}^{t×n} has Φ_{h(i),i} = 1 and is zero elsewhere, and D is diagonal with D_{ii} ∈ {±1} uniform. Then (SA)_{h(i),·} accumulates ±A_{i,·} over all rows i hashing to the same bucket. Apply cost: one add per nonzero. O(nnz(A)). 

Why the random signs? Without them, ‖Sy‖² = Σ_j (Σ_{i: h(i)=j} y_i)² and the off-diagonal of that square gives 2 Σ_{i<i', h(i)=h(i')} y_i y_{i'} — a bias that does not vanish in expectation. With the signs, ‖ΦDy‖² = Σ_j (Σ_{i:h(i)=j} D_{ii} y_i)², and the cross terms y_i y_{i'} D_{ii} D_{i'i'} have mean zero because E[D_{ii} D_{i'i'}] = 0 for i ≠ i'. So E[‖ΦDy‖²] = Σ_i y_i² = ‖y‖². Unbiased. That's the same random-sign-hash trick Alon–Matias–Szegedy used for second-moment estimation, and the same structure as the CountSketch data structure of Charikar–Chen–Farach-Colton for finding frequent items — one signed counter per element. I've literally just written down CountSketch and called it a sketch matrix.

Now the obvious worry, and I should take it seriously before getting excited: does this thing even work as an embedding? Let me try the standard proof and see if it goes through. The standard recipe for "S preserves a whole subspace": (1) show ‖Sy‖ = (1±ε)‖y‖ for a *fixed* y with failure probability e^{−Ω(r)}; (2) put an ε-net on the unit sphere of the r-dimensional column space — it has e^{O(r)} points; (3) union bound over the net; (4) extend to all vectors by linearity. Step (3) needs the per-point failure to beat the net size e^{O(r)}, i.e. failure ≤ e^{−Ω(r)}.

Does ΦD give a fixed vector failure of e^{−Ω(r)}? Take a single adversarial vector — say y = e₁, a standard basis vector. Then ‖ΦDy‖² = 1 always; that's fine. But take y with two big equal coordinates, y = (e₁ + e₂)/√2. With probability 1/t the two coordinates collide in the same bucket, and then ‖ΦDy‖² = (D₁₁/√2 + D₂₂/√2)² which is either 2 or 0 — a 100% distortion — with probability 1/t. So the failure probability is only ~1/t, polynomial, not e^{−Ω(r)}. The union bound over e^{O(r)} net points needs t ≥ e^{Ω(r)}, which is absurd. The standard proof is dead. CountSketch simply does *not* preserve the norms of an arbitrary set of e^{O(r)} worst-case vectors. Wall.

So either CountSketch is not an embedding, or the standard proof is the wrong proof. Let me look at what just killed me. The fatal configuration was two big coordinates colliding. The standard net argument treats the e^{O(r)} net points as if they were arbitrary adversarial vectors — and against arbitrary vectors with heavy coordinates that can be anywhere, CountSketch is hopeless. But the net points are not arbitrary. They all live in one fixed d-dimensional (rank r) subspace, C(A). Maybe that subspace constrains *where* the heavy coordinates can be, and that constraint is exactly what I'm throwing away by treating the net points as worst-case.

Let me chase that. Where can a unit vector y ∈ C(A) have a large coordinate? Write y = Ux for a unit x (U orthonormal basis). Then y_i = U_{i,·} · x, so by Cauchy-Schwarz y_i² ≤ ‖U_{i,·}‖² ‖x‖² = u_i, the i-th leverage score. So coordinate i of any unit column-space vector is bounded by √u_i — and this bound is *fixed*, it depends only on the subspace, not on which y I picked. Now Σ_i u_i = r. So if I call a coordinate "heavy" when u_i > α, the number of heavy coordinates is at most r/α (since each contributes more than α to a sum that totals r). The set H of coordinates that could *ever* be large, over all unit y in the subspace, is a single fixed set of size ≤ r/α. That's the structural fact the net argument ignored. The heavy coordinates can't roam — they're pinned to the large leverage scores.

This reframes the whole problem. The collision disaster (two heavy coordinates landing in one bucket) only involves coordinates in H, and |H| is small. So if I make t large enough that the |H| heavy coordinates get hashed *without any collision among themselves*, the disaster can't happen for the heavy part. By the birthday bound, |H| items into t buckets have a collision with probability ≤ |H|²/t, so t ≳ |H|² makes them perfectly hashed with good probability. On that event, Φ restricted to the heavy coordinates is just a signed permutation — an exact isometry: ‖ΦD y^H‖ = ‖y^H‖ exactly, where y^H is y restricted to H. The heavy part costs nothing.

That leaves the light part, y^L = y restricted to [n]\H, every coordinate of which has y_i² ≤ u_i ≤ α — small ∞-norm. And small ∞-norm is exactly the regime where a sparse hashing map *does* concentrate. This is the heavy/light split from streaming norm estimation: track the heavy coordinates exactly (here: perfect hashing), and handle the many small coordinates by random aggregation. So the plan is: split y = y^H + y^L by a leverage threshold; perfect-hash the heavy part to get an exact isometry; bound the light part by concentration; bound the cross term; sum up. Let me actually do each piece, because the constants and the threshold are where all the difficulty hides.

Set up notation that will make the threshold visible. Order the rows so the leverage scores u_i are non-increasing (just for the analysis; the algorithm never sorts). Pick a threshold T and let s = min{i : u_i ≤ T} — so coordinates 1..s−1 are the heavy ones (u_i > T) and s..n are light (u_i ≤ T). For a vector u write u_{s:n} for u with coordinates < s zeroed, etc.

Take the heavy part first. Let E_B be the event that h(i) ≠ h(i') for all i, i' < s — the heavy coordinates are perfectly hashed. Each pair collides with probability 1/t, so by a union bound the failure probability is δ_B ≡ 1 − Pr[E_B] ≤ s²/t. On E_B, distinct heavy coordinates go to distinct buckets, so ‖ΦD y_{1:(s−1)}‖² = Σ_{i<s} (D_{ii} y_i)² = Σ_{i<s} y_i² = ‖y_{1:(s−1)}‖². Exact. Good. This already tells me I'll need t ≳ s².

Now the light part. I want ‖ΦD y_{s:n}‖² ≈ ‖y_{s:n}‖². Write z = diag(D) ∈ {±1}^n. Then ‖ΦDy^L‖² = Σ_j (Σ_{i≥s, h(i)=j} z_i y_i)² = Σ_{i,i'≥s} z_i z_{i'} y_i y_{i'} 𝟙[h(i)=h(i')] = zᵀBz, where B_{ii'} = y_i y_{i'} 𝟙[h(i)=h(i')] for i,i' ≥ s. This is a quadratic form in the ±1 vector z, with tr(B) = Σ_{i≥s} y_i² = ‖y^L‖². So I want a tail bound for zᵀBz around its trace — Hanson–Wright is exactly that: for symmetric B and integer ℓ ≥ 2,
  E|zᵀBz − tr(B)|^ℓ ≤ (CQ)^ℓ, with Q = max{√ℓ ‖B‖_F, ℓ ‖B‖₂}.
So I need ‖B‖_F and ‖B‖₂. Both are controlled by the *bucket load* of the light coordinates. Define, for a bucket j, its light load Σ_{i≥s, h(i)=j} y_i². Let me first show that, with the threshold chosen right, every bucket's light load is small — call this event E_h and the bound W.

For E_h: I want a uniform bound W ≥ max_j Σ_{i≥s, h(i)=j} u_i (using u_i to dominate y_i², which holds since y_i² ≤ u_i). Fix a bucket j, let X_i = u_i 𝟙[h(i)=j, i≥s]. Then 0 ≤ X_i ≤ T (light coords have u_i ≤ T), E[Σ X_i] = Σ_{i≥s} u_i / t ≤ r/t, and the variance V = Σ_{i≥s} E[X_i²] = Σ_{i≥s} u_i²/t = ‖u_{s:n}‖²/t. Now Bernstein in the convenient form: for X_i ∈ [0,T] with V ≤ LT²/6, Pr[Σ X_i ≥ Σ E[X_i] + LT] ≤ e^{−L}. Take L = log(t/δ_h). The condition V ≤ LT²/6 becomes ‖u_{s:n}‖²/t ≤ LT²/6, i.e. t ≥ 6‖u_{s:n}‖²/(L T²). Then with failure δ_h/t per bucket, and a union bound over t buckets, with probability ≥ 1 − δ_h every bucket has light load ≤ W where
  W ≡ T log(t/δ_h) + r/t.
So condition on E_h. Then, since y_i² ≤ u_i, the per-bucket light *squared-mass* Σ_{i≥s, h(i)=j} y_i² ≤ W for every j.

Now feed E_h into ‖B‖_F and ‖B‖₂.
‖B‖_F² = Σ_{i,i'≥s} (y_i y_{i'})² 𝟙[h(i)=h(i')] = Σ_{i≥s} y_i² · (Σ_{i' ≥ s, h(i')=h(i)} y_{i'}²) ≤ Σ_{i≥s} y_i² · W ≤ W (using Σ y_i² ≤ 1). 
For ‖B‖₂: for a fixed bucket j, the vector z(j) with z(j)_i = y_i 𝟙[h(i)=j, i≥s] is an eigenvector of B with eigenvalue ‖z(j)‖² = Σ_{i≥s, h(i)=j} y_i², and these eigenvectors span the column space of B (B is block-diagonal across buckets). So ‖B‖₂ = max_j Σ_{i≥s, h(i)=j} y_i² ≤ W. Both ≤ W. 

Plug into Q: with √ℓ ‖B‖_F ≤ √ℓ √W = √(ℓW) and ℓ ‖B‖₂ ≤ ℓW, if ℓW ≤ 1 then √(ℓW) ≥ ℓW, so Q ≤ √(ℓW). (This is why I'll want ℓ ≤ 1/W — so the √ℓ‖B‖_F term, not the ℓ‖B‖₂ term, dominates Q.) Now Markov on the ℓ-th moment with ℓ = log(1/δ_L):
  Pr[|‖ΦDy^L‖² − ‖y^L‖²| ≥ eC√(ℓW)] ≤ e^{−ℓ} = δ_L.
So with failure δ_L, |‖ΦDy^L‖² − ‖y^L‖²| ≤ K_L √(W log(1/δ_L)) for an absolute constant K_L. To make this O(ε) I need W = O(ε² / log(1/δ_L)). Good — the threshold T sets W, and W sets the error.

(I should note the route I'm taking. The original sparse-JL analysis of Dasgupta–Kumar–Sarlós handled the small-∞-norm regime directly; the moment route through Hanson–Wright in the spirit of Kane–Nelson is tighter, and crucially it lets me use the *per-bucket* mass bound W coming from the subspace, instead of re-deriving a concentration for each individual y. That's the difference between treating the e^{O(r)} net points one at a time and conditioning on a single structural event E_h. Tighter, and it's what shrinks t later.)

That leaves the cross term. ‖ΦDy‖² = ‖ΦDy^H‖² + ‖ΦDy^L‖² + 2⟨ΦDy^H, ΦDy^L⟩, and I've handled the first two. The cross term is 2 y_{1:(s−1)}ᵀ DΦᵀΦD y_{s:n}. On E_B each heavy coordinate sits alone in its bucket, so a heavy coordinate i' < s only interacts with a light coordinate i ≥ s if h(i) = h(i'); for each light i there is at most one such heavy partner. Define z_i = y_{i'} D_{i'i'} if i ≥ s collides with a heavy i' < s, else 0. Then the cross term is Σ_{i≥s} y_i D_{ii} z_i — a signed sum. Khintchine on its 2p-th moment:
  E[(Σ_{i≥s} y_i D_{ii} z_i)^{2p}]^{1/p} ≤ C_p Σ_{i≥s} y_i² z_i² = C_p Σ_{i'<s} y_{i'}² Σ_{i≥s, h(i)=i'} y_i² ≤ C_p W,
where C_p = Γ(p+1/2)^{1/p} = O(p), the last step uses E_h (each heavy bucket's light load ≤ W) and Σ_{i'<s} y_{i'}² ≤ 1. Markov with p = log(1/δ_C):
  |y_{1:(s−1)}ᵀ DΦᵀΦD y_{s:n}| ≤ K_C √(W log(1/δ_C)) with failure δ_C.
Again O(ε) once W = O(ε²/log).

Putting the fixed-y bound together: on E_h and E_B,
  |‖ΦDy‖² − ‖y‖²| ≤ |‖ΦDy^L‖² − ‖y^L‖²| + 0 + 2|cross| ≤ K_L√(W log(1/δ_L)) + 2K_C√(W log(1/δ_C)).
Set δ_L = δ_C = δ_y/2 and W ≤ K_y ε²/log(1/δ_y) for a small enough constant K_y; then the right side is ≤ 3ε√(K_y)(K_L + K_C) ≤ ε with K_y ≤ 1/(9(K_L+K_C)²). So: conditioned on E_h, E_B, a *fixed* unit y ∈ C(A) has ‖ΦDy‖ = (1±ε)‖y‖ with failure δ_y. And here's the payoff — δ_y can be pushed down to e^{−Ω(r)} by taking ℓ = Θ(r) in the moment bounds, because the cost of a smaller δ is only a larger ℓ (more moments), and the per-bucket mass W doesn't depend on δ. So the fixed-vector guarantee really does have failure e^{−Ω(r)} — the very thing the naive analysis could not get for worst-case vectors, recovered by using the subspace structure.

Now the net. With e^{−Ω(r)} fixed-vector failure, the union bound over an e^{O(r)} net is finally affordable. Let me do it cleanly with a grid net rather than a sphere-covering, since I need to control a bilinear form. Let E = {w ∈ (γ/√r) Z^r : ‖w‖ ≤ 1} be a γ/√r-grid inside the unit ball of R^r. Two standard facts (Arora–Hazan–Kale): |E| ≤ e^{cr} with c = (1/γ + 2); and for any r×r matrix J, if |uᵀJv| ≤ ε for all u,v ∈ E, then |wᵀJw| ≤ ε/(1−γ)² for every unit w. Set J = UᵀSᵀSU − I_r. For fixed x, y ∈ E, apply the fixed-vector bound to Ux, Uy, and U(x+y) (all in C(A), all of norm ≤ 1): with failure 3δ_y, ‖SUx‖² = (1±ε/6)‖Ux‖², similarly for Uy and U(x+y); expanding ‖SU(x+y)‖² = ‖SUx‖² + ‖SUy‖² + 2⟨SUx,SUy⟩ and using ‖Ux‖,‖Uy‖ ≤ 1 gives |xᵀJy| ≤ ε/2. Choose γ = 1 − 1/√2; union-bound over all pairs in E (there are ≤ e^{2cr} of them, but δ_y K_{sub}^r dominates for an absolute constant K_{sub}): with probability ≥ 1 − δ_y K_{sub}^r, |xᵀJy| ≤ ε/2 for all x,y ∈ E. Then by the second fact, |wᵀJw| ≤ ε for all unit w, i.e. ‖SUw‖² = (1±ε)‖w‖² for all w — a subspace embedding. The net works only because the dimension is r: the net has e^{O(r)} points and the per-point failure is e^{−Ω(r)}; in the ambient R^n a net would have e^{O(n)} points and this would fail. The subspace is doing all the work.

What t do I need? Collect the constraints. (i) E_B perfect hashing: δ_B ≤ s²/t small, so t = Ω(s²). (ii) The fixed-vector error: W = T log(t/δ_h) + r/t ≤ K_y ε²/log(1/δ_y), with δ_y = K_{sub}^{−r}/30 (so the net union bound gives ≤ 1/10 total). With δ_y this small, log(1/δ_y) = Θ(r), so I need W = O(ε²/r). Since W has a term T log t, I need T = O(ε²/(r log t)). (iii) Bernstein condition t ≥ 6‖u_{s:n}‖²/(log(t/δ_h) T²): but ‖u_{s:n}‖² = Σ_{i≥s} u_i² ≤ T Σ_{i≥s} u_i ≤ Tr (light scores are ≤ T), so the requirement becomes t = O((r/ε)² log t) — mild. (iv) How big is s? s is the number of heavy coordinates, those with u_i > T. Since Σ u_i = r, s ≤ r/T. With T = Θ(ε²/(r log)), s = Θ(r²/(ε² ) · log). Then t = Ω(s²) = Ω((r/ε)⁴ log²). So
  t = O((r/ε)⁴ log²(r/ε))
suffices, and ΦD is an ε-subspace embedding for A with probability ≥ 9/10, applied in O(nnz(A)) time — a subspace embedding whose application cost is proportional to the number of nonzeros, with no polynomial dependence on n. That's the result I was after.

Let me pause on the shape of that bound, because the (r/ε)⁴ bothers me. Where does the ⁴ come from? It's t = Ω(s²) and s = Θ((r/ε)²). The s² is the price of *perfectly* hashing all the heavy coordinates — no collisions allowed among s = (r/ε)² items, which by the birthday bound costs s² buckets. But do I really need *all* heavy coordinates perfectly hashed? The worst case that forces s to be large is a tower of leverage scores: about r scores ≈ 1, about 2r scores ≈ 1/2, about 4r ≈ 1/4, … down to T. The top r (value ≈ 1, e.g. when A contains I_d as a submatrix) genuinely must be perfectly hashed — a single collision of two unit-leverage coordinates is the (e₁+e₂)/√2 disaster, full distortion. But the scores of value 1/2, 1/4, … are individually small; a few of *them* colliding is survivable, as long as the total mass that any column-space vector can place on the colliding set is small. And "the mass a column-space vector can place on a set of rows" is exactly the spectral norm of the corresponding submatrix of U.

So refine: partition the heavy scores into geometric groups G_j = {i : 2^{−j} < u_i ≤ 2^{−j+1}}, j = 1..q with q = log₂(1/T) = O(log(r/ε)), β_j ≡ 2^{−j}. Since Σ u_i = r, |G_j| = n_j ≤ r/β_j = r·2^j. For each group, instead of demanding zero collisions, let G'_j ⊆ G_j be the coordinates of G_j that *do* collide with another G_j-member under h, and bound the damage they do. The damage from within-group collisions is controlled by the spectral norm of Û_j, the submatrix of U on rows G'_j: a unit column-space vector y can put squared mass at most ‖Û_j‖₂² on those rows. So I need ‖Û_j‖₂² small for all j.

To bound ‖Û_j‖₂² I'll use matrix Bernstein. The collision set is a random subset of G_j (an item is in G'_j iff it shares its bucket with another G_j-item). Analyze a with-replacement sampling proxy: draw ℓ_j ≈ (4e²)k_j + Θ(q) rows of B (the rows of U in G_j, normalized so ‖B‖₂ ≤ 1, each row squared-norm in [β_j, 2β_j]) and form Ĥ_m = B_{i:}ᵀB_{i:}, E[Ĥ_m] = (1/n_j)BᵀB. With H_m = Ĥ_m − E[Ĥ_m], compute ρ_m² = ‖E[H_m H_m]‖₂ ≤ 2β_j/n_j + 1/n_j² and M = ‖H_m‖₂ ≤ 2β_j + 1/n_j. Matrix-Bernstein (Recht's form): log Pr[‖Σ_m H_m‖₂ > τ] ≤ log 2r − (τ²/2)/(Σ ρ_m² + Mτ/3). Setting τ = Θ(q(β_j + 1/n_j + √(r/t))) drives this below 1/r, giving, with the E[Ĥ_m] contribution added back,
  ‖Σ_m Ĥ_m‖₂ = O(q(β_j + 1/n_j + √(r/t) + n_j/t)),
and ‖Σ_m Ĥ_m‖₂ is exactly the squared spectral norm of the submatrix of U on the ℓ_j sampled rows. A balls-and-bins argument (Azuma) shows ℓ_j with-replacement samples contain ≥ k_j distinct rows, so this bounds ‖Û_j‖₂². The number of collisions k_j ≈ n_j²/t (expected, by Markov), and the spectral-norm contribution is small precisely when t ≳ r²/ε² · polylog — no longer needing the full s² = (r/ε)⁴. So with this group-by-group treatment the embedding dimension drops to
  t = r² ε^{−2} polylog(r/ε),
which is the right additive overhead for the downstream solvers. The point of the refinement: spend perfect hashing only on the O(r) genuinely-unit leverage scores; let smaller scores collide a little and pay only their (small) spectral norm.

There's a third dial I want, for a different reason. For ℓ_p regression and for low-rank I'll need the embedding to succeed not with constant probability but with probability 1 − 1/poly(n) — I'll be union-bounding over many blocks. The basic ΦD gives constant probability. To get high probability while keeping nnz time, replace the single ±1-per-column by a *small JL transform inside each bucket*. Hash n coordinates into about d²/log n buckets (balls and bins: with d² balls in d²/log n bins, each bin gets O(log n) balls whp), and within each bucket run an O(log n / ε²)-dimensional JL map. Now each bucket holds only O(log n) heavy coordinates, so I can net all vectors supported on a bucket with a net of size only poly(n) and union-bound; the per-bucket JL preserves its norm with failure 1/poly(n). Kane–Nelson's construction lets the in-bucket JL be applied in O(log n / ε) time per coordinate, so total time O(nnz(A) log n / ε), and the embedding now holds with probability 1 − 1/poly(n). Concretely the generalized matrix is block-diagonal: a permutation P groups the rows by an outer hash h:[n]→[q], then independent KN12 maps B^{(i)} (each the concatenation of √v̄ Φ_j D_j sparse blocks, v̄ = Θ(ε⁻¹ log(r/εγ)), v = Θ(ε⁻¹)) act on each group. The small-∞-norm analysis reduces to the basic one (each "slice" N_j is itself a sparse embedding matrix), and the high-probability JL handles the heavy part per bucket. This is the version that powers the high-probability applications.

OK — I have the embedding in three flavors (basic ΦD; partition-refined t; generalized high-probability). Now spend it.

Start with least squares. Form the embedding of the column space of [A | b] (rank ≤ r+1). Compute ΦDA and ΦDb in O(nnz(A)) time, then solve the small problem argmin_x ‖ΦDAx − ΦDb‖₂. The embedding guarantees ‖ΦD(Ax−b)‖ = (1±ε)‖Ax−b‖ for all x, so the sketch minimizer is a (1+ε)-approximation. The small problem has t = O(d²ε⁻² polylog) rows and d columns; solve it with a Fast-JL-accelerated dense least-squares (Sarlós) in O(t d log d + d³ ε⁻¹ log d). Total O(nnz(A)) + Õ(d³ε⁻²). The dependence on nnz(A) is *linear* — exactly the dream.

I can do better on the ε dependence. The Õ(d³ε⁻²) carries ε⁻² because the embedding error feeds directly into the answer. If I instead use the embedding only to find a good *preconditioner* and then iterate, I can get log(1/ε). Estimate the leverage scores of A to constant factor (next paragraph), which as a side effect produces a change-of-basis R so that AR has all singular values 1±ε₀ — extremely well-conditioned. Then the simple fixed-point x^{(m+1)} = x^{(m)} + Rᵀ Aᵀ(b − ARx^{(m)}) contracts: using the normal equations and the SVD AR = UΣVᵀ,
  AR(x^{(m+1)} − x*) = (AR − ARRᵀAᵀAR)(x^{(m)} − x*) = U(Σ − Σ³)Vᵀ(x^{(m)} − x*),
and since every σ_i = 1±ε₀, the diagonal entries of Σ − Σ³ are σ_i(1 − σ_i²) in magnitude at most σ_i((1+ε₀)² − 1) = σ_i(2ε₀ + ε₀²) ≤ 3ε₀ σ_i for ε₀ ≤ 1. So ‖AR(x^{(m+1)} − x*)‖ ≤ 3ε₀ ‖AR(x^{(m)} − x*)‖; that contracts as long as 3ε₀ < 1, so pick a small enough constant ε₀ (say ε₀ = 1/4, factor ≤ 3/4) and O(log(1/ε)) iterations reach ε relative error. Each iteration costs a constant number of nnz(A)-time products. Total O(nnz(A) log(n/ε) + r³ log²r + r² log(1/ε)) — log dependence on ε, never worse than CG, and condition-independent because the preconditioner R came from the embedding, not from A's spectrum.

Leverage scores next — I used these above, so let me actually compute them fast. The recipe (Drineas–Magdon-Ismail–Mahoney–Woodruff): pick a subspace embedding Π₁, compute Π₁A, find a change-of-basis R so that Π₁AR has orthonormal columns (a QR factorization), and then the row norms ‖(AR)_{i,·}‖² equal u_i(1±ε) — because AR is, up to the embedding error, an orthonormal basis for C(A). To read off all n row norms cheaply, multiply AR on the right by a tiny r×O(log n) JL matrix Π₂ and take row norms of A(RΠ₂). The only change I make to their recipe is Π₁: take Π₁ = F·S, where S is my sparse embedding (t = O(r² log r)) and F is a Fast-JL bringing it down to O(r log²r) rows. Then SA costs O(nnz(A) log r), F(SA) costs O(r³ log²r), the QR of FSA costs O(r³ log²r), RΠ₂ costs O(r² log n), and A(RΠ₂) costs O(nnz(A) log n). Total O(nnz(A) log n + r³ log²r + r² log n). Composition of two embeddings is still an embedding (errors add, failure probabilities add), so FSA preserves the column space and the row-norm reading is correct. That's the constant-factor leverage approximation; oversampling lifts it to (1±ε) when needed.

Low rank is the subtler one — it needs generalized (multiple-response) regression: min_X ‖AX − B‖_F. The clean way is *affine embeddings*: a single S with two properties — it's a subspace embedding for A, and it satisfies approximate matrix multiplication ‖AᵀSᵀSB − AᵀB‖_F ≤ (ε/√r)‖A‖_F‖B‖_F — makes argmin_X ‖S(AX − B)‖ a (1+ε)-proxy for argmin_X ‖AX − B‖. The proof is Pythagoras plus the normal equations: with A orthonormal, ‖AỸ − B‖² = ‖AY* − B‖² + ‖A(Ỹ − Y*)‖², and the AMM property bounds ‖A(Ỹ − Y*)‖ ≤ 2√ε ‖B − AY*‖, giving (1+O(ε)). My sparse embedding satisfies AMM too (the Thorup–Zhang second-moment bound for hashed sketches: E[X_{ij}] = 0, Var[X_{ij}] = O(1/t)‖A_i‖²‖B_j‖², so t = Θ(ε⁻²) suffices via Chebyshev), so it qualifies. Now the algorithm: (1) right-sketch C(AR^⊤) with a sparse embedding R (r = k) — this gives a rank-k space U whose best fit to A is within (1+ε) of Δ_k = ‖A − A_k‖_F, because by generalized regression AR^⊤(A_kR^⊤)⁻A_k is a (1+ε)-good rank-k approximation. (2) Left-affine-embed with S = (SRHT)∘(sparse), v = Θ(ε⁻⁴k²polylog), v' = Θ(ε⁻³k polylog), so that argmin_{rank k} ‖S(UX − A)‖ is a (1+ε)-proxy for argmin_{rank k} ‖UX − A‖. (3) The rank-k restriction inside the sketch is solved by SVD: if SU = ŨΣ̃Ṽ⊤, the best rank-k fit is [Ũᵀ SA]_k (cw09), and lifting through Ṽ Σ̃⁻ gives the answer LDW⊤ with ‖LDW⊤ − A‖ ≤ (1+ε)²Δ_k. Total O(nnz(A)) + Õ(nk²ε⁻⁴ + k³ε⁻⁵) — the nnz term is the cheap sparse right-sketch, everything else is small-dimensional.

And ℓ_p regression rides on the high-probability generalized embedding. Block the rows of A into n/w blocks of w = Θ(r⁶ log n(r+log n)) rows; embed each block to ℓ₂ with the generalized sparse embedding S (high probability, so union-bound over all n/w blocks holds). A subspace embedding for ℓ₂ on each block lets me build a *well-conditioned basis* for ℓ_p (the Dasgupta–Kumar–Sarlós–Mahoney machinery — an (α,β,p)-well-conditioned basis with α,β = poly(r)), then sample rows with probability ∝ their ℓ_p well-conditioned-basis norms and solve the small weighted ℓ_p problem. The connection (that an ℓ₂ embedding as a black box yields ℓ_p regression) is the CDMMMW reduction; with my embedding the leading O(nr⁵ log n) of the basis construction drops to O(nnz(A) log n). Total O(nnz(A) log n) + poly(r/ε), any constant p ∈ [1,∞). Crucially this needs the *high-probability* embedding — the basic constant-probability ΦD does not survive the union bound over blocks.

So the causal chain, top to bottom: every one of these tasks reduces to "embed the column space, then solve small." The embedding's cost is the cost of S·A. To make S·A cost O(nnz(A)), make S have one nonzero per column — that's CountSketch. CountSketch fails the worst-case net argument because two heavy coordinates can collide, but the column-space constraint y_i² ≤ u_i pins every heavy coordinate to a fixed small set H = the large leverage scores; perfectly hash H (an exact isometry on the heavy part), control the light part by Hanson–Wright (its small ∞-norm gives small per-bucket mass W), bound the cross term by Cauchy–Schwarz/Khintchine, and close with a net over the *r*-dimensional subspace where e^{O(r)} net points meet e^{−Ω(r)} per-point failure. Refine the perfect-hashing of the heavy set group-by-group (spectral norm of collided submatrices) to bring t down to r²ε⁻² polylog; put a JL inside each bucket to get 1−1/poly(n) success for the union-bound-heavy applications. Then plug the embedding into the known sketch-and-solve, leverage-score, affine-embedding, and well-conditioned-basis recipes to get input-sparsity-time regression, low rank, leverage scores, and ℓ_p.

Now the concrete code. The basic embedding is one nonzero per column with a random sign — built as a sparse CSC matrix — and applied as S @ A; regression is sketch-and-solve on [A | b]. This is exactly the construction analyzed above.

```python
import numpy as np
from scipy import sparse
from scipy import linalg


def build_sketch(t, n, rng):
    """The sparse embedding S = Phi D: one nonzero (+1 or -1) per column.

    Column i has its single nonzero in row h(i) = rows[i] (the hash), with value
    signs[i] (the random sign, = D_ii). Applying S @ A then hashes each nonzero of A
    into a bucket with a sign and adds -- O(nnz(A)) time. This is CountSketch.
    """
    rows = rng.integers(0, t, size=n)           # h : [n] -> [t], the bucket of each coordinate
    cols = np.arange(n + 1)                      # CSC pointers: exactly one entry per column
    signs = rng.choice([1, -1], size=n)         # diag(D): the random +-1 sign per column
    return sparse.csc_matrix((signs, rows, cols), shape=(t, n))


def sketch_apply(S, A):
    """Form S A in O(nnz(A)) time (sparsity-aware product)."""
    return S @ A


def embedding_dimension(r, eps, refined=True):
    """Rows t for an eps-subspace-embedding on a rank-r column space.

    Basic analysis (perfect-hash all heavy leverage scores, t = Omega(s^2)):
        t = O((r/eps)^4 log^2(r/eps)).
    Partition-refined (perfect-hash only the unit scores; let smaller groups collide
    and pay their submatrix spectral norm):
        t = O(r^2 / eps^2 * polylog(r/eps)).
    """
    if refined:
        return int(np.ceil((r ** 2 / eps ** 2) * max(1.0, np.log(r / eps)) ** 6))
    return int(np.ceil((r / eps) ** 4 * max(1.0, np.log(r / eps)) ** 2))


def regression_sketch_and_solve(A, b, t, rng):
    """(1+eps)-approximate argmin_x ||A x - b||_2 via the sparse embedding.

    Embed the column space of [A | b] (rank <= r+1), then solve the tiny problem on
    the sketch. Theorem: the sketch minimizer is a (1+eps)-approximation.
    """
    Ab = _adjoin(A, b)                                  # work in C([A | b])
    S = build_sketch(t, Ab.shape[0], rng)
    SAb = sketch_apply(S, Ab)                           # O(nnz(A)) time
    SA, Sb = SAb[:, :-1], SAb[:, -1]
    x, *_ = linalg.lstsq(_dense(SA), _dense(Sb).ravel())  # small dense least squares
    return x


def regression_iterative(A, b, t, rng, n_iter=None):
    """log(1/eps) variant: use the embedding for a preconditioner, then iterate.

    A constant-factor embedding makes the preconditioned system well-conditioned; the
    fixed point x <- x + R^T A^T (b - A R x) then contracts the residual by ~3*eps0 < 1
    per step, so O(log(1/eps)) steps suffice. Here R is read off the sketched QR.
    """
    A = _dense(A); b = _dense(b).ravel()
    S = build_sketch(t, A.shape[0], rng)
    SA = _dense(sketch_apply(S, A))
    # change of basis so that A @ Rinv is well-conditioned (all singular values ~ 1)
    Q, Rmat = np.linalg.qr(SA)
    Rinv = np.linalg.inv(Rmat)                          # R in the derivation: AR well-conditioned
    AR = A @ Rinv
    y = np.zeros(AR.shape[1])                           # solve min ||AR y - b||, x = Rinv y
    if n_iter is None:
        n_iter = 40
    for _ in range(n_iter):
        y = y + AR.T @ (b - AR @ y)                     # x^{(m+1)} = x^{(m)} + (AR)^T (b - AR x^{(m)})
    return Rinv @ y


def leverage_scores(A, t, rng):
    """Constant-factor approximation to all u_i = ||U_{i,*}||^2 simultaneously.

    Embed, compute a basis-change R so that (A R) has orthonormal columns up to the
    embedding error; then the row norms ||(A R)_{i,*}||^2 = (1 +- eps) u_i. A tiny JL
    Pi2 reads all n row norms quickly via A (R Pi2).
    """
    A = _dense(A); n, d = A.shape
    S = build_sketch(t, n, rng)
    SA = _dense(sketch_apply(S, A))
    _, Rmat = np.linalg.qr(SA)
    R = np.linalg.inv(Rmat)                             # (A R) ~ orthonormal basis of C(A)
    p = max(1, int(np.ceil(8 * np.log(max(n, 2)))))     # JL width O(log n)
    Pi2 = rng.standard_normal((R.shape[1], p)) / np.sqrt(p)
    Y = A @ (R @ Pi2)                                   # rows carry (1 +- eps) the true row norms
    return np.sum(Y ** 2, axis=1)


def _adjoin(A, b):
    b = np.asarray(b).reshape(-1, 1)
    if sparse.issparse(A):
        return sparse.hstack([A, sparse.csc_matrix(b)]).tocsc()
    return np.hstack([np.asarray(A), b])


def _dense(M):
    return M.toarray() if sparse.issparse(M) else np.asarray(M)
```
