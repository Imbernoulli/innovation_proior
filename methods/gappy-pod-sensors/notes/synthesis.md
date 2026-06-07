# Synthesis — Data-driven sparse sensor placement via QR pivoting on a POD/SVD basis

## The pain point / research question
Pick p ≪ n point sensors (rows of identity, i.e. individual components x_{γ_i} of the state) so the full
high-dimensional state x ∈ R^n can be reconstructed from y = Cx ∈ R^p. Exact optimal placement is a
combinatorial search over C(n,p) subsets — intractable. Need a scalable, near-optimal, easy-to-implement scheme.

## Core derivation chain (the heart of reasoning.md)
1. **Tailored low-rank basis.** Data X=[x_1...x_m] ∈ R^{n×m}. SVD X = ΨΣVᵀ; keep r leading left singular
   vectors Ψ_r (POD modes). Eckart–Young: Ψ_rΣ_rV_rᵀ is the optimal rank-r LS approximation. So x ≈ Ψ_r a,
   a = Ψ_rᵀx ∈ R^r. (r ≪ n because dominant coherent structures.)
2. **Reconstruct from few measurements.** y = Cx ≈ CΨ_r a = Θ a, Θ = CΨ_r ∈ R^{p×r}. Estimate a by
   pseudoinverse: â = Θ†y (= Θ⁻¹y if p=r). x̂ = Ψ_r â. This *is* gappy POD (Everson–Sirovich 1995): the
   point measurements pick out the observed entries, and a is fit by least squares to those entries.
3. **Why placement matters — conditioning.** â = Θ†y. With sensor noise y = Θa + ξ, ξ~N(0,η²I),
   Var(a − â) = η²(ΘᵀΘ)⁻¹. Worst-case error amplification = κ(Θ) = σ_max/σ_min (Sidebar: condition number;
   SNR_out = SNR_in/κ). If Θ=CΨ_r is ill-conditioned (some σ_min(Θ)→0), reconstruction blows up. So choose
   the rows of C (the sensor locations γ) to make Θ as well-conditioned / large-volume as possible.
4. **Exact objective is combinatorial.** Optimal-design criteria on M_γ = ΘᵀΘ:
   - E-optimal: max σ_min(M) = min ||M⁻¹||₂  (eqn e_opt)
   - A-optimal: max tr(M) = Σλ_i           (eqn a_opt)
   - D-optimal: max |det M| = ∏σ_i(M)      (eqn d_opt)  ← the volume / variance-determinant criterion.
   Each needs C(n,p) evaluations. D-optimal ⇔ minimize volume of error-covariance ellipsoid
   η²(ΘᵀΘ)⁻¹ (Joshi & Boyd 2009). Convex relaxation (Joshi–Boyd): max_β logdet Σ β_i θ_iᵀθ_i s.t.
   Σβ_i=p, 0≤β_i≤1 — but O(n³) per Newton iteration, O(n²) storage, redo for each p.
5. **Greedy surrogate = column-pivoted QR.** Businger–Golub (1965) pivoted QR: A Cᵀ = QR; at each step
   pick the column of largest residual 2-norm, then deflate (subtract its projection from all remaining
   columns via a Householder reflector). This greedily maximizes the volume ∏|r_ii| = ∏σ_i = |det| of the
   selected submatrix, and enforces the diagonal-dominance |r_ii|² ≥ Σ_{j=i}^k |r_jk|² (Drmač–Gugercin).
   So QR pivoting of Ψ_rᵀ greedily solves the D-optimal subset selection: the first r pivots = sensor rows
   that maximize |det Θ| (p=r case). This is exactly Q-DEIM (Drmač–Gugercin 2016).
6. **Oversampling p>r.** Θ now p×r, tall; D-optimal wants max det(ΘᵀΘ) = ∏_{i=1}^r σ_i(ΘΘᵀ). Pivot instead
   on the n×n Gram matrix Ψ_rΨ_rᵀ (rows = points, (Ψ_rΨ_rᵀ)_{ij}=ψ_i·ψ_j). First p pivots = p points whose
   row-submatrix has maximal volume in the r-dim row space → increases leading r singular values of ΘΘᵀ →
   increases det(ΘᵀΘ). Cost O(n³) (vs O(nr²) for p=r), but one factorization yields a *hierarchical* ranked
   list of all pivots, reusable for any p.
7. **Algorithm.** if p==r: γ = qrPivot(Ψ_rᵀ, r); elif p>r: γ = qrPivot(Ψ_rΨ_rᵀ, p). C = [e_{γ_1}...e_{γ_p}]ᵀ.

## Load-bearing ancestors (elaborate, with the gap each leaves)
- **Gappy POD — Everson & Sirovich 1995 (JOSA A 12:1657).** Reconstruct masked data in a KL/POD basis.
  Mask vector n (n_i=1 observed, 0 missing); gappy inner product (u,v)_n = Σ n_i u_i v_i; fit coefficients
  b by minimizing E=||g−g̃||²_n → small LS system Ma=f, M=(Φ,Φ)_n. GAP: which points to keep was left to
  random/heuristic sub-sampling — no principled placement, no conditioning guarantee.
- **EIM — Barrault et al. 2004; DEIM — Chaturantabut & Sorensen 2010 (SISC 32:2737).** Greedy interpolation
  point selection for nonlinear ROM terms. DEIM (Algorithm 1): p_1=argmax|u_1|; for each new mode u_j, solve
  for its representation in already-chosen rows, take residual r_j=u_j−U_{j-1}z, pick p_j=argmax|r_j|. Gives
  f̂=U(SᵀU)⁻¹Sᵀf, error ||f−f̂||₂ ≤ ||(SᵀU)⁻¹||₂·||(I−UUᵀ)f||₂, with the bound
  c=||(SᵀU)⁻¹||₂ ≤ (1+√(2n))^{m-1}/||u_1||_∞ — exponentially pessimistic. GAP: p=r only (square SᵀU),
  basis-order-dependent, recomputes a residual every iteration; no oversampling.
- **Q-DEIM — Drmač & Gugercin 2016 (SISC 38:A631; arXiv 1505.00370).** Replace DEIM's residual greedy by
  column-pivoted QR of Uᵀ. Sharper bound ||(SᵀU)⁻¹||₂ ≤ √(n−m+1)·(√(4^m+6m−1)/3); selection invariant to
  unitary basis change. Diagonal dominance (2.7): |T_ii|²≥Σ_{j=i}^k|T_jk|²; from RRᵀ=I_m get
  min_i|T_ii|=|T_mm|≥1/√(n−m+1). GAP: still p=r (square); restricted to ROM nonlinear-term interpolation,
  not framed as general sensor placement, no oversampling.
- **Businger & Golub 1965 (Numer. Math. 7:269).** Column-pivoted QR / Householder LS — the engine. Pivot =
  max residual column norm; QR step maps it to e_i R_ii, other column norms ≤ |R_ii|. This is the greedy
  volume maximizer that underlies all the above.
- **Joshi & Boyd 2009 (IEEE TSP 57:451).** Sensor selection by Boolean→[0,1] relaxation, max logdet of the
  Fisher/information matrix Σθ_iᵀθ_i (D-optimal). Provides a lower-bound certificate, but O(n³)/iter, no
  global-optimality bound after rounding, doesn't exploit POD structure. GAP: expensive, per-p recompute.
- **D/A/E-optimal design (classical DoE).** The variance criteria QR approximates; combinatorial exact.

## Design decisions → why
- Left singular vectors as basis: Eckart–Young optimal rank-r LS fit; orthonormal Ψ_rᵀΨ_r=I → clean
  projection a=Ψ_rᵀx and κ(Θ)=κ(CΨ_r) controlled purely by row selection.
- D-optimal (det) over A/E: det = volume of error-cov ellipsoid = product of all axes; QR's ∏|r_ii| is
  exactly a determinant, so det is the criterion QR greedily and natively optimizes. (Also: D-optimal is
  the right criterion for minimum-variance estimation under Gaussian noise.)
- Pivot on Ψ_rᵀ for p=r (O(nr²)); pivot on Ψ_rΨ_rᵀ for p>r (O(n³)) — Gram matrix lets pivoting select among
  n candidate *points* while implicitly maximizing det(ΘᵀΘ); single factorization → hierarchical pivot list.
- Why QR not convex opt: one O(n³) (or O(nr²)) factorization total, native in LAPACK/NumPy/MATLAB, vs
  O(n³)/iteration and per-p recompute for SDP/Newton; comparable or better accuracy; ranked list for free.
- r chosen by singular-value thresholding (e.g. Gavish–Donoho optimal hard threshold) — beyond informative
  r, QR places sensors on noise modes → overfitting; accuracy plateaus then degrades.
- Oversample p>r when σ decay is slow / under noise: extra rows lift σ_min(Θ), shrink (ΘᵀΘ)⁻¹, average out
  noise — bounded error growth.

## Code landing (pysensors, dynamicslab)
- SVD basis: TruncatedSVD → basis_matrix_ = components_.T = Ψ_r (n×r).
- QR optimizer: `scipy.linalg.qr(basis_matrix.conj().T, pivoting=True)` → pivots_ = ranked sensors.
- SSPOR.fit: basis.fit → basis_matrix_; optimizer.fit(basis_matrix_).get_sensors(); singular_values from
  ||X Ψ_r||/√m. predict: unregularized = lstsq/solve of Ψ_r[sensors,:] then x̂=Ψ_r â; or regularized
  (prior-cov + ΘᵀΘ/η²)⁻¹.
- p=r square solve; p>r rectangular lstsq. Oversampled QR in MATLAB supplement: qr(Psi_r*Psi_r','vector').

## Empirical facts (context only, never fabricate)
- Cylinder flow: n≈90000 (downsampled to 3600), r=42 intrinsic rank, singular values in pairs (vortex
  shedding harmonics). Yale B faces: 1024 px, r=166 Gavish–Donoho threshold. SST NOAA OISST: r=302.
  (These are reported settings/spectra; the proposed-method reconstruction-error wins are OUT of scope.)
