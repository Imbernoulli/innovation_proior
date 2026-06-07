# DEIM synthesis (from TR09-05 primary + pyMOR code + EIM/POD/gappy lineage)

## Sources retrieved this run
- PRIMARY: Chaturantabut & Sorensen, "Discrete Empirical Interpolation for Nonlinear Model Reduction", Rice CAAM TR09-05 (April 8 2009) — full text extracted via pdf_extract.py (the freely-available tech-report version of the SISC 2010 paper). Read end to end: §1 intro, §2 problem/POD/complexity, §3 DEIM (algorithm, error bound proof, error estimate σ_{m+1}, FD application, general nonlinearity, complexity tables), §4 FitzHugh–Nagumo.
- BACKGROUND: POD/SVD (primary §2.1 + sibling gappy-pod-sensors); Galerkin projection ROM (primary §2); EIM (Barrault–Maday–Nguyen–Patera 2004) — PDF NOT obtainable (numdam direct PDF 404/cert fail, HAL returned access-denied HTML); essentials grounded from primary §3.1 + search snippet (greedy "magic points", interpolation replaces projection, lower-triangular interpolation matrix, non-affine parametrized function approximated by M basis funcs interpolated at M points). gappy-POD (Everson–Sirovich 1995) from sibling.
- THIRD-PARTY: pyMOR `src/pymor/algorithms/ei.py` `deim()` function (canonical Python implementation; exactly mirrors Algorithm 1 — greedy residual, amax, incremental interpolation matrix). SciML ModelOrderReduction.jl tutorial (abstracts deim()). Ulm project sheet (German, just an assignment).

## The pain (research question)
- High-dim nonlinear ODE from FD/FEM: dy/dt = A y + F(y), y ∈ R^n, n huge (1024, 15000…). F componentwise: F(y)=[F(y_1),…,F(y_n)]^T.
- POD-Galerkin: y ≈ V_k ỹ, ỹ ∈ R^k, k≪n. Reduced: dỹ/dt = Ã ỹ + V_k^T F(V_k ỹ), Ã = V_k^T A V_k (k×k, precomputed once).
- THE PROBLEM (§2.2): the nonlinear term Ñ(ỹ)=V_k^T F(V_k ỹ). To evaluate it each step you must (a) lift V_k ỹ to R^n [nk flops], (b) evaluate F at all n components [α(n)], (c) project back [nk]. Cost O(α(n)+4nk) — STILL depends on n. The linear part reduced offline once, but the nonlinear part is re-evaluated at all n points every step → reduced model isn't actually cheap. For Newton on steady problems even worse: Jacobian V_k^T diag{F'(V_k ỹ)} V_k costs O(α(n)+4nk+2nk^2).

## The intellectual move
- Approximate f(τ):=F(V_k ỹ(τ)) ∈ R^n in its OWN POD basis U=[u_1…u_m]∈R^{n×m} (m≪n), built from nonlinear snapshots {F(y(t_1)),…,F(y(t_ns))}: f ≈ U c(τ).
- DON'T determine c by orthogonal projection c=U^T f — that needs all n entries of f (back to square one). Instead determine c by INTERPOLATION at m selected indices ℘_1…℘_m. P=[e_{℘_1}…e_{℘_m}]∈R^{n×m}. Match at those rows: P^T f = (P^T U) c ⇒ c=(P^T U)^{-1} P^T f. So
    f ≈ f̂ = U (P^T U)^{-1} P^T f.    (eq 16)
- This is an OBLIQUE projector P_proj = U(P^T U)^{-1}P^T (idempotent, range U, but not orthogonal). Exact at the interpolation rows: P^T f̂ = P^T f.
- KEY for componentwise F: P^T F(V_k ỹ) = F(P^T V_k ỹ). So you only ever evaluate F at the m rows ℘ of V_k ỹ — extract m rows of V_k (call V_℘ = P^T V_k ∈ R^{m×k}), form V_℘ ỹ ∈ R^m, evaluate F there [α(m)], multiply by precomputed B = V_k^T U (P^T U)^{-1} ∈ R^{k×m}. Cost O(α(m)+4km) — INDEPENDENT of n. (eqs 53–54.)
  - General (non-componentwise) F: each output component i depends on n_i inputs via index set p_i; only need rows ℘ of F, each needing Σ n_{℘_i} input components; effective iff Σ n_{℘_i} ≪ n. Sparse CSR-style structure (irstart/jrow).

## DEIM greedy point selection (Algorithm 1) — picks argmax residual
- ℘_1 = argmax_i |u_1(i)|  (largest-magnitude entry of first mode).  U=[u_1], P=[e_{℘_1}].
- for ℓ=2..m: solve (P^T U) c = P^T u_ℓ for c; residual r = u_ℓ − U c; ℘_ℓ = argmax_i |r(i)|; append u_ℓ to U, e_{℘_ℓ} to P.
- r is the part of mode ℓ not explained by interpolating previous modes at previous points; r(℘_i)=0 for i<ℓ; linear independence ⇒ r≠0 ⇒ ρ:=r(℘_ℓ)≠0 ⇒ P^T U nonsingular ⇒ method well-defined. Indices hierarchical, non-repeated.
- Basis order matters: feed POD basis ordered by dominant singular values.
- Once indices fixed, the DEIM approx is independent of which basis spans Range(U) (eq 17): U=QR ⇒ U(P^TU)^{-1}P^T = Q(P^TQ)^{-1}P^T.

## Error bound (Lemma 3.2) — exact, must reproduce
‖f − f̂‖_2 ≤ ‖(P^T U)^{-1}‖_2 · E_*(f),  E_*(f)=‖(I−UU^T)f‖_2  (best 2-norm error from Range(U)).
Proof:
- f_* = UU^T f best approx; w = f − f_* = (I−UU^T)f. f = w + f_*.
- Projector P_proj = U(P^TU)^{-1}P^T. f̂ = P_proj f = P_proj w + P_proj f_*. Since f_*∈Range(U) and P_proj is identity on Range(U), P_proj f_* = f_*. So f̂ = P_proj w + f_*.
- f − f̂ = (I − P_proj) w. ‖f−f̂‖ ≤ ‖I−P_proj‖ ‖w‖.
- For any projector ≠0,I: ‖I−P_proj‖_2 = ‖P_proj‖_2 (Kato/Szyld identity). And ‖P_proj‖_2 = ‖U(P^TU)^{-1}P^T‖_2 ≤ ‖U‖‖(P^TU)^{-1}‖‖P^T‖ = ‖(P^TU)^{-1}‖_2 since ‖U‖=‖P‖=1 (orthonormal columns / selection cols).
- ⇒ ‖f−f̂‖_2 ≤ ‖(P^TU)^{-1}‖_2 ‖w‖_2 = ‖(P^TU)^{-1}‖_2 E_*(f).  ∎
Recursive constant C bound (very pessimistic, theoretical): C_1 = ‖u_1‖_∞^{-1} = 1/|e_{℘_1}^T u_1|; C_ℓ = (1+√(2n)) C_{ℓ-1}; C=C_m. So ‖(P^TU)^{-1}‖_2 ≤ C.
Block-inverse derivation of the recursion (must reproduce):
- M=P^TU. Partition M = [[M̄, P̄^T u],[a^T, p^T u]], a^T=p^T Ū. Factor M = [[M̄,0],[a^T,ρ]][[I,c],[0,1]] with c=M̄^{-1}P̄^T u, ρ=p^T u − a^T c = p^T(u − Ū M̄^{-1} P̄^T u) = p^T r. |ρ|=‖r‖_∞ (since ℘_ℓ=argmax|r|).
- M^{-1} = [[I,−c],[0,1]] [[M̄^{-1},0],[−ρ^{-1}a^T M̄^{-1}, ρ^{-1}]].
- Rearranged: M^{-1} = [[M̄^{-1},0],[0,0]] + ρ^{-1} [c;−1][a^T,−1]·(block w/ M̄^{-1}). Norm bound:
  ‖[c;−1][a^T,−1]‖ with the [Ū,u] factor: ‖[Ū,u][c;−1]‖·‖[a^T,−1]‖ = ‖Ūc−u‖·‖[a^T,−1]‖ ≤ √(1+‖a‖^2)·√n·‖Ūc−u‖_∞ ≤ √(2n)|ρ|. (‖a‖≤1 since a=Ū^T p selects/combines orthonormal cols; ‖Ūc−u‖_∞=‖r‖_∞=|ρ|; the √n converts ∞→2 norm.)
- ⇒ ‖M^{-1}‖ ≤ (1+√(2n))‖M̄^{-1}‖. Recursion (22). ∎
- DEIM picks ℘_ℓ = argmax|r| ⇒ maximizes |ρ| ⇒ minimizes 1/|ρ| ⇒ minimizes the stepwise growth of ‖(P^TU)^{-1}‖_2. So the greedy is literally minimizing the error-bound magnification factor.
Practical error estimate: E_*(f) ≈ σ_{m+1} (next singular value of nonlinear snapshot matrix), so avg DEIM error ≲ C̄ σ_{m+1} with C̄=‖(P^TU)^{-1}‖_2 (used directly, not the pessimistic C).

## Why this and not alternatives
- vs orthogonal projection c=U^T f: needs all n entries of f → no complexity reduction. Interpolation needs only m. DEIM is "nearly as good as orthogonal projection" per the bound (E_* IS the orthogonal-projection error; DEIM only inflates it by ‖(P^TU)^{-1}‖).
- vs EIM (Barrault 2004): continuous, function-space, greedy "magic points"; DEIM is its discrete finite-dim simplification — same interpolation-replaces-projection idea, but posed on R^n with POD basis, clean matrix form U(P^TU)^{-1}P^T, and a clean error bound. DEIM basis & points differ from EIM's.
- vs gappy-POD (Everson–Sirovich): gappy fits coefficients by LEAST SQUARES over a (often random/ad-hoc) mask, b=(M)^{-1}f with M=Θ^TΘ. DEIM uses square interpolation (m points = m modes, P^TU square) and PROVIDES the point-selection greedy that gappy left open. Different pain though: gappy reconstructs a sparsely-SAMPLED field; DEIM cheapens a nonlinear-TERM evaluation inside a ROM. Do NOT frame DEIM as sensor placement.
- Why POD basis for U: optimal (Eckart–Young) low-rank fit of the nonlinear snapshots; ordered by singular values, which the error analysis needs; gives the σ_{m+1} estimate.

## Demo (ground the code): FitzHugh–Nagumo, n=1024→k=m=5
ε v_t = ε^2 v_xx + f(v) − w + c,  w_t = b v − γ w + c,  f(v)=v(v−0.1)(1−v).
L=1, ε=0.015, b=0.5, γ=2, c=0.05, stimulus i0(t)=50000 t^3 e^{−15t}, Neumann BC v_x(0)=−i0, v_x(L)=0.
FD in space → ODE system dy/dt = A y + F(y) form. 100 snapshots over t∈[0,8]. Singular values decay fast (~40). POD=5/DEIM=5 captures limit cycle, ~1000x speed, negligible err.
For a SELF-CONTAINED small demo I'll use a single scalar reaction-diffusion (e.g. v_t = ε^2 v_xx + f(v) + stimulus with cubic f), n≈512–1024, build V_k from solution snapshots, U from F-snapshots, run DEIM point selection, integrate POD vs POD-DEIM, report relative error + speedup vs full F-evals. Mirror pyMOR deim() (greedy amax, incremental interp matrix) + the U(P^TU)^{-1}P^T formula.

## Scaffold (pre-method, piece-for-piece with final code)
- pod_basis(snapshots, k): SVD → leading left sing vecs. (exists: numpy.linalg.svd / scipy)
- galerkin_project(A, V): Ã=V^T A V. (exists)
- SLOT: choose m interpolation indices from a basis U (the DEIM greedy). pass
- SLOT: cheap reduced nonlinear term from m samples + precomputed B. pass
- integrate(...) forward Euler loop (exists), filling reduced RHS.
