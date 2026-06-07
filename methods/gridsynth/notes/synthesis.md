# Synthesis notes — gridsynth (Ross–Selinger optimal ancilla-free Clifford+T approx of z-rotations)

## Pain point / research question
Fault-tolerant QC: Clifford gates cheap, T-gate (=π/8 = diag(1, e^{iπ/4})) is the expensive non-Clifford one (magic-state distillation). Cost metric = **T-count**. Need: given a z-rotation Rz(θ)=diag(e^{-iθ/2},e^{iθ/2}) and ε>0, find a single-qubit Clifford+T circuit U with ‖Rz(θ)-U‖≤ε (operator norm) of minimal T-count.

## State of field / ancestors (load-bearing, with limitations)
- **Solovay–Kitaev** (geometric recursive net-then-refine). Works (universal), but circuit length O(log^c(1/ε)), c = log5/log(3/2) ≈ 3.97 (>3). Practically wasteful: huge constants, far above the info-theoretic lower bound. Asymptotically fine, constant-factor disaster.
- **Info-theoretic lower bound**: ~3 log₂(1/ε) T-gates needed (counting argument: number of distinct Clifford+T operators of T-count m grows like a constant·(something)^m; to ε-cover SU(2) directionally you need ≥3log₂(1/ε)). Concretely K+3log₂(1/ε).
- **Exhaustive search** (Fowler 2011; Kliuchnikov–Maslov–Mosca "practical" 2016): optimal T-count but EXPONENTIAL runtime. Fowler feasible to ε≈1e-4, KMM-practical to ε≈1e-17. Dead end for high precision.
- **Exact synthesis** (Kliuchnikov–Maslov–Mosca 2013, "Fast and efficient exact synthesis…", arXiv:1206.5236): a 2×2 unitary is **exactly** representable in Clifford+T **iff all entries ∈ D[ω]=Z[1/√2,i]**. And there's an efficient algorithm + the minimal T-count is determined by the **least denominator exponent k** of the entries (T-count = 2k-2 or 2k for the off-phase case). This is the key enabler: it tells you EXACTLY which unitaries are reachable and how cheaply.
- **First number-theoretic approx synth** (KMM 2013 approx, arXiv:1212.0822): O(log(1/ε)) gates, poly runtime, but uses ancillas.
- **newsynth / Selinger 2015** (arXiv:1212.6253): ancilla-free, T-count K+4log₂(1/ε), K≈10. Solves the same grid+Diophantine reduction but enumerates only a SUBSET of grid solutions and finds candidate u by a cruder method. Leaves a ×4/3 gap to the 3log₂ bound in the typical case.

## The method (discovery order)
1. **Reduce to representable unitary.** Approximate Rz(θ) by U=[[u,-t†],[t,u†]], u,t∈D[ω], unitary (u†u+t†t=1). Lemma (ℓ=0 wlog when ε<|1-e^{iπ/8}|). T-count = 2k-2 (k=least denom exp of u; TUT† trick removes the +2). So: minimize k = denom exp of u.
2. **Re-express the norm condition on u.** With z=e^{-iθ/2}: ‖Rz-U‖² = 2-2Re(z†u). So ‖·‖≤ε ⟺ Re(z†u) ≥ 1-ε²/2 ⟺ ⃗z·⃗u ≥ 1-ε²/2. This carves the **ε-region** R_ε out of the unit disk: a thin sliver bounded by the unit circle and a chord at distance ε²/2, normal direction ⃗z. Width ε²/2, contains a disk of radius ε²/4.
3. **Two constraints, two conjugations.** Need u∈R_ε (closeness) AND, for unitarity to be solvable, the √2-conjugate u•∈ unit disk D̄ (because u†u≤1 ⟹ also (u•)†(u•)≤1). (-)• is the ring automorphism √2 ↦ -√2. So: find u∈Z[ω] (scaled) with **u∈A=R_ε and u•∈B=D̄**. This is a **two-dimensional grid problem**.
4. **Solve the Diophantine eq.** For each candidate u, set ξ=1-u†u∈D[√2], solve t†t=ξ for t∈D[ω] (relative norm equation). Reduces to: factor n=(ξ•ξ)·2^ℓ ∈ Z; classify primes in Z[√2] and Z[ω] (Euclidean domains). Solvable iff ξ doubly-positive AND every odd-multiplicity prime factor p of n satisfies p=2 or p≡1,3,5 mod 8 (p≡7 mod 8 to an odd power kills it). Factoring is the only hard part ⟹ optional **factoring oracle** (Shor) ⟹ absolute optimality; without it, classical factoring + accept primes ⟹ near-optimality (n≡1 mod 8 by Lemma, so prime n always solvable; PNT says ~1/k_j chance prime ⟹ expected O(log 1/ε) candidates tried; extra T-count O(loglog 1/ε)).

## Grid-problem machinery (the technical innovation, Appendix A)
- 1D grid problem (α∈Z[√2], α∈A, α•∈B): efficient — rescale by λ=1+√2 so width<1, enumerate b in [(x0-y1)/√2³,(x1-y0)/√2³], unique a per b. O(1) per solution. Density bound: distinct α,β∈Z[√2] ⟹ |α-β||α•-β•|≥1.
- 2D upright rectangles → two 1D problems (via Z[ω]=Z[√2]+Z[√2]i ⊕ ω-coset, Lemma).
- **Uprightness** up(A)=area(A)/area(BBox(A)). M-upright sets: enumerate BBox grid, filter; O(1/M²) per solution.
- **Grid operators** G: real-linear, G(Z[ω])⊆Z[ω], "special" = det ±1. Preserve solutions: u solves (A,B) ⟺ Gu solves (G(A),G•(B)). So transform a skinny non-upright problem into an upright one.
- **Theorem (ellipse):** for ellipses A,B there's a grid operator G making G(A),G•(B) 1/6-upright, computable in O(log 1/M) ops. Proof = repeated **Step Lemma**.
- **Step Lemma / skew reduction:** state = pair of det-1 SPD matrices (D,Δ), each = [[eλ^{-z}, b],[b, eλ^z]], e²=b²+1. skew = b²+β², bias = ζ-z. Small skew ⟺ high uprightness (up=π/(4√(b²+1))). If skew≥15, one special grid operator G drops skew by factor ≤0.9. Proof: case-distinction on (z,ζ)-plane covered by regions, each with a fixed operator from {R, A^n, B^n, K, σ-shift, X, Z}:
  - **Shift** σ,τ (not grid ops, but conjugation gives grid ops): set bias∈[-1,1].
  - **R** = (1/√2)[[1,-1],[1,1]] (Hadamard-like rotation): both z,ζ near 0. skew(·R)=(b²+1)sinhλ²(z)+(β²+1)sinhλ²(ζ). r-bound 0.8.
  - **A^n** = [[1,-2],[0,1]]^n: z,ζ ≥ -0.3. uses g(x)=(1-2x)².
  - **B^n** = [[1,√2],[0,1]]^n: b≤0≤β, z,ζ≥-0.2. uses h(x)=(1-√2x)².
  - **K** = (1/√2)[[-λ^{-1},-1],[λ,1]]: z≤-0.3, ζ≥0.8. uses coshλ.
  - **X,Z** symmetries reduce # cases (Z negates anti-diag, X swaps diag).
- **Enclosing ellipse** (Prop): any bounded convex set inscribes in ellipse of area ≤ (4π/(3√3))≈2.418× (sharp, equilateral triangle). So general convex A,B reduce to the ellipse case.
- **Scaled grid problem**: u∈(1/√2^k)Z[ω], enumerate in increasing k (increasing denom exp). Lower bounds: rR≥(1+√2)²/2^k ⟹ ≥2 sols; ≥2 sols at k ⟹ ≥2^ℓ+1 at k+2ℓ (convex combination of two sols, exponential growth). This drives k₂=O(log 1/ε) and the expected-candidates bound.

## Worst case / typical
- Worst case still K+4log₂(1/ε) (matches newsynth), but only attained where it's actually optimal (when tan(θ/2)∈Q(√2): ε-region parallel to grid lines, # sols ∝ width not area, k~K+2log → 2k=K+4log). Typical (tan(θ/2)∉Q(√2)): #sols ∝ area ε³, grid density 4^k ⟹ 4^{-k}≤cε³ ⟹ 2k≥K+3log ⟹ T-count K+3log₂(1/ε) — matches info-theoretic bound. Conjecture, with experimental support.
- Up-to-phase variant: only λ∈{1, e^{iπ/8}} matter (Lemma: discrete determinant ⟹ phase ∈ e^{inπ/8}). Run both algos, take smaller. Interleave for increasing T-count (alg1 even, alg2 odd T-counts).

## Complexity
O(polylog(1/ε)) expected, with or without oracle. up(R_ε)=Ω(ε⁴) ⟹ log(1/M)=O(log 1/ε); candidates j₀=O(log 1/ε); each Diophantine polylog(n), n≤4^k. Experimentally ~O(log³(1/ε)). ε=1e-1000 achievable, T-count <10000, <50s.

## Empirical facts → context (recall, don't measure)
- SK c≈3.97; Fowler feasible ε≈1e-4; KMM-practical ε≈1e-17; newsynth K+4log₂, K≈10, T-count 13300 / 504.8s at some ε on i5-3570. These are PRIOR-ART facts → context.md. The proposed method's own benchmark table (its T-counts/runtimes) → EXCLUDED from reasoning.

## Code map (pygridsynth, canonical Python port of Selinger newsynth)
- gridsynth.py: EpsilonRegion, UnitDisk, _gridsynth_exact (loop over k), _gridsynth_up_to_phase.
- region.py: ConvexSet/Ellipse/Rectangle/Interval.
- to_upright.py: _step_lemma (the case distinction w/ skew≤15, bias regions; ops Z,X,S(σ),R,K,A,B), to_upright_ellipse_pair.
- tdgp.py / odgp.py: 2D / 1D grid problem solvers (scaled, with parity).
- grid_op.py: GridOp over Z[ω], special (det±1), inv, conj_sq2, adj.
- diophantine.py: diophantine_dyadic, prime classification mod 8, Pollard-rho factor (_find_factor), sqrt(-1)/sqrt(±2) mod p, _adj_decompose.
- synthesis_of_cliffordT.py: decompose_domega_unitary = KMM exact synthesis (denominator reduction by H/T from left).
- ring.py: ZRootTwo, DRootTwo, ZOmega, DOmega arithmetic, conj (†), conj_sq2 (•), denomexp.

## Final code for deliverables: a faithful, self-contained reduction mirroring pygridsynth's structure (rings, EpsilonRegion/UnitDisk, to_upright Step-Lemma loop, TDGP, diophantine, exact synth, main k-loop).
