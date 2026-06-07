# Synthesis — MAX CUT via SDP + random hyperplane rounding (GW)

Grounding sources (all retrieved this run, under refs/ and code/):
- refs/gw95-jacm.pdf / .txt — PRIMARY. Goemans & Williamson, JACM 42(6):1115–1145, 1995. Read §1 (motivation), §2 (algorithm), §3 (analysis: Thm 3.1, Lem 3.2, Thm 3.3, Lem 3.4, Lem 3.5), §4 (SDP relaxation (SD), dual (D), Delorme–Poljak (EIG) equivalence).
- refs/delorme-poljak-1993.pdf / .txt — ANTECEDENT. Delorme & Poljak, Math. Prog. 62:557–574, 1993. Laplacian eigenvalue upper bound φ(G), Rayleigh principle, correcting vectors, min over correcting vectors.
- refs/toronto-sdp-lecture12.pdf / .txt — ANALYSIS. Toronto CSC2411 Lecture 12. LP-not-tight motivation, IQP→VP relaxation, C3 example, full rounding analysis (Pr[sep]=θ/π, α=0.878).
- refs/utah-lec14-maxcut.pdf / .txt — ANALYSIS (secondary).
- code/rigetti-gw.py — CANONICAL CVXPY implementation (Rigetti quantumflow-qaoa). PSD variable, diag==1, maximize trace(0.25·L·X), eigh recover vectors, random Gaussian hyperplane, sign rounding.

## Pain point / research question
MAX CUT: partition V to maximize cut weight w(S,S̄)=Σ_{i∈S,j∉S} w_ij. NP-complete (Karp 1972). Best approximation since 1976 was Sahni–Gonzales: assign greedily / equivalently flip a fair coin per vertex → expected cut = (1/2)Σ w_ij ≥ (1/2)·OPT. So 1/2-approx. Twenty years, no improvement on the constant. The various 1/2 + 1/(poly) bounds (Vitányi, Poljak–Turzík, Haglin–Venkatesan, Hofmeister–Lefmann) only beat 1/2 by additive lower-order terms; the worst-case constant stayed 1/2.

Why stuck at 1/2: the analysis compares the cut to the TOTAL weight Σ w_ij, and Σ w_ij can be ≈ 2·OPT (an almost-bipartite graph), so any "fraction of total weight" bound caps at 1/2. (GW §1: "previous approximation algorithms compared the value... to the total sum of the weights... This sum can be arbitrarily close to twice the value of the maximum cut.") Need a TIGHTER upper bound on OPT to compare against.

## The LP wall (analysis notes)
Natural LP/combinatorial relaxations of MAX CUT are not tight (Toronto §2.2–2.3): bad integrality gap. Canonical small witness: triangle C3. OPT_cut(C3)=2. The vector/SDP relaxation gives 9/4 (three unit vectors pairwise at angle 2π/3). LP can't beat the loose bounds. So we need a relaxation strictly between combinatorial LP and the integer program — one whose optimum is a genuinely tight upper bound on OPT.

## Antecedent that pointed the way: Delorme–Poljak eigenvalue bound
DP 1993: Laplacian L with L_ij=−w_ij (i≠j), L_ii=Σ_j w_ij. For S, x_S∈{±1}^n, x_S^T L x_S = Σ w_ij(x_i−x_j)^2 = 4·w(S,S̄). Rayleigh: λ_max(M) ≥ x^T M x / x^T x. With "correcting vector" u (Σu_i=0, U=diag(u)), x_S^T U x_S = Σ u_i = 0, so 4·mc(G) = x_S^T(L+U)x_S ≤ λ_max(L+U)·n. Hence mc(G) ≤ (n/4)λ_max(L+U) for every correcting u; minimize over u → φ(G), a polynomial-time-computable upper bound that is empirically very tight (worst case they knew: C5, ratio 32/(25+5√5)=0.88445). They could not prove a worst-case ratio better than 0.5.

Crucial clue: there EXISTS a tight, efficiently computable upper bound on OPT. GW §4.3 proves their SDP relaxation (SD) is *equivalent* to DP's (EIG) (equivalence via Poljak–Rendl 1995a). The SDP and the eigenvalue bound are the same object viewed two ways.

## The leap (the heart of reasoning.md)
1. MAX CUT as integer quadratic program (Q): max ½Σ w_ij(1−y_i y_j), y_i∈{±1}.
2. y_i∈{±1} = unit vector in R^1. RELAX: let y_i become a unit VECTOR u_i∈S_n, replace y_i y_j by inner product u_i·u_j. Objective ½Σ w_ij(1−u_i·u_j) = (P), reduces to (Q) in 1-D, valid relaxation; Z*_P ≥ Z*_MC.
3. (P) solvable: with Y=(u_i·u_j) = Gram matrix, Y⪰0 and y_ii=1 ⇔ unit vectors. So (P) ≡ SDP (SD): max ½Σw_ij(1−y_ij) s.t. y_ii=1, Y⪰0. Solvable to ε in poly time (ellipsoid / interior point). Recover vectors by Cholesky/eigendecomposition of Y.
4. ROUNDING: vectors on a sphere; need ±1. Random hyperplane through origin: r uniform on sphere, S={i: u_i·r ≥ 0}. Motivation: (P) is rotation-invariant, so a rotation-invariant rounding is natural.
5. ANALYSIS. Edge (i,j) cut iff u_i,u_j on opposite sides. Lemma 3.2: Pr[sgn(u_i·r)≠sgn(u_j·r)] = θ_ij/π, θ_ij=arccos(u_i·u_j). Proof: project to the 2-plane spanned by u_i,u_j; the normal's projection is uniform on the circle; the two arcs that separate them subtend angle θ each → measure 2θ/2π = θ/π.
6. E[W] = Σ w_ij · θ_ij/π. Per-edge ratio to relaxation term ½(1−cos θ): (θ/π)/(½(1−cosθ)) = (2/π)·θ/(1−cosθ). Minimize over θ∈(0,π]:
   α = min_{0<θ≤π} (2/π)·θ/(1−cosθ).
7. Lemma 3.5: α = 0.87856..., attained at θ* = 2.331122 rad (root of cos θ + θ sin θ = 1). Then E[W] ≥ α·Σ½w_ij(1−u_i·u_j) = α·Z*_P ≥ α·Z*_MC. 0.878-approx.

Key sign/constant checks (verified against primary):
- α = min_θ (2/π) θ/(1−cosθ) = min_θ (θ/π)/((1−cosθ)/2). Same thing. ✓
- θ* = 2.331122 rad ≈ 133.56°, root of cos θ + θ sin θ = 1. ✓ (GW Lem 3.5)
- Per-edge probability θ/π = arccos(u_i·u_j)/π. ✓
- Edge contribution to relaxation = ½ w_ij(1 − u_i·u_j) = w_ij(1−cosθ)/2. ✓
- For 0<θ≤π/2 the ratio 2θ/(π(1−cosθ)) > 1; concavity of 1−cosθ on [π/2,π] bounds below; min at θ*. ✓

## Design decisions → why
- Why vectors/SDP not LP? LP/combinatorial relaxations are not tight (integrality gap), capping near 1/2; SDP (PSD Gram matrix) is strictly tighter, still poly-solvable, and coincides with DP's empirically-tight eigenvalue bound.
- Why relax to UNIT vectors (Gram diag=1)? y_i∈{±1} ⇔ unit 1-D vector; smallest faithful relaxation keeps norm 1, lifts dimension. Diagonal=1 ⇔ u_i·u_i=1.
- Why a RANDOM hyperplane (uniform r)? Relaxation invariant under orthonormal transforms of vectors → rounding should be rotation-invariant; uniform hyperplane is. Also makes Pr[cut] depend only on θ_ij — exactly what the objective rewards.
- Why r ~ i.i.d. N(0,1) coords? Normalized Gaussian is uniform on sphere; sign test ignores normalization, so r~N(0,I) suffices (Knuth).
- Why 0.878 and not better with same scheme? Worst angle θ*≈133° edge realizes ratio; GW note straightforward modifications won't beat it; C5 gives 0.88446 just above.

## Code grounding (rigetti-gw.py)
- laplacian = 0.25·L(G); Maximize trace(L/4 · X) equals cut objective (x^T L x/4 = cut). PSD variable X, constraint diag(X)==1.
- evals,evects=eigh(X); vectors from eigenvectors (sign rounding needs only directions).
- random_vector ~ randn, normalize; colors = sign(vec·r); score = colors^T (L/4) colors.
This is the piece-for-piece target for the final code.
