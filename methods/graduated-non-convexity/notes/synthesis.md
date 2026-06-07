# GNC synthesis

## Pain point / problem
Robust estimation: fit a model to data corrupted by OUTLIERS (gross errors) and recover
piecewise-smooth signals with DISCONTINUITIES. Least squares (quadratic ρ) is non-robust: a
single outlier with residual r contributes r², influence ψ=ρ'=2r grows without bound → one bad
point drags the whole fit. Need a ρ that DOWN-WEIGHTS large residuals (redescending influence).
But the robust ρ that does this (truncated quadratic, Geman-McClure) is NON-CONVEX → riddled
with local minima; gradient/coordinate descent lands in the wrong one, sensitive to init.

## Background / ancestors (knowable pre-method, citable)
- Tikhonov regularization: ill-posed inverse problems → add stabilizer; E(u)=Σ(u_s-d_s)² + λ Σ(u_s-u_t)². Convex, unique min, but SMOOTHS OVER discontinuities.
- Geman & Geman 1984: binary LINE PROCESS l_st∈{0,1} on dual lattice; smoothness term (u_s-u_t)²(1-l_st)+α l_st. Discontinuity present (l=1) → pay α, drop smoothness. Non-convex; solved by stochastic simulated annealing (expensive).
- Hinton 1978: "weak constraint" — smoothness enforced only while neighbors similar enough; beyond threshold dropped.
- Robust statistics (Huber 1981; Hampel et al. 1986): M-estimator min_a Σ ρ((d_i - u(i;a))/σ). Influence function = ψ = ρ'. Quadratic ψ=2r unbounded. Huber minimax: quadratic for |r|<ε, linear beyond — bounded ψ but NON-redescending (Ψ(0)=∞, low breakdown). Redescending: ψ→0 for large r (skipped mean = truncated quadratic; Lorentzian; Tukey biweight; Geman-McClure). These reject outliers but are non-convex.
- Shulman & Hervé 1989: discontinuities are outliers; Huber → convex but weak rejection.
- IRLS (Beaton-Tukey 1974, Campbell 1980): solve M-estimate by reweighting, weight z=ρ'(x)/(2x); but IRLS just minimizes Σ x²z with z a function of x — no explicit z variable, no constraints.

## The GNC primary content (Blake & Zisserman 1987, via Black-Rangarajan eqs 34-38)
Weak string/membrane energy: data term Σ(u_s-d_s)² + smoothness Σ g(u_s-u_t), where the neighbor
interaction is the TRUNCATED QUADRATIC g(t)=min(λ²t², α) — quadratic for small gradient, capped
at α (= a discontinuity). Eliminating the binary line process from Geman-Geman gives exactly this.
Non-convex (the cap creates the local minima).

GNC = build a one-parameter family ρ_c approximating the truncated quadratic, controlled by c:
piecewise-polynomial (BZ "GNC function"), with a QUADRATIC region, a smooth CONCAVE bridge
(negative-curvature middle), and a CONSTANT region. The parameter c controls the slope/curvature
of the bridge. BR eq (34), λ=weight, c=control, threshold barc/(1+barc):
  ρ(x,λ,c) = { λ²x²                                       0 ≤ λ²x² < c/(1+c)
             { 2λ|x|√(c(1+c)) − c(1+λ²x²)                 c/(1+c) ≤ λ²x² < (1+c)/c
             { 1                                          otherwise
As c→∞ ρ→truncated quadratic (the true cost). As c→0 the concave bridge flattens; below a
threshold the whole energy (data + smoothness) becomes CONVEX (BZ's convexity theorem: the
data-term curvature 2 plus the smoothness-term most-negative curvature ≥ 0). Start at small c
(convex → unique global min, no init needed), minimize, INCREASE c slowly, re-minimize from the
previous solution, TRACK the minimizer to c→∞. Deterministic continuation = cheaper than SA.

NOTE on direction: BZ/BR use c↑ from 0 (convex) to ∞ (true cost). Yang's μ for TLS also goes
μ→0 (convex) to μ→∞ (true cost); Yang's μ for GM goes μ→∞ (convex) to μ→1 (true cost). The
"increasing non-convexity" continuation is the invariant.

## Black-Rangarajan duality (1996) — the bridge to outlier weights / IRLS
Claim: min_x ρ(x) ≡ min_{x, z∈[0,1]} [ x² z + Ψ(z) ] for a class of robust ρ. The weight z is an
explicit "outlier process": z≈1 inlier, z≈0 outlier; Ψ(z) penalizes declaring an outlier.
Derivation (eqs 16-29):
- E(x,z)=x²z+Ψ(z). Want min over z to reproduce ρ(x): ρ(x)=min_z(x²z+Ψ(z)).
- ∂E/∂z=0: x² + Ψ'(z)=0  (19).  Also at the min, dρ/dx matches: ρ'(x)/(2x)=z  (17). This z is exactly the IRLS weight.
- Define φ(x²)≜ρ(x). Then φ'(x²)=ρ'(x)/(2x) (22), so z=φ'(x²) (26).
- −x²=Ψ'(φ'(x²)) (23). Integrate by parts (multiply by 2xφ''): Ψ(φ'(x²))=−x²φ'(x²)+φ(x²) (25).
- Substituting z=φ'(x²): Ψ(z)=φ((φ')⁻¹(z)) − z(φ')⁻¹(z) (27). i.e. Ψ is (minus) the convex conjugate of φ.
Conditions for validity (Fig 10 mechanism): with φ(w)=ρ(√(w/τ)),
  lim_{w→0} φ'(w)=1, lim_{w→∞} φ'(w)=0, φ''(w)<0 (φ concave).
- φ concave ⟺ Ψ''(φ'(x²))=−1/φ''(x²)>0 ⟺ z=φ'(x²) is a MINIMUM of E over z (28-29). Beautiful.
Catalog (exact, verified):
  Geman-McClure: ρ(x)=x²/(1+x²), ψ=2x/(1+x²)², φ(w)=w/(1+w), φ'=1/(1+w)², (φ')⁻¹(z)=−1+1/√z, Ψ(z)=(−1+√z)², E=x²z+Ψ(z). Weight z=1/(1+x²)²=φ'(x²).
  Lorentzian: ρ=log(1+½(x/σ)²), Ψ(z)=z−1−log z, weight z=1/(1+½(x/σ)²) … (non-redescending: Ψ(0)=∞).
  Truncated quadratic (skipped mean): linear Ψ(z)=1−z (BR §5.4: c→∞ limit of GNC), binary outlier z∈{0,1}.
  GNC function: Ψ(z,c)=c(1−z)/(c+z) (eq 37), retains control parameter c.
  L1: Ψ=1/(4z); Huber: Ψ=ε/(2z); Tukey: Ψ=⅓−z+⅔z^{3/2}; Leclerc: Ψ=z log z − z + 1.

## Yang 2020 modern revival (target — DO NOT cite as artifact)
Combine GNC + BR duality + non-minimal (global) solvers for the weighted LS variable update.
- robust problem: min_{x∈X} Σ ρ(r(y_i,x)). GNC surrogate ρ_μ. BR-dual: min_{x, w_i∈[0,1]} Σ [w_i r²(y_i,x) + Φ_{ρ_μ}(w_i)].
- ALTERNATE: (1) variable update x ← argmin Σ w_i r²(y_i,x) (weighted LS, solved globally by non-minimal solver / or normal equations for linear); (2) weight update w_i ← closed form.
- GM surrogate: ρ_μ(r)=μ c̄² r²/(μ c̄² + r²); Φ=μ c̄²(√w−1)²; weight w_i=(μ c̄²/(r̂_i²+μ c̄²))². Convex μ→∞ (quadratic), true GM at μ=1. Init μ=2 r²_max/c̄², μ←μ/1.4, stop μ<1.
- TLS surrogate: ρ_μ piecewise, quadratic for r²∈[0, μ/(μ+1) c̄²], smooth bridge 2c̄|r|√(μ(μ+1)) − μ(c̄²+r²) for r²∈[μ/(μ+1)c̄², (μ+1)/μ c̄²], const c̄² beyond. ρ''_μ=−2μ→0 convex as μ→0; true TLS as μ→∞. Φ=μ(1−w)/(μ+w) c̄².
  Weight update: w=0 if r²≥(μ+1)/μ c̄²; w=1 if r²≤μ/(μ+1) c̄²; else w=√(c̄² μ(μ+1)/r²)−μ.
  Init μ=c̄²/(2 r²_max−c̄²), μ←1.4μ, stop when Σ w_i r̂_i² converges (or weights binary).
- c̄ = max inlier error (noise bound), e.g. χ²_inv(0.99,dof)·σ². "residuals" in code = SQUARED residuals r².

## Canonical code (GNC-and-ADAPT, MATLAB → port to Python, TLS, linear regression)
gnc.m + gncWeightsUpdate.m + leastSquareNorm2.m:
- weighted LS: x=(AᵀWA)⁻¹AᵀWy (normal equations).
- TLS thresholds th1=(μ+1)/μ·c̄², th2=μ/(μ+1)·c̄²; weight branches as above.
- init μ=1/(2 max_res/c̄² − 1) = c̄²/(2 max_res − c̄²); μ←1.4μ; stop cost_diff<thresh or binary weights.
- inliers = w>1−eps.

## Demo to build (Python, self-contained)
1D robust line fit y=ax+b with 80% outliers; plain LS fails, GNC-TLS recovers. Plus the GM
weight-update variant and (optionally) a 1D weak-membrane reconstruction. Show convexity at start,
continuation, recovery.

## Design-decision → why
- Why non-convex ρ at all? redescending influence (ψ→0) needed for high breakdown; convex Huber has low breakdown. The non-convexity is the price of strong rejection.
- Why start convex? unique global min reachable from ANY init — kills the init-sensitivity.
- Why graduate slowly (not jump to true cost)? jumping = optimizing the non-convex cost from arbitrary point = back to local-min trap. Slow deformation keeps tracked min near global.
- Why the BR-dual weight form? converts each inner minimization into alternating weighted-LS (cheap, global) + closed-form weight — and the weight IS the IRLS/influence weight, so it's interpretable as soft outlier rejection. Adds explicit z that allows constraints (spatial coherence) IRLS can't.
- Why Φ exactly (−1+√z)² etc.? forced by the conjugate construction (eq 27) — not a free heuristic; that's the whole point of duality vs heuristic switchable-constraint SLAM.
- Why μ init = 2r²_max/c̄² (GM)? makes surrogate convex over the actual residual range at iter 0.
- Continuation factor 1.4: geometric, scale-free; slow enough to track, fast enough to finish.
