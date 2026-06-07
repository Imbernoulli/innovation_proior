# ART/SART synthesis notes

## Sources retrieved (this run)
- PRIMARY: Andersen & Kak 1984, "SART: A superior implementation of the ART algorithm", Ultrasonic Imaging 6, 81-94 (full text, Purdue PDF). Eq.(12) sequential ART; Eq.(13) simultaneous SART update; bilinear elements; longitudinal Hamming window Eq.(16); partial weights so Σ_i a_ij = L_j.
- PRIMARY (reproduces GBH 1970 ART exactly): Kak & Slaney, "Principles of Computerized Tomographic Imaging", Ch.7 (full text, Purdue PDF). Eq.(4) Kaczmarz projection; Eq.(13)-(16) general correction; Eq.(17) binary ART (= GBH 1970); Eq.(18) length-normalized ART; Eq.(31) sequential ART; Eq.(32) SART. Kaczmarz [Kac37], Tanabe [Tan71] convergence.
- BACKGROUND: Kaczmarz 1937 update + convergence (HandWiki/Grokipedia): x_{k+1}=x_k+λ(b_i-⟨a_i,x_k⟩)/‖a_i‖² a_i; orthogonal projection onto hyperplane a_i·x=b_i; consistent system from x_0=0 → minimum-norm solution; randomized rate E‖x_k-x‖²≤(1-κ⁻²)^k‖x_0-x‖². Radon transform / FBP as analytic alternative (ramp filter |ω|, projection-slice theorem).
- THIRD-PARTY explainer: Kak-Slaney textbook itself is the canonical explainer; HandWiki Kaczmarz.
- CODE grounding: hanyoseob/python-ART (block form x←x+μ Aᵀ(b-Ax)/AᵀA); scikit-image SART. I build explicit sparse A via parallel-beam pixel-intersection lengths (standard).

## Exact math (verified)
Kaczmarz / ART row update for ray i:
  x ← x + λ (b_i − a_i·x)/‖a_i‖² · a_i
- a_i = i-th row of A (intersection lengths of ray i with each pixel), b_i = measured ray-sum.
- ‖a_i‖² = Σ_j a_ij². λ=1 is pure projection; λ<1 underrelaxation for noise.
- Binary GBH form (Kak-Slaney Eq.17): a_ij∈{0,1}, ‖a_i‖²=N_i (count), correction (p_i−q_i)/N_i to each hit pixel.
- Length-normalized (Eq.18): Δf_j = w_ij (p_i−q_i)/(L_i N_i).

Geometry: each equation a_i·x=b_i is a hyperplane in R^N; nearest point of x to it is the orthogonal projection (derivation: minimize ‖x−z‖² s.t. a_i·z=b_i → Lagrange → z=x+((b_i−a_i·x)/‖a_i‖²)a_i). Cyclic projection = Kaczmarz. Converges to a point in the intersection if consistent; to min-norm solution closest to x_0 if underdetermined (Tanabe); oscillates near intersection if overdetermined+noisy.

Convergence rate depends on angles between hyperplanes; orthogonal → 1 sweep; near-parallel → slow. Hence Hounsfield's trick: order rays so successive projections are far apart in angle (73.8° in the paper).

SART (Andersen-Kak Eq.13 / Kak-Slaney Eq.32): for all rays i in one projection (view) simultaneously,
  g_j ← g_j + [ Σ_{i∈view} a_ij (p_i − q_i)/(Σ_k a_ik) ] / ( Σ_{i∈view} a_ij )
where q_i = Σ_k a_ik g_k (current forward projection), Σ_k a_ik = L_i (ray length).
- Inner: per-ray residual normalized by ray length L_i = Σ_k a_ik, back-distributed with weight a_ij.
- Outer normalization: divide by Σ_{i∈view} a_ij (column sum = total weight pixel j receives in this view).
- This is: residual normalized by ROW sums (ray lengths), back-projected, then normalized by COLUMN sums (pixel total weight in view). Row/column weighting = the SART signature.
- Three SART ingredients: (1) simultaneous per-view update (vs ray-by-ray); (2) longitudinal Hamming window on back-distribution (w_ij = Σ h_im d_ijm Δs); (3) bilinear elements + partial first/last weights so Σ_i a_ij = L_j.
- Optional relaxation λ in front of the bracket.

## Design-decision → why
- Why iterate not invert: A is ~65000×65000 sparse, noisy, often under/over-determined; direct inversion or least-squares infeasible at that size; iterative row-action is O(nnz) per sweep, low memory (can compute weights on the fly).
- Why ‖a_i‖² normalization: it's exactly what makes the step the orthogonal projection onto the hyperplane (closest point), so each step is geometrically optimal for that one equation and provably non-expansive.
- Why relaxation λ<1: inconsistent equations (discretization + noise) make pure projections (λ=1) chase contradictory hyperplanes → salt-and-pepper; underrelaxation averages the contradictions, smoother at cost of more sweeps.
- Why order rays far apart in angle: near-parallel hyperplanes → tiny angle → slow zig-zag convergence; orthogonal → fast. Adjacent rays/views are nearly parallel, so interleave widely separated angles.
- Why SART simultaneous: ray-by-ray, each new ray undoes pixels the previous ray just set (competing updates along a stripe) → salt-and-pepper; averaging all corrections in a view removes the per-ray stripe ambiguity, uniquely defines each pixel's correction → smooth in one pass.
- Why row-sum (L_i) then column-sum normalization in SART: row-sum makes the per-ray residual a proper average correction independent of ray length; column-sum makes the total correction to a pixel independent of how many rays in the view happen to cross it (uniformity), and keeps dimensions of g.
- Why bilinear elements: pixel basis → discontinuous f, poor ray-integral approximation → larger inconsistencies; bilinear (pyramid) basis → continuous f, better forward model, fewer inconsistencies, can use fewer rays.
- Why Hamming window along ray: corrections near where a ray enters/exits the circle are based on partial-length, less reliable cells; emphasize mid-ray corrections, de-emphasize ends → less edge noise.
- Why nonneg/support constraint: physical attenuation ≥0 and zero outside object; projecting onto these convex sets each sweep is free extra information, accelerates and regularizes.
