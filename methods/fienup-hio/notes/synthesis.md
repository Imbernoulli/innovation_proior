# Synthesis — Fienup HIO

## Pain point / setting
Single-intensity phase retrieval (astronomy / crystallography): measure |F(u)| (Fourier modulus
from speckle interferometry power spectrum, or structure-factor magnitudes), object f(x) is real
and NON-NEGATIVE with finite SUPPORT. Want phase φ(u) / equivalently f(x). Only ONE modulus
known (unlike GS/electron-microscopy two-modulus case) — the second domain's "data" is replaced
by the a-priori constraints (real, ≥0, support).

DFT convention (Fienup 1982): F(u)=Σ_x f(x)exp(-i2πu·x/N), f(x)=N^{-2}Σ_u F(u)exp(i2πu·x/N)
(2-D, M=2 dims, so N^{-2}). Parseval: N^{-2}Σ_u|G|² = Σ_x|g|².

## Constraint sets
- Fourier set M = {y : |FT y| = |F|}, NON-CONVEX (product of circles of prescribed radius).
  Projection P_M: FT, keep phase, reset modulus to |F|, IFT. = first 3 steps of GS.
- Object set S (support only) = {y : y=0 outside D}, CONVEX (linear subspace). Projection: zero
  outside support.
- Object set S+ (support + nonneg) = {y : y=0 outside D, y≥0}. CONVEX. Projection:
  max(0,y) inside D, 0 outside.

## Error-reduction algorithm (generalized GS) — the baseline
Four steps (Eqs 6-9 in primary):
  G_k = FT(g_k);  G'_k = |F| exp(i·arg G_k);  g'_k = IFT(G'_k);
  g_{k+1} = project g'_k onto object constraints.
For single-intensity (Eq 10): g_{k+1}(x) = g'_k(x) for x∉γ;  0 for x∈γ,
  where γ = set where g'_k violates object constraints (negative, or outside support/diameter).
Fourier squared error: B_k = E_F,k = N^{-2}Σ_u|G_k − G'_k|² = N^{-2}Σ_u(|G_k|−|F|)².
Object squared error: E_O,k = Σ_x|g_{k+1}−g'_k|² = Σ_{x∈γ}[g'_k(x)]² (single-intensity).

### Convergence proof (Fienup 1982 §II, reproduce in reasoning)
By Parseval E_F,k² = N^{-2}Σ|G_k−G'_k|² = Σ|g_k−g'_k|². Since g_{k+1} is the NEAREST object
satisfying constraints to g'_k, |g_{k+1}−g'_k| ≤ |g_k − g'_k| pointwise  ⇒ E_O,k ≤ E_F,k.
Then Parseval again E_O,k² = N^{-2}Σ|G_{k+1}−G'_k|² and G'_{k+1} nearest to G_{k+1} in Fourier
set, G'_k also in that set ⇒ |G_{k+1}−G'_{k+1}| ≤ |G_{k+1}−G'_k| ⇒ E_F,k+1 ≤ E_O,k.
Chain:  E_F,k+1 ≤ E_O,k ≤ E_F,k.  Monotone non-increasing. (Eq 22)

### ER = steepest descent (Fienup 1982 §III, reproduce)
Minimize B=E_F²=N^{-2}Σ(|G|−|F|)² over the N² real values g(x). Gradient:
  ∂B/∂g(x) = 2[g(x) − g'(x)]   (Eq 28), where g'=IFT(|F|·G/|G|).
So gradient is computed FREE by the same 3 transform steps. Steepest descent step
g(x)−g_k(x) = −h ∂B/∂g. First-order Taylor optimum gives h=1/2 (Eq 32), but B is QUADRATIC
in g so linear approx underestimates step by half ⇒ DOUBLE-LENGTH step h=1 ⇒ g̃=g'_k, at which
B=0 exactly (Eq 33). Then impose object constraints (Eq 10). So ER ≡ double-length-step
steepest descent. This is WHY ER stalls: it's gradient descent on a non-convex B, gradient
becomes tiny on plateaus, h fixed → crawls. Also the constraint set is hit every step ("constantly
running into the object-domain constraints").

## The diagnostic phenomena (context, pre-method facts)
- ER error drops fast first ~30 iters, then PLATEAUS (0.16, then 0.02, then 0.003) — 2000+ iters
  unsatisfactory (Fienup 1982 Fig 2). Local, not global, minimum — convergence problem not
  uniqueness problem.
- STRIPES: stagnates at local min with low-contrast stripe pattern across image (Fienup 1981
  report, cited; §VII). Algorithm convergence problem, not uniqueness.
- TWIN IMAGE: f(x) and f*(−x) share the same |F| ⇒ a reconstruction can lock onto a
  superposition of object and its inverted conjugate. (background fact; centrosymmetry of |F|.)
- OUTPUT-OUTPUT stagnation: output unchanged over successive iterations though far from solution.

## Input-output viewpoint (§V) — the key reframing
The first 3 steps = a NONLINEAR SYSTEM: input g, output g'=IFT(|F|·FT(g)/|FT g|). Property:
output g' ALWAYS has correct Fourier modulus (lies in M). If output also satisfies object
constraints → it's a solution. So the input g need NOT be the current best estimate — it is a
DRIVING FUNCTION chosen to push the output where we want.
Empirical fact (refs 6,7,17): small input change Δg ⇒ output change ≈ α·Δg (plus nonlinear
terms). So to get a desired output change Δg', drive the input by β·Δg', β ideally ≈ −1 ... but
sign convention: we want output at violating points driven to ZERO, so desired Δg'(x) = −g'_k(x)
there. Logical next input = g + βΔg.

### Desired output change (Eq 41)
Δg_k(x) = 0 where constraints satisfied (x∉γ);  = −g'_k(x) where violated (x∈γ).

### The four variants (object-domain update rules — get EXACTLY right)
Let γ = points where g'_k violates object constraints (negative inside support, or outside support).
- BASIC input-output (Eq 42): g_{k+1}(x) = g_k(x)            x∉γ
                                          = g_k(x) − β g'_k(x)  x∈γ
- OUTPUT-OUTPUT (Eq 43):       g_{k+1}(x) = g'_k(x)            x∉γ
                                          = g'_k(x) − β g'_k(x)  x∈γ
   (β=1 ⇒ ER. So ER is suboptimal output-output.)
- HYBRID input-output HIO (Eq 44): combine upper line of (43) with lower line of (42):
                                g_{k+1}(x) = g'_k(x)            x∉γ
                                          = g_k(x) − β g'_k(x)  x∈γ
   Outside γ: take the output (already correct modulus + satisfies constraints).
   Inside γ:  feedback — input minus β·output. If output stays negative many iters, that input
   point grows without bound until the output must go non-negative → escapes the output-output
   stagnation. This is the cure.
β typically near 1 (Fienup 1982 found β≈unity best with the alternating strategy); modern
practice β∈[0.5,1], ~0.9 common. Large β → faster but unstable (multiple local minima in E_O
vs β). HIO is UNSTABLE: E_O can INCREASE for first iterations (0.071→0.137 over 4 iters in the
example) yet image quality improves; a few ER iterations afterward drop E_O sharply to a level
consistent with image quality. So practice = ALTERNATE 10–30 HIO iters with 5–10 ER iters.
For input-output, E_F is meaningless (input not an estimate); use object error E_O = Σ_{γ}[g'_k]².

## Practical (§VII)
- Support/diameter from autocorrelation = IFT(|F|²); object diameter = half autocorr diameter.
  In 2-D support of autocorr does NOT uniquely give object support (ref 10) → loose mask.
- Mask strategy: tight mask early (speed), looser later (avoid truncating off-center object,
  which itself causes stagnation).
- Starting input: randomized demagnified-thresholded autocorrelation (random unbiased, right
  size/shape). Constant phase + centrosymmetric data ⇒ stuck → use random.
- Restart with different random seed if stagnates; reconstruct 2-3× to confirm uniqueness.

## Third-party convex recasting (BCL — IN-FRAME use only: the convex-projection viewpoint,
## alternating projection onto convex sets, reflectors; do NOT cite Levi-Stark/BCL as posterior)
- ER (support only) = POCS  T = P_S P_M (alternating projection). Converges for convex sets;
  here M non-convex so no guarantee → plateaus.
- Reflector R_X = 2P_X − I.
- HIO support-only β=1: x_{n+1} = ½(R_S R_M + I)x  — the Douglas–Rachford / "reflect-reflect-
  average" map. General β: x_{n+1} = ½[R_S(R_M+(β−1)P_M)+I+(1−β)P_M]x.
  This is WHY HIO escapes where ER stalls: ER averages a point with a projection (drifts into
  the nonconvex trap); DR reflects across BOTH sets and averages → can overshoot a constraint
  set, stepping out of a basin ER cannot leave. The reflection is the relaxation that escapes
  the local minimum. (P_M midpoint of x and R_M x.)
- BIO = Dykstra's algorithm (nonconvex instance). (mention as sibling, lightly.)
NOTE: Douglas-Rachford (1956, heat-conduction splitting) and alternating projection onto convex
sets (von Neumann; Bregman 1965; Gubin–Polyak–Raik 1967) PREDATE 1982, so the
projection/reflection geometry is fair to reason with in-frame. The explicit "HIO IS Douglas-
Rachford" identification is later — present the GEOMETRY (reflect both sets, average) as the
narrator's own derivation, not as a cited theorem.

## Code (numpy, grounded in standard HIO implementations)
- forward = fft2, P_M = keep phase reset |F|, backward = ifft2.
- support mask boolean. ER: g'=P_M output; g_{k+1}= g' inside (support ∧ ≥0), 0 else.
- HIO: gamma = (g' < 0) | (~support).  g_{k+1} = g' where ~gamma; g_k − β g' where gamma.
- alternate HIO blocks with ER blocks; β≈0.9; support from thresholded autocorrelation.
- error monitor: object-domain E_O = sqrt(Σ_gamma g'² / Σ g'²).
