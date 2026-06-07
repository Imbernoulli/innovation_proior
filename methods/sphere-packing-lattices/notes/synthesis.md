# Synthesis — Sphere packing, Cohn–Elkies LP bound, Viazovska magic function

## The problem
Find sup over packings P of equal balls in R^d of density Δ_P. Known exactly only d=1 (Δ=1), d=2 (hexagonal, π/√12, Thue/Fejes Tóth), d=3 (Kepler, π/√18, Hales). For d>3 unknown until 2016. Want: a *certificate* (upper bound) that matches a known lattice's density.

## Load-bearing ancestors
- **Delsarte LP bound for codes (1972, Delsarte72)**: for error-correcting codes, choose an auxiliary function positive-definite on the Hamming scheme; its positivity (dual) gives an upper bound on code size. The pattern: certify a packing-type bound by a single well-chosen function with sign conditions on the function and its "dual transform" (here the Fourier/zonal-spherical transform).
- **Kabatiansky–Levenshtein (1978, KabLev)**: deduced sphere-packing bounds from spherical-code bounds — indirect, loses sharpness.
- **Poisson summation**: ∑_{x∈Λ} f(x) = (1/covol Λ) ∑_{t∈Λ*} f̂(t). This is the engine: it relates values of f on a lattice to values of f̂ on the dual.
- **Modular forms / theta series**: Θ_Λ(z)=∑_{x∈Λ} e^{πi‖x‖²z} is a modular form whose weight is d/2; this is the classical bridge between lattices and SL₂(Z). Eisenstein E₄, E₆, E₂ (quasimodular), Jacobi thetanulls θ₀₀,θ₀₁,θ₁₀, Δ=(E₄³−E₆²)/1728, j=1728E₄³/(E₄³−E₆²).
- **Gaussian self-duality of Fourier transform**: F(e^{πi‖x‖²z})(y)=z^{−d/2} e^{πi‖y‖²(−1/z)}. The Fourier transform acts on the modular variable z by S: z↦−1/z. THIS is why modular forms are the right tool: building f as ∫ (modular form)(z) e^{πi r² z} dz turns the Fourier transform into the modular S-action, so modular transformation laws become Fourier-eigenfunction statements.

## The LP bound (Cohn–Elkies 2003, ElkiesCohn / math/0110009)
THEOREM. f:R^d→R admissible, not ≡0, with
(i) f(x) ≤ 0 for ‖x‖ ≥ r (here r=1),
(ii) f̂(x) ≥ 0 for all x.
Then density ≤ (f(0)/f̂(0))·vol B_d(0, r/2) = (f(0)/f̂(0))·(r/2)^d π^{d/2}/Γ(d/2+1).
Equivalently center density δ ≤ f(0)/(2^d f̂(0)).

PROOF (periodic packing = N translates v_1..v_N of lattice Λ, min distance r=1):
density = N·vol B(½)/covol(Λ). Apply Poisson to each shifted lattice:
∑_{x∈Λ} f(x+v_j−v_k) = (1/covolΛ) ∑_{t∈Λ*} e^{2πi t·(v_j−v_k)} f̂(t).
Sum over j,k=1..N. LHS: the only terms with argument of norm <1 are x=0,j=k, giving ≤ N f(0) by (i). RHS: = (1/covolΛ) ∑_t f̂(t) |∑_j e^{2πi t·v_j}|² ≥ (1/covolΛ)·f̂(0)·N² by (ii) (t=0 term). So N f(0) ≥ N² f̂(0)/covolΛ ⟹ covolΛ/N ≥ f̂(0)/f(0) ⟹ density ≤ (f(0)/f̂(0)) vol B(½). Reduce to radial f by rotational averaging (hypotheses rotation-invariant; FT preserves radiality).

NUMERICS: parametrize f(r)=p(r²)e^{−πr²} (p polynomial; Laguerre/Hermite basis = Gaussian × poly are FT-friendly since Hermite functions are FT-eigenfunctions); impose (i),(ii) at sample radii ⟹ finite LP. Cohn–Elkies got d=8 bound only 1.000001× the E8 density; d=24 within 1.000707 (later Cohn–Kumar 2009 to ~1+10^{−29} in d=8). Best known bounds d=4..36.

## Magic-function forcing (Conjecture 8.1 of ElkiesCohn; lived out in Viazovska §2)
If equality holds for E8 (self-dual/unimodular, scaled to min length √2):
Poisson on E8: ∑_{Λ8} g = ∑_{Λ8} ĝ (self-dual). With g(0)=ĝ(0)=1, g(ℓ)≤0 for ‖ℓ‖≥√2, ĝ≥0:
∑ g ≤ g(0)=1 and ∑ ĝ ≥ ĝ(0)=1, and they're equal ⟹ g(ℓ)=ĝ(ℓ)=0 for ALL ℓ∈Λ8∖0. Also g≤0 near these zeros forces g'(ℓ)=0 too ⟹ DOUBLE zeros at every lattice length √(2n), n≥2; SINGLE zero (sign change) at the minimal length √2.
E8 vectors: ‖x‖²∈{0,2,4,6,...}=2Z≥0. So required: g(r) and ĝ(r) vanish to order 2 at r=√(2n) for n≥2, simple at √2.

## Viazovska's construction in d=8 (1603.04246)
Build g = α a + β b, a=+1 Fourier eigenfunction, b=−1 eigenfunction, each a radial Schwartz fn that ALREADY has the forced double zeros.

The double-zero factor: for r>√2,
 a(r) = −4 sin²(πr²/2) ∫_0^{i∞} φ₀(−1/z) z² e^{πi r² z} dz,
 b(r) = −4 sin²(πr²/2) ∫_0^{i∞} ψ_I(z) e^{πi r² z} dz.
sin²(πr²/2) vanishes to order 2 exactly at r²∈2Z, i.e. at all √(2n) — that's the magic double-zero structure built in. (Identity used: −4 sin²(πr²/2) = e^{πir²}−2+e^{−πir²}, the symmetric second difference in the e^{πir²z} variable.)

Eigenfunction mechanism: a(x)=∫ Φ(z) e^{πi‖x‖²z} dz with Φ a (quasi)modular combination. Apply F: Gaussian self-duality sends e^{πi‖x‖²z}→z^{−4} e^{πi‖y‖²(−1/z)}; change variable w=−1/z (the S-action); the weight-matched modular transformation of Φ under S returns ±the same integral ⟹ â=a (eigenvalue +1) or b̂=−b (−1). Weight bookkeeping: contour pieces carry (cz+d) factors; the relevant weight is 2−d/2 (= −2 for d=8) so that z^{−d/2} from the Gaussian cancels against the form's automorphy factor.

The +1 piece (a): from weakly-holomorphic forms φ_{−2}=−1728 E₄E₆/(E₄³−E₆²), φ_{−4}=1728 E₄²/(E₄³−E₆²). Set
 φ_{−4}=φ_{−4}, φ_{−2}=φ_{−4}E₂+φ_{−2}(weak), φ₀=φ_{−4}E₂²+2φ_{−2}(weak)E₂+j−1728.
φ₀ is NOT modular but quasimodular: φ₀(−1/z)=φ₀(z)−(12i/π)(1/z)φ_{−2}(z)−(36/π²)(1/z²)φ_{−4}(z). a(x) defined as a 4-term contour integral (eqn a definition); Proposition shows â=a using exactly this transformation law and 1-periodicity. Values: a(0)=−i·8640/π, a(√2)=0, a'(√2)=i·72√2/π.

The −1 piece (b): h=128(θ₀₀⁴+θ₀₁⁴)/θ₁₀⁸ ∈ M!_{−2}(Γ₀(2)). Define ψ_I=h−h|_{−2}ST, ψ_T=ψ_I|_{−2}T, ψ_S=ψ_I|_{−2}S; Jacobi identity ⟹ ψ_T+ψ_S=ψ_I and ψ_I|S=ψ_S, ψ_S|S=ψ_I, ψ_T|S=−ψ_T. b(x) is a 4-term contour integral; b̂=−b. Values: b(0)=0, b(√2)=0, b'(√2)=2√2 π i.

Assembly: g = (πi/8640) a + (i/(240π)) b. Then g(0)=ĝ(0)=1 (from a(0),b(0)). For r>√2:
 g(r)=(π/2160) sin²(πr²/2) ∫_0^∞ A(t) e^{−πr²t} dt, A(t)=−t²φ₀(i/t)−(36/π²)ψ_I(it).
Need A(t)<0 on (0,∞) ⟹ g≤0 (condition (i)). And ĝ(r)=(π/2160) sin²(πr²/2)∫ B(t) e^{−πr²t} dt with B(t)=−t²φ₀(i/t)+(36/π²)ψ_I(it); need B(t)>0 ⟹ ĝ≥0 (condition (ii)). Both verified by asymptotic expansions (A₀^{(6)},A_∞^{(6)}, etc.) + Fourier-coefficient bounds |c(n)|≤2e^{4π√n} (Bruinier convergent expansion) + interval arithmetic on a compact range. Done ⟹ Δ₈ = π⁴/384 ≈ 0.25367, attained by E8 (centers ½^{1/2}Λ8... actually (1/√2)Λ8). Uniqueness of densest periodic packing follows (Conjecture 8.1 conclusions).

## d=24 parallel (1603.06518)
Same skeleton, different modular data; Leech min length scaled to 2 (squared 4), vectors length √(2n), n≥2, double zeros at n≥3? — minimal vectors squared norm 4 = 2·2, so the simple zero is at r=2 (n=2) and double zeros at √(2n), n≥3 — wait: Leech min length √4=2, lengths are √(2n) n=2,3,... so simple at √4=2, double at √(2n) for n≥3.
+1 eigenfunction a: weakly-holomorphic quasimodular φ of weight −8, depth 2 for SL₂(Z):
 φ = [(25E₄⁴−49E₆²E₄)+48E₆E₄²E₂+(−49E₄³+25E₆²)E₂²]/Δ²; quasimodularity z⁸φ(−1/z)=φ+φ₁/z+φ₂/z². a(r)=−4 sin²(πr²/2)∫₀^{i∞} φ(−1/z) z^{10} e^{πir²z} dz (note z^{10}, weight bookkeeping for d=24).
−1 eigenfunction b: weakly-holomorphic modular form weight −10 for Γ(2):
 ψ_I = (7Θ₀₁²⁰Θ₁₀⁸+7Θ₀₁²⁴Θ₁₀⁴+2Θ₀₁²⁸)/Δ²; ψ_S=ψ_I|_{−10}S = −(7Θ₁₀²⁰Θ₀₁⁸+...)/Δ², ψ_T=ψ_I|_{−10}T; ψ_S+ψ_T=ψ_I. b(r)=−4 sin²(πr²/2)∫₀^{i∞} ψ_I(z) e^{πir²z} dz.
Assembly g = (π/28304640) a + (1/(65520π)) b. Sign conditions reduce to:
 A(t)=(π/28304640) t^{10}(φ(i/t)+(432/π²)ψ_S(i/t)) ≤ 0,
 B(t)=(π/28304640) t^{10}(φ(i/t)−(432/π²)ψ_S(i/t)) ≥ 0,
i.e. φ(it)+(432/π²)ψ_S(it) ≤ 0 and φ(it)−(432/π²)ψ_S(it) ≥ 0. ψ_S(it)≤0 is immediate from the theta product being ≤0; the φ inequalities need work. Δ₂₄ = π¹²/12! = 0.0019295743..., attained by Leech. Posted 1 week after d=8.

## Design decisions → why
- LP bound via Poisson, not direct geometry: Poisson is the only handle that converts "no two centers within distance 1" + lattice structure into a single inequality on f vs f̂. Sign conditions (i),(ii) are precisely what makes both sides of Poisson controllable.
- Two sign conditions, one for f one for f̂: dictated by which side of Poisson you're bounding — f≤0 bounds the geometric (primal) side from above, f̂≥0 bounds the dual side from below.
- Why double zeros: forced by the equality case of Poisson on a self-dual lattice (not a choice).
- Why modular forms: the Fourier transform's action on Gaussians IS the modular S-transform z↦−1/z; an integral transform of a weight-(2−d/2) form is automatically a Fourier eigenfunction. No other function class has this exact compatibility.
- Why sin²(πr²/2): cheapest analytic factor with double zeros exactly on r²∈2Z (= the e^{πir²}−2+e^{−πir²} second difference), matching lattice lengths.
- +1/−1 eigenfunction split: a single magic function must satisfy BOTH a sign on f and a sign on f̂; splitting into Fourier eigenfunctions lets you control f and f̂ via the SAME modular object (since f̂=±f), collapsing two conditions to checking A(t),B(t) signs.
- E₂ quasimodular (not modular): needed to reach negative weight / the right depth; its anomaly term −(6i/π)(1/z) is exactly absorbed into the contour-deformation identity that produces the 2φ₀(z) (the "2" in the second difference).
- Weight 2−d/2: the Gaussian contributes z^{−d/2}; to land an eigenfunction the form must have automorphy weight that, combined with the contour (z+1)^{...} factors, reproduces the integral — forcing weight 2−d/2.

## Code reality
No "the method exists" repo predates it. The grounded code is: (1) the LP-bound numerical solver — radial FT via Hankel/Laguerre, build the LP, solve — which is what told everyone d=8,24 are special; (2) a SageMath/PARI script that builds φ₀, ψ_I from E₄,E₆,E₂,θ's, forms a(r),b(r) via the Laplace/Fourier integral of the q-expansion (term e^{2πinz}→ Gaussian-integral closed form), assembles g, and numerically verifies A(t)<0, B(t)>0 and g(0)=ĝ(0)=1. Both are standard, reproducible with mpmath/PARI.

## URLs read
- math/0110009 (Cohn–Elkies, ar5iv)
- 1603.04246 (Viazovska d=8, arXiv LaTeX source — primary)
- 1603.06518 (CKMRV d=24, arXiv LaTeX source — primary)
- 1607.02111 (Cohn, "Search for Magic Functions", ar5iv) — discovery narrative
- Quanta 2016 — historical narrative

## Uncertainty flags
- Leech double-zero indexing: simple zero at r=2 (minimal), double at √(2n) for n≥3. Verify wording.
- d=24 sign-condition statement φ(it)±(432/π²)ψ_S(it): signs taken from grep of source lines 633–664; double-check in answer.
- The "(1/√2)Λ8 centers" vs "½ scaling": density π⁴/384 is the invariant; present min distance scaling carefully.
