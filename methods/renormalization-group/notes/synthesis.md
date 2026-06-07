# Synthesis — Wilson's Renormalization Group (grounded in his own Nobel-lecture account)

## Sources actually read this run (refs/)
- `wilson-nobel-lecture.txt/.pdf` — Wilson's OWN 1982 Nobel lecture "The Renormalization Group and Critical Phenomena." HIGHEST PRIORITY. Three parts: (I) simplified RG+ε exposition; (II) "history as I remember it"; (III) post-1971. The backbone.
- `mit-ocw-position-space-rg.txt/.pdf` — Kardar MIT 8.334 Lec.13: EXACT 1D Ising decimation RG with additive constant g, recursion K'=(1/2)ln cosh 2K, fixed points K*=0 (stable, ξ=0) and K*=∞ (unstable, ξ=∞), tanh K'=(tanh K)², linearization. The worked-code grounding.
- `pedagogical-rg-intro.txt/.pdf` (arXiv 1402.6837) — Kadanoff block-spin picture; Wilson momentum-shell φ⁴ recursion r'=4(r+3c u/(1+r)), u'=2^{4-d}(u−9c u²/(1+r)²); linearization → ν; fixed point; ε-structure.
- `wilson-origins-eft-history.txt/.pdf` (arXiv 2111.03148, Rivat) — historical analysis corroborating the path: momentum-slice (1965) breakthrough, phase-space cells, Kadanoff 1966 catalyst, "many couplings not just two" (Wilson 2002 interview quote).
- `kronfeld-wilson-memorial.txt/.pdf` (arXiv 1312.6861) — lineage/dates; Wilson coined "anomalous dimension," "relevant/irrelevant/marginal" (1975 Kondo RMP), "integrate out," "Wilson loop."
- `wilson-rg-paradigmatic-shift.txt/.pdf` (arXiv 1402.3437) — secondary corroboration.

GAP: Kadanoff 1966 (Physics 2, 263) and the Wilson 1971 PRB papers themselves are APS-paywalled; could not download primary PDFs. Mitigation: Wilson's Nobel lecture narrates Kadanoff's idea + his own 1971 construction in his own words (lines 791–802, 873–906); the pedagogical + MIT + history refs supply the exact math. Noted in notes/gaps.

## The pain point (research question)
Near a critical point (Curie point of a ferromagnet; liquid–vapor critical point) fluctuations occur on ALL length scales simultaneously — from atomic up to the (diverging) correlation length ξ. Ordinary methods fail: analytic perturbation theory wants to expand around ONE scale; numerical integration dies past ~10 variables; the partition function is a sum of analytic Boltzmann factors yet the magnet shows NON-analytic behavior at Tc (β≈1/3, ν≈2/3, not Landau's 1/2). Landau/mean-field theory assumes only atomic-scale fluctuations matter and averages them out into a continuum M(x) — it gives β=1/2, ν=1/2, WRONG below 4 dimensions. The challenge: explain non-analyticity and the universal non-mean-field exponents.

## Antecedents (load-bearing)
- **Landau / Landau-Ginzburg (1937, 1950):** F = V{R M² + U M⁴}, minimize over M. R∝(T−Tc), U>0. Gives β=1/2 (eq.4), ν=1/2. Gradient form F=∫[ (∇M)² + R M² + U M⁴ − BM]. Assumes analyticity survives averaging space-dependent fluctuations. Fails for d<4.
- **Gell-Mann–Low (1954) QED:** physical charge e measured at long distance, bare charge e₀ at short distance; a family of effective charges e_λ at momentum scale λ obeys λ dē/dλ = ψ(ē); forerunner of Wilson's eqs. The "renormalization group." But: FIXED number of couplings (just e). Wilson's eq.(39) analogue.
- **Kadanoff (1966) block spins:** group 2×2×2 atoms into a block acting like one effective moment with effective T_L, h_L; T_{2L}, h_{2L} analytic functions of T_L, h_L; at Tc these reach L-independent fixed values. Derives Widom's scaling laws & exponent relations. BUT: postulates the block keeps the SAME nearest-neighbor form with just two parameters; gives no method to COMPUTE the transformation.
- **Widom (1965) scaling:** homogeneous equation of state accommodating non-mean-field exponents; no theoretical basis (Wilson "puzzled by the absence of any theoretical basis").
- **Onsager (1944):** exact 2D Ising, ν=1 ≠ 1/2 — proof that mean field is wrong.
- **Stueckelberg–Petermann (1953):** named "groupe de normalisation."

## Wilson's actual discovery path (from the Nobel lecture, in order)
1. Background = QFT, not stat mech. Renormalization (Bethe/Schwinger/Feynman/Dyson) + Gell-Mann–Low RG = "principal inspiration."
2. **1965 momentum-slice breakthrough** (Phys Rev 140, B445): fixed-source meson model, replace momentum continuum by well-separated slices 1, Λ, Λ²,…,Λⁿ. Treat highest slice as unperturbed H₀, lower slices as perturbation. Solving+eliminating the top slice → effective H for n−1 slices, SAME form but g renormalized. "For the first time I had found a natural basis for renormalization group analysis: namely the solution and elimination of one momentum scale." Reminiscent of Dyson: do high energies before low.
3. **"What is a field theory" / phase-space cells:** φ⁴ Hamiltonian — kinetic term diagonal in Fourier φ_k, interaction diagonal in φ(x). Compromise: wavefunctions of minimum phase-space volume (uncertainty principle → unit cells). Momentum on a LOG scale (slices 1<|k|<2, 2<|k|<4…). Must cut off at large k ⇒ lattice theory. "To understand field theories I would have to understand field theories on a lattice."
4. **Aspen 1966:** worked Onsager; realized RG ideas apply to critical phenomena; told he'd been "scooped by Kadanoff." Reads Kadanoff block-spin preprint.
5. **The liberation (the key move):** Previously all RG transformations he knew had a FIXED number of couplings (Gell-Mann–Low: just e; Kadanoff: just T,h). "I had tried many ways to derive transformations just for these fixed number of couplings, without success." When he did the momentum-slice analysis to all orders in 1/Λ (since real Λ=2 not large), an "infinitely complicated effective Hamiltonian was generated, with an infinite set of coupling constants" at each step — yet he could prove the higher couplings had small, boundable effect. "Liberated from this restriction, it turned out to be easy to define renormalization group transformations." Locality gives a natural order-of-importance (truncate to couplings fitting a 3³ or 4³ region). Restate Kadanoff: nearest-neighbor is the most important coupling because it's most localized, but OTHER couplings appear too. (History ref + Wilson 2002: "plenty of couplings, not just two as Kadanoff assumed.")
6. **1971 recursion formula** (PRB 4, 3174 & 3184): Widom asked him to lecture (Fall 1970); to give a computable example he applied the phase-space-cell analysis to Landau-Ginzburg, simplified to a nonlinear integral recursion on a function of ONE variable, iterated on a computer → fixed point, exponents, Widom scaling derived from fixed-point formalism.
7. **ε-expansion with Fisher (1972):** showing Fisher numerical results, they realized TOGETHER that the nontrivial fixed point becomes trivial at d=4 and is easy to study near 4D; d enters the recursion as a simple parameter; ε=4−d. "Critical Exponents in 3.99 Dimensions." Then Feynman-graph expansion (1972) for the full Landau-Ginzburg, small coefficient of φ⁴ = expansion parameter.

## The method as landed on (in-frame, the structure to re-derive)
- Coarse-grain: integrate out fluctuations in one shell (a length scale L→L+δL, or in real space, decimate spins). This produces a new effective Hamiltonian/free-energy functional F_L of the same KIND but with changed couplings.
- Rescale to dimensionless form so the transformation repeats identically → a transformation R on coupling space (the renormalization GROUP, really a semigroup).
- Iterate. The flow has FIXED POINTS H*: R(H*)=H*. A fixed point is scale-invariant ⇒ ξ either 0 or ∞; the critical fixed point has ξ=∞.
- UNIVERSALITY: many microscopically different systems flow to the same fixed point ⇒ same exponents.
- EXPONENTS from LINEARIZATION about H*: eigenvalues λ_i of the linearized map. Relevant (λ>1) directions = the few knobs (T−Tc, h) you must tune; irrelevant (λ<1) directions decay = why microscopic details don't matter. ν = ln b / ln λ_t.
- ε=4−d makes the nontrivial (Wilson-Fisher) fixed point perturbatively accessible.

## Worked example (runnable code), grounded in MIT OCW Lec.13
1D Ising chain, H = −K Σ σ_i σ_{i+1} (units of kT; K=J/kT). Decimate every other spin (b=2). Sum s=±1 between fixed neighbors σ₁,σ₂:
Σ_{s} exp[K s(σ₁+σ₂)] = 2 cosh(K(σ₁+σ₂)).  Demand this equal z' exp[K' σ₁σ₂].
Match the two cases σ₁σ₂=+1 (→2cosh2K) and σ₁σ₂=−1 (→2):
  e^{K'} z' = 2 cosh 2K ;  e^{−K'} z' = 2  ⇒  K' = (1/2) ln cosh 2K,  with additive g' = g + (1/2)ln(4 cosh 2K)/... (free-energy bookkeeping).
Fixed points: K*=0 (stable, ξ=0, disordered sink) and K*=∞ (unstable, ξ=∞, T=0 order). Any finite K flows to 0 ⇒ no finite-T transition in 1D — the flow ITSELF tells you d=1 has no ordering. Equivalent compact form: tanh K' = (tanh K)². Track free energy per spin via the accumulated g; compare to exact transfer-matrix f = −ln(2 cosh K).

## Design decisions → why
- Integrate out ONE scale at a time, not all at once: the all-scales average is the hopeless coupled integral; one shell is a single tractable integration that you ITERATE. (Nobel: "tackle the problem in steps, one step for each length scale.")
- Allow arbitrarily many couplings (vs Kadanoff's two): forcing the form to stay nearest-neighbor is exactly what blocked every prior attempt; the transformation only becomes DEFINABLE once you let new couplings appear. Locality bounds them ⇒ truncation is controlled.
- Rescale lengths/fields to dimensionless form: so the same map repeats and fixed points exist; without rescaling each step's lattice grows and you can't compare.
- Fixed point, not the partition function directly: the fixed point is where ξ=∞ (criticality) and is universal; exponents come from how you LEAVE it (linearization), not from the (non-universal) approach.
- ε=4−d: above 4D Landau is exact (Gaussian fixed point stable); at 4D the φ⁴ coupling is marginal; below, a new fixed point splits off at O(ε), small ⇒ perturbative. Makes non-trivial exponents computable.
- Real-space decimation for the worked example: exactly soluble in 1D, needs no truncation, shows the whole machine (flow, fixed points, free-energy constant) with runnable arithmetic.
