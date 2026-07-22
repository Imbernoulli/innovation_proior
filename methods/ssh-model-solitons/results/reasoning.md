Let me start from the thing that won't sit still: the electron-spin-resonance line in undoped trans-(CH)_x. It's there even when nobody has doped the material — a few hundred parts per million of paramagnetic centers, g almost exactly the free-electron value 2.00263, a single narrow Lorentzian only about 1.65 Oe wide, and it stays narrow all the way down to 10 K. A narrow Lorentzian that survives to 10 K is the fingerprint of *motional narrowing*: whatever carries this spin is moving, fast, sampling many environments so the hyperfine and dipolar broadenings average out. So I have an object that is neutral (no dopant put it there, it's intrinsic), carries spin one-half (it's an ordinary ESR signal at g≈2), and is highly mobile. And then the other half of the puzzle: when you dope the stuff, the conductivity climbs by orders of magnitude, but the Curie-law spin susceptibility barely moves. The carriers you're adding don't bring spin with them. Spinless charge. A mobile neutral spin, and spinless charge — those are exactly the two things a normal band picture can't give me. In a band insulator a carrier is an electron in the conduction band or a hole in the valence band, and either way it's spin-½ and charged. The observations have the spin and the charge *decoupled*, even *anticorrelated*. I need a theory of the elementary excitations that does that.

What is special about this chain? It dimerizes. One π electron per CH unit, so the π band is exactly half full, Fermi points at ±π/2a — and a half-filled one-dimensional band is the textbook setup for the Peierls instability. The carbons displace into an alternating short/long bond pattern, period-doubled, and that opens a gap right at the Fermi level, lowering the energy of every filled state. Peierls' argument is that in one dimension the electronic gain always beats the elastic cost, so the uniform chain is never stable; it always dimerizes. I should re-derive that for myself because the *amount* of dimerization, and the energetics, are going to matter. But before the energetics, the structural fact that grabs me: there are two ways to alternate. Call the bond pattern that goes "=−=−" the A phase and "−=−=" the B phase. They're related by sliding the whole alternation over by one bond, and by the chain's symmetry they have identical energy. Two degenerate ground states.

Two degenerate ground states. The moment I see that, I want to ask what happens at the boundary between a stretch of A and a stretch of B. Somewhere in the middle the alternation has to switch sign. That's a domain wall. You can't smooth it away by a small local wiggle — to the left the displacements are locked into one minimum, to the right into the other, and the wall is the region where they cross over. It's a topological object: pinned by the two distinct vacua on either side. Could *this* be my mobile neutral spin? It's intrinsic to the dimerized chain (no dopant needed), and a wall has no obvious reason to be charged. I don't yet know its spin, its energy, its width, or its mass. So let me build the smallest honest model that contains the dimerization, and then put a wall into it and compute.

The model. I want only the degrees of freedom that matter at the ≤0.5 eV scale of these excitations. The σ electrons live in bonds with a ~10 eV gap to their antibonding states — they never get excited, they just resist when I push the carbons around. So I keep the σ system only as an elastic medium: expand its energy to second order about the undimerized geometry and I get a harmonic spring, ½K Σ_n (u_{n+1} − u_n)², where u_n is the displacement of the n-th CH unit along the chain axis and K is the effective spring constant. The linear term vanishes because the undimerized chain is the σ equilibrium. The π electrons I treat tight-binding, one orbital per site, nearest-neighbour hopping only — that's the minimal thing that has a half-filled band. The one essential coupling is that the hopping depends on bond length: squeeze two carbons together and their π orbitals overlap more, t goes up. For the small displacements here (I'll find them to be ~0.04 Å, a few percent of a bond) the linear term is enough, t_{n+1,n} = t_0 − α(u_{n+1} − u_n), with α the electron–lattice coupling — positive α so that bringing sites closer (u_{n+1} − u_n < 0) raises the hopping. And the carbons are heavy compared to electrons, so I treat them classically with kinetic energy ½M Σ u̇_n² and let the electrons follow adiabatically — Born–Oppenheimer. Put it together:

H = − Σ_{n,s} t_{n+1,n} (c†_{n+1,s} c_{n,s} + h.c.) + ½K Σ_n (u_{n+1} − u_n)² + ½M Σ_n u̇_n².

Three pieces: π hopping with displacement-dependent t, σ elastic spring, lattice kinetic energy. That's it. I'm deliberately leaving out the Coulomb repulsion between π electrons — partly hoping the screened, effective t_0 and α absorb it, partly because if the on-site U were dominant I'd need a completely different (large-U) starting point, and the small observed dimerization and the band-like gap suggest I'm not there. I'll flag that as the model's known weak point and move on.

Now solve the perfectly dimerized chain to get the ground state and the gap, in Born–Oppenheimer with the nuclei clamped. The dimerized configuration is u_n = (−1)^n u — alternate carbons displaced ±u. Plug into the hopping: t_{n+1,n} = t_0 − α(u_{n+1} − u_n) = t_0 − α[(−1)^{n+1}u − (−1)^n u] = t_0 + (−1)^n 2αu. So the hopping alternates: on one set of bonds t_0 + 2αu, on the other t_0 − 2αu. Define t_1 = 2αu; the two bond strengths are t_0 ± t_1. Good — dimerization literally means two alternating hopping amplitudes, "strong bond" and "weak bond."

Diagonalize. With a two-atom unit cell the natural thing is a reduced Brillouin zone, −π/2a ≤ k ≤ π/2a. Let me Fourier transform on the original lattice but keep track of the two sublattices through the (−1)^n. Write the electron operators in k-space; because the hopping has a piece that's uniform (t_0) and a piece that alternates ((−1)^n 2αu), the alternating piece scatters k into k + π/a, i.e. couples the "valence-like" and "conduction-like" Bloch states of the reduced zone. Concretely, with the zero-order (u = 0) bands ε_k = −2t_0 cos ka folded into the reduced zone, the Hamiltonian in the 2×2 space of the two folded states at each k is

H_k = [ ε_k, −iΔ_k ; +iΔ_k, −ε_k ] with ε_k = −2t_0 cos ka and Δ_k = 4αu sin ka.

Let me check the off-diagonal. The alternating hopping term Σ_n (−1)^n 2αu (c†_{n+1}c_n + h.c.) — when I Fourier transform, (−1)^n = e^{iπn/a·a}... I'll just track magnitudes: the matrix element coupling the two zone-folded states picks up 2αu times the structure factor (e^{ika} − e^{−ika})-type combination, giving something ∝ sin ka, and the constant 4 comes out from summing both bonds in the cell with their phases. So Δ_k = 4αu sin ka. Diagonalizing the 2×2:

E_k = ±(ε_k² + Δ_k²)^{1/2}.

There it is — a gap. The bands ε_k = −2t_0 cos ka would cross zero at k = ±π/2a; Δ_k = 4αu sin ka is *maximal* exactly there, Δ_{π/2a} = 4αu. So the gap opens precisely at the Fermi points, E_g = 2·(4αu) at the band edge... wait, let me be careful. At k = π/2a, ε_k = 0 and Δ_k = 4αu, so E_k = ±4αu; the gap between the top of the lower band and bottom of the upper band is 2·4αu = 8αu? No — the gap is between E = −4αu and E = +4αu, so E_g = 8αu? Let me recompute. Δ_k at the zone edge is 4αu. The two eigenvalues are ±|Δ_k| = ±4αu there. So the full gap is from −4αu to +4αu = 8αu. But t_1 = 2αu, so the band-edge value 4αu = 2t_1, and the full gap E_g = 8αu = 4t_1. Yes: E_g = 4t_1, and the half-gap (band edge measured from mid-gap) is Δ = 2t_1 = 4αu. With the observed optical gap E_g ≈ 1.4 eV that gives t_1 ≈ 0.35 eV and Δ ≈ 0.7 eV. The dimerized chain is a small-gap semiconductor, exactly as it should be. The coefficients that diagonalize the 2×2 are the usual Bogoliubov-like α_k = [½(1 + ε_k/E_k)]^{1/2}, β_k = [½(1 − ε_k/E_k)]^{1/2} sgn k, with E_k = (ε_k² + Δ_k²)^{1/2}; the valence band is filled, the conduction band empty.

Now the energetics — does dimerization actually pay, and how much? Ground-state energy as a function of u: sum the filled (negative) band over the reduced zone and add the elastic cost. The conduction/valence eigenvalues come in ±E_k pairs, the valence (−E_k) is doubly occupied (spin), so

E_0(u) = −2 Σ_k E_k + 2NKu².

The elastic term: ½K Σ_n (u_{n+1} − u_n)² with u_n = (−1)^n u gives ½K · N · (2u)² = 2NKu². Good. Turn the sum into an integral over the reduced zone, length L = Na:

E_0(u) = −(2L/π) ∫_0^{π/2a} [(2t_0 cos ka)² + (4αu sin ka)²]^{1/2} dk + 2NKu².

Factor out 2t_0 and write z = t_1/t_0 = 2αu/t_0:

= −(4Nt_0/π) ∫_0^{π/2} [cos²θ + z² sin²θ]^{1/2} dθ + 2NKu²
= −(4Nt_0/π) ∫_0^{π/2} [1 − (1 − z²) sin²θ]^{1/2} dθ + (NKt_0² z²)/(2α²).

(I used 2αu = t_0 z so u² = t_0²z²/4α², and 2NKu² = NKt_0²z²/2α².) The integral is the complete elliptic integral E(1 − z²). So

E_0(u) = −(4Nt_0/π) E(1 − z²) + NKt_0² z² / (2α²).

The Peierls argument is now a statement about the small-z behaviour of E(1 − z²). For small z (nearly undimerized),

E(1 − z²) ≅ 1 + ½(ln(4/|z|) − ½) z² + …

Stare at the |z| inside the logarithm. The electronic energy −(4Nt_0/π)E(1−z²) has a piece −(4Nt_0/π)·½ ln(4/|z|)·z² ∝ +z² ln|z| (it goes *down* as a z²ln(1/|z|), i.e. the energy is lowered). The elastic cost is +(NKt_0²/2α²) z², plain quadratic. As z → 0, the derivative of z² ln(1/|z|) is z(2 ln(1/|z|) − 1), which dominates the linear-in-z derivative of the pure z² elastic term — the electronic gain wins for small enough z. So dE_0/du < 0 just above u = 0: the undimerized chain is a *local maximum*, and the chain rolls down to a finite ±u_0. That is the Peierls theorem, and I've watched it fall out of the logarithm in the elliptic integral. The energy E_0(u) is a symmetric double well in u, with two minima at ±u_0 — confirming the two degenerate ground states I argued for structurally.

Minimize to fix u_0. Setting dE_0/du = 0 with K = 21 eV/Å² (from independent vibrational estimates) and the gap pinned to 4t_1 = 1.4 eV gives α ≈ 4.1 eV/Å, comparable to quantum-chemical values, and u_0 ≈ 0.042 Å. The actual bond-length change from the alternation is √3 u_0 ≈ 0.073 Å (geometry of the zig-zag: the displacement along the chain axis projects onto the bond), small — a few percent of a bond, which retroactively justifies my linearizing the hopping. The condensation energy, the depth of the well, is −E_c/N = (E_0(u_0) − E_0(0))/N ≈ −0.015 eV per CH. Modest, which already tells me the *barrier between A and B vacua isn't huge* — promising for a low-energy domain wall.

So now the real question. Put a wall into this double-well system and compute its energy, width, mass, spin, and charge. Let me set up the order parameter properly. The raw displacement u_n isn't the cleanest variable — what's physical is the *sign* and *amplitude* of the dimerization, so define the staggered order parameter

ψ_n = (−1)^n u_n,

which equals −u_0 in the A phase, +u_0 in the B phase (or vice versa), and is zero by symmetry at the center of a wall. The two ground states are ψ_n = ±u_0. A soliton is a configuration where ψ_n goes from −u_0 at n → −∞ to +u_0 at n → +∞. I need a trial shape. The natural monotonic interpolation between two equal-and-opposite values is a tanh:

ψ_n = u_0 tanh(n/ℓ),

with ℓ the width in units of the lattice spacing — one variational parameter. Why tanh and not something else? Because the underlying field theory of a real scalar with a symmetric double well — which is what E_0(u) *is*, a φ⁴-type double well in the amplitude u — has the kink ∝ tanh as its exact static solution; tanh is the shape that balances the gradient energy against the potential. And this is the moment to be careful about a wrong turn I could take.

The tempting wrong picture: think of the dimerization as a charge-density wave and the soliton as a *phase* slip of that wave — a sine-Gordon kink. Interchanging double and single bonds *looks* like shifting the CDW phase by π, and a π phase slip in a CDW carries charge ±e. If I went down that road I'd conclude the soliton is charged. But that's wrong here, and I can see why if I look at what the order parameter actually is. In trans-(CH)_x the dimerization is described by a *real* amplitude — ψ_n is a real number sitting in one of two discrete minima ±u_0 — not the phase of a complex order parameter that can wind continuously. A real scalar with two degenerate minima is a φ⁴ field, not sine-Gordon. Its kink interpolates between the two minima as an *amplitude distortion*, the field passing through zero at the center. There's no phase to shift, no winding number, no built-in charge. So I expect the bare soliton to be *neutral*, and the charge (if any) to come from how electrons fill the levels — not from the lattice configuration itself. This distinction — amplitude soliton (φ⁴), not phase soliton (sine-Gordon) — is the hinge of the whole problem, and getting it right is what's going to give me a neutral spin-½ object instead of a charged one. Good. Onward with the φ⁴ / tanh picture.

Compute the soliton energy E(ℓ). Naively I'd diagonalize the electronic Hamiltonian for the tanh displacement on a long chain, sum the filled levels, add the elastic energy, subtract the perfect-chain ground state, and minimize over ℓ. But there's a subtlety that bit me when I first set it up: a *single* soliton on a finite chain changes which phase terminates the chain. If the chain starts in A phase on the left, after one soliton the right end is in B phase. So the energy I compute mixes the soliton energy with an "end-effect" energy from the chain now terminating in the wrong phase, and that end energy depends on the chain's boundary, which is exactly the kind of thing I don't want polluting a supposedly localizable bulk excitation. The fix: create a *soliton–antisoliton pair*, S and S̄. Going left to right the order parameter goes A → B (at S̄) → A (at S), so both ends are A phase, the end-effect cancels, and if S and S̄ are far apart and non-interacting the total energy change is just 2E_s. Compute the pair, halve it. (This also keeps the total electron count and total spin sensible — I'll come back to why that matters for Kramers' theorem.)

For the actual numbers, the clean way to get the energy of a localized perturbation without diagonalizing the whole chain is a Green's-function determinant. The soliton is a local change V̂ in the hopping relative to a reference perfect chain. The shift in the total ground-state energy of the filled sea due to a localized perturbation V̂ is

ΔE = (2/π) ∫_{−∞}^{μ} Im ln det[1 − G⁰(ω) V̂] dω,

where G⁰ is the Green's function of the unperturbed (perfectly dimerized) chain and μ = 0 is the chemical potential (mid-gap, half filling). The beauty is that det[1 − G⁰V̂] has dimension equal only to the spatial extent of V̂ — a few tens of sites where the hopping differs from the reference, say a (2ν+1)×(2ν+1) determinant with 2ν+1 of order 41–61 — not the whole chain, and it converges fast. To use it I need G⁰ for the perfect dimerized lattice, which I get from the eigenstates I already found:

G⁰_{nn'}(ω) = Σ_{k,λ} ψ^λ_k(n) ψ^{λ*}_k(n') / (ω − E^λ_k + iδ),

λ = conduction/valence, ψ^λ_k the Bloch eigenfunctions [α_k ± iβ_k(−1)^n] e^{ikan}/√N. Summing over k gives closed forms — for the diagonal element, for example,

G⁰_{nn}(ω) = −iω / [(4t_0² − ω²)(ω² − Δ²)]^{1/2}, for Δ ≤ |ω| ≤ 2t_0 (with corresponding pieces for |ω| < Δ and |ω| > 2t_0),

and a nearest-neighbour element G⁰_{n,n+1}(ω) with the same square-root branch structure. The density of states reads off from −(1/π) sgn(E) Im G⁰_{nn}(E):

ρ_0(E) = (N/π) |E| / [(4t_0² − E²)(E² − Δ²)]^{1/2}, for Δ ≤ |E| ≤ 2t_0, else 0,

with the characteristic 1/√(E²−Δ²) square-root divergences at the band edges ±Δ and a hard gap |E| < Δ. Good, the machinery is consistent. To handle the soliton I split the chain into three segments — perfect A to the right of the wall, perfect B to the left, and the soliton region S in between — and build G⁰ for the broken-into-segments reference by placing an infinite diagonal potential to decouple the segments, then taking U → ∞:

G⁰_{nn'} = G^d_{nn'} − G^d_{n,ν−1} G^d_{ν−1,n'} / G^d_{ν−1,ν−1}, for the A segment, and the mirror with t_1 → −t_1 for B.

Then evaluate ΔE numerically for the tanh trial with the missing-hopping perturbation V̂_{n,n+1} = t_0 + (−1)^n α(ψ_{n+1} + ψ_n) and vary ℓ. The result: the energy E(ℓ) for a single soliton has a shallow minimum. For the 1.4-eV gap (4t_1 = 1.4 eV) the minimum sits at ℓ ≈ 7 with E_s ≈ 0.4 eV; for a smaller dimerization (4t_1 = 1.0 eV) the wall is even wider, ℓ ≈ 9, and for a stronger one (4t_1 = 2.0 eV) it tightens to ℓ ≈ 5. So the soliton is *diffuse* — about 2ℓ ≈ 14 lattice sites of extent, not one bond. That diffuseness matters: a wide, smooth wall has a tiny pinning energy from the discrete lattice, so it can slide nearly freely.

Now the electronic structure of the soliton, which is where the spin and charge live. I claimed the φ⁴ kink should be neutral with the charge decided by electron filling — let me find the levels. Because the wall is a region where the gap parameter passes through zero, I expect a state pulled out of the bands into the gap. Let me hunt for a state exactly at mid-gap, E = 0. The eigenvalue equation for the gap-center amplitude φ₀(n), using the bond-dependent hopping t_{n+1,n} = t_0 + (−1)^n α(ψ_{n+1} + ψ_n), is

t_{n+1,n} φ₀(n) + t_{n+1,n+2} φ₀(n+2) = 0,

i.e. a recursion connecting n to n+2 only — a state on a *single sublattice*. (E = 0 decouples the even and odd sites because the diagonal is zero and hopping only connects neighbours.) Two linearly independent solutions: one decaying, one growing. For a soliton centered at n = 0, the normalizable one lives on even sites, φ₀(odd) = 0, and

φ₀(n+2) = −(t_{n+1,n}/t_{n+2,n+1}) φ₀(n),

which telescopes:

φ₀(n) = (−t_{n−1,n}/t_{n,n+1})(−t_{n−3,n−2}/t_{n−2,n−1}) ⋯ (−t_{1,0}/t_{2,1}) φ₀(0).

Each ratio is the local ratio of "weak side" to "strong side" hopping; away from the wall it's a number < 1, so φ₀ decays exponentially into both A and B phases, and it's modulated site-to-site by the sublattice structure. Working out the continuum/WKB form for the smooth tanh wall — φ₀ tunnels through a barrier of height Δ and is modulated by the zone-edge Bloch factor — gives the clean envelope

φ₀(n) ≅ (1/ℓ) sech(n/ℓ) cos(½πn),

normalizable, peaked at the soliton center, decaying as sech, with cos(½πn) just selecting the even sublattice and putting nodes on the odd sites. It sits at E = 0, dead center of the gap — a *nonbonding* state, neither bonding (valence) nor antibonding (conduction). And there's a deeper reason it must be exactly at zero: the Hamiltonian on this bipartite lattice anticommutes with the sublattice (chiral) operator that flips the sign on odd sites — C⁻¹HC = −H. That symmetry pairs every state of energy +E with one of energy −E; an odd state out, the localized one trapped on the wall, must sit at the self-conjugate point E = 0. The chiral symmetry *guarantees* a mid-gap state.

Now count electrons to get charge and spin — this is the payoff. The total number of one-electron states in the π system is fixed (it's the number of sites), so creating the soliton can only *redistribute* states, not create or destroy them: ∫ Δρ(E) dE = 0, where Δρ is the change in density of states from the soliton. The mid-gap state is a brand-new δ-function of weight 1 (one state per spin) sitting at E = 0. Where did it come from? Conservation says it was pulled out of the bands. And the change Δρ(E) is symmetric in E (again the chiral symmetry: states removed from the valence band are mirrored by states removed from the conduction band). So to supply the one mid-gap state, the valence band gives up *half* a state and the conduction band gives up *half* a state — per spin. The mid-gap level is literally half-valence, half-conduction: a nonbonding state floating between the bonding and antibonding manifolds. Let me make that local statement sharp, because it's what guarantees charge neutrality of the neutral soliton. The local density of states satisfies a completeness sum rule ∫ ρ_{nn}(E) dE = 1 at every site; differencing soliton vs perfect chain and using Δρ_{nn}(E) = Δρ_{nn}(−E),

2 ∫_{−∞}^{−Δ} Δρ_{nn}(E) dE + |φ₀(n)|² = 0.

Read it: at each site n, the electron density *missing* from the valence band, 2∫Δρ_{nn} over the filled band, is exactly compensated by |φ₀(n)|², the density of the mid-gap state. So if the mid-gap state holds exactly one electron, the chain is locally and globally charge neutral.

Now fill the level three ways. The valence band is fully occupied in all cases (we don't touch it beyond the redistribution already counted). The mid-gap state φ₀ can hold 0, 1, or 2 electrons:

— **One electron** in φ₀: the missing half-from-valence-half-from-conduction is exactly replenished by that one electron, so the soliton is **neutral, Q = 0**. But that electron is *unpaired* (the valence band is spin-paired, φ₀ has a single electron), so the soliton has **spin ½**. There it is — the mobile neutral spin-½ defect, sitting right at the gap center, intrinsic to the undoped chain. The narrow ESR line is this.

— **Empty** φ₀: the chain is missing the electron that would have neutralized it, so **Q = +e**, and with no electron in φ₀ and a spin-paired valence band the soliton has **spin 0** — a spinless positive charge.

— **Doubly occupied** φ₀: one extra electron beyond neutral, **Q = −e**, again spin-paired, **spin 0** — a spinless negative charge.

So the soliton's charge and spin are *reversed* from an ordinary electron:

(Q, s) = (0, ½), (±e, 0).

A neutral object that carries spin, charged objects that don't. That is precisely the experimental signature — the undoped ESR is the neutral spin-½ soliton; the spinless charged solitons are the doping carriers that raise conductivity without raising the Curie susceptibility. The φ⁴-amplitude-soliton picture *had* to give this; a sine-Gordon phase soliton would have handed me a charged soliton and gotten the spin–charge assignment wrong.

I should double-check this against Kramers' theorem, which once tripped me up. Kramers says a system with an odd number of electrons has half-odd-integer total spin. If I make a single neutral soliton, I've added an object with spin ½ — but I haven't changed the total electron number, so how can the total spin become half-odd-integer? It can't, on a finite ring. The resolution is the same S–S̄ pairing as before: on a ring of N CH units a soliton S that takes B → A must be accompanied by an antisoliton S̄ that takes A → B, and each carries spin ½; far apart they're independent excitations, but together they're an integer number of states removed and the spin counting is consistent — each of S and S̄ removes half a state from the valence band, and S plus S̄ together remove one complete state, an integer, as Kramers demands. The half-state-per-soliton is real, but it only becomes a *sharp* count when you have the pair (or when the single soliton is far from the chain ends and an extra half-spin is created or destroyed at the end). This is also why the mid-gap state being "half valence, half conduction" is not a contradiction: it's a bookkeeping that only resolves into integers globally.

Now the mass — because "mobile" needs a number. Let the wall move slowly: ψ_n(t) = u_0 tanh[(na − v_s t)/ℓa], a rigid translation at speed v_s. The lattice kinetic energy is ½M Σ_n u̇_n². With u_n = (−1)^n ψ_n and ψ_n the moving tanh, u̇_n = ψ̇_n = u_0 · (−v_s/ℓa) · sech²[(na − v_s t)/ℓa]. (Time-reversal symmetry tells me any change in the wall *shape* under motion is O(v_s²) and can't contribute to the leading kinetic energy, so I only keep the rigid translation — that's why a single width parameter ℓ is enough at this order.) So

½ M Σ_n u̇_n² = ½ M (u_0² v_s² / ℓ² a²) Σ_n sech⁴(n/ℓ).

I want to read this as ½ M_s v_s², which *defines* the soliton mass:

M_s = M (u_0² / ℓ² a²) Σ_n sech⁴(n/ℓ).

The lattice sum Σ_n sech⁴(n/ℓ): for a wide wall (ℓ large) replace by an integral, ∫ sech⁴(x/ℓ) dx = ℓ ∫ sech⁴ y dy = ℓ · (4/3), since ∫_{−∞}^{∞} sech⁴ y dy = 4/3. So Σ_n sech⁴(n/ℓ) ≈ (4/3) ℓ, and

M_s = (4/3ℓ) (u_0/a)² M.

Plug in the 1.4-eV-gap numbers, ℓ ≈ 7, u_0/a = 0.042/1.22 ≈ 0.0344, M = mass of a CH unit (13 proton masses ≈ 13·1836 m_e):

M_s ≈ (4/(3·7))·(0.0344)²·(13·1836) m_e ≈ 0.19·0.00118·23870 m_e ≈ 5 m_e.

A few electron masses. Astonishingly small — the soliton is *lighter than a single carbon by a factor of thousands*, because the mass is suppressed by both (u_0/a)² (the displacements are tiny) and 1/ℓ (the wall is wide, so few sites move much). A handful-of-electron-masses object must be treated as a quantum particle, and it will be extremely mobile — which is exactly the motional narrowing in the ESR. The smallness of M_s is a direct consequence of u_0 ≪ a; I should make sure I trust it, and I do: the equality of this kinetic mass (from ½M_s v_s²) with the inertial mass that enters the soliton's equation of motion follows from a work–energy theorem for the soliton, so the M_s I just defined is the genuine dynamical mass.

And the activation energy for moving it — does the discrete lattice pin it? Translate the rigid tanh through the lattice by sliding the soliton center across a lattice site and watch how E_s changes with center position. Because the wall is wide and smooth (ℓ ≈ 7, so it spans many sites), the energy varies only by ~0.002 eV between commensurate positions. That tiny Peierls–Nabarro barrier means nearly free translation down to temperatures of order 20–40 K — and again, consistent with the ESR staying motionally narrowed to 10 K.

Now the doping mechanism, which closes the loop on the spinless carriers. When I dope, I add a charge to the chain. Two options. Option one, the band picture: inject an electron into the conduction band (or a hole into the valence band), which costs the band-edge energy Δ = half the gap, and the carrier is spin-½ and charged. Option two: form a *charged soliton* — pull the electron out of (or add it to) a mid-gap state of a soliton, which costs E_s, and the carrier is spinless. Compare the energies. If E_s < Δ, it's cheaper to make a charged soliton than to put a carrier at the band edge, so doping proceeds through soliton formation; if E_s > Δ, ordinary band doping wins. With E_s ≈ 0.4 eV (for the 1.4-eV gap, and in the range 0.3–0.6 eV across reasonable parameters) and Δ = 0.7 eV, indeed E_s < Δ — soliton doping is favored. So each dopant transfers a charge to the chain by creating a charged, spinless soliton, raising conductivity without contributing to the Curie susceptibility. The spin–charge puzzle is resolved by energetics: the charged excitation is a spinless soliton, not a band carrier, precisely because the soliton costs less than the band edge.

A few finishing pieces. In the lightly doped chain the charged soliton is bound to its parent dopant by the Coulomb attraction. With the soliton's charge density e|φ₀(n)|² and an impurity of charge ∓e a distance d from the chain, the binding energy is

ΔE_I(n_s) = −(e²/ε) Σ_n |φ₀(n − n_s)|² / [(na)² + d²]^{1/2},

screened by the macroscopic dielectric constant ε ≈ 10; for d ≈ 2.4 Å this gives a binding energy of order 0.3 eV, comparable to the measured conductivity activation energy ≈ 0.3 eV in the dilute alloy — the carrier is a soliton that has to unbind from its dopant to conduct. And the soliton's local vibration against that impurity, treated to second order, gives a bound-state spring constant K_b and a vibrational quantum ℏω_s = ℏ(K_b/M_s)^{1/2} ≈ 0.07 eV; with M_s so small this oscillator energy is sizable and carries large dipole oscillator strength (a full electronic charge sloshing), which is a natural candidate for the ~0.1 eV infrared feature. And since the order parameter ψ_n → 0 at the wall center, the C–C bonds there are intermediate between single and double — a local stretch frequency between the two — consistent with the strong mode seen near 1370 cm⁻¹ in the lightly doped polymer.

One last connection, the one that tells me the half-state bookkeeping isn't an artifact of my lattice. Take the continuum limit: ψ_n becomes a smooth amplitude field, and the π electrons near the Fermi points become a one-dimensional Dirac fermion whose mass is set by the local order parameter. A soliton is a kink in that mass field, interpolating −u_0 → +u_0 and passing through zero — the mass changes sign across the wall. A Dirac fermion in a kink background has a conjugation symmetry C with C⁻¹HC = −H (the same chiral/sublattice symmetry I used), so positive- and negative-energy states pair up, and there's exactly one unpaired, self-conjugate zero mode bound to the kink — the continuum echo of my mid-gap φ₀. Filling the negative-energy sea and asking for the fermion number relative to the uniform vacuum: the zero mode is shared between the soliton and antisoliton sectors, and the regularized fermion number of a single isolated soliton comes out to ±½. So in the continuum, an isolated soliton carries *half* an electronic charge, e/2 — the lattice "half a state pulled from each band" sharpened to a fractional charge when soliton and antisoliton are taken infinitely far apart and each keeps half of the one removed state. On the finite lattice the eigenvalues are still integer (Q = 0, ±e) — the ½ is the expectation value that becomes a sharp fraction only in the infinite, well-separated limit — but the same nonbonding mid-gap state, the same conjugation symmetry, the same conservation sum rule are doing the work. The mobile neutral spin and the spinless charge were never paradoxes; they are what an amplitude domain wall in a Peierls-dimerized, half-filled chain must look like once you fill its one nonbonding mid-gap state.

Let me put the whole chain of reasoning in one line. A half-filled 1D π band must dimerize (Peierls), giving two degenerate bond-alternation vacua; a domain wall between them is a φ⁴-type amplitude soliton, not a sine-Gordon phase soliton, so it is intrinsically neutral; the chiral/sublattice symmetry of the bipartite chain forces a localized nonbonding state at the exact gap center, built half from the valence and half from the conduction band; filling that one state with 0, 1, or 2 electrons gives a spinless +e, a neutral spin-½, or a spinless −e soliton; the wall is wide and the displacements tiny, so its mass is only a few electron masses and it moves almost freely; and because the soliton formation energy E_s ≈ 0.4 eV is less than the band-edge cost Δ = 0.7 eV, doping proceeds by making spinless charged solitons — explaining at once the mobile neutral spin in the undoped chain and the spinless charge carriers on doping.

Here is the worked evaluation of the final formulas for the standard parameter set — a check, not a simulation: building the gap-center wavefunction from its closed form, verifying the lattice sum Σ sech⁴ → (4/3)ℓ, and evaluating the soliton mass.

```python
import numpy as np

# parameters fixed from independent data
t0   = 2.5            # eV  (pi bandwidth W = 4 t0 ~ 10 eV)
Eg   = 1.4            # eV  optical gap = 4 t1
t1   = Eg / 4.0       # eV  -> 0.35
Delta = 2 * t1        # eV  half-gap = 0.70
K    = 21.0           # eV/Ang^2  sigma spring constant
a    = 1.22           # Ang  CH spacing along chain axis
u0   = 0.042          # Ang  dimerization amplitude (from minimizing E0(u))
ell  = 7              # soliton half-width (sites), from minimizing E(ell)
M    = 13 * 1836.0    # CH-unit mass in electron masses (13 amu)

# --- gap-center (mid-gap) state:  phi0(n) = (1/ell) sech(n/ell) cos(pi n / 2) ---
def phi0(n, ell):
    return (1.0/ell) * (1.0/np.cosh(n/ell)) * np.cos(0.5*np.pi*n)

n = np.arange(-60, 61)
phi = phi0(n, ell)
phi /= np.sqrt(np.sum(phi**2))          # normalize on the lattice
# lives only on even sites: cos(pi n/2) = 0 for odd n
print("max |phi0|^2 at n=0 :", phi[n == 0][0]**2)
print("weight on odd sites :", np.sum(phi[n % 2 == 1]**2))   # ~0

# --- the lattice sum that sets the mass:  sum_n sech^4(n/ell) -> (4/3) ell ---
S = np.sum((1.0/np.cosh(n/ell))**4)
print("sum sech^4(n/ell)   :", round(S, 3), "  vs (4/3)ell =", round(4*ell/3, 3))

# --- soliton mass  M_s = (4 / 3 ell) (u0/a)^2 M  (electron masses) ---
M_s = (4.0/(3.0*ell)) * (u0/a)**2 * M
print("soliton mass M_s    :", round(M_s, 2), "m_e")

# --- which doping channel wins:  charged soliton (E_s) vs band edge (Delta) ---
E_s = 0.42            # eV, soliton formation energy at the E(ell) minimum (Eg=1.4)
print("E_s < Delta ?       :", E_s, "<", round(Delta,2), "->", E_s < Delta,
      " => spinless charged solitons carry the doping charge")

# --- charge / spin of the soliton from the occupation of the mid-gap level ---
for occ, label in [(0, "empty"), (1, "single"), (2, "double")]:
    Q = (1 - occ)            # +e empty, 0 neutral, -e double  (in units of e)
    spin = 0.5 if occ == 1 else 0.0
    print(f"phi0 {label:6s}: Q = {Q:+d} e,  s = {spin}")
```
