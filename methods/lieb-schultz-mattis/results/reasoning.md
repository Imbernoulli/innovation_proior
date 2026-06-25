Let me start where the trouble actually is. The one-dimensional Heisenberg antiferromagnet, H = Σ_i **S**_i·**S**_{i+1} with spin-½ on every site, is "solved" — Bethe wrote down the eigenstates in 1931, Hulthén got the ground-state energy in 1938 — and yet the two things I most want to know I cannot read off. Is the ground state unique? And is there a gap above it as the chain grows long? Bethe's ansatz hands me eigenfunctions as superpositions of magnon plane waves with amplitudes pinned by two-body scattering phases and those coupled transcendental rapidity equations, but it has never yielded the long-range order in closed form, and the way it presents the spectrum makes the N→∞ scaling of the lowest excitation opaque. In any finite chain a unique ground state is automatically separated from the rest by *some* gap; the whole question is whether that gap survives as N→∞, and the ansatz doesn't put that scaling in my hand. Spin-wave theory (Anderson, Kubo) presupposes a broken Néel state and is untrustworthy in one dimension. The variational tradition — Hulthén, Kasteleijn, Marshall, Ruijgrok and Rodriguez — gets beautiful energies and contradictory verdicts on order. The bitter lesson everyone has learned is that a trial state can nail the energy and badly misrepresent the order; energy is a weak diagnostic.

So I will not try to out-Bethe Bethe. Let me instead see how much I can settle by structure and by inequalities, because inequalities are exactly what a variational principle gives me, and a variational principle survives even when the exact spectrum is out of reach.

First a model I can actually solve all the way, to get my hands dirty and to see what the answers even look like. The thing standing between me and a solution of the full Heisenberg chain is the S^z S^z term: in **S**_i·**S**_{i+1} = S^z_i S^z_{i+1} + ½(S^+_i S^-_{i+1} + S^-_i S^+_{i+1}) the longitudinal piece is what will make any fermionization quartic, i.e. interacting. So throw it away and keep only the transverse part. That gives me

H_γ = Σ_i [(1+γ) S^x_i S^x_{i+1} + (1−γ) S^y_i S^y_{i+1}],

with γ an anisotropy I'll keep general for now (γ=0 is the isotropic transverse case; γ→1 the Ising limit). Only the x- and y-components appear, so call it the XY model. It's strikingly close to Heisenberg, and crucially it's *quadratic* in the spin operators with the troublesome longitudinal coupling gone. If I can diagonalize this exactly I'll have an honest gapless-or-not example to point at, and it will tell me whether the isotropic point is special.

How do I diagonalize a quadratic spin Hamiltonian? Introduce the lowering operators a_i = S^x_i − iS^y_i, a_i^† = S^x_i + iS^y_i, so S^x_i = (a_i^† + a_i)/2, S^y_i = (a_i^† − a_i)/2i, and S^z_i = a_i^† a_i − ½. In these,

H_γ = ½ Σ_i [(a_i^† a_{i+1} + γ a_i^† a_{i+1}^†) + h.c.].

Quadratic — good. Now here's the wall. These a's are *almost* fermions: on a single site {a_i, a_i^†} = 1 and a_i² = 0 (you can flip a spin at most once — the hard-core constraint), which is fermionic. But on different sites [a_i, a_j] = [a_i, a_j^†] = 0, which is *bosonic*. They are paulions, halfway between. And a quadratic form in operators with no single consistent algebra cannot be brought to normal form by a linear canonical transformation — because a linear canonical transformation preserves whatever algebra you feed it, and here there is no one algebra to preserve. A principal-axis rotation of the a's just produces more paulions. Dead end if I treat them as they are.

Could I just declare them bosons? No — a_i² = 0 is the hard core; bosons don't have it, and ignoring it is exactly the uncontrolled step I'm trying to avoid. Could I declare them fermions? Then I'd be wrong about the off-site commutators, which are bosonic. The mismatch is *only* in the relative statistics of operators at different sites: same-site they're fine as fermions, cross-site they commute when fermions should anticommute. So what I really need is to fix up the cross-site sign without disturbing the on-site algebra. I want to attach to a_i something that anticommutes once for every fermion sitting to its left, so that two operators at different sites pick up the extra minus sign that turns commutation into anticommutation.

The object that counts fermions to the left and flips sign accordingly is a string of (−1)^{n_j}. Since exp(πi a_j^† a_j) = 1 − 2 a_j^† a_j (because a_j^† a_j is 0 or 1, and e^{πi·0}=1, e^{πi·1}=−1), that's exactly (−1)^{n_j}. So define

c_i = exp[πi Σ_{j=1}^{i−1} a_j^† a_j] a_i, c_i^† = a_i^† exp[−πi Σ_{j=1}^{i−1} a_j^† a_j].

This is the Jordan–Wigner string (Jordan and Wigner 1928 used precisely this to relate spin chains and fermion fields; Kramers' book describes it). Let me check it does what I claim. On the same site the string Σ_{j<i} doesn't include i, so the exponential commutes with a_i and c_i^† c_i = a_i^† a_i — the number operator is untouched, good, the physics on a site is preserved. Now the cross-site anticommutator. Take j > i. The string of c_j includes site i, and using the on-site fact {a_i, 1−2a_i^†a_i} = 0 — check it: a_i(1−2a_i^†a_i) + (1−2a_i^†a_i)a_i = 2a_i − 2a_i a_i^† a_i = 2a_i − 2(1−a_i^†a_i)a_i = 0 since a_i² = 0 — the string element at site i anticommutes with a_i, while string elements away from i commute with a_i. Carrying this through,

{c_i, c_j^†} = (a_i a_j^† − a_j^† a_i) exp[πi Σ_{k=i}^{j−1} a_k^† a_k] = [a_i, a_j^†] exp[...] = 0 for i ≠ j,

and {c_i, c_i^†} = a_i a_i^† + a_i^† a_i = 1, and {c_i, c_j} = 0 likewise. So the c's are *genuine* fermions, {c_i, c_j^†} = δ_ij, {c_i, c_j} = 0. The string traded the awkward cross-site bosonic sign for the fermionic one, on-site algebra intact, and now the quadratic form is in operators with a single consistent algebra.

Now rewrite H_γ in the c's. For a free-ended chain, 1 ≤ i ≤ N−1, the string factors between i and i+1 cancel because they're adjacent, and one finds a_i^† a_{i+1} = c_i^† c_{i+1} and a_i a_{i+1}^† = c_i^† c_{i+1} too (let me just track one: a_i^† a_{i+1} = c_i^† exp[πi Σ_{k<i} n_k] exp[−πi Σ_{k<i+1} n_k] c_{i+1} = c_i^† exp[−πi n_i] c_{i+1} = c_i^†(1−2n_i)c_{i+1} = c_i^† c_{i+1} since c_i^† n_i = 0 forces... carefully, c_i^†(1−2c_i^†c_i) = c_i^† because c_i^† c_i^† c_i = 0). So

H_γ = ½ Σ_{i=1}^{N−1} [(c_i^† c_{i+1} + γ c_i^† c_{i+1}^†) + h.c.].

Free spinless fermions: a nearest-neighbor hopping c_i^† c_{i+1} and, when γ ≠ 0, a pairing term c_i^† c_{i+1}^† that creates fermions in pairs. And the total fermion number is

𝔑 = Σ_i c_i^† c_i = Σ_i a_i^† a_i = Σ_i (S^z_i + ½),

so the number of fermions is the total S^z plus N/2 — each fermion carries one unit of up-spin. Conservation of S^z became conservation (mod the pairing term) of fermion number; for γ=0 the pairing vanishes and number is exactly conserved.

A subtlety for the cyclic chain I should flag, because it bit me when I tried to just impose periodic boundary conditions on the c's. The bond between site N and site 1 carries the full string Σ_{j=1}^{N} n_j, so a_N^† a_1 = −c_N^† c_1 exp(iπ𝔑): the boundary fermion bond comes with a sign exp(iπ𝔑)+1 set by the *parity* of the total fermion number. Periodic spin boundary conditions thus map to periodic or antiperiodic fermion conditions depending on whether 𝔑 is odd or even — the "a-cyclic" problem. The boundary term it adds is O(1/N) in macroscopic quantities, so for the bulk spectrum I can work with the simpler "c-cyclic" problem (drop the boundary term, impose plain periodicity) and remember the parity caveat. I'll do that.

Now diagonalize the quadratic form in general, because I'll need it cleanly. The shape is

H = Σ_{ij} [c_i^† A_ij c_j + ½(c_i^† B_ij c_j^† + h.c.)],

with A Hermitian (so the hopping is Hermitian) and B antisymmetric (B_ij = −B_ji, forced by the anticommutation of the c's in the pairing term); here both are real. I want a linear canonical transformation

η_k = Σ_i (g_ki c_i + h_ki c_i^†),

with real g, h, that keeps the η's fermionic and gives H = Σ_k Λ_k η_k^† η_k + const. The condition for a normal mode is [η_k, H] = Λ_k η_k. Computing the commutator and matching coefficients of c_i and c_i^† separately gives

Λ_k g_ki = Σ_j (g_kj A_ji − h_kj B_ji), Λ_k h_ki = Σ_j (g_kj B_ji − h_kj A_ji).

These couple g and h. Take their sum and difference: define φ_k = g_k + h_k and ψ_k = g_k − h_k (as N-component row vectors). Adding and subtracting,

φ_k (A − B) = Λ_k ψ_k, ψ_k (A + B) = Λ_k φ_k,

a beautifully symmetric pair. Eliminate ψ:

φ_k (A − B)(A + B) = Λ_k² φ_k,

and likewise ψ_k (A + B)(A − B) = Λ_k² ψ_k. Since A is symmetric and B antisymmetric, (A + B)^T = A − B, so (A − B)(A + B) = (A + B)^T (A + B) is symmetric and positive semidefinite. Therefore Λ_k² ≥ 0 — all the mode energies are real, and I can choose the φ_k real and orthogonal. For Λ_k ≠ 0 I solve the eigenproblem for φ_k and get ψ_k from φ_k(A−B) = Λ_k ψ_k; for Λ_k = 0 the relative sign of φ and ψ is free (it just swaps the definition of occupied and empty for that zero mode, which can't change the energy). The constant: from the invariance of tr H under the canonical transformation, tr H = 2^{N−1} Σ_i A_ii = 2^{N−1} Σ_k Λ_k + 2^N·const, so

H = Σ_k Λ_k η_k^† η_k + ½(Σ_i A_ii − Σ_k Λ_k).

That's the whole diagonalization machine (this is the general machinery I'll need — the Bogoliubov rotation for a general real quadratic Fermi form).

Apply it to the c-cyclic XY model. The hopping matrix A is ½ on nearest neighbors (cyclically), the pairing B is ½γ antisymmetric on nearest neighbors. The combination (A − B)(A + B) is a real symmetric circulant-like matrix, so its eigenvectors are the plane waves φ_kj = √(2/N) sin kj or √(2/N) cos kj with k = 2πm/N, m = −N/2,…,N/2−1. Plugging in, the eigenvalues come out

Λ_k² = 1 − (1 − γ²) sin²k.

I take Λ_k ≥ 0 (the sign is free; choosing it positive makes the ground state the fermion vacuum with η_k Ψ_0 = 0 and every excitation positive — the particle-hole convention, which simplifies the bookkeeping; the alternative is a filled Fermi sea for |k| > π/2). The ground state is η_k Ψ_0 = 0 for all k, and

E_0 = −½ Σ_k Λ_k → E_0/N = −(1/π) ∫_0^π dk [1 − (1−γ²)sin²k]^{1/2} = −(1/π) ℰ(1−γ²),

a complete elliptic integral, smoothly interpolating from E_0/N = −1/π at the isotropic point γ=0 to −½ at the Ising limit γ=1. Sanity check passed: a clean closed-form energy from a genuinely quantum model.

Now the payoff. Look at the gap. The minimum of Λ_k² = 1 − (1−γ²)sin²k over k is at sin²k = 1, i.e. k = ±π/2, where Λ²_{π/2} = 1 − (1−γ²) = γ². So the smallest excitation energy is |γ|. For any γ ≠ 0 the spectrum is gapped, but **at γ = 0 the gap closes**: Λ_k = |cos k| vanishes at k = ±π/2, and the dispersion is linear there, Λ_{π/2+q} ≈ |sin q| ≈ |q|. The isotropic transverse model is gapless, and only the isotropic case is. And the ground state is nondegenerate (for N even, not a multiple of 4, so that no exact zero mode sits at the Fermi points). So in this soluble caricature I have exactly the picture I wanted: a unique ground state with a gapless spectrum at the isotropic point.

But — and this is the honest wall — the XY model is *not* the Heisenberg model. I threw away S^z S^z to make the fermions free. Reinstating it, S^z_i S^z_{i+1} = (n_i − ½)(n_{i+1} − ½), is a nearest-neighbor density-density *interaction* among the fermions; the problem is no longer free and my diagonalization doesn't apply. So I have a gapless soluble example, but I have not proven anything about the genuine isotropic Heisenberg chain, which is the thing I actually care about. I need an argument that doesn't require solving the model.

Back to inequalities. I have two questions — uniqueness and the gap — and I'll attack each with the variational principle, because that's the one tool that survives the interaction.

Uniqueness first. Work in the S^z_total = 0 sector (allowed since [S^z_total, H] = 0; it's where the antiferromagnetic ground state lives). A complete basis there is the set of Ising configurations Φ_μ with N/2 up and N/2 down. Any state Ψ = Σ_μ C_μ Φ_μ. Before anything else, rotate every spin on the B sublattice by π about the z-axis (S^x_j → −S^x_j, S^y_j → −S^y_j, S^z_j → S^z_j). Under this canonical transformation the diagonal S^z S^z part is unchanged but the off-diagonal flip-flop flips sign, so the Hamiltonian becomes

H' = Σ S^z_i S^z_j − ½ Σ (S^+_i S^-_j + S^-_i S^+_j),

now with a *minus* on the hopping. In the Ising basis every off-diagonal matrix element of H' is therefore ≤ 0. Schrödinger's equation in this representation is the coupled set

(E − ε_μ) C_μ = ½ Σ_{μ'(μ)} C_{μ'},

where ε_μ is the diagonal (S^z S^z) energy of configuration μ and the sum runs over configurations μ' that the flip-flop connects to μ. H is real, so I can take all C_μ real.

Claim (the first lemma): for any ground state in this sector, all C_μ ≠ 0. Suppose not — suppose some ground state Ψ_0 at energy E_0 has C_μ = 0 for μ in some set {μ_1,…,μ_r}. For those, the equation reduces to 0 = ½ Σ_{μ'(μ)} C_{μ'}. In at least one of these, say at μ_p, some of the connected C_{μ'} are nonzero — otherwise H' would break into a block with no matrix elements connecting {μ_1,…,μ_r} to the rest, which is impossible because the flip-flop ultimately connects every S^z=0 configuration to every other. So that equation forces nonzero C's of *both* signs (they must sum to zero). Now build the trial state Ψ_0' = Σ_μ |C_μ| Φ_μ. On the one hand, Ψ_0' is not an eigenstate: |C_{μ_p}| = 0 but Σ_{μ'(μ_p)} |C_{μ'}| ≠ 0, so it fails the eigenvalue equation, and the variational principle gives strictly E_0' > E_0. On the other hand, evaluate the energies explicitly. With every off-diagonal element of H' negative,

E_0' = Σ_μ ε_μ C_μ² − ½ Σ_μ Σ_{μ'(μ)} |C_μ| |C_{μ'}|, E_0 = Σ_μ ε_μ C_μ² − ½ Σ_μ Σ_{μ'(μ)} C_μ C_{μ'}.

Term by term |C_μ||C_{μ'}| ≥ C_μ C_{μ'}, and the off-diagonal terms enter with a minus, so E_0' ≤ E_0. The strict E_0' > E_0 contradicts E_0' ≤ E_0. Hence no C_μ can vanish.

The second lemma falls out of the same inequality: for E_0' = E_0 to be possible — i.e. for a genuine ground state — equality must hold in E_0' ≤ E_0, which requires every connected product C_μ C_{μ'} to be positive (they're all nonzero by the first lemma). So all the C_μ that are linked through the interaction share one sign, and since every configuration is ultimately linked to every other, *all* C_μ have the same sign. This is the Marshall–Peierls sign rule, here as a strict statement. Now uniqueness is immediate: if there were two ground states in the S^z=0 sector, both would have all-positive coefficients (after fixing the overall sign), and two all-positive vectors cannot be orthogonal — contradiction, since distinct eigenstates are orthogonal. So there is exactly one ground state with S^z=0. Marshall already showed at least one ground state has total S=0; any *additional* ground state, of whatever multiplicity, would carry some S^z=0 member, which I've just shown is impossible. Therefore the ground state is nondegenerate — strictly stronger than Marshall, who left the door open to extra degenerate, possibly non-singlet, ground states. And this argument used only bipartiteness and the sign of the bonds, so it generalizes to any bipartite lattice in any dimension.

One question down. The hard one is the gap, and I want it for the genuine Heisenberg chain, where I have no spectrum. Variational again, but now I need an *upper* bound on the energy of the *first excited* state — which means I need a trial state that is (a) low in energy and (b) provably orthogonal to the ground state. Orthogonality is the trap: any trial that fails to be orthogonal to Ψ_0 just bounds the ground-state energy and tells me nothing about a gap.

What deformation of Ψ_0 stays low in energy? A *uniform* spin rotation exp(iθ Σ_n S^z_n) commutes with H (the model is rotation-symmetric about z), so it sends Ψ_0 straight back to Ψ_0 — useless, gives the same state. I need a rotation that is *not* a symmetry, so it actually changes the state, but is as gentle as possible so the energy barely rises. The gentlest non-symmetry I can imagine is a rotation whose angle *winds slowly along the chain*: rotate the spin at site n by an angle proportional to its position, k·n, about the z-axis. Adjacent spins are then rotated by almost the same angle, differing only by k, so each bond is distorted by O(k) and the total energy cost should be O(k²) per bond times N bonds — order Nk². If I take k ∝ 1/N, that's O(1/N): vanishingly small as the chain grows. (This is the same operator Bloch used to argue about persistent currents and flux — a slow global twist.) So define the twisted trial state

Ψ_k = exp(i k Σ_n n S^z_n) Ψ_0 ≡ 𝒪^k Ψ_0.

Now I have to *earn* both properties.

Orthogonality. Use translation. Let U_z be the operator that shifts every spin one site cyclically, U_z **S**_i U_z^{−1} = **S**_{i+1}, with S_{N+1} = S_1. The Hamiltonian is translation-invariant, [H, U_z] = 0, and since I just proved Ψ_0 nondegenerate, Ψ_0 must be an eigenstate of U_z: U_z Ψ_0 = e^{iα} Ψ_0, some phase α (translation eigenvalue). Then

⟨Ψ_0 | Ψ_k⟩ = ⟨Ψ_0 | 𝒪^k | Ψ_0⟩ = ⟨Ψ_0 | U_z 𝒪^k U_z^{−1} | Ψ_0⟩,

where I inserted U_z^{−1} U_z and used U_z^{−1}Ψ_0 = e^{−iα}Ψ_0, U_z Ψ_0 = e^{iα}Ψ_0 so the phases cancel. Now compute how the twist transforms under one translation. Shifting n → n+1 in the exponent Σ_n n S^z_n, the term at the wrap-around site N produces an extra N S^z_1, and the overall shift produces an extra −Σ_n S^z_n:

U_z 𝒪^k U_z^{−1} = 𝒪^k exp(i k N S^z_1) exp(−i k Σ_n S^z_n).

The last factor is harmless: Ψ_0 is a singlet, Σ_n S^z_n Ψ_0 = 0, so exp(−ik Σ_n S^z_n) Ψ_0 = Ψ_0. That leaves the single-site factor exp(ikN S^z_1). S^z_1 has eigenvalues ±½, so in its eigenbasis

exp(ikN S^z_1) = diag(e^{+ikN/2}, e^{−ikN/2}).

Choose k = 2πm/N. Then kN/2 = πm, and e^{±iπm} = (−1)^m. If I pick m *odd*, this is −1 in both entries — i.e. exp(ikN S^z_1) = −1 acting on a spin-½. Therefore

⟨Ψ_0 | Ψ_k⟩ = −⟨Ψ_0 | Ψ_k⟩ = 0.

Orthogonal. Where did that minus sign come from? It came from S^z_1 = ±½ being *half-integer*: kN S^z_1 = ±πm = ±π·(odd), and e^{iπ·odd} = −1. If the spin were *integer*, S^z_1 ∈ ℤ, then kN S^z_1 = 2π·(integer) and the factor would be +1, the orthogonality argument would collapse, and I'd have no theorem. The gaplessness is not a generic fact about spin chains — it is bolted to the half-integer value of the spin. (That distinction — half-integer gapless, integer not obviously so — is itself a striking thing I'll want to think harder about.)

Energy. Now bound ⟨Ψ_k|H|Ψ_k⟩ = ⟨Ψ_0| 𝒪^{−k} H 𝒪^k |Ψ_0⟩. The twist rotates the in-plane spin components by angle kn at site n:

𝒪^{−k} S^x_n 𝒪^k = S^x_n cos kn + S^y_n sin kn, 𝒪^{−k} S^y_n 𝒪^k = −S^x_n sin kn + S^y_n cos kn, 𝒪^{−k} S^z_n 𝒪^k = S^z_n.

The S^z S^z part of H is untouched. For the transverse part, the rotation of two neighboring spins by angles kn and k(n+1) differing by k gives, after collecting terms,

𝒪^{−k} H 𝒪^k = H + (cos k − 1) Σ_n (S^x_n S^x_{n+1} + S^y_n S^y_{n+1}) + sin k Σ_n (S^x_n S^y_{n+1} − S^y_n S^x_{n+1}).

Take the expectation in Ψ_0 and read the three pieces. The first is ⟨Ψ_0|H|Ψ_0⟩ = E_0. The second: cos k − 1 = −½ k² + O(k⁴) = −½(2π/N)² + O(N^{−4}) for k = 2π/N, multiplying the in-plane bond sum Σ_n⟨S^x_n S^x_{n+1} + S^y_n S^y_{n+1}⟩, which is O(N) and bounded; so this piece is bounded by (2π/N)²·(N/2) + O(N^{−3}) = 2π²/N + O(N^{−3}). (Here I'm using k=2π/N, the smallest winding, m=1, which is odd — so the same k serves orthogonality and the energy bound at once.) The third piece vanishes: Σ_n(S^x_n S^y_{n+1} − S^y_n S^x_{n+1}) is, up to a factor, the commutator [Σ_n n S^z_n, H], so its expectation in the energy eigenstate Ψ_0 is ⟨Ψ_0|[Σ n S^z_n, H]|Ψ_0⟩ = (E_0 − E_0)⟨...⟩ = 0. Putting it together,

⟨Ψ_k | H | Ψ_k⟩ ≤ E_0 + 2π²/N.

So I have produced, for every even N, a state Ψ_k orthogonal to the nondegenerate ground state whose energy lies within 2π²/N of E_0. The first excited energy is at most E_0 + 2π²/N. As N → ∞ that gap closes. There is no energy gap. Combined with the uniqueness theorem: the isotropic spin-½ Heisenberg antiferromagnetic chain has a unique ground state and a gapless spectrum above it.

And this is a *clean* result for the genuine model — no Bethe ansatz, no spectrum, just two variational inequalities and the half-integer minus sign. It also dovetails with the soluble XY model, which exhibited exactly this — unique ground state, gapless at the isotropic point — by an entirely independent (free-fermion) route, so the two arguments corroborate each other.

Let me check how far the twist argument reaches. The energy bound only needed: translation invariance (for orthogonality), a nondegenerate singlet ground state, and a *transverse* (in-plane) coupling for cos k − 1 to act on — the S^z S^z part dropped out of the deformation entirely. So the no-gap half goes through for any Hamiltonian with these features, not just nearest-neighbor and not just one dimension. In two dimensions, take a square lattice of N sites along x and M = O(N^ν) along y, 0 < ν < 1, wrapped on a torus, and twist along x: 𝒪^k = exp(ik Σ_{n,m} n S^z_{n,m}), the same gradual winding applied to every row. The orthogonality goes through as before, and the energy bound becomes

⟨Ψ_k | H | Ψ_k⟩ ≤ E_0 + 2π²/N^{1−ν}.

Still vanishing as the lattice grows. I should be honest about the limit of this: for a genuinely two-dimensional N×N lattice the same trial state isn't sufficiently like a true low-lying excitation to force the conclusion — the twist is too crude a probe there — so the cleanest statement is the one-dimensional one, with the higher-dimensional version flagged as suggestive rather than decisive.

Let me restate the causal chain so it's airtight. The Heisenberg chain is solved but its order and gap are inaccessible, and variational energies are untrustworthy diagnostics — so I switch to inequalities. To even see the answer I solve a nearby model exactly: the Jordan–Wigner string converts the on-site-fermionic/off-site-bosonic spin operators into true fermions, the XY Hamiltonian becomes free fermions, the Bogoliubov diagonalization gives Λ_k = [1−(1−γ²)sin²k]^{1/2}, a gap |γ| that closes only at the isotropic point, where the dispersion is linear — a unique gapless ground state in the soluble case. For the real Heisenberg model I prove two things variationally: the sublattice-rotated Hamiltonian has all-negative off-diagonal elements, so the |C| trial state forces every ground-state amplitude to be nonzero and same-signed (Marshall–Peierls), which makes the ground state nondegenerate; and a slowly-winding twist 𝒪^k=exp(ik Σ n S^z_n) at k=2π/N produces a state whose energy exceeds E_0 by at most 2π²/N and which is orthogonal to the ground state *because* the spin is half-integer (the e^{ikNS^z_1} = −1 sign), so the spectrum is gapless. Unique ground state, no gap, for the genuine isotropic spin-½ chain.
