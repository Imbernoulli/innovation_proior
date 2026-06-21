I want the field dependence of the magnetization inside a ferromagnetic domain. Concretely: I have a lattice of atomic spins, they line up below the Curie point because of the exchange interaction, and I apply a field H along the magnetization. The magnetization is not quite saturated — thermal agitation has tilted some spins — and the field pulls them back. I want M(T,H), and especially the susceptibility ∂M/∂H, all the way up to large fields where the dipolar physics should show through. So I need the *excited states* of the spin Hamiltonian, populated by ordinary statistics, with the field and the dipole–dipole term carried along. The aligned state is trivial; everything interesting lives in the low-lying excitations. Let me start from those.

The Hamiltonian. Heisenberg gave me the right starting point: the alignment isn't magnetic, it's electrostatic. Coulomb repulsion plus the Pauli principle makes the two-electron energy depend on whether the spins are parallel or antiparallel — the Heitler–London picture of H₂, where the symmetric and antisymmetric spatial states split by twice an exchange integral and the spin state is dragged along by the antisymmetry of the total wavefunction. Coarse-grain that to a lattice and the spin-dependent energy is H = −Σ_⟨ij⟩ J S_i·S_j with J>0 for a ferromagnet. Add the field: each moment is −gμ_B S, so a field H along z costs gμ_B H per unit of S^z, giving a Zeeman term gμ_B H Σ_i S_i^z. And there's the dipole–dipole term between the actual magnetic moments — weak compared to exchange, but it's the thing that makes the magnetization depend on field and sample shape in the measurable way I care about. So the real object is exchange + Zeeman + dipolar. Let me get the machinery right on the exchange part first; the field and dipolar terms I'll see how to attach once the framework exists.

Write the exchange dot product in ladder form: S_i·S_j = S_i^z S_j^z + ½(S_i^+ S_j^- + S_i^- S_j^+), with S^± = S^x ± iS^y and the algebra [S^z,S^±]=±S^±, [S^+,S^-]=2S^z. Take everything aligned along +z, every spin at m=+S. Is |all S⟩ an eigenstate? The flip terms S_i^+S_j^- each contain an S^+ acting on a state that's already maximally raised — that gives zero. Only S^zS^z survives, eigenvalue S², so H|all S⟩ = −Σ_⟨ij⟩ JS² |all S⟩. Good, the fully aligned state is exact. That's the classical ground state.

Now one step up. Flip a single spin at site n down by one unit: |…, S, S−1, S, …⟩. Apply H. The diagonal S^zS^z piece is fine. But look at S_n^- S_{n+1}^+ acting on this: S_{n+1}^+ raises the already-maxed neighbor → zero, fine; but S_{n+1}^- S_n^+ takes the flip at n and *moves it to n+1*. So a localized flip is not stationary — it hops to its neighbors with equal amplitude. The single-site reversed-spin state is not an eigenstate. This is the whole difficulty in miniature: the off-diagonal exchange terms delocalize excitations, so I can't just count localized flips.

What hops with equal amplitude to each neighbor on a periodic lattice diagonalizes by Fourier transform — this is Bloch's spin wave. Form |k⟩ = N^{−1/2} Σ_n e^{ik·r_n}|flip at n⟩. The hopping matrix is circulant, so plane waves are its eigenvectors, and for a 1D chain with nearest-neighbor J the single-excitation energy comes out ε_k = 2JS(1−cos kd), which is ≈ JS(kd)² at long wavelength. Each such excitation lowers the total S^z by one and carries spin −1. Populate these as independent excitations and you get the spontaneous magnetization falling off as T^{3/2} at low temperature. Beautiful — and it tells me the excitations exist and are wavelike and quadratic at small k.

But I can't build my whole calculation on Bloch's construction. What he did is diagonalize the *one-flip* sector by writing down an explicit basis and finding its eigenvectors. For one excitation that's exact. For two, I'm in trouble: I have to track that two flips can't pile up arbitrarily on one site (a spin-S can be reversed at most 2S times), I have to worry about two-magnon bound states, and the bookkeeping is by hand. Worse, the moment I put in the field and the dipolar term — which is the actual problem — the convenient single-flip eigenbasis stops being the natural object, and I'd be re-doing the diagonalization for each new term. There's no *operator-level* representation in which the whole Hamiltonian becomes something I can expand systematically and then correct order by order. The dot product S_i·S_j is a product of non-commuting operators; it isn't a set of oscillators I can just "fill with quanta." I want a framework, not a sequence of bespoke diagonalizations.

So let me stare at the excitations themselves and ask what kind of object they really are. An excitation at a site is a reversal: it lowers S^z there by an integer. Let n_i = S − ⟨S_i^z⟩ be the number of units the spin at i is reversed from the top. By construction n_i is a non-negative integer, and it's bounded above: I can flip a spin-S at most 2S times, from m=+S down to m=−S, so 0 ≤ n_i ≤ 2S. Each unit of n_i carries spin −1. Independent excitations, integer-valued, additive per site, lowering the spin one unit at a time…

Integer occupation numbers, additive, non-negative — that's the spectrum of a harmonic oscillator's number operator. a†a has eigenvalues 0,1,2,…. The matching is staring at me: write S_i^z = S − a_i†a_i, with a a boson per site. The bosons are excitations of the spin — "spin deviations" — and a magnon is the quantized spin wave once I diagonalize. Why bosons and not fermions? Because for S>½ a single site can hold more than one deviation (up to 2S), so the occupancy is not Pauli-limited to one; and the collective mode carries integer spin −1. Bosons, with [a_i,a_j†]=δ_ij, [a_i,a_j]=[a_i†,a_j†]=0.

I'll set S^z and let the algebra force the rest. I have S^z = S − a†a. I need S^+ and S^-, and the only non-negotiable constraint is that the spin commutation relations come out *exactly*, otherwise I've changed the physics. Let me see what the spin operators do in the deviation labeling and match.

In the spin space, label states by n through m = S − n, n = 0,1,…,2S — and note this is a genuine choice. I could have written m = −S + n′ instead, counting up from the bottom. But n counts deviations from the *aligned* ground state, and the ground state should be the boson vacuum n=0; the −z reference would make the vacuum the fully-reversed state, which is unphysical for a ferromagnet aligned along +z. So m = S − n it is, vacuum = top. Then S^z|n⟩ = (S−n)|n⟩ matches S^z = S − a†a exactly, since a†a|n⟩ = n|n⟩.

Now S^+. From angular momentum, S^+|m⟩ = √(S(S+1) − m(m+1)) |m+1⟩. In the n labeling, |m⟩↦|n⟩ and |m+1⟩↦|n−1⟩ (raising m lowers n — the deviation count goes down when I raise the spin). Substitute m = S − n:

S(S+1) − m(m+1) = S(S+1) − (S−n)(S−n+1).

Expand: (S−n)(S−n+1) = S² − 2Sn + n² + S − n. So S(S+1) − that = S² + S − S² + 2Sn − n² − S + n = 2Sn − n² + n = n(2S − n + 1) = n(2S − (n−1)). Therefore

S^+|n⟩ = √(n(2S − (n−1))) |n−1⟩.

Compare with the boson a|n⟩ = √n |n−1⟩. The √n factor is exactly the boson annihilation; what's left over is √(2S − (n−1)). Since the final state is |n−1⟩, that leftover must be the operator √(2S − a†a) acting on |n−1⟩ — because on |n−1⟩ the operator √(2S − a†a) gives √(2S − (n−1)). So

S^+ = √(2S − a†a) · a.

Order matters: a sits on the right and annihilates first, then √(2S − a†a) evaluates on the lowered state, reproducing √(2S − (n−1)). And S^- is the Hermitian conjugate,

S^- = a† √(2S − a†a).

There's the transformation:

  S^z = S − a†a,   S^+ = √(2S − a†a)·a,   S^- = a†·√(2S − a†a),

which I can also write as S^+ = √(2S)·√(1 − a†a/2S)·a and S^- = √(2S)·a†·√(1 − a†a/2S), pulling out the √(2S).

Why is that square-root factor there, and why does it have to be the *square root* and not a bare √(2S)? Two reasons, and they're the same reason. First, the truncation: when n reaches 2S, the spin is at the bottom m=−S and S^+ should still raise it — but more importantly the boson space must not be allowed to exceed n=2S, because the spin space is only (2S+1)-dimensional. Look at S^- = a†√(2S−a†a): acting on |2S⟩ it gives a†√(2S−2S)|2S⟩ = 0. The square-root factor √(2S − a†a) *kills the state at full occupancy*, so the boson Fock space is automatically cut off at the right place to match the finite spin. A bare √(2S)a† would happily keep creating bosons past n=2S into states that don't exist for a real spin. Second, the algebra: I claimed the constraint is exact preservation of the spin commutators, so let me actually verify it, because if it fails the whole construction is wrong.

Check [S^z, S^+] = S^+. With S^z = S − n̂ (n̂ = a†a) and S^+ = √(2S−n̂) a:
[S^z, S^+] = [S − n̂, √(2S−n̂) a] = −[n̂, √(2S−n̂) a]. Now √(2S−n̂) is a function of n̂ alone so it commutes with n̂; thus [n̂, √(2S−n̂) a] = √(2S−n̂)[n̂, a]. And [n̂, a] = [a†a, a] = a†[a,a] + [a†,a]a = −a. So [n̂, √(2S−n̂) a] = −√(2S−n̂) a, and [S^z,S^+] = +√(2S−n̂) a = S^+. Good. Same way [S^z,S^-] = −S^-.

Now the hard one, [S^+, S^-] = 2S^z. Write S^+ = √(2S−n̂) a, S^- = a† √(2S−n̂), and abbreviate f = √(2S−n̂). I want [f a, a† f] = f a a† f − a† f f a = f a a† f − a† (2S − n̂) a. For the first term I need a a† = a†a + 1 = n̂ + 1, so f a a† f = f(n̂+1)f = (n̂+1) f² because f and n̂ commute, = (n̂+1)(2S − n̂). For the second term, a† (2S−n̂) a: I have to be careful, (2S−n̂) doesn't commute with a. Use a† g(n̂) a where g(n̂)=2S−n̂. Note a† n̂ a = a† a a† a − a† a = … let me just push n̂ through a. Since [n̂,a] = −a, we have n̂ a = a(n̂−1), so (2S−n̂)a = a(2S−(n̂−1)) = a(2S−n̂+1). Then a†(2S−n̂)a = a† a (2S−n̂+1) = n̂(2S−n̂+1). So

[S^+,S^-] = (n̂+1)(2S−n̂) − n̂(2S−n̂+1).

Expand: (n̂+1)(2S−n̂) = 2Sn̂ − n̂² + 2S − n̂. And n̂(2S−n̂+1) = 2Sn̂ − n̂² + n̂. Subtract: (2S − n̂) − (n̂) = 2S − 2n̂ = 2(S − n̂) = 2S^z. Exactly. The square-root factor is precisely what's needed for [S^+,S^-]=2S^z to hold *for all n*, not just at low occupancy. So this representation is not an approximation — it is an exact rewriting of the spin algebra in boson language. That's the payoff: I've turned the non-commuting spin operators into ordinary boson operators with the constraint built into one square-root factor.

Let me sanity-check the whole thing numerically before I build on it, because a sign error here propagates everywhere. Build the spin-S matrices in the |S,m⟩ basis, build truncated boson matrices a, a† on the same (2S+1)-dimensional space with n=0,…,2S (ordering |n⟩ to match m=S−n), form S^z = S−a†a, S^+ = √(2S−a†a) a, S^- = a†√(2S−a†a), and compare to the spin matrices and check [S^+,S^-]−2S^z. For S = ½, 1, 3/2, 2 the exact map reproduces the spin matrices to machine precision and the commutator [S^+,S^-]−2S^z and [S^z,S^+]−S^+ vanish to ~10^{−15}. The exact transformation is verified.

The reason to do all this is not to rewrite the algebra for its own sake — it's that the *square root can be expanded*. At low temperature, in the ordered phase, the number of deviations is tiny, ⟨n̂⟩ ≪ S (and for large spins ⟨n̂⟩/2S is small anyway). So I can expand √(1 − a†a/2S) ≈ 1 − a†a/4S + …. To leading order,

  S^+ ≈ √(2S) a,   S^- ≈ √(2S) a†,   S^z = S − a†a.

Substitute these into the Heisenberg Hamiltonian and the product of spins becomes a *quadratic* form in a, a† plus higher-order corrections. Let me do it carefully, because the bookkeeping of which terms are which order in S is the whole content of linear spin-wave theory, and I want to see the interactions fall out too, not just assert them.

H = −J Σ_⟨ij⟩ [S_i^z S_j^z + ½(S_i^+ S_j^- + S_i^- S_j^+)]. Take the diagonal piece first with S^z = S − n̂:

S_i^z S_j^z = (S − n̂_i)(S − n̂_j) = S² − S(n̂_i + n̂_j) + n̂_i n̂_j.

The flip piece: ½(S_i^+ S_j^- + S_i^- S_j^+). If I keep the full square roots I'd get the exact thing; let me instead organize by orders of S. Using the expansion S^+ = √(2S)(1 − n̂/4S)a + …, S^- = √(2S) a†(1 − n̂/4S) + …, the product S_i^+ S_j^- = 2S (1 − n̂_i/4S) a_i a_j† (1 − n̂_j/4S) = 2S a_i a_j† − ½(a_i a_j† n̂_j + n̂_i a_i a_j†) + O(1/S). Symmetrize with S_i^- S_j^+ and collect.

Let me just track the two leading orders. The O(S²) term is −J Σ S² = −NqJS²/2 (N sites, coordination q, the ½ for counting each bond once) — the classical ground-state energy. The O(S) terms: from S^zS^z I get −J Σ S(−)(n̂_i+n̂_j) = +JS Σ_⟨ij⟩ (n̂_i+n̂_j); from the flip piece at leading order ½·2S(a_i a_j† + a_i† a_j) = S(a_i a_j† + a_i† a_j) = S(a_i† a_j + a_j† a_i) up to the commutator constant. Putting the quadratic part together,

H_1 = −J Σ_⟨ij⟩ S [ a_i† a_j + a_j† a_i − n̂_i − n̂_j ] = JS Σ_⟨ij⟩ (a_i† − a_j†)(a_i − a_j).

I like that last form — it's manifestly the cost of a *difference* between neighboring deviations, exactly a wave's stiffness, and it's manifestly positive (J>0). And the leftover terms, the ones with four boson operators like n̂_i n̂_j and a_i† n̂ a_j, are O(S⁰): these are H_2, the magnon–magnon interactions. So the structure is

  H = −NqJS²/2 + H_1 + H_2 + O(1/S),

ground state, then free magnons, then interactions, organized as a clean 1/S expansion. At low T I keep H_1; the corrections are systematically smaller. *This* is what Bloch's counting couldn't give me: not just the leading spectrum but a controlled framework for the corrections, in operator form, so I can also hang the field and dipolar terms on it without redoing anything.

Diagonalize H_1. Translational invariance means momentum is good, so Fourier transform a_j = N^{−1/2} Σ_k e^{ik·r_j} a_k (and a_j† = N^{−1/2} Σ_k e^{−ik·r_j} a_k†), which preserves [a_k, a_k'†] = δ_{kk'}. Write the nearest-neighbor sum with displacement vectors δ. The hopping −JS Σ_⟨ij⟩(a_i†a_j + a_j†a_i) becomes, after the Fourier substitution and using Σ_j e^{i(k−k')·r_j} = N δ_{kk'},

−JS Σ_k Σ_δ cos(k·δ) a_k† a_k,

while the +qJS Σ_i n̂_i diagonal piece becomes +qJS Σ_k a_k† a_k. Combine:

H_1 = Σ_k qJS [ 1 − (1/q) Σ_δ cos(k·δ) ] a_k† a_k ≡ Σ_k ε_k a_k† a_k,

with the magnon dispersion

  ε_k = qJS (1 − γ_k),  γ_k = (1/q) Σ_δ cos(k·δ).

For a 1D chain, δ = ±a, so γ_k = cos(ka) and ε_k = 2JS(1 − cos ka) — exactly Bloch's spin-wave dispersion, now dropping out as the diagonal form of a free-boson Hamiltonian rather than from a hand diagonalization of the one-flip sector. And the small-k limit ε_k ≈ JS(ka)², the quadratic, free-particle-like spectrum. So the magnon is a boson with energy ε_k; the linear-spin-wave Hamiltonian is a gas of non-interacting magnons. The whole many-body spin problem has collapsed into independent oscillators.

Let me double-check the validity of dropping the square root, because the linear map is only an approximation. Numerically: build S^+ from the linear map √(2S)a and compare its matrix elements to the exact spin S^+. Near the top of the ladder (n small, m≈+S) the linear map is essentially exact — the matrix element ⟨S|S^+|S−1⟩ agrees. Near the bottom (n≈2S, m≈−S) the error is order one and grows with S. That's exactly the regime that linear spin-wave theory assumes is empty: at low T only the top of the ladder is occupied, n≪2S, so the linear map is accurate where it's used and fails only in the highly-excited sector that's thermally inaccessible. Good — the approximation is controlled, and the boundary of its validity is visible.

Now hang the external field on it — this is the actual problem I started with. The Zeeman term is gμ_B H Σ_i S_i^z = gμ_B H Σ_i (S − n̂_i) = gμ_B H (NS − Σ_i n̂_i) = gμ_B H NS − gμ_B H Σ_k a_k† a_k. The constant gμ_B H NS just shifts the ground-state energy; the operator part subtracts gμ_B H per magnon — or, depending on the sign convention for which way the field lowers the energy, *adds* a uniform gap gμ_B H to every magnon. Either way, the field enters trivially:

  ε_k(H) = qJS(1 − γ_k) + gμ_B H.

This is the whole reason the boson framework is worth the trouble. In Bloch's eigenstate-counting I would have to re-diagonalize with the field present; here the field is diagonal in the same magnon basis and just rigidly shifts the spectrum by a k-independent gap. Every magnon now costs at least gμ_B H to excite, so a field suppresses the thermal magnon population, which raises ⟨S^z⟩ back toward saturation — the field dependence of the magnetization is now a statistical-mechanics calculation over a free-boson gas with a field-tunable gap. Each thermally excited magnon lowers the total spin by one unit, so

  M(T,H) = gμ_B (NS − Σ_k ⟨a_k† a_k⟩) = gμ_B (NS − Σ_k 1/(e^{ε_k(H)/k_BT} − 1)),

the magnons being bosons with the Bose occupation. Setting H=0 and taking the low-T limit, the integral Σ_k n_B(ε_k) with ε_k≈JS(ka)² gives the saturation deficit ∝ T^{3/2} — Bloch's law re-derived, now as a corollary. Turning H on, the gap gμ_B H cuts off the small-k magnons (the cheapest ones), so the deficit shrinks with field; differentiating with respect to H gives the intrinsic susceptibility, and once the long-range dipolar term is folded into the spectrum the high-field falloff of that susceptibility is the quantity I set out to find. The dipolar interaction itself, being bilinear in the moments, is also a sum of products of spin components, so under the same linearization it adds further quadratic terms to H_1 — k-dependent and anisotropic because the dipolar coupling depends on the bond direction relative to the magnetization — modifying γ_k → an effective dispersion with a dipolar contribution, but it enters the *same* free-magnon framework rather than breaking it. That's the point: field and dipolar terms are perturbations of a diagonal quadratic Hamiltonian, not new diagonalization problems.

One more thing worth nailing down, because it shows the framework's reach: what if the order is *antiferromagnetic*, J effectively favoring antiparallel neighbors on two sublattices? Then I'd put deviations relative to *up* on sublattice A and relative to *down* on sublattice B — S^z = S − a†a on A but S^z = b†b − S on B, with S^+ = b†√(2S−b†b) there. The exchange now connects an up-reference site to a down-reference site, and the leading quadratic Hamiltonian contains *anomalous* terms a_i b_j + a_i† b_j† — pair creation and annihilation — which Fourier transform won't diagonalize because they don't conserve magnon number. Those need a Bogoliubov rotation: define α_k = u_k a_k − v_k b_{−k}†, β_k = u_k b_{−k} − v_k a_k†, with u_k, v_k real and u_k² − v_k² = 1 to keep the new operators bosonic, and choose u_k, v_k to kill the off-diagonal a b + a† b† terms. The condition that the coefficient of (α_k β_k + α_k† β_k†) vanish, γ_k(u_k²+v_k²)+2u_kv_k=0, together with u_k²−v_k²=1, solves to u_k² = ½(1/√(1−γ_k²)+1), v_k² = ½(1/√(1−γ_k²)−1), giving the antiferromagnetic magnon energy ε_k = qJS√(1−γ_k²), which for a 1D chain is 2JS|sin ka| — *linear* in k at small k, not quadratic, and with a zero-point energy lowering that tells me the classical Néel state is not the true ground state. I don't need this for the ferromagnet, but it confirms the representation is the general tool: once spins are bosons, the whole apparatus of quadratic-Hamiltonian diagonalization — Fourier plus Bogoliubov — is available.

So the path is: the localized flip isn't an eigenstate because exchange hops it; Bloch's plane-wave magnons fix the one-excitation sector but give no extensible framework, and choke once I add more excitations or the field and dipolar terms. The deviation count n_i is a non-negative integer bounded by 2S, which is exactly a boson occupation, so I set S^z = S − a†a and let the spin algebra fix S^± = √(2S−a†a)a and a†√(2S−a†a); the square-root factor both truncates the boson space to the finite spin and makes the commutators exact. Expanding the square root in the low-deviation regime turns the Heisenberg + Zeeman + dipolar Hamiltonian into a quadratic free-magnon Hamiltonian plus a controlled 1/S series of interactions; Fourier transform diagonalizes it to ε_k = qJS(1−γ_k), the field enters as a rigid gap ε_k → ε_k + gμ_B H, and the magnetization and its field-dependent susceptibility follow from Bose statistics over this magnon gas — with the dipolar term and the antiferromagnetic case both fitting into the same quadratic-diagonalization framework.

```python
import numpy as np

def spin_matrices(S):
    # standard (2S+1)-dim spin-S matrices in the |S,m> basis, m = S,...,-S
    dim = int(round(2 * S + 1))
    m = np.array([S - k for k in range(dim)])
    Sz = np.diag(m).astype(complex)
    Sp = np.zeros((dim, dim), dtype=complex)
    for k in range(1, dim):                       # S^+|m> = sqrt(S(S+1)-m(m+1))|m+1>
        Sp[k - 1, k] = np.sqrt(S * (S + 1) - m[k] * (m[k] + 1))
    return Sz, Sp, Sp.conj().T

def boson_matrices(S):
    # truncated bosons on the same (2S+1)-dim space, n = 0..2S, |n> <-> m = S-n
    dim = int(round(2 * S + 1))
    a = np.zeros((dim, dim), dtype=complex)
    for n in range(1, dim):                        # a|n> = sqrt(n)|n-1>
        a[n - 1, n] = np.sqrt(n)
    return a, a.conj().T

def msqrt(M):                                       # sqrt of a PSD operator
    w, V = np.linalg.eigh(M)
    return (V * np.sqrt(np.clip(w.real, 0, None))) @ V.conj().T

def hp_exact(S):
    # S^z = S - a^dag a ; S^+ = sqrt(2S - a^dag a) a ; S^- = a^dag sqrt(2S - a^dag a)
    a, ad = boson_matrices(S)
    n = ad @ a
    root = msqrt(2 * S * np.eye(n.shape[0]) - n)    # the truncation/algebra factor
    return S * np.eye(n.shape[0]) - n, root @ a, ad @ root

def hp_linear(S):
    # leading order: S^+ ~ sqrt(2S) a, S^- ~ sqrt(2S) a^dag  (valid for <n> << 2S)
    a, ad = boson_matrices(S)
    n = ad @ a
    return S * np.eye(n.shape[0]) - n, np.sqrt(2 * S) * a, np.sqrt(2 * S) * ad

def comm(A, B):
    return A @ B - B @ A

if __name__ == "__main__":
    for S in [0.5, 1.0, 1.5, 2.0]:
        Sz, Sp, Sm = spin_matrices(S)
        ez, ep, em = hp_exact(S)
        repro = max(np.abs(Sz - ez).max(), np.abs(Sp - ep).max(), np.abs(Sm - em).max())
        alg = np.abs(comm(ep, em) - 2 * ez).max()    # exact: [S^+,S^-] = 2 S^z
        lz, lp, lm = hp_linear(S)
        dim = int(round(2 * S + 1))
        top = np.abs(lp[0, 1] - Sp[0, 1])            # n small (m ~ +S): linear map exact
        bot = np.abs(lp[dim - 2, dim - 1] - Sp[dim - 2, dim - 1])  # n ~ 2S: linear map fails
        print(S, repro, alg, top, bot)
```
