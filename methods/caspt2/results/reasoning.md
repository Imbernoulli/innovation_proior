I have a converged CASSCF wave function in front of me, and it is good for exactly one thing and useless for almost everything else. Good: it has the static correlation. At a stretched bond, where one determinant has σ² and another has σ*² and the two are nearly degenerate, the CAS mixes them with the right weights and the dissociation curve has the right shape and the right spin. Useless: the number it gives me is quantitatively wrong by an electron-volt or more, because the active space is small — a dozen orbitals — and almost all of the correlation energy is the short-range dynamic stuff, the in-and-out jiggling of pairs of electrons avoiding each other near the cusp, and that lives out in the enormous space of excitations into the inactive and virtual orbitals that the CAS never touches. I cannot just make the active space bigger; the CI dimension explodes combinatorially and I lose the calculation entirely. So I need to add the dynamic correlation on top, after the fact, cheaply.

I already know how to do this for a single determinant. Take Hartree–Fock, perturb it. Møller and Plesset showed the way: the zeroth-order Hamiltonian is the sum of one-electron Fock operators, H₀ = Σ_p F̂(p), the HF determinant is its ground eigenstate, and — this is the part that makes it work — every other determinant built from the HF spin-orbitals is *also* an eigenstate of H₀, with eigenvalue equal to the sum of the occupied orbital energies. So I have a soluble zeroth-order problem with a complete, ready-made spectrum, and the perturbation V = Ĥ − H₀ is the fluctuation potential, the difference between the true instantaneous repulsion and the averaged mean field already baked into H₀. Through first order I just get HF back; the first correlation correction is second order; only doubles survive; and the answer is one closed-form sum,

  E⁽²⁾ = Σ_{i<j,a<b} |⟨ij‖ab⟩|² / (ε_i + ε_j − ε_a − ε_b),

cheap, non-iterative, size-extensive. That is exactly the *shape* of correction I want. The question is whether I can do the same thing with the CASSCF state sitting where the HF determinant used to sit.

Let me be honest about why this is not a five-minute port. MP2 leans completely on one fact: the reference is an eigenfunction of a simple one-electron operator, and that same operator hands me the spectrum of excited states for free. My CASSCF state is not like that. It is a variationally optimized linear combination of many active-space configurations. It is an eigenfunction of the *full* Hamiltonian restricted to the active CI space, sure, but that is a complicated many-electron object, not a one-electron operator, and I do not get a clean ladder of excited-state energies out of it. If I want to run Rayleigh–Schrödinger perturbation theory I need a zeroth-order Hamiltonian Ĥ₀ such that my CASSCF state |Ψ₀⟩ is an exact eigenfunction of it, with a known eigenvalue, and such that the rest of the spectrum is cheap. That is the whole problem in one sentence.

Let me write down what I am unwilling to give up, because the constraints are going to drive the construction. First, I want standard RSPT with intermediate normalization — ⟨Ψ₀|Ψ⟩ = 1 — because that is the non-iterative, order-by-order machine I trust and it is what MP2 used. Second, the zeroth-order state has to be the CASSCF state itself; the whole point is to keep the static correlation in zeroth order so I am not perturbing around a broken single determinant. Third, Ĥ₀ should be an *effective one-electron operator*. Why insist on one-electron? Because that is what makes the perturbation step cheap and gives me a Fock-like spectrum I can actually compute with; a two-electron H₀ would be more accurate against some pathologies but it would be expensive and it would not look like MP2 at all. And fourth — this is a sanity anchor I refuse to drop — when I shrink the active space down to nothing, so the CAS is just a single closed-shell or open-shell HF determinant, the whole method has to collapse back exactly to MP2. If it does not reduce to the established single-reference theory in that limit, I do not believe it.

Now I can already feel the second and third constraints fighting. A general one-electron operator does *not* have a multiconfigurational state as an eigenfunction. So I cannot just write down some Fock-like Ĥ₀ and hope |Ψ₀⟩ comes out an eigenstate; in general it won't. I will have to force it somehow — but I cannot force anything until I know what operator I am forcing, so let me get the *operator* right first, the one-electron object whose spectrum I want, by demanding the MP2 limit, and come back to the eigenfunction problem once I have it.

What is the Fock operator, really, when I stop thinking of it as "h plus Coulomb minus exchange" and ask what physical thing its diagonal elements measure? For a closed-shell determinant, the orbital energy ε_p is minus the ionization potential if p is occupied, and minus the electron affinity if p is empty — Koopmans. So the diagonal Fock element is an operator that *snaps* between −(IP) and −(EA) depending on whether the orbital is full or empty. If I want a Fock-like operator for a reference that is partially open-shell and multiconfigurational and not even built from canonical orbitals, maybe I should build it from that snapping property directly, from first principles, rather than copy the closed-shell formula and hope.

So let me actually derive it. Take the N-electron reference |Ψ_N⟩. The ionization potential for removing an electron from orbital p is

  (IP)_p = ⟨Ψ^p_{N−1}|Ĥ|Ψ^p_{N−1}⟩ − ⟨Ψ_N|Ĥ|Ψ_N⟩,

where |Ψ^p_{N−1}⟩ = â_{pσ}|Ψ_N⟩ is the (N−1)-electron state with an electron pulled out of p. Write it with the creation and annihilation operators. Since â†_{pσ}â_{pσ}|Ψ_N⟩ = |Ψ_N⟩ when p is occupied,

  −(IP)_p = ⟨Ψ_N|(â†_{pσ}â_{pσ}Ĥ − â†_{pσ}Ĥ â_{pσ})|Ψ_N⟩ = −⟨Ψ_N| â†_{pσ}[Ĥ, â_{pσ}] |Ψ_N⟩.

And the electron affinity for adding an electron to a virtual p, |Ψ^p_{N+1}⟩ = â†_{pσ}|Ψ_N⟩:

  −(EA)_p = ⟨Ψ_N|(â_{pσ}Ĥ â†_{pσ} − â_{pσ}â†_{pσ}Ĥ)|Ψ_N⟩ = ⟨Ψ_N| â_{pσ}[Ĥ, â†_{pσ}] |Ψ_N⟩.

Now I see the single operator that does both jobs at once. Define

  f̂_{pp} = Σ_σ ( â_{pσ}[Ĥ, â†_{pσ}] − â†_{pσ}[Ĥ, â_{pσ}] ).

For a closed-shell single determinant, if p is doubly occupied the first term annihilates (â†_{pσ} on a full orbital gives zero) and only the second survives, returning −(IP); if p is empty the second term annihilates and the first survives, returning −(EA). The operator automatically picks the right one. Good — that is the closed-shell Fock element rederived from the snapping property, with no assumption beyond Slater–Condon for a single determinant.

But my reference is not a single determinant, and I want off-diagonal elements too, and I want it spin-balanced so it does not artificially prefer α over β. So I generalize: symmetrize the two pieces and average over spin,

  f̂_{pq} = ½ Σ_σ ( â_{pσ}[Ĥ, â†_{qσ}] − â†_{pσ}[Ĥ, â_{qσ}] ).

This is a guess in form, but it is the natural one: it reduces to the diagonal construction when p = q, it is symmetric in the two index roles, and the ½ is just the average of the two ways of writing the same physics. The real test is whether it evaluates to something Fock-like and whether it gives back MP2. Let me grind the commutators with the second-quantized Hamiltonian

  Ĥ = Σ_{pq} h_{pq} Ê_{pq} + ½ Σ_{pqrs} (pq|rs) ê_{pqrs},   Ê_{pq} = Σ_σ â†_{pσ}â_{qσ}.

Take the commutators piece by piece. For the one-electron part, [Ê_{mn}, â†_{qσ}] = δ_{nq} â†_{mσ}, so [Σ_{mn} h_{mn} Ê_{mn}, â†_{qσ}] = Σ_m h_{mq} â†_{mσ}, and contracting with â_{pσ} and summing spin gives Σ_σ â_{pσ}(Σ_m h_{mq} â†_{mσ}) → Σ_m h_{mq}(δ_{pm} + …) ; the number-conserving piece that survives the expectation value is h_{pq}. The symmetric −â†_{pσ}[Ĥ, â_{qσ}] piece, with [Ê_{mn}, â_{qσ}] = −δ_{mq} â_{nσ}, produces the same h_{pq} contribution, and the ½ averages the two into one h_{pq}. For the two-electron part, [ê_{mnrs}, â†_{qσ}] lowers ê (rank two) to a rank-one operator with one index forced to q; the Coulomb labelling (pq|rs) comes from the direct contraction and the (pr|qs) from the exchange contraction, and the spin average over σ is exactly what supplies the factor ½ on the exchange term — single-spin exchange, not doubled. The remaining open index pair contracts against Ê_{rs}. So what drops out is

  f̂_{pq} = h_{pq} + Σ_{rs} Ê_{rs} [ (pq|rs) − ½ (pr|qs) ],

and taking its expectation value over the reference, with the one-particle reduced density matrix D_{rs} = ⟨Ψ₀|Ê_{rs}|Ψ₀⟩,

  f_{pq} = h_{pq} + Σ_{rs} D_{rs} [ (pq|rs) − ½ (pr|qs) ].

Now I have to check that ½ exchange-like term, because if I got the factor wrong everything downstream is wrong. The cleanest check is the MP2 limit. Let the reference be a closed-shell single determinant, so D_{rs} = 2 δ_{rs} over the occupied orbitals and zero otherwise. Then

  f_{pq} = h_{pq} + Σ_k 2 [ (pq|kk) − ½ (pk|qk) ] = h_{pq} + Σ_k [ 2(pq|kk) − (pk|qk) ],

summed over occupied k. That is exactly the canonical closed-shell Fock matrix in chemists' notation — the Coulomb term 2(pq|kk) and the exchange term (pk|qk) — so the ½ is correct precisely because the factor of 2 from the closed-shell density cancels it into the standard exchange. The fourth constraint is satisfied at the operator level: my generalized Fock reduces to the MP2 Fock. The ½ was never a free parameter; it is the spin-average of the exchange and the closed-shell density doubles it back to one. Good.

So I have a one-electron operator F̂ = Σ_{pq} f_{pq} Ê_{pq} built from the reference's own density. Let me look at its matrix f_{pq} blocked by orbital class — inactive (i), active (t), virtual (a). The diagonal blocks (inactive–inactive, active–active, virtual–virtual) I can diagonalize independently if I want pseudocanonical orbitals within each class. The inactive–virtual block is zero — that is the generalized Brillouin / variational stationarity of the reference, the same fact that kills the singles in MP2. But here is the new feature compared with closed-shell HF: the inactive–active and active–virtual blocks are *not* zero. For a multiconfigurational reference the active orbitals are fractionally occupied, the stationarity conditions are different, and those cross blocks carry real coupling — physically, the orbital relaxation between the active set and the rest.

So I have a fork. I could throw away those off-diagonal inactive–active and active–virtual blocks and keep f diagonal-in-classes; that would make Ĥ₀ trivially diagonal and the perturbation cheap. Call that the diagonal option. Or I could keep them. Let me reason about whether dropping them is safe. Consider what happens as the active occupations approach the single-reference limit — an active orbital with occupation near 2 (essentially inactive) or near 0 (essentially virtual). In that regime the active orbital is barely distinguishable from a true inactive or virtual one, and the coupling between the active set and the rest *is* the orbital relaxation that should be there in the single-reference limit. If I drop those blocks, the operator no longer relaxes correctly exactly where the system is most single-reference-like — precisely where I expect the method to be most accurate and where I have the MP2 limit to honor. So dropping them poisons the limit I care most about. I keep the off-diagonal blocks. F̂ = Σ_{pq} f_{pq} Ê_{pq} with all blocks.

Back to the eigenfunction problem. F̂ = Σ f_{pq} Ê_{pq} is a perfectly good one-electron operator, but does it have my CASSCF state as an eigenfunction? No — there is no reason a one-electron operator should have a particular multiconfigurational CI vector as an exact eigenstate, and in general it does not. If |Ψ₀⟩ is not an eigenfunction of Ĥ₀, then E⁽⁰⁾ = ⟨Ψ₀|Ĥ₀|Ψ₀⟩ is not even well-defined as "the zeroth-order energy of an eigenstate," the first-order equation (Ĥ₀ − E⁽⁰⁾)|Ψ⁽¹⁾⟩ = −(V − E⁽¹⁾)|Ψ₀⟩ is built on sand, and the whole RSPT apparatus is invalid. I am stuck. I have the right operator and the wrong eigenstructure.

Stare at what "eigenfunction" actually demands. I need Ĥ₀|Ψ₀⟩ = E⁽⁰⁾|Ψ₀⟩, i.e. Ĥ₀ must map |Ψ₀⟩ onto a multiple of itself with no leakage into other states. F̂|Ψ₀⟩ does leak — applying Σ f_{pq} Ê_{pq} to the CI vector scatters it into other configurations, both inside and outside the active space. What if I simply *forbid* the leakage by hand? Project. Split the whole N-electron Hilbert space into the pieces I care about and let F̂ act only block by block, throwing away every cross-block term that would carry |Ψ₀⟩ out of its own subspace.

What are the right blocks? There is V₀, the one-dimensional space of the reference |Ψ₀⟩ itself. There is V_K, all the *other* configurations you can build inside the active space with the same number of active electrons and the same spin — the rest of the CAS CI space, excluding the reference. There is V_SD, everything reached by single and double replacements that take electrons out of the inactive orbitals or put them into the virtuals — the external space. And V_TQ…, triples and higher into the external space. Now define

  Ĥ₀ = P̂₀ F̂ P̂₀ + P̂_K F̂ P̂_K + P̂_SD F̂ P̂_SD + P̂_TQ F̂ P̂_TQ,

where each P̂ projects onto the corresponding subspace. This operator is block-diagonal across the four subspaces by construction: it never connects V₀ to anything else. Therefore |Ψ₀⟩ ∈ V₀ is an eigenfunction, with eigenvalue E⁽⁰⁾ = ⟨Ψ₀|F̂|Ψ₀⟩ = Σ_{pq} f_{pq} D_{pq}. The clash is resolved: the operator is still one-electron (within each block it is just Σ f_{pq} Ê_{pq}), the reference is now an exact eigenstate, and I have paid for it only by deleting the cross-block matrix elements of a one-electron operator — which is exactly the freedom the partitioning of perturbation theory gives me, since I can choose Ĥ₀ however I like as long as Ĥ₀ + V = Ĥ. The perturbation V = Ĥ − Ĥ₀ absorbs everything I projected out.

With |Ψ₀⟩ an eigenstate and a clean Ĥ₀, RSPT is back on its feet. The first-order wave function lives in the orthogonal complement of |Ψ₀⟩, and the second-order energy is E⁽²⁾ = ⟨Ψ₀|V|Ψ⁽¹⁾⟩ as usual. The only question left is which configurations actually contribute, and how to solve for the coefficients.

Which configurations couple to |Ψ₀⟩ through the full Ĥ at first order? Let me go through the four blocks. V₀ is the reference itself — that fixes the normalization, not the correction. V_K, the other configurations inside the active space: these do *not* couple to |Ψ₀⟩ over Ĥ, because the CASSCF state is variationally optimal within the active CI space — that is the generalized Brillouin statement, ⟨Ψ₀|Ĥ|χ⟩ = 0 for any χ in the active CI space orthogonal to |Ψ₀⟩. So V_K is a null space, just as the singles are null in MP2; no contribution. V_TQ…, triples and higher into the external space: Ĥ is a two-body operator, so ⟨Ψ₀|Ĥ| triple ⟩ vanishes by the generalized Slater–Condon argument — you cannot connect two configurations differing by three or more replacements with a two-electron operator. No contribution. That leaves V_SD, the single and double replacements into the external space, as the *only* space that interacts with the reference at first order. This is the analogue of "only doubles" in MP2, but richer, because now the "occupied" side of an excitation can be an inactive orbital *or* an active orbital, and the "virtual" side can be active or truly virtual.

Here is where I have to be careful, because I have been burned before. The natural first instinct, coming from MP2, is to excite electrons out of the inactive orbitals into the virtuals — treat the inactive electrons as the ones that need dynamic correlation and leave the active electrons alone, since the CAS already did the active part. Let me try that: restrict V_SD to single and double replacements out of the inactive orbitals only. … It does not work. The energies it gives are wrong, and once I look at what is missing I see why. The active electrons are *also* correlated electrons. They have a short-range cusp too; pairs within the active space, and pairs of one active and one inactive electron, also avoid each other dynamically, and those correlations are exactly the doubles that move active electrons into the virtuals or scatter inactive-active pairs. The CASSCF only correlated the active electrons *among the active orbitals* — that is static correlation. The dynamic correlation of the active electrons reaches *outside* the active space and I left it out. So the first-order interacting space must come from single and double replacements of *any* electron, active or inactive, not just the inactive ones. That single correction is the difference between a method that fails and a method that works.

Now let me enumerate V_SD properly. A general double replacement is Ê_{pq} Ê_{rs}|Ψ₀⟩, with the four orbital indices each being inactive (i,j), active (t,u,v), or virtual (a,b). The case where all four indices are active is not in V_SD at all — it stays inside the active space, so it belongs to V_K. Throw that out. What remains, classified by how many of the excitations land in the virtual space, splits into three families. Internal, with no electron put into a virtual: Ê_{ti}Ê_{uv}|Ψ₀⟩ (call it class A) and Ê_{ti}Ê_{uj}|Ψ₀⟩ (class B). Semi-internal, with one electron in a virtual: Ê_{at}Ê_{uv}|Ψ₀⟩ (C), the pair Ê_{ai}Ê_{tu}|Ψ₀⟩ together with Ê_{ti}Ê_{au}|Ψ₀⟩ (D), and Ê_{ti}Ê_{aj}|Ψ₀⟩ (E). External, with two electrons in virtuals: Ê_{at}Ê_{bu}|Ψ₀⟩ (F), Ê_{ai}Ê_{bt}|Ψ₀⟩ (G), and Ê_{ai}Ê_{bj}|Ψ₀⟩ (H). Eight classes. The fully external class H, Ê_{ai}Ê_{bj}|Ψ₀⟩ with all four indices inactive/virtual, is exactly the MP2 double excitation — another sign the construction is right, because in the single-determinant limit only H survives and it must reproduce MP2.

For the classes where the two excitation operators carry different index pairs, the two ways of coupling spin are not equivalent, and I should form the spin-symmetry-adapted combinations. For class G, for instance,

  Ê_{ai}Ê_{bt}|Ψ₀⟩ ± Ê_{bi}Ê_{at}|Ψ₀⟩,

the symmetric combination being singlet-coupled and the antisymmetric one triplet-coupled; the two do not mix and can be handled as separate orthogonal sets. Same idea for B, E, F, H.

Now solve for the first-order correction. Write it as a linear combination of the interacting configurations,

  |Ψ⁽¹⁾⟩ = Σ_{j∈SD} C_j |Ψ_j⟩.

The Rayleigh–Schrödinger first-order equation, projected onto each interacting function |Ψ_i⟩, is

  Σ_{j∈SD} C_j ⟨Ψ_i|(Ĥ₀ − E⁽⁰⁾)|Ψ_j⟩ = −⟨Ψ_i|Ĥ|Ψ₀⟩,

for every i in SD. The right-hand side uses the full Ĥ rather than V because ⟨Ψ_i|Ĥ₀|Ψ₀⟩ = 0 — the projected Ĥ₀ does not connect SD to V₀ — so ⟨Ψ_i|V|Ψ₀⟩ = ⟨Ψ_i|Ĥ|Ψ₀⟩. Write the system compactly as

  (H₀ − E⁽⁰⁾ S) C = −V,   V_i = ⟨Ψ_i|Ĥ|Ψ₀⟩,

and then the second-order energy is simply the contraction

  E⁽²⁾ = ⟨Ψ₀|Ĥ|Ψ⁽¹⁾⟩ = V† C.

That looks done, but I have quietly written an S in there, an overlap matrix, and that S is the next wall. In MP2 the doubles are orthonormal determinants and S is the identity. Here the interacting functions are *internally contracted*: each one is an excitation operator applied to the *whole* multiconfigurational |Ψ₀⟩, not to a single determinant. Why do it that way at all? Because the alternative — exciting each CAS configuration separately and treating the results as independent — would make the first-order space astronomically large, scaling with the CI length of the reference; the contracted functions instead scale with the number of orbital pairs, independent of how many configurations the CASSCF has. That is what keeps the method affordable for a real multiconfigurational reference. But the price is that Ê_{pq}Ê_{rs}|Ψ₀⟩ for different index sets are *not* orthogonal, and worse, within a class they can be linearly dependent or nearly so — applying two different excitation strings to the same correlated state can produce the same (or nearly the same) function. So S is non-trivial and possibly singular, and I cannot just invert (H₀ − E⁽⁰⁾).

The fix is symmetric orthonormalization with a cutoff. Diagonalize the overlap within each class,

  Λ_S = U† S U,

throw away the eigenvectors whose eigenvalue is at or below a small threshold (these are the linear-dependent or near-dependent directions that carry no independent physics and would otherwise blow up the inverse), and build the transformation onto the surviving, orthonormal directions,

  Ω = U Λ_S^{−1/2}.

Transform the whole system into this clean basis: H₀′ = Ω† H₀ Ω, V′ = Ω† V, C′ = Ω† C (more precisely the reduced coefficients in the new basis), and solve

  (H₀′ − E⁽⁰⁾ I) C′ = −V′,   E⁽²⁾ = V′† C′.

Now S has become the identity on the surviving space and the resolvent is well-conditioned. If I go one step further and diagonalize H₀′ within each class too, the equations decouple into independent modes α with zeroth-order energies F_α, and each coefficient is just

  C_α = −V_α / (F_α − E⁽⁰⁾),

which is the MP2 formula generalized: a coupling in the numerator over an energy-denominator difference. In the single-determinant limit the only surviving class is H, F_α − E⁽⁰⁾ becomes ε_a + ε_b − ε_i − ε_j, and I recover MP2 exactly. The construction holds together end to end.

I should record what these matrix elements cost, because it is the price of the multiconfigurational reference. The right-hand side V_i and the H₀′ blocks for the fully external classes F, G, H need only the one- and two-particle density matrices of the reference. But the internal and semi-internal classes A through E, which carry active indices on both excitation operators, require contractions up to the three-particle and even four-particle density matrices of the reference. That is the real computational signature of CASPT2: the energy is a single second-order sum like MP2, but assembling it needs the high-order reduced density matrices of the CAS wave function, and those are what make it heavier than MP2 while still vastly cheaper than diagonalizing in the external space.

Now I run a thought-test on the operator before trusting it on a hard case: a tiny open-shell system, three electrons in three orbitals — one inactive orbital i doubly occupied, one active orbital t with a single α electron, one virtual a. The reference is a single doublet CSF, |Ψ₀⟩ = (i)²(t)¹. The one-particle density is diagonal with entries 2, 1, 0 on i, t, a. My generalized Fock element is

  f_{pq} = h_{pq} + 2(pq|ii) − (pi|qi) + (pq|tt) − ½(pt|qt),

the inactive orbital contributing a full closed-shell Coulomb-minus-exchange, the half-occupied active orbital contributing a Coulomb term with only *half* the exchange because there is one electron, not two. That half-exchange on the open-shell orbital is the whole reason an ordinary closed-shell Fock operator would mis-handle an open shell, and here it falls out automatically from the density weighting. The operator snaps correctly between the occupied-orbital IP regime and the empty-orbital EA regime through the same density-weighted formula. I am convinced the operator is right.

One more wall, and it is a serious one: intruder states. Look again at C_α = −V_α/(F_α − E⁽⁰⁾). Nothing guarantees the denominator stays away from zero. There can be a configuration in the external space whose zeroth-order energy F_α happens to lie right on top of E⁽⁰⁾. When that happens the denominator vanishes, the coefficient C_α explodes, and the second-order energy becomes physically meaningless — a single near-degenerate perturber swamps the whole correction. This is not a numerical accident; it is a structural feature of perturbing a real molecule, where the external space is dense and some diffuse or low-lying configuration can drift into resonance with the reference, especially for excited states or transition metals.

Let me understand it with the smallest possible model and then fix it. Two states, the reference with zeroth-order energy α and a perturber with zeroth-order energy β, coupled by δ:

  Ĥ(z) = [[α, 0],[0, β]] + z [[0, δ],[δ, 0]].

The exact eigenvalues are

  E_±(z) = (α+β)/2 ± ½ √((β−α)² + 4 z² δ²).

The perturbation series in z is the Taylor expansion of that square root. A √(1 + x) expansion converges only for |x| < 1, and here x = 4 z² δ² / (β−α)². At the physical point z = 1 the series therefore converges iff

  4 δ² / (β−α)² < 1   ⇔   |β − α| > 2 |δ|.

So the second-order expansion is trustworthy exactly when the zeroth-order energy gap exceeds twice the coupling. When the gap collapses below 2|δ|, the branch point of the square root, at z = ± i (β−α)/(2δ), moves inside the unit circle and the series diverges. That branch point is the intruder, sitting on the imaginary axis. The diagnosis is precise: the trouble is a small *denominator* gap, and I have two levers — widen the gap, or shrink the coupling.

Widening the gap is the cheap fix and it respects the structure. Add a shift ε to the perturber's zeroth-order energy in Ĥ₀ and subtract the same ε from the coupling block in V, so that Ĥ₀ + V is unchanged at z = 1 — I am only re-partitioning, not changing the physics:

  Ĥ₀ = [[α, 0],[0, β + ε]],   V = [[0, δ],[δ, −ε]].

The first-order coefficient becomes

  C_i = ⟨Ψ₀|V|Ψ_i⟩ / (−ε_i − E⁽⁰⁾ + ε),

with the shift ε in the denominator keeping it finite even when the original gap −ε_i − E⁽⁰⁾ collapses. Solving the shifted two-state model for the convergence condition gives a minimum shift

  ε_c = [ 4 δ² − (β−α)² ] / [ 2 (β−α) ],

so for any gap and coupling there is a shift that pushes the branch point back outside the unit circle and kills the intruder. That is a real-valued level shift.

But a constant real shift has a flaw I can see immediately from C_i: it changes *every* coefficient, including the well-behaved ones with large gaps where I did not want any change, and it does not actually remove the singularity — it just moves it, because if the original denominator is negative the shifted one can pass through zero somewhere else. A cleaner instrument is an imaginary shift, iε. The denominator becomes Δ_i + iε with Δ_i = ε_i − E⁽⁰⁾, which can never be zero for real Δ_i, so the singularity is gone entirely. Keeping only the real part of the coefficient, since the energy must be real,

  Re(C_i) = −⟨Ψ₀|V|Ψ_i⟩ · Δ_i / (Δ_i² + ε²).

This damps the dangerous near-resonant terms smoothly toward zero while leaving the safe terms with large |Δ_i| essentially untouched — exactly the behavior I want, a shift that acts only where it is needed.

There is one last refinement to make the shifted answer trustworthy. The straightforward second-order energy E⁽²⁾ = V′†C′ is *linear* in the coefficients, so it is sensitive to the shift to first order — change ε a little and the answer drifts. I can do better by evaluating the energy through the Hylleraas functional,

  E⁽²⁾ = ⟨Ψ⁽¹⁾|(Ĥ₀ − E⁽⁰⁾)|Ψ⁽¹⁾⟩ + 2 ⟨Ψ₀|V|Ψ⁽¹⁾⟩,

which is *stationary* with respect to variations of |Ψ⁽¹⁾⟩ at the true first-order solution. Because it is stationary, a small error in the coefficients introduced by the shift only perturbs the energy at second order, so the Hylleraas energy is far less sensitive to the value of the shift than the bare projected expression. I will use the Hylleraas form as the reported energy and treat the projected form V†C as a diagnostic.

Let me trace the whole causal chain one more time so I am sure nothing dangles. I want dynamic correlation on a CASSCF reference, cheaply, valid where single-reference theory fails. Perturbation theory is the cheap, size-extensive instrument, but it needs a soluble Ĥ₀ with the reference as an eigenfunction. I demand four things — RSPT, CASSCF reference, one-electron Ĥ₀, MP2 limit — and the MP2 limit forces the *form* of the operator: a generalized, spin-averaged Fock built from the snapping IP/EA commutators, f_{pq} = h_{pq} + Σ_{rs} D_{rs}[(pq|rs) − ½(pr|qs)], with all blocks kept so the orbital relaxation survives toward the single-reference limit. A bare Σ f_{pq}Ê_{pq} does not have the CASSCF state as an eigenfunction, so I project it into the reference, the rest of the active space, the singles-and-doubles external space, and the higher excitations, taking only the block-diagonal part — which makes the reference an exact eigenstate while keeping the operator one-electron. Only the singles-and-doubles external space interacts at first order, because the rest of the active space is killed by the variational stationarity of the reference and the higher excitations by the two-body nature of Ĥ — and crucially that space must be built from excitations of *all* electrons, active and inactive, not just the inactive ones. The interacting functions are internally contracted, so they are non-orthogonal and linearly dependent; I orthonormalize with an overlap diagonalization and a small-eigenvalue cutoff, solve the resulting well-conditioned linear system, and contract to the second-order energy, which reduces term-by-term to MP2 when the active space collapses. Finally, near-degenerate perturbers make the denominators vanish; an imaginary level shift removes the singularity and the Hylleraas functional makes the shifted energy stationary and robust. That is the method.

```python
import numpy as np
from scipy.sparse.linalg import cg, LinearOperator

# Inputs from a converged CASSCF: orbital partition into inactive (i,j),
# active (t,u,v), virtual (a,b); MO integrals h[p,q] and eri[p,q,r,s] = (pq|rs)
# in chemists' notation; the reference one- and two-particle reduced density
# matrices dm1, dm2 over the ACTIVE orbitals; and the CASSCF energy E0.

class CASReference:
    def __init__(self, ncore, nact, nvirt, h, eri, dm1, dm2, E0):
        self.ncore, self.nact, self.nvirt = ncore, nact, nvirt
        self.h, self.eri, self.E0 = h, eri, E0
        self.dm1, self.dm2 = dm1, dm2
        self.nmo = ncore + nact + nvirt
        # index ranges by class
        self.core = range(0, ncore)
        self.actv = range(ncore, ncore + nact)
        self.virt = range(ncore + nact, self.nmo)

    def full_dm1(self):
        """1-RDM over all MOs: core doubly occupied, active from CAS, virtual empty."""
        D = np.zeros((self.nmo, self.nmo))
        for i in self.core:
            D[i, i] = 2.0
        ca = self.ncore
        D[ca:ca+self.nact, ca:ca+self.nact] = self.dm1
        return D

def generalized_fock(ref):
    """f_pq = h_pq + sum_rs D_rs [ (pq|rs) - 1/2 (pr|qs) ]   -- the one-electron operator.
    Reduces to the closed-shell Fock when D_rs = 2*delta over the occupied space (MP2 limit)."""
    D = ref.full_dm1()
    eri = ref.eri
    coulomb  = np.einsum('rs,pqrs->pq', D, eri)          # sum_rs D_rs (pq|rs)
    exchange = np.einsum('rs,prqs->pq', D, eri)          # sum_rs D_rs (pr|qs)
    return ref.h + coulomb - 0.5 * exchange

def first_order_space(ref):
    """The single+double replacements E_pq E_rs |Psi0> that interact at first order,
    classified A..H by how many indices are virtual (none / one / two). The all-active
    case is excluded (it lies in V_K and is killed by the reference's variational
    stationarity); triples+ are excluded (H is two-body). Excitations of ANY electron,
    active or inactive, are included -- not only the inactive ones."""
    c, a, v = list(ref.core), list(ref.actv), list(ref.virt)
    classes = {
        'A': [('ti', 'uv')], 'B': [('ti', 'uj')],                 # internal
        'C': [('at', 'uv')], 'D': [('ai', 'tu'), ('ti', 'au')],   # semi-internal
        'E': [('ti', 'aj')],
        'F': [('at', 'bu')], 'G': [('ai', 'bt')], 'H': [('ai', 'bj')],  # external
    }
    return classes  # each class expands over its index ranges (c,a,v) at assembly time

def build_class_system(ref, fock, E0):
    """For one excitation class, assemble the overlap S, the zeroth-order operator
    block H0 = projected generalized Fock, and the coupling vector V_i = <Psi_i|H|Psi0>.
    S and H0 contract the reference RDMs (up to 3- and 4-particle for the internal and
    semi-internal classes); V contracts the integrals with the RDMs. Returns (S, H0, V)."""
    # ... contractions of fock, eri with dm1, dm2 (and dm3, dm4 for A..E) ...
    raise NotImplementedError  # class-by-class RDM contractions

def orthonormalize(S, H0, V, cutoff=1e-10):
    """Symmetric orthonormalization with a small-eigenvalue cutoff: the internally
    contracted functions are non-orthogonal and (near-)linearly dependent."""
    w, U = np.linalg.eigh(S)
    keep = w > cutoff                       # drop the null / near-null directions
    Omega = U[:, keep] / np.sqrt(w[keep])   # Omega = U Lambda^{-1/2}
    H0p = Omega.T @ H0 @ Omega
    Vp  = Omega.T @ V
    return H0p, Vp, Omega

def solve_class(H0p, Vp, E0, imag_shift=0.0):
    """Solve (H0' - E0 I) C' = -V' with an optional imaginary level shift to remove
    intruder-state singularities, then report the second-order energy three ways."""
    n = H0p.shape[0]
    diag = np.diag(H0p) - E0                              # zeroth-order denominators
    if imag_shift > 0.0:
        # imaginary shift: Re C = -V * Delta / (Delta^2 + eps^2), never singular
        Cp = -Vp * diag / (diag**2 + imag_shift**2)
    else:
        A = H0p - E0 * np.eye(n)
        Cp, _ = cg(A, -Vp, atol=1e-12)                   # initial guess C = V/diag
    E2_diagonal      = -np.sum((Vp / diag) * Vp)         # diagonal estimate
    E2_nonvariational = -np.dot(Cp, Vp)                  # V . C
    # Hylleraas variational form: stationary in C', insensitive to the shift
    E2_variational = 2.0 * E2_nonvariational + Cp @ ((H0p - E0 * np.eye(n)) @ Cp)
    return E2_variational, E2_nonvariational, E2_diagonal, Cp

def caspt2_energy(ref, imag_shift=0.0):
    """Total second-order dynamic-correlation correction to the CASSCF energy."""
    fock = generalized_fock(ref)
    E0 = ref.E0
    classes = first_order_space(ref)
    E2 = 0.0
    for label in classes:
        S, H0, V = build_class_system(ref, fock, E0)       # per-class assembly
        H0p, Vp, _ = orthonormalize(S, H0, V)
        e2, _, _, _ = solve_class(H0p, Vp, E0, imag_shift)
        E2 += e2
    return E0 + E2, E2
```
