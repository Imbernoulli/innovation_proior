Let me start from the thing that actually goes wrong, because that is where the method has to come from. Take a closed-shell molecule with one covalent bond — H₂ is the cleanest — and stretch it. At equilibrium the restricted Hartree–Fock picture is fine: both electrons sit in the bonding orbital σ, the determinant |σσ̄| is a decent description. Now pull the nuclei apart. Physically the two electrons must end up one on each atom, the system must dissociate into two neutral hydrogens, and the wavefunction must stay a singlet. What does RHF give me? It keeps forcing both electrons into σ. Write σ ≈ (1s_A + 1s_B)/√2; then |σσ̄| expands into 1s_A1s_A + 1s_B1s_B + 1s_A1s_B + 1s_B1s_A pieces with equal weight. Half of that is ionic — both electrons on the same atom, H⁻···H⁺ — and that ionic weight never goes away no matter how far I pull. So RHF dissociates to something with a huge spurious ionic contribution and an energy far above two neutral H atoms. The curve is qualitatively wrong.

I know the fix conceptually: at large separation the bonding configuration σ² and the antibonding configuration σ*² become degenerate, and the correct singlet is their combination. With σ* ≈ (1s_A − 1s_B)/√2, the combination (σ² − σ*²)/√2 is exactly 1s_A1s_B + 1s_B1s_A — two neutral atoms, no ionic terms, and it is a proper singlet. So one determinant is structurally incapable of this; I need at least two configurations, c₁ σ² + c₂ σ*², with c₁ ≈ 1, c₂ ≈ 0 near equilibrium smoothly going to c₁ = −c₂ = 1/√2 at dissociation. This is "static" correlation — it is not a small dynamical-correlation correction, it is a qualitative near-degeneracy that one determinant gets categorically wrong.

Could I escape with UHF instead? Let the α electron localize on A and the β on B. Yes, the curve shape comes out roughly right — but the determinant is now a mix of singlet and triplet, it is not an eigenfunction of Ŝ². Spin contamination. I want spin-pure energies across the whole surface, so UHF is not the answer; I have to keep the multi-configuration, spin-adapted form. So the target is settled: a wavefunction that is a CI expansion Ψ = Σ_I c_I Φ_I in configuration state functions, with the c_I determined variationally, and — this is the part single-reference CI never does — the orbitals themselves re-optimized for this multiconfiguration energy rather than frozen at their Hartree–Fock values. If I freeze the orbitals at HF and only vary the c_I, I am doing ordinary CI; that recovers correlation but it does not let the *orbitals* relax to the multiconfiguration situation, which matters because the optimal σ, σ* at a stretched geometry are not the HF orbitals. So I want to make the energy stationary with respect to both the c_I and the orbitals at once. That is the multiconfiguration SCF problem.

Now, how big is this expansion and which configurations do I keep? This is the question that decides whether the method is usable. The honest general-MCSCF answer is "the user picks a list of important configurations." I have seen where that leads: the answer depends on the human's choice, two people get two surfaces, the procedure is fragile, and convergence of the coupled orbital-plus-CI optimization is touchy because the chosen list and the orbitals interact. I want something I can hand a molecule and an orbital window and turn the crank — black-box.

Let me think about what a clean, choice-free CI space would be. Partition the molecular orbitals into three groups. Some orbitals are doubly occupied in *every* configuration — call them inactive (the chemically inert core and the deep bonding pairs that never break). Some are empty in *every* configuration — virtual. And in the middle, a handful of orbitals where the action is: the near-degenerate frontier orbitals that carry the static correlation. For H₂ that middle set is just {σ, σ*}. Call these active. Then the choice-free CI space is: take the active electrons and distribute them over the active orbitals in *all possible ways*, generating *every* configuration that lives in the active space. A complete CI within the active orbitals. For H₂ with two electrons in two active orbitals that is exactly the configurations σ², σσ*, σ*² (and the spin-adapted singlet combinations) — which contains the σ² and σ*² I needed. The user no longer selects configurations; the user selects *which orbitals are active*, and the chemistry knowledge goes into that smaller, more natural decision.

Why is "complete within the active space" the right move and not just a convenience? Because completeness buys an invariance. If I take any unitary rotation of the orbitals *within* the active set, the set of all distributions of the active electrons over the active orbitals is unchanged — it just gets re-expressed in a rotated basis — so the energy is invariant under active-active rotations. Likewise it is invariant under rotations within the inactive set (all doubly occupied, a closed shell is rotation-invariant) and within the virtual set (all empty). That invariance is exactly what kills the pathology of hand-selected MCSCF, where the selection and the orbitals fight each other. Here the wavefunction is the variationally best one *in the chosen valence space*, full stop — this is the full-valence / full-reaction-space idea, made into an algorithm. And spin adaptation comes for free if I build the active CI out of CSFs rather than raw determinants, so Ŝ² is exact, which I need for the low-spin open-shell and dissociation cases.

The price is obvious and I should name it: the number of active configurations grows combinatorially — binomially — in the active size, so the active space has to stay small. That is fine; the whole design is "small active space, complete within it, orbitals optimized." The exponential wall is pushed into one human decision (how many orbitals are active) and otherwise the method is automatic.

Good. Now the real work: how do I actually optimize the orbitals for this energy, efficiently? Let me get the energy in a form I can differentiate. Write the Hamiltonian in second quantization with the spin-summed generators Ê_pq = a†_{pα}a_{qα} + a†_{pβ}a_{qβ}. The two-electron part needs the operator ê_pqrs = Ê_pq Ê_rs − δ_qr Ê_ps (the δ corrects the spurious term when the indices coincide, because Ê_pq Ê_rs is not normal-ordered). So

  Ĥ = Σ_pq h_pq Ê_pq + ½ Σ_pqrs (pq|rs) ê_pqrs.

Take the expectation value in the CASCI state |Ψ⟩. Define the one- and two-particle reduced density matrices

  γ_pq = ⟨Ψ| Ê_pq |Ψ⟩,  Γ_pqrs = ⟨Ψ| ê_pqrs |Ψ⟩.

Then

  E = Σ_pq h_pq γ_pq + ½ Σ_pqrs (pq|rs) Γ_pqrs.

This is the key structural fact, and I want to stare at it. The energy is *linear* in γ and Γ. Everything about the CI vector enters only through these density matrices. The CI expansion can be enormous — thousands of configurations — but the energy, and as I'll see the orbital gradient too, sees the CI vector *only* through γ and Γ, whose indices run over the occupied (inactive + active) orbitals. The inactive part is trivial: in every configuration the inactive orbitals are doubly occupied, so γ_ij = 2δ_ij there, and Γ over inactive indices is fixed. The only nontrivial density-matrix elements are those with all indices in the *active* set: γ_tu and Γ_tuvw, a small object set by the active size, not by the CI length. Hold onto that — it is going to be the whole efficiency story.

First let me dispatch the inactive orbitals so I don't carry them around. Split every index into inactive {i,j,k,l}, active {t,u,v,x}, virtual {a,b,c,d}, general {p,q,r,s}. Since the inactive density is fixed (doubly occupied), the pieces of E that involve inactive indices collapse into a constant core energy plus a modification of the one-electron operator seen by the active electrons. Concretely the inactive electrons contribute a closed-shell Coulomb-minus-exchange potential; folding it in gives an effective one-electron integral in the active space,

  h^eff_tu = h_tu + Σ_i [ 2(tu|ii) − (ti|iu) ],

and a core energy E_core = Σ_i [2 h_ii + Σ_j (2(ii|jj) − (ij|ji))] (plus nuclear repulsion). With these, the active CI is just a full CI of the active electrons under (h^eff, active two-electron integrals), and

  E = E_core + Σ_tu h^eff_tu γ_tu + ½ Σ_tuvw (tu|vw) Γ_tuvw.

So the CI step is small and self-contained, and it hands me back γ_tu, Γ_tuvw. Now everything that follows about orbitals must be expressible through those.

Parametrize the orbital change. I want C → C·U with U unitary so orthonormality is automatic and I never have to drag Lagrange multipliers around to enforce ⟨φ_p|φ_q⟩ = δ_pq. The clean way to get an unconstrained, automatically-unitary parametrization is the exponential of an anti-Hermitian operator:

  U = e^κ,  κ† = −κ,  κ̂ = Σ_{p>q} κ_pq (Ê_pq − Ê_qp) ≡ Σ_{p>q} κ_pq Ê⁻_pq.

The anti-Hermiticity makes e^κ unitary to all orders; the κ_pq for p>q are independent real variables; and rotating the state is |Ψ̃⟩ = e^{−κ̂}|Ψ⟩, equivalently transforming the orbitals. (If I ever truncate the exponential I'll have to re-orthonormalize, but as a parametrization for taking derivatives it is exactly what I want.) Why not just vary the MO coefficients directly with orthonormality constraints? Because then I'm back to constrained optimization with multipliers, and the formalism gets opaque; the exponential turns it into free variables.

Which κ_pq actually matter? Here the completeness invariance pays off again. Rotations within the inactive set, within the virtual set, and within the active set leave E unchanged — I argued that from completeness. So κ_ij, κ_ab, and κ_tu are *redundant*: they change the orbitals but not the energy, and including them just creates a singular optimization (zero gradient, zero curvature) and numerical trouble. Drop them. What's left are the inter-space rotations:

  κ̂ = Σ_{ti} κ_ti Ê⁻_ti + Σ_{ai} κ_ai Ê⁻_ai + Σ_{at} κ_at Ê⁻_at,

i.e. core→active (ti), core→virtual (ai), and active→virtual (at). Three blocks. (I should note to myself: the active-active block κ_tu being redundant is special to a *complete* active space. If the active CI were truncated — not complete — then rotating within the active set *would* change the energy and I'd have to keep κ_tu. The completeness is exactly what removes that block and simplifies the optimization. Good, another reason the "complete" in complete-active-space is load-bearing, not cosmetic.)

Now the orbital gradient. I expand the rotated energy E(κ) = ⟨Ψ| e^{−κ̂} Ĥ e^{κ̂} |Ψ⟩ to first order. The derivative at κ = 0 is

  G_pq = ∂E/∂κ_pq |₀ = ⟨Ψ| [Ê⁻_pq, Ĥ] |Ψ⟩ = ⟨Ψ| [Ê_pq − Ê_qp, Ĥ] |Ψ⟩.

Let me evaluate ⟨Ψ|[Ê_pq, Ĥ]|Ψ⟩. Using the energy form, this commutator's expectation contracts the integrals against γ and Γ. Carrying it through, the convenient object that appears is the *generalized Fock matrix*

  F_pq = Σ_r h_pr γ_rq + Σ_rst (pr|st) Γ_qrst,

and the gradient is its antisymmetric part:

  G_pq = 2 ( F_pq − F_qp ).

This is worth dwelling on. For ordinary Hartree–Fock, γ_rq = 2δ_rq over occupied and Γ collapses to products of γ, and F reduces to the usual Fock matrix; the stationarity condition F_pq − F_qp = 0 with occupied p, virtual q is exactly the Brillouin theorem (the occupied–virtual block of the Fock matrix vanishes). So my multiconfiguration stationarity condition

  G_pq = 0  ⇔  F_pq = F_qp  (F symmetric)

is the *generalized* Brillouin condition: at the optimum the generalized Fock matrix is symmetric, equivalently the reference does not couple through Ĥ to any singly-orbital-excited state. And — crucial — F is built by contracting integrals with γ and Γ whose nontrivial indices are active. The orbital gradient depends on the CI vector *only* through the small active-space density matrices. The CI can be huge; the gradient does not care.

So now I have a stationarity condition and a gradient that are both cheap. How do I take the step? Let me first try the most natural multiconfiguration generalization of "diagonalize the Fock matrix," because Hartree–Fock just diagonalizes F and reads off improved orbitals.

The super-CI route. The generalized Brillouin theorem says: at the optimum, ⟨Ψ| Ĥ Ê⁻_pq |Ψ⟩ = 0 for all inter-space (p,q) — the reference is decoupled from the singly-excited states |Ψ^q_p⟩ = Ê⁻_pq |Ψ⟩. Away from the optimum it is not, and the size of the coupling is exactly −G_pq/… So: build the small secular problem of Ĥ in the space spanned by {|Ψ⟩} ∪ {|Ψ^q_p⟩}, the "super-CI," diagonalize it, and the coefficients of the singly-excited states in the lowest eigenvector tell me how much of each single excitation to fold into the reference — i.e. they *are* the orbital rotation parameters κ_pq. Rotate the orbitals by those amounts, re-solve the active CI to refresh γ, Γ, rebuild the super-CI, iterate. At convergence the singles have zero coefficient — that is the generalized Brillouin condition satisfied — and the orbitals are optimal. This is exactly the trick that already existed for general MCSCF references.

In the original super-CI, though, the singly-excited states |Ψ^q_p⟩ = Ê⁻_pq|Ψ⟩ are built on the *entire* reference expansion, and the super-CI secular matrix elements like ⟨Ψ^q_p| Ĥ |Ψ^s_r⟩ are evaluated over that full expansion. So the work per orbital step scales with the *length of the CI*. With a complete active space the CI is long by design — that is the whole combinatorial cost — so a super-CI whose every matrix element is a sum over all CI pairs would put the giant CI length right back into the orbital step, and the method would be dead on arrival.

Let me look harder at those super-CI matrix elements, because the density-matrix structure I noticed in the energy might rescue this. The off-diagonal coupling of the reference to a single excitation is

  ⟨Ψ| Ĥ Ê⁻_pq |Ψ⟩ = −G_pq = −2(F_pq − F_qp),

which I already know depends on the CI only through γ, Γ. So the *gradient* part of the super-CI is already CI-length-independent. What about the diagonal block, ⟨Ψ^q_p| Ĥ |Ψ^s_r⟩? Written out, this is ⟨Ψ| Ê⁻_{qp} Ĥ Ê⁻_{rs} |Ψ⟩ — an expectation value, in the reference, of a string of three operators. The trouble is that Ĥ in the middle drags in *all* orbitals, so naïvely this is again a sum over the full CI. I need to reorganize it so that what survives is again a contraction with the small active-space density matrices.

If I replace the bare Ĥ between the singly-excited states by a one-electron-like effective operator that has the reference as an eigenstate, the three-operator string collapses. Build a generalized (averaged) Fock operator — the same F-style object, contracted with γ and Γ — so that, by construction, this effective operator acting on |Ψ⟩ returns E|Ψ⟩ (it reproduces the reference energy). Concretely use the generalized Fock built from the inactive/core potential plus the active density-weighted part,

  f_pq = h_pq + Σ_i [ 2(pq|ii) − (pi|iq) ] + Σ_tu [ (pq|tu) − ½(pt|uq) ] γ_tu,

i.e. an inactive (closed-shell) Fock plus an active-density-weighted Fock, with a constant shift C chosen so that the effective Hamiltonian gives E on the reference. Now every super-CI matrix element reduces to a commutator/expectation of this effective one-electron operator with the single-excitation generators, and *those* contract the integrals only against γ_tu and Γ_tuvw. The diagonal block elements become things like

  ⟨Ψ^q_p| Ĥ_eff |Ψ^s_r⟩  →  built from f_pq and the active-space γ, Γ,

with all the heavy summations confined to the active indices. The super-CI secular problem is now *small* and its construction cost is set by the active-space density matrices, independent of how many configurations the CI actually contains. That is the resolution: formulate the super-CI in terms of the active-space density matrices. The orbital optimization is decoupled from the CI length. The CI step (a direct full CI in the active space, done with the unitary-group bookkeeping so it is spin-adapted and never stores the Hamiltonian, and so that it spits out γ, Γ directly from the same machinery) is the only thing that scales with the active size; the orbital step rides entirely on the small density matrices.

Let me make the actual orbital-update formula concrete in the simplest useful approximation, because that is what I'd code first. If I take the super-CI diagonal block in its leading (diagonal) approximation, the rotation parameters come out as gradient over an orbital-energy denominator. For the core→virtual block, where the relevant orbitals look like ordinary doubly-occupied and empty ones, this is

  κ_ai = G_ai / (f_ii − f_aa),

a steepest-descent-with-Fock-denominator step — exactly the structure of an SCF orbital update, but driven by the *generalized* Fock gradient. For the core→active and active→virtual blocks the denominators carry the active-orbital occupations and the diagonalization mixes in the active density-matrix eigenvectors, but the shape is the same: rotation = (generalized-Fock gradient) divided by a (generalized-Fock energy difference), all evaluated from γ, Γ. Collect the κ_pq into the anti-Hermitian κ, form U = e^κ, rotate the orbitals C ← C·U, re-solve the active CI for fresh γ, Γ, recompute F and the super-CI, and repeat until ‖G‖ → 0. When the gradient vanishes the generalized Brillouin condition holds and I am at a stationary point of the joint energy.

Let me sanity-check the limit. If the active space is empty (no active orbitals), every orbital is inactive or virtual, F reduces to the ordinary closed-shell Fock matrix, G_ai = 2(F_ai − F_ia) is the occupied–virtual Fock block, the update κ_ai = G_ai/(f_ii − f_aa) is the standard SCF orbital correction, and the generalized Brillouin condition is the ordinary Brillouin theorem. So this method contains Hartree–Fock as the no-active-space special case. That is the kind of consistency I want to see — the new thing is a strict generalization, not a parallel construction.

Now, the super-CI / generalized-Brillouin step is first-order: cheap per iteration but it converges linearly, and far from the solution it can be slow or stall. Is there a second-order alternative I should keep in my back pocket? Yes — Newton–Raphson on the joint parameter set. Treat the variables as both the orbital rotations κ and the CI coefficient changes, expand the energy to second order at the current point,

  E(ξ) ≈ E⁽⁰⁾ + ξᵀ g + ½ ξᵀ H ξ,  ξ = (CI-coefficient change, κ),

and solve H ξ = −g for the Newton step. In block form,

  [ H_cc  H_co ] [ δc ]   = − [ g_c ]
  [ H_oc  H_oo ] [ κ  ]       [ g_o ],

where g_c = 2⟨I|Ĥ − E|Ψ⟩ is the CI gradient (the CI residual), g_o = G_pq is the orbital gradient I derived, and the Hessian blocks are commutators: H_cc = 2⟨I|Ĥ−E|J⟩, H_oc,pq,I = 2⟨Ψ|[Ê⁻_pq, Ĥ]|I⟩ couples orbital and CI variations, and the orbital–orbital block

  H_oo, pq,rs = ½⟨Ψ|[Ê⁻_pq,[Ê⁻_rs, Ĥ]]|Ψ⟩ + ½⟨Ψ|[Ê⁻_rs,[Ê⁻_pq, Ĥ]]|Ψ⟩,

which again contracts the integrals against γ, Γ. This has the equivalent eigenvalue (augmented-Hessian) form

  [ 0   g ᵀ ] [ 1 ]   = ω [ 1 ]
  [ g   H   ] [ ξ ]         [ ξ ],

whose lowest root gives a step damped toward a trust region — this is just the super-CI secular problem promoted to second order, the reference plus the singly-excited (and CI) directions, diagonalized. Near the solution this converges quadratically. The cost: I need the Hessian-vector products, but I never have to form H explicitly — I can apply it directly (one-index transformations of the integrals), so even the second-order scheme stays feasible for long CI expansions. In practice I'd alternate: a macro-iteration that re-solves the active full CI to refresh γ, Γ (and the CI vector), wrapped around micro-iterations that take orbital steps (super-CI or Newton) at fixed CI. The first-order super-CI is the cheap default; the second-order Newton step is the robust closer near convergence.

Let me run the whole thing on the one case I started from, to make sure it does what I demanded — H₂, minimal basis, two active orbitals σ, σ* and two active electrons: a complete active space of two electrons in two orbitals. No inactive orbitals, no virtuals beyond σ*, so there are no core→active or active→virtual or core→virtual rotations to do for the active electrons themselves once σ, σ* are the active set; the active CI is the whole story. The spin-adapted singlet configurations are |σ²⟩ and |σ*²⟩ (the open-shell σσ* singlet doesn't mix with the closed singlets by symmetry along the dissociation coordinate). Solve the 2×2 active CI:

  Ψ = c₁ |σ²⟩ + c₂ |σ*²⟩, with the 2×2 Hamiltonian [[E(σ²), K],[K, E(σ*²)]],

where K is the σ²–σ*² coupling, the exchange-type integral (σσ*|σσ*). Near equilibrium E(σ²) ≪ E(σ*²), so c₁ ≈ 1, c₂ small and negative — a small admixture of σ*² that bends the curve below RHF; that admixture is the static correlation already showing up. As I stretch, E(σ²) and E(σ*²) approach each other, the 2×2 problem becomes degenerate, and the lowest eigenvector goes to c₁ = −c₂ = 1/√2. Then

  Ψ_diss = (|σ²⟩ − |σ*²⟩)/√2 = 1s_A1s_B + 1s_B1s_A spin-singlet — two neutral H atoms, no ionic term.

And meanwhile the orbital optimization is relaxing σ, σ* to the best shapes for this mixture at each geometry. The energy dissociates correctly, the wavefunction stays a singlet throughout, and at equilibrium it smoothly reduces to essentially RHF (c₂ → 0). That is exactly the qualitative correctness I could not get from one determinant, and it came out of "complete CI in {σ,σ*} with orbitals optimized." The method does what it was designed to do, and on the smallest nontrivial case I can see every piece working.

Let me also be honest about what the two-by-two illustration hides, because the general machinery is precisely what makes it scale: in H₂ there were no inactive orbitals to fold into a core Fock and no long CI to worry about, so the density-matrix trick had nothing to do. The trick earns its keep when the active space is, say, several electrons in several orbitals on top of a large inactive core and many virtuals — then the inactive electrons fold into h^eff and E_core, the active full CI produces γ_tu, Γ_tuvw, and the entire orbital optimization (gradient G = 2(F−Fᵀ), super-CI denominators, or the Newton Hessian-vector products) runs off those small active-space density matrices and the integrals, never touching the length of the CI. That is the property that makes "complete active space, orbitals optimized" practical rather than just correct.

So the recipe is: choose the active orbitals; partition into inactive/active/virtual; fold the inactive electrons into a core Fock h^eff and energy E_core; solve the active full CI (spin-adapted, direct/unitary-group, no stored Hamiltonian) to get the CI vector and the active density matrices γ, Γ; build the generalized Fock F and the orbital gradient G = 2(F − Fᵀ) over the three inter-space rotation blocks; take an orbital step — first-order density-matrix super-CI κ = G/Δf in the simplest form, or a second-order augmented-Hessian Newton step near convergence; rotate the orbitals by U = e^κ; iterate until the gradient (the generalized Brillouin residual) vanishes. Let me write it as code, with the active full CI and density matrices coming from the existing solver and the orbital optimizer the new piece.

```python
import numpy as np
from scipy.linalg import expm

# Orbital partition is the chemistry input: which orbitals are active.
# inactive = doubly occupied in every configuration; active = the CI ranges over them;
# virtual = empty in every configuration.

def inactive_fock_and_core_energy(h_mo, eri_mo, ncore):
    """Fold the doubly-occupied inactive electrons into an effective one-electron
    operator and a constant core energy (gamma_ij = 2 delta_ij there):
    f^I_pq = h_pq + sum_i [ 2(pq|ii) - (pi|iq) ]."""
    j = np.einsum('pqii->pq', eri_mo[:, :, :ncore, :ncore])      # Coulomb from core
    k = np.einsum('piiq->pq', eri_mo[:, :ncore, :ncore, :])      # exchange from core
    fock_inactive = h_mo + 2.0*j - k                             # closed-shell core Fock
    e_core = np.einsum('ii->', h_mo[:ncore, :ncore] + fock_inactive[:ncore, :ncore])
    return fock_inactive, e_core

def active_integrals(fock_inactive, eri_mo, ncore, ncas):
    """Effective one-electron integrals seen by the active electrons + active ERIs."""
    o = slice(ncore, ncore + ncas)
    return fock_inactive[o, o], eri_mo[o, o, o, o]               # h^eff_tu, (tu|vw)

def full_density_matrices(dm1_act, dm2_act, ncore, ncas, nmo):
    """Embed the active-space RDMs into the full MO space; the inactive part is fixed
    (gamma_ij = 2 delta_ij), inactive-inactive/inactive-active 2-RDM = products of gamma."""
    nocc = ncore + ncas
    g1 = np.zeros((nmo, nmo))
    g1[:ncore, :ncore] = 2.0*np.eye(ncore)              # inactive doubly occupied
    g1[ncore:nocc, ncore:nocc] = dm1_act               # active 1-RDM from the CI
    g2 = np.zeros((nmo, nmo, nmo, nmo))
    g2[ncore:nocc, ncore:nocc, ncore:nocc, ncore:nocc] = dm2_act   # active-active 2-RDM
    for i in range(ncore):
        for j in range(ncore):
            g2[i, i, j, j] += 4.0
            g2[i, j, j, i] -= 2.0
        g2[i, i, ncore:nocc, ncore:nocc] += 2.0*dm1_act
        g2[ncore:nocc, ncore:nocc, i, i] += 2.0*dm1_act
        g2[i, ncore:nocc, ncore:nocc, i] -= dm1_act
        g2[ncore:nocc, i, i, ncore:nocc] -= dm1_act
    return g1, g2

def generalized_fock(h_mo, eri_mo, g1, g2):
    """F_pq = sum_r h_pr gamma_rq + sum_rst (pr|st) Gamma_qrst."""
    return h_mo @ g1 + np.einsum('prst,qrst->pq', eri_mo, g2)

def orbital_gradient(F):
    """Generalized Brillouin gradient G_pq = 2 (F_pq - F_qp)."""
    return 2.0*(F - F.T)

def rotation_blocks(nmo, ncore, ncas):
    """Non-redundant inter-space rotations only: core->active, core->virtual, active->virtual.
    Within-space rotations (core-core, active-active, virtual-virtual) are redundant for a
    COMPLETE active space and are excluded."""
    nocc = ncore + ncas
    pairs  = [(t, i) for t in range(ncore, nocc) for i in range(ncore)]        # active<-core
    pairs += [(a, i) for a in range(nocc, nmo)   for i in range(ncore)]        # virtual<-core
    pairs += [(a, t) for a in range(nocc, nmo)   for t in range(ncore, nocc)]  # virtual<-active
    return pairs

def super_ci_step(G, fock_diag, pairs):
    """First-order density-matrix super-CI step: kappa_pq = G_pq / (f_qq - f_pp)
    (generalized-Fock denominator), built only from active-space RDMs via F and G."""
    kappa = np.zeros_like(G)
    for (p, q) in pairs:
        kappa[p, q] = G[p, q] / (fock_diag[q] - fock_diag[p])
        kappa[q, p] = -kappa[p, q]              # anti-Hermitian
    return kappa

def casscf(h_mo, eri_mo, mo, ncore, ncas, nelec_act, fci_solve, ao2mo_update,
           v_nuc=0.0, max_macro=50, tol=1e-7):
    nmo = mo.shape[1]
    pairs = rotation_blocks(nmo, ncore, ncas)
    e_tot = None
    for _ in range(max_macro):
        # --- fold inactive electrons; build the active-space problem ---
        fock_in, e_core = inactive_fock_and_core_energy(h_mo, eri_mo, ncore)
        h_eff, eri_act = active_integrals(fock_in, eri_mo, ncore, ncas)
        # --- macro step: full CI in the active space (spin-adapted, direct/UGA) ---
        ci, dm1, dm2 = fci_solve(h_eff, eri_act, ncas, nelec_act)   # gamma_tu, Gamma_tuvw
        e_tot = (e_core + v_nuc
                 + np.einsum('tu,tu->', h_eff, dm1)
                 + 0.5*np.einsum('tuvw,tuvw->', eri_act, dm2))
        # --- orbital gradient = generalized Brillouin residual, from the RDMs ---
        g1, g2 = full_density_matrices(dm1, dm2, ncore, ncas, nmo)
        F = generalized_fock(h_mo, eri_mo, g1, g2)
        G = orbital_gradient(F)
        if np.linalg.norm([G[p, q] for (p, q) in pairs]) < tol:
            return e_tot, mo, ci                 # generalized Brillouin satisfied
        # --- micro step: orbital rotation from the density-matrix super-CI ---
        kappa = super_ci_step(G, np.diag(F), pairs)   # cheap first-order step
        mo = mo @ expm(kappa)                          # U = e^kappa keeps orthonormality
        h_mo, eri_mo = ao2mo_update(mo)                # re-transform integrals
    return e_tot, mo, ci
```

The causal chain, start to finish: one determinant cannot represent two near-degenerate configurations, so a stretched bond dissociates wrongly and a spin-pure description forces a multiconfiguration wavefunction. Choosing the CI to be *complete* within a small active orbital window makes the method black-box and, through completeness, makes the energy invariant to within-space rotations — so only three inter-space rotation blocks are non-redundant and the active-active block vanishes. The energy is linear in the one- and two-particle density matrices, which depend on the CI vector only through small active-space objects; the orbital gradient is the antisymmetric part of a generalized Fock matrix built from those same density matrices, and its vanishing is the generalized Brillouin condition. Folding the inactive electrons into a core Fock leaves a self-contained active full CI (solved spin-adapted, without storing the Hamiltonian, yielding the density matrices directly), and the orbital optimization — a density-matrix-formulated super-CI in first order, or an augmented-Hessian Newton step in second order — rides entirely on those active-space density matrices, so its cost is set by the active size and not by the length of the CI. Iterating the active CI and the orbital step to a vanishing gradient yields, self-consistently, both the configuration mixing and the orbitals: complete active space, optimized.
