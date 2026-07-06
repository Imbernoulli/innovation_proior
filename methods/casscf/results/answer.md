# CASSCF — the complete active space self-consistent field method

## Problem

A single Slater determinant (Hartree–Fock) cannot represent situations where two or more electron
configurations are nearly degenerate: a covalent bond being broken (σ² and σ*² become degenerate),
diradicals, low-spin open-shell states, excited states with strong configurational mixing. RHF
dissociates with a spurious ionic tail; UHF fixes the curve shape but is spin-contaminated. We want
a variational, spin-pure wavefunction that captures this *static* (non-dynamical) correlation,
optimizes both the configuration mixing and the orbitals self-consistently, requires no hand
selection of configurations, and whose orbital-optimization cost does not grow with the length of
the configuration expansion.

## Key idea

Partition the molecular orbitals into **inactive** (doubly occupied in every configuration),
**active** (a small chosen window of near-degenerate frontier orbitals), and **virtual** (empty in
every configuration). Take the wavefunction to be a **full CI within the active space** — distribute
the active electrons over the active orbitals in all possible (spin-adapted) ways — and optimize the
**orbitals** simultaneously with the CI coefficients. Completeness within the active space makes the
energy invariant under rotations *within* each subspace, so only inter-space orbital rotations are
non-redundant and the active–active rotation block drops out. Because the energy and the orbital
gradient depend on the CI vector only through the **active-space one- and two-particle reduced
density matrices**, the orbital optimization is reformulated entirely in terms of those small density
matrices and is therefore **independent of the number of configurations** in the CI.

## The method

**Wavefunction.** With orbital classes inactive {i,j,k,l}, active {t,u,v,x}, virtual {a,b,c,d},
general {p,q,r,s}, and spin-summed generators Ê_pq = a†_{pα}a_{qα} + a†_{pβ}a_{qβ},

  |Ψ⟩ = Σ_I c_I |Φ_I⟩,   {Φ_I} = all CSFs of the active electrons in the active orbitals (CASCI).

**Energy.** Ĥ = Σ_pq h_pq Ê_pq + ½ Σ_pqrs (pq|rs) ê_pqrs, ê_pqrs = Ê_pq Ê_rs − δ_qr Ê_ps, gives

  E = Σ_pq h_pq γ_pq + ½ Σ_pqrs (pq|rs) Γ_pqrs,

with the one- and two-particle reduced density matrices γ_pq = ⟨Ψ|Ê_pq|Ψ⟩,
Γ_pqrs = ⟨Ψ|ê_pqrs|Ψ⟩. The inactive density is fixed (γ_ij = 2δ_ij), so the inactive electrons fold
into an inactive Fock matrix and a core energy,

  f^I_pq = h_pq + Σ_i [ 2(pq|ii) − (pi|iq) ],   E_core = Σ_i ( h_ii + f^I_ii ) + V_nuc,

leaving a self-contained active full CI with effective integrals (h^eff_tu = f^I_tu, (tu|vw)):

  E = E_core + Σ_tu f^I_tu γ_tu + ½ Σ_tuvw (tu|vw) Γ_tuvw.

**Orbital parametrization.** Rotate orbitals by U = e^κ, κ̂ = Σ_{p>q} κ_pq (Ê_pq − Ê_qp), κ† = −κ
(unitary, orthonormality automatic, unconstrained variables). Only inter-space blocks survive:

  κ̂ = Σ_{ti} κ_ti Ê⁻_ti + Σ_{ai} κ_ai Ê⁻_ai + Σ_{at} κ_at Ê⁻_at

(core→active, core→virtual, active→virtual; within-space and active–active rotations are redundant
for a complete active space).

**Generalized Fock matrix and gradient.** The orbital gradient is

  G_pq = ∂E/∂κ_pq |₀ = ⟨Ψ| [Ê⁻_pq, Ĥ] |Ψ⟩ = 2 ( F_pq − F_qp ),
  F_pq = Σ_r h_pr γ_rq + Σ_rst (pr|st) Γ_qrst.

Stationarity G_pq = 0 ⇔ F symmetric is the **generalized Brillouin condition**; for an empty active
space it reduces to the ordinary Fock matrix and the Brillouin theorem, so Hartree–Fock is the
no-active-space special case. F is built from γ, Γ whose nontrivial indices are active, so G is
independent of the CI length.

**Orbital step — density-matrix super-CI (first order).** Solve the small secular problem of Ĥ in
{|Ψ⟩} ∪ {Ê⁻_pq|Ψ⟩}, with Ĥ between singles replaced by a generalized Fock operator (shifted so |Ψ⟩
is its eigenfunction), so every matrix element contracts only the active-space density matrices. In
the leading diagonal approximation the rotation is a generalized-Fock-gradient/denominator step,

  κ_ai = G_ai / (f_ii − f_aa)  (and analogous active-coupled forms for κ_ti, κ_at),

so the orbital step is independent of the CI length. Update C ← C·e^κ, re-solve the active CI,
iterate until ‖G‖ → 0.

**Orbital step — Newton–Raphson (second order, alternative).** Expand E to second order in the
joint (CI, orbital) variables ξ and solve

  [ H_cc  H_co ] [ δc ]   = − [ g_c ]          (equivalently the augmented-Hessian eigenproblem
  [ H_oc  H_oo ] [ κ  ]       [ g_o ],          [[0, gᵀ],[g, H]] [1; ξ] = ω [1; ξ]),

g_c = 2⟨I|Ĥ−E|Ψ⟩, g_o = G, H_oo,pq,rs = ½⟨Ψ|[Ê⁻_pq,[Ê⁻_rs,Ĥ]]|Ψ⟩ + (pq↔rs); Hessian-vector
products are formed by one-index transformations without ever building H. Quadratic convergence
near the solution.

**Driver.** Macro-iteration: solve the active full CI (spin-adapted, direct/unitary-group, no stored
Hamiltonian) to refresh γ, Γ. Micro-iterations: take orbital steps at fixed CI. Iterate to a
vanishing gradient.

## Worked illustration: H₂ in a CAS(2,2)

Two active electrons in {σ, σ*}, no inactive or extra virtual orbitals. The spin-adapted singlet
active CI is the 2×2 problem in {|σ²⟩, |σ*²⟩}:

  Ψ = c₁ |σ²⟩ + c₂ |σ*²⟩,   H = [[E(σ²), K],[K, E(σ*²)]], K = (σσ*|σσ*).

Near equilibrium E(σ²) ≪ E(σ*²) ⇒ c₁ ≈ 1, c₂ small (static-correlation admixture; smoothly reduces
to RHF as c₂ → 0). At dissociation E(σ²) → E(σ*²) ⇒ c₁ = −c₂ = 1/√2, and with
σ = (1s_A+1s_B)/√2, σ* = (1s_A−1s_B)/√2,

  Ψ_diss = (|σ²⟩ − |σ*²⟩)/√2 = (1s_A 1s_B + 1s_B 1s_A) singlet — two neutral H, no ionic term.

The orbitals σ, σ* are re-optimized at each geometry. The curve dissociates correctly and stays a
spin singlet throughout — the static correlation that one determinant cannot represent.

## Reference implementation

```python
import numpy as np
from scipy.linalg import expm

# Orbital partition is the chemistry input: which orbitals are active.
#   inactive (ncore): doubly occupied in every configuration
#   active   (ncas) : the full CI ranges over them
#   virtual         : empty in every configuration

def inactive_fock_and_core_energy(h_mo, eri_mo, ncore):
    """Fold the doubly-occupied inactive electrons into an effective one-electron
    operator and a constant core energy (gamma_ij = 2 delta_ij there).
    f^I_pq = h_pq + sum_i [ 2(pq|ii) - (pi|iq) ]."""
    j = np.einsum('pqii->pq', eri_mo[:, :, :ncore, :ncore])     # core Coulomb
    k = np.einsum('piiq->pq', eri_mo[:, :ncore, :ncore, :])     # core exchange
    fock_inactive = h_mo + 2.0 * j - k
    e_core = np.einsum('ii->', h_mo[:ncore, :ncore]
                              + fock_inactive[:ncore, :ncore])  # + V_nuc outside
    return fock_inactive, e_core

def active_integrals(fock_inactive, eri_mo, ncore, ncas):
    """Effective one-electron integrals seen by the active electrons + active ERIs."""
    o = slice(ncore, ncore + ncas)
    return fock_inactive[o, o], eri_mo[o, o, o, o]

def full_density_matrices(dm1_act, dm2_act, ncore, ncas, nmo):
    """Embed the active-space RDMs into the full MO space; inactive part is fixed."""
    nocc = ncore + ncas
    g1 = np.zeros((nmo, nmo))
    g1[:ncore, :ncore] = 2.0 * np.eye(ncore)            # inactive doubly occupied
    g1[ncore:nocc, ncore:nocc] = dm1_act               # active 1-RDM from the CI
    g2 = np.zeros((nmo, nmo, nmo, nmo))
    # active-active 2-RDM
    g2[ncore:nocc, ncore:nocc, ncore:nocc, ncore:nocc] = dm2_act
    # inactive-inactive and inactive-active pieces are products of the 1-RDM
    ci = range(ncore)
    for i in ci:
        for j in ci:
            g2[i, i, j, j] += 4.0
            g2[i, j, j, i] -= 2.0
    for i in ci:
        g2[i, i, ncore:nocc, ncore:nocc] += 2.0 * dm1_act
        g2[ncore:nocc, ncore:nocc, i, i] += 2.0 * dm1_act
        g2[i, ncore:nocc, ncore:nocc, i] -= dm1_act
        g2[ncore:nocc, i, i, ncore:nocc] -= dm1_act
    return g1, g2

def generalized_fock(h_mo, eri_mo, g1, g2):
    """F_pq = sum_r h_pr gamma_rq + sum_rst (pr|st) Gamma_qrst."""
    F = h_mo @ g1
    F += np.einsum('prst,qrst->pq', eri_mo, g2)
    return F

def orbital_gradient(F):
    """Generalized Brillouin gradient G_pq = 2 (F_pq - F_qp)."""
    return 2.0 * (F - F.T)

def rotation_pairs(nmo, ncore, ncas):
    """Non-redundant inter-space rotations only: core->active, core->virtual,
    active->virtual.  Within-space (incl. active-active) rotations are redundant for a
    COMPLETE active space and are excluded."""
    nocc = ncore + ncas
    pairs  = [(t, i) for t in range(ncore, nocc) for i in range(ncore)]
    pairs += [(a, i) for a in range(nocc, nmo)   for i in range(ncore)]
    pairs += [(a, t) for a in range(nocc, nmo)   for t in range(ncore, nocc)]
    return pairs

def super_ci_step(G, fock_diag, pairs):
    """First-order density-matrix super-CI step: kappa_pq = G_pq / (f_qq - f_pp),
    built only from the active-space RDMs via F and G; CI-length independent."""
    kappa = np.zeros_like(G)
    for (p, q) in pairs:
        kappa[p, q] = G[p, q] / (fock_diag[q] - fock_diag[p])
        kappa[q, p] = -kappa[p, q]                      # anti-Hermitian
    return kappa

def casscf(h_mo, eri_mo, mo, ncore, ncas, nelec_act, fci_solve, ao2mo_update,
           v_nuc=0.0, max_macro=50, tol=1e-7):
    nmo = mo.shape[1]
    pairs = rotation_pairs(nmo, ncore, ncas)
    e_tot = None
    for _ in range(max_macro):
        # fold inactive electrons; build the active-space problem
        fock_in, e_core = inactive_fock_and_core_energy(h_mo, eri_mo, ncore)
        h_eff, eri_act = active_integrals(fock_in, eri_mo, ncore, ncas)
        # macro step: spin-adapted full CI in the active space (direct/UGA, no stored H)
        ci, dm1, dm2 = fci_solve(h_eff, eri_act, ncas, nelec_act)   # gamma_tu, Gamma_tuvw
        e_tot = (e_core + v_nuc
                 + np.einsum('tu,tu->', h_eff, dm1)
                 + 0.5 * np.einsum('tuvw,tuvw->', eri_act, dm2))
        # orbital gradient = generalized Brillouin residual, from the RDMs
        g1, g2 = full_density_matrices(dm1, dm2, ncore, ncas, nmo)
        F = generalized_fock(h_mo, eri_mo, g1, g2)
        G = orbital_gradient(F)
        if np.linalg.norm([G[p, q] for (p, q) in pairs]) < tol:
            return e_tot, mo, ci                         # generalized Brillouin satisfied
        # micro step: density-matrix super-CI rotation, then U = e^kappa
        kappa = super_ci_step(G, np.diag(F), pairs)
        mo = mo @ expm(kappa)
        h_mo, eri_mo = ao2mo_update(mo)                  # re-transform integrals
    return e_tot, mo, ci
```
