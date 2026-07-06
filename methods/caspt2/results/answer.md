# CASPT2 — second-order perturbation theory on a CASSCF reference

## Problem

A CASSCF wave function captures *static* correlation (near-degeneracy: bond breaking, open shells,
excited states) but, with a tractable active space, misses most of the *dynamic* correlation — the
short-range electron–electron correlation living in the inactive/virtual space. MP2 adds dynamic
correlation cheaply to a single Hartree–Fock determinant but is single-reference and fails wherever
no determinant dominates. CASPT2 adds MP2-style dynamic correlation *on top of* a multiconfigurational
CASSCF reference, non-iteratively and size-extensively.

## Key idea

Run Rayleigh–Schrödinger second-order perturbation theory with the CASSCF state as the zeroth-order
function, under four constraints: standard RSPT with intermediate normalization; reference = the
CASSCF state; the zeroth-order Hamiltonian Ĥ₀ is an effective *one-electron* operator; and Ĥ₀ reduces
to the MP2 Fock partition when the active space collapses to a single determinant.

The obstruction is that a one-electron operator does not have the multiconfigurational CASSCF state as
an eigenfunction. CASPT2 resolves this by (1) building a generalized, spin-averaged Fock operator from
the reference's own density, and (2) *projecting* it block-by-block so the reference becomes an exact
eigenfunction while the operator stays one-electron.

## Method

**Generalized Fock operator.** Built from the IP/EA commutator construction
f̂_pq = ½ Σ_σ (â_pσ[Ĥ, â†_qσ] − â†_pσ[Ĥ, â_qσ]), whose expectation over the reference is

  f_pq = h_pq + Σ_rs D_rs [ (pq|rs) − ½ (pr|qs) ],

with D the reference one-particle density matrix. For a closed-shell determinant (D_rs = 2δ_rs over
occupied) this reduces to the canonical Fock matrix h_pq + Σ_k [2(pq|kk) − (pk|qk)], so the MP2 limit
holds. The inactive–virtual block of f vanishes (generalized Brillouin), but the inactive–active and
active–virtual blocks are nonzero and are *kept* (CASPT2N) — dropping them (CASPT2D) fails when active
occupations approach 0 or 2. Define F̂ = Σ_pq f_pq Ê_pq.

**Projected zeroth-order Hamiltonian.** Partition the N-electron space into V₀ (reference), V_K (rest
of the CAS CI space), V_SD (single+double replacements into the external space), V_TQ.. (higher), with
projectors P̂. Then

  Ĥ₀ = P̂₀ F̂ P̂₀ + P̂_K F̂ P̂_K + P̂_SD F̂ P̂_SD + P̂_TQ F̂ P̂_TQ,

block-diagonal across the four subspaces, so |Ψ₀⟩ is an exact eigenfunction with
E⁽⁰⁾ = ⟨Ψ₀|F̂|Ψ₀⟩ = Σ_pq f_pq D_pq. The perturbation is V = Ĥ − Ĥ₀.

**First-order interacting space.** Only V_SD couples to |Ψ₀⟩ at first order: V_K vanishes by the
reference's variational stationarity (generalized Brillouin), V_TQ.. vanishes because Ĥ is two-body
(Slater–Condon). V_SD is built from single and double replacements of *any* electron — active or
inactive — via internally contracted functions Ê_pq Ê_rs |Ψ₀⟩, classified A–H by orbital type
(indices: core i,j; active t,u,v; virtual a,b):

| Family | Class | Function |
|---|---|---|
| Internal | A | Ê_ti Ê_uv |Ψ₀⟩ |
|          | B | Ê_ti Ê_uj |Ψ₀⟩ |
| Semi-internal | C | Ê_at Ê_uv |Ψ₀⟩ |
|               | D | Ê_ai Ê_tu |Ψ₀⟩, Ê_ti Ê_au |Ψ₀⟩ |
|               | E | Ê_ti Ê_aj |Ψ₀⟩ |
| External | F | Ê_at Ê_bu |Ψ₀⟩ |
|          | G | Ê_ai Ê_bt |Ψ₀⟩ |
|          | H | Ê_ai Ê_bj |Ψ₀⟩ |

Class H (all core/virtual indices) is the MP2 double; in the single-determinant limit only H survives.
Classes B, E, F, G, H further split into singlet/triplet spin couplings (e.g. for G,
Ê_ai Ê_bt |Ψ₀⟩ ± Ê_bi Ê_at |Ψ₀⟩). Assembling the matrix elements needs reference reduced density
matrices up to 3- and 4-particle for the internal/semi-internal classes; only 1- and 2-particle for
F, G, H.

**First-order equations and second-order energy.** Write |Ψ⁽¹⁾⟩ = Σ_{j∈SD} C_j |Ψ_j⟩. Standard RSPT
gives

  Σ_j C_j ⟨Ψ_i|(Ĥ₀ − E⁽⁰⁾)|Ψ_j⟩ = −⟨Ψ_i|Ĥ|Ψ₀⟩,    i.e.   (H₀ − E⁽⁰⁾ S) C = −V,

with V_i = ⟨Ψ_i|Ĥ|Ψ₀⟩ and S the overlap of the (non-orthogonal, internally contracted) functions.
Because S is non-orthogonal and (near-)linearly dependent, diagonalize Λ_S = U†S U, drop
near-null eigenvectors, form Ω = U Λ_S^{−1/2}, transform H₀′ = Ω†H₀Ω, V′ = Ω†V, and solve
(H₀′ − E⁽⁰⁾ I) C′ = −V′. The second-order correction is E⁽²⁾ = V′†C′.

**Intruder states and shifts.** When an external configuration has F_α ≈ E⁽⁰⁾, the denominator
F_α − E⁽⁰⁾ → 0 and the amplitude diverges. The two-state model Ĥ(z) = diag(α,β) + z·offdiag(δ) has
E_± = (α+β)/2 ± ½√((β−α)²+4z²δ²); the series converges iff |β−α| > 2|δ|, with branch point
ζ = ±i(β−α)/(2δ). A level shift restores convergence. The imaginary shift iε removes the singularity
entirely, taking the real part of the amplitude,

  Re(C_i) = −⟨Ψ₀|V|Ψ_i⟩ · Δ_i / (Δ_i² + ε²),   Δ_i = ε_i − E⁽⁰⁾,

and the energy is reported through the stationary Hylleraas functional

  E⁽²⁾ = ⟨Ψ⁽¹⁾|(Ĥ₀ − E⁽⁰⁾)|Ψ⁽¹⁾⟩ + 2 ⟨Ψ₀|V|Ψ⁽¹⁾⟩,

which is insensitive to the shift to first order. Total energy: E = E_CASSCF + E⁽²⁾.

## Worked check — the MP2 limit and the open-shell Fock

Collapse the active space to a single closed-shell determinant: D_rs = 2δ_rs over occupied, so
f_pq = h_pq + Σ_k[2(pq|kk) − (pk|qk)] is the closed-shell Fock; classes A–G vanish (no active
orbitals), only H = Ê_ai Ê_bj |Ψ₀⟩ survives, S = I, and E⁽²⁾ = Σ_{i<j,a<b} |⟨ij‖ab⟩|²/(ε_i+ε_j−ε_a−ε_b)
— exactly MP2. For a doublet with one inactive orbital i (occ 2), one active t (occ 1), one virtual a,
the density is diag(2,1,0) and f_pq = h_pq + 2(pq|ii) − (pi|qi) + (pq|tt) − ½(pt|qt): the half-occupied
active orbital contributes only *half* the exchange, automatically handling the open shell that a
closed-shell Fock operator would mishandle.

## Reference implementation (single-state CASPT2)

```python
import numpy as np
from scipy.sparse.linalg import cg

class CASReference:
    """Outputs of a converged CASSCF."""
    def __init__(self, ncore, nact, nvirt, h, eri, dm1, dm2, E0):
        self.ncore, self.nact, self.nvirt = ncore, nact, nvirt
        self.h, self.eri, self.E0 = h, eri, E0       # h[p,q], eri[p,q,r,s]=(pq|rs)
        self.dm1, self.dm2 = dm1, dm2                 # active 1- and 2-RDMs
        self.nmo = ncore + nact + nvirt
        self.core = range(0, ncore)
        self.actv = range(ncore, ncore + nact)
        self.virt = range(ncore + nact, self.nmo)

    def full_dm1(self):
        D = np.zeros((self.nmo, self.nmo))
        for i in self.core:
            D[i, i] = 2.0
        ca = self.ncore
        D[ca:ca+self.nact, ca:ca+self.nact] = self.dm1
        return D

def generalized_fock(ref):
    """f_pq = h_pq + sum_rs D_rs [ (pq|rs) - 1/2 (pr|qs) ].
    Reduces to the closed-shell Fock for D_rs = 2*delta (the MP2 limit)."""
    D = ref.full_dm1()
    coulomb  = np.einsum('rs,pqrs->pq', D, ref.eri)   # sum_rs D_rs (pq|rs)
    exchange = np.einsum('rs,prqs->pq', D, ref.eri)   # sum_rs D_rs (pr|qs)
    return ref.h + coulomb - 0.5 * exchange

def first_order_space(ref):
    """Single+double replacements E_pq E_rs |Psi0> that interact at first order,
    classified A..H by number of virtual indices. All-active excluded (V_K, killed by
    the reference's variational stationarity); triples+ excluded (two-body H). Includes
    excitations of ANY electron, active or inactive."""
    return {
        'A': [('ti', 'uv')], 'B': [('ti', 'uj')],                 # internal
        'C': [('at', 'uv')], 'D': [('ai', 'tu'), ('ti', 'au')],   # semi-internal
        'E': [('ti', 'aj')],
        'F': [('at', 'bu')], 'G': [('ai', 'bt')], 'H': [('ai', 'bj')],  # external
    }

def build_class_system(ref, fock, E0):
    """Per class: overlap S, projected zeroth-order block H0, coupling V_i=<Psi_i|H|Psi0>.
    S, H0, V contract the reference RDMs (up to 3-/4-particle for A..E; 1-/2-particle for
    F,G,H) with fock and eri. Returns (S, H0, V)."""
    raise NotImplementedError  # class-by-class RDM contractions

def orthonormalize(S, H0, V, cutoff=1e-10):
    """Symmetric orthonormalization with a small-eigenvalue cutoff: the internally
    contracted functions are non-orthogonal and (near-)linearly dependent."""
    w, U = np.linalg.eigh(S)
    keep = w > cutoff
    Omega = U[:, keep] / np.sqrt(w[keep])            # Omega = U Lambda^{-1/2}
    return Omega.T @ H0 @ Omega, Omega.T @ V, Omega

def solve_class(H0p, Vp, E0, imag_shift=0.0):
    """Solve (H0' - E0 I) C' = -V'; optional imaginary shift removes intruder singularities.
    Report E2 three ways; the Hylleraas (variational) form is the robust one."""
    n = H0p.shape[0]
    diag = np.diag(H0p) - E0
    if imag_shift > 0.0:
        Cp = -Vp * diag / (diag**2 + imag_shift**2)  # Re C = -V Delta/(Delta^2+eps^2)
    else:
        Cp, _ = cg(H0p - E0 * np.eye(n), -Vp, atol=1e-12)
    E2_diagonal       = -np.sum((Vp / diag) * Vp)
    E2_nonvariational = -np.dot(Cp, Vp)              # V . C
    E2_variational    = 2.0 * E2_nonvariational + Cp @ ((H0p - E0 * np.eye(n)) @ Cp)
    return E2_variational, E2_nonvariational, E2_diagonal, Cp

def caspt2_energy(ref, imag_shift=0.0):
    """Total second-order dynamic-correlation correction to the CASSCF energy."""
    fock = generalized_fock(ref)
    E2 = 0.0
    for _ in first_order_space(ref):
        S, H0, V = build_class_system(ref, fock, ref.E0)
        H0p, Vp, _ = orthonormalize(S, H0, V)
        e2, _, _, _ = solve_class(H0p, Vp, ref.E0, imag_shift)
        E2 += e2
    return ref.E0 + E2, E2
```
