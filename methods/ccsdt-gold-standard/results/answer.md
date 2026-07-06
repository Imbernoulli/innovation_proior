# CCSD(T): coupled-cluster singles and doubles with a non-iterative connected-triples correction

## Problem

Coupled-cluster singles and doubles (CCSD) is size-extensive and affordable (∝N⁶ iterative) but omits **connected triple excitations**, which first contribute to the correlation energy at fourth order in Møller–Plesset theory; this leaves a systematic error in quantitative properties. Solving for triple-excitation amplitudes iteratively (CCSDT, CCSDT-n) cures the error but costs ∝N⁷–N⁸ *per iteration* — unaffordable for routine use. The goal is the chemistry of connected triples at a cost only modestly above CCSD.

## Key idea

Treat triples as a **one-off perturbation on the converged CCSD solution**: build the triple-excitation amplitude once from the converged singles (T₁) and doubles (T₂) amplitudes, contract it back, and add the resulting energy. The iteration stays ∝N⁶; one extra ∝N⁷ pass is paid at the end, and triples are never stored. Crucially, the triple is allowed to couple back to **both singles and doubles** — matching the equal-footing treatment of S and D in CCSD itself. The doubles-only channel (the fourth-order double–triple term E_T⁴) is necessarily negative; the single–triple channel (the fifth-order term E_ST⁵) carries the opposite sign. Their **near-cancellation** is what makes the correction robust: a doubles-only correction keeps only the one-signed half and overshoots, producing — on the correlation-sensitive ozone asymmetric stretch — an imaginary frequency, whereas the full S+D correction restores a real frequency.

## Final method

Layer on a converged CCSD calculation the triples correction

  ΔE_T(CCSD) = (Σ_s^S + Σ_s^D) Σ_t^T Σ_u^D (E₀ − E_t)⁻¹ a_s V_{st} V_{tu} a_u ,

so that the total energy is

  E[CCSD(T)] = E_HF + E_corr(CCSD) + ΔE_T(CCSD).

The outer index s runs over single and double substitutions; the inner index u over doubles; t over triples; (E₀ − E_t)⁻¹ is the resolvent denominator and V the fluctuation potential; all amplitudes a are the converged CCSD values. This is the augmented quadratic-CI triples formula

  ΔE_T(QCISD) = (2 Σ_s^S + Σ_s^D) Σ_t^T Σ_u^D (E₀ − E_t)⁻¹ a_s V_{st} V_{tu} a_u

with the **singles coefficient changed from 2 to 1**: CCSD already contains half of the 2E_ST⁵ single–triple energy through its nonlinear amplitude terms (whereas QCISD contains none), so only the missing half is added.

In explicit orbital labels, with the triple excitation ijk → abc and denominator Δ_{ijk}^{abc} = ε_i + ε_j + ε_k − ε_a − ε_b − ε_c, the three sibling corrections unify as

  ΔE_T = (1/36) Σ_{ijk} Σ_{abc} (Δ_{ijk}^{abc})⁻¹ [ (2 − x) ã_{ijk}^{abc} + ū_{ijk}^{abc} ] · ā_{ijk}^{abc} ,

where ū is the doubles-derived part of the triple amplitude, ã the singles-derived part (from the partial-triples amplitude Δa_s = (E₀ − E_s)⁻¹ Σ_t^S V_{st} a_t), ā the full symmetrized triple amplitude, and x = 0, 1, 2 selects QCISD(T), CCSD(T), and BD(T). The Brueckner case x = 2 gives coefficient (2 − 2) = 0: with T₁ = 0 by construction the single–triple energy is already in the reference, so no singles contribution is added — a consistency check on the (2 − x) counting.

## Cost and behavior

CCSD(T) is "iterative N⁶ plus one non-iterative N⁷" (one O(n³N⁴) pass). It is fully correct through fourth order in S, D, T, Q, and correct in the fifth-order parts linear in the higher (triple, quadruple) substitutions. On the ozone asymmetric stretching frequency (polarized double-zeta basis), the doubles-only correction returns an imaginary frequency; CCSD(T) returns **977 cm⁻¹** (alongside QCISD(T)'s 933 cm⁻¹), against the experimental 1089 cm⁻¹.

## Reference implementation (schematic, spin-orbital antisymmetrized form)

```python
import numpy as np
from itertools import product

def ccsd_pt_correction(t1, t2, eri, eps, occ, vir):
    """
    Non-iterative connected-triples correction on a converged CCSD solution.
    t1[i,a]      : converged singles amplitudes
    t2[i,j,a,b]  : converged doubles amplitudes
    eri[p,q,r,s] : antisymmetrized MO integrals <pq||rs>
    eps[p]       : orbital energies
    occ, vir     : occupied / virtual spin-orbital index lists
    Returns dE_T(CCSD). One O(n^3 N^4) pass; no triples are stored.
    """
    e_t = 0.0
    for i, j, k in product(occ, occ, occ):
        for a, b, c in product(vir, vir, vir):
            denom = eps[i] + eps[j] + eps[k] - eps[a] - eps[b] - eps[c]

            # doubles-derived triple amplitude: 4th-order, necessarily negative
            u = _connected_triple_from_doubles(i, j, k, a, b, c, t2, eri, vir, occ)
            # singles-derived triple amplitude: 5th-order, opposite sign
            a_tilde = _partial_triple_from_singles(i, j, k, a, b, c, t1, eri, vir, occ)

            x = 1                                   # CCSD(T): add the missing half
            a_bar = (u + a_tilde) / denom           # full triple amplitude
            e_t += ((2 - x) * a_tilde + u) * a_bar / denom

    return e_t / 36.0

def total_energy_ccsd_pt(e_hf, e_ccsd, t1, t2, eri, eps, occ, vir):
    """E[CCSD(T)] = E_HF + E_corr(CCSD) + dE_T(CCSD)."""
    return e_hf + e_ccsd + ccsd_pt_correction(t1, t2, eri, eps, occ, vir)
```
