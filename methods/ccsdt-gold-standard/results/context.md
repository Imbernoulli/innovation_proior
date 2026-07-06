# Context: an affordable, size-extensive route to high-accuracy electron correlation

## Research question

A Hartree–Fock (HF) single determinant treats each electron in the mean field of the others and so misses the *instantaneous* electron–electron interaction — the correlation energy. Recovering it accurately matters: chemical accuracy (≈1 kcal/mol) for reaction energies, and faithful potential-energy surfaces for properties like vibrational frequencies, hinge on it. The open problem is to build a correlated method that is simultaneously (i) **size-extensive** — the energy of *n* non-interacting copies of a system must equal *n* times the energy of one (truncated configuration interaction fails this, and the failure worsens with system size); (ii) **systematically accurate** enough that quantitative properties come out right, which in practice means capturing the effect of connected *triple* excitations, not just singles and doubles; and (iii) **affordable** enough to apply to molecules of perhaps fifteen heavy atoms, i.e. with computational cost that does not explode. The methods available at the singles-and-doubles level meet (i) but fall short on (ii); the methods that include triples iteratively meet (ii) but violate (iii). Closing that gap — getting the triples effect at a cost only modestly above the singles-and-doubles level — is the target.

## Background

Electron correlation can be organized as a perturbation on the HF problem. In Møller–Plesset (MP) theory the Hamiltonian is split H = F + V, with F the Fock operator and V the fluctuation potential, and the correlated wavefunction and energy are expanded in powers of V. Writing the full correlated state as Ψ = Ψ₀ + Σ_s a_s Ψ_s over single, double, triple, quadruple, … substitutions (S, D, T, Q) of the reference, the amplitudes a_s carry an MP expansion a_s = a_s¹ + a_s² + …. By Brillouin's theorem and the two-body nature of V, only **double** substitutions appear at first order, a_s¹ = (E₀ − E_s)⁻¹ V_{s0}; at second order, singles, doubles, triples, and quadruples all switch on through a_s² = (E₀ − E_s)⁻¹ Σ_t V_{st} a_t¹. The energy through low orders then partitions cleanly by excitation type. The second and third orders contain only doubles, so every standard iterative method is automatically correct to third order. The fourth-order energy splits as E⁴ = E_S⁴ + E_D⁴ + E_T⁴ + E_Q⁴ — and the appearance of E_T⁴ is the key fact: **connected triple excitations first contribute at fourth order**. The fifth order subdivides further into single–single, single–double, double–double, single–triple, double–triple, triple–triple, double–quadruple, triple–quadruple, and quadruple–quadruple couplings (denoted E_SS⁵, E_SD⁵, E_DD⁵, E_ST⁵, E_DT⁵, E_TT⁵, E_DQ⁵, E_TQ⁵, E_QQ⁵), with the off-diagonal cross terms carrying a factor of two (e.g. 2E_SD⁵, 2E_ST⁵). This term-by-term partition is the yardstick for asking *which pieces of the exact correlation energy a given approximate method actually contains*.

Two structural ideas dominate the field. The **exponential (coupled-cluster) ansatz** Ψ = e^T Ψ₀, with T a sum of excitation operators, builds in size-extensivity automatically: the exponential generates disconnected products (a doubles-squared term, etc.) that exactly track how the energy must factor across non-interacting fragments, so any truncation of T still gives an extensive energy. The competing **linear (configuration-interaction) ansatz** Ψ = (1 + C)Ψ₀ does not have this property under truncation. Computational cost is counted in the number of occupied orbitals *n* and virtual orbitals *N*: a singles-and-doubles iterative method has leading cost O(n²N⁴) + O(n³N³) (commonly written ∝N⁶) per iteration; a method that solves for triple-excitation amplitudes iteratively jumps to O(n³N⁵) (∝N⁸) per iteration, with intermediate variants at ∝N⁷.

A concrete, correlation-sensitive diagnostic sets the empirical stage: the **asymmetric stretching frequency of ozone**, long known to be acutely sensitive to the level of correlation treatment. The experimental value is 1089 cm⁻¹ (Barbe, Secroun, Jouve 1974). With a polarized double-zeta basis, the singles-and-doubles methods land within about 15% but with **errors of opposite sign** depending on the method, and the available augmented (triples-corrected) schemes give wildly different answers for this frequency — one of them so far off that it predicts the molecule has no symmetric minimum at all (an imaginary frequency, signalling a spurious asymmetric structure). The frequency is thus a sharp test of *how* triples are folded in (Stanton, Lipscomb, Magers, Bartlett 1989).

## Baselines

**Configuration interaction with singles and doubles (CISD)** (Shavitt 1977). Linear variational expansion in the space of all single and double substitutions: ⟨Ψ₀|H̄|T₂Ψ₀⟩ = E_corr, ⟨Ψ_i^a|H̄|(T₁+T₂)Ψ₀⟩ = a_i^a E_corr, ⟨Ψ_{ij}^{ab}|H̄|(1+T₁+T₂)Ψ₀⟩ = a_{ij}^{ab} E_corr (with H̄ = H − E_HF). Its core limitation is that it is **not size-extensive**: the energy of separated subsystems is not additive, and the error grows with the number of electrons, so an approximate (Davidson) correction must be bolted on afterward. It also contains no triple-excitation information.

**Coupled-cluster singles and doubles (CCSD)** (Purvis, Bartlett 1982). Takes Ψ = e^T Ψ₀ with T = T₁ + T₂, giving projection equations ⟨Ψ₀|H̄|(T₂ + ½T₁²)Ψ₀⟩ = E_corr, ⟨Ψ_i^a|H̄|(T₁ + T₂ + ½T₁² + T₁T₂ + ⅙T₁³)Ψ₀⟩ = a_i^a E_corr, and an analogous (more involved) doubles equation. The exponential makes it **rigorously size-extensive** and exact for a two-electron system, at ∝N⁶ iterative cost; it is correct to fourth order in the combined singles-doubles-quadruples space. What it does not contain is the effect of **connected triple excitations**: it misses part of E_T⁴ and everything at higher order that involves triples.

**Quadratic configuration interaction with singles and doubles (QCISD)** (Pople, Head-Gordon, Raghavachari 1987). Modifies the CISD equations by adding precisely the *minimal* quadratic terms — a T₁T₂ term in the singles projection and a ½T₂² term in the doubles projection — that restore exact size-extensivity: ⟨Ψ_i^a|H̄|(T₁+T₂+T₁T₂)Ψ₀⟩ = a_i^a E_corr, ⟨Ψ_{ij}^{ab}|H̄|(1+T₁+T₂+½T₂²)Ψ₀⟩ = a_{ij}^{ab} E_corr. It is exact for two electrons, correct to fourth order, and sits between CISD and CCSD in complexity and content (CCSD carries additional nonlinear terms QCISD omits). It shares the missing-triples limitation.

**Augmented singles-and-doubles with non-iterative triples.** Two schemes add a one-off triples estimate on top of a converged singles-and-doubles solution, so the expensive triples step is done once rather than every iteration. (a) For QCISD, an augmented procedure (Pople, Head-Gordon, Raghavachari 1987) forms a triples correction from the converged amplitudes, ΔE_T(QCISD) = (2 Σ_s^S + Σ_s^D) Σ_t^T Σ_u^D (E₀ − E_t)⁻¹ a_s V_{st} V_{tu} a_u — a sum running over both single and double substitutions on the outer index. (b) For CCSD, the scheme CCSD+T(CCSD) (Urban, Noga, Cole, Bartlett 1985) forms its triples correction T(CCSD) = Σ_s^D Σ_t^T Σ_u^D (E₀ − E_t)⁻¹ a_s V_{st} V_{tu} a_u using **only the converged doubles amplitudes** on the outer index. Both run at ∝N⁶ iterative cost plus one ∝N⁷ step. The two corrections thus differ in what the triple is allowed to couple back into on the outer index. On the ozone diagnostic this difference is stark: the QCISD augmentation gives a real frequency in fair agreement with experiment, whereas the doubles-only CCSD correction badly overshoots, even returning an imaginary frequency that wrongly predicts an asymmetric structure — its triples contribution comes out too large.

**Iterative-triples coupled cluster (CCSDT-n, CCSDT)** (Lee, Kucharski, Bartlett 1984; Noga, Bartlett 1987; Scuseria, Schaefer 1988). Solve for triple-excitation amplitudes iteratively: the CCSDT-n family folds in at least the leading triples terms at ∝N⁷ per iteration, and full CCSDT includes the triple–triple interaction at ∝N⁸ per iteration. These are accurate but, because the costly triples step recurs every iteration, restricted to small systems.

## Evaluation settings

The natural yardstick is the term-by-term **fifth-order Møller–Plesset partition** above: for each method, tabulate which of E_SS⁵, E_SD⁵, E_DD⁵, E_ST⁵, E_DT⁵, E_TT⁵, E_DQ⁵, E_TQ⁵, E_QQ⁵ it contains fully, half, or only partially, alongside its computational class (iterative ∝N⁶, ∝N⁷, ∝N⁸, with or without a one-off non-iterative step). For small systems where it is feasible, the absolute benchmark is **full configuration interaction** (full CI) energies, against which a method's error in milli-hartree can be read directly (small molecules at minimal and double-zeta bases, including geometries with stretched bonds that stress the correlation treatment). The physical-property test is the **ozone asymmetric stretching frequency** with a polarized double-zeta basis, compared to the experimental 1089 cm⁻¹. Scaling with *n* and *N* is the affordability metric.

## Code framework

Assume a working singles-and-doubles correlation solver already exists — it converges the cluster/CI amplitudes and provides molecular-orbital two-electron integrals, orbital energies, and the converged single (`t1`) and double (`t2`) amplitudes. We build a non-iterative post-step on top of it.

```python
import numpy as np

class SinglesDoublesSolver:
    """Existing iterative SD-level correlation solver (CCSD/QCISD-type)."""
    def __init__(self, eri, orb_energies, n_occ, n_vir):
        self.eri = eri                # antisymmetrized MO integrals <pq||rs>
        self.eps = orb_energies       # orbital energies
        self.no, self.nv = n_occ, n_vir
        self.t1 = None                # converged singles amplitudes
        self.t2 = None                # converged doubles amplitudes

    def solve(self):
        # iterate the SD projection equations to convergence (cost ~ N^6 per iter)
        # ... fills self.t1, self.t2, returns E_corr (correct through 4th order in SDQ)
        raise NotImplementedError

def orbital_energy_denominator(eps, i, j, k, a, b, c):
    """(E0 - E_t)^{-1} type resolvent for a triple excitation ijk->abc."""
    # epsilon_i + epsilon_j + epsilon_k - epsilon_a - epsilon_b - epsilon_c
    return eps[i] + eps[j] + eps[k] - eps[a] - eps[b] - eps[c]

def post_sd_correction(solver: SinglesDoublesSolver) -> float:
    """
    A one-off, non-iterative correction layered on the converged SD solution,
    using the already-available t1, t2, integrals, and orbital energies.
    """
    t1, t2, eri, eps = solver.t1, solver.t2, solver.eri, solver.eps
    # TODO: the correction we will design here
    raise NotImplementedError

def total_energy(e_hf: float, solver: SinglesDoublesSolver) -> float:
    e_corr_sd = solver.solve()
    e_post = post_sd_correction(solver)
    return e_hf + e_corr_sd + e_post
```
