# Solitons in dimerized polyacetylene (the SSH model)

## Problem

Undoped trans-(CH)_x carries a dilute, highly mobile, **neutral spin-½** defect (a narrow motionally-narrowed ESR line surviving to 10 K), and on doping its conductivity rises while the Curie-law spin susceptibility does not — the doping carriers are **spinless and charged**. A band-insulator picture cannot give a mobile neutral spin or spinless charge. The goal is a microscopic theory of the elementary excitations of the dimerized chain that produces both, with quantitative energy, width, mass, charge, and spin, and that explains the doping mechanism.

## Key idea

Model the half-filled π band of the chain by tight-binding electrons whose hopping depends on the carbon displacements, with the σ framework as an elastic spring and the heavy CH units treated classically (adiabatic). The half-filled 1D band is Peierls-unstable and **dimerizes**, opening a gap and producing **two degenerate bond-alternation ground states** (A and B). A domain wall between them is a topological **soliton**. Because the dimerization order parameter is a *real amplitude* with two discrete minima, the soliton is a **φ⁴-type amplitude kink (not a sine-Gordon phase soliton)** and is intrinsically neutral. The bipartite (chiral) symmetry of the chain forces a localized **nonbonding electronic state at the exact gap center**, built half from the valence and half from the conduction band. Occupying that one mid-gap state with 0, 1, or 2 electrons gives the reversed charge–spin assignment (Q,s) = (+e,0), (0,½), (−e,0). The wall is wide and the displacements tiny, so the soliton mass is only a few electron masses; and since its formation energy is below the band-edge cost, doping proceeds by making spinless charged solitons.

## Model Hamiltonian

$$H = -\sum_{n,s} t_{n+1,n}\,(c^\dagger_{n+1,s}c_{n,s} + \text{h.c.}) + \tfrac12 K\sum_n (u_{n+1}-u_n)^2 + \tfrac12 M\sum_n \dot u_n^2,$$

with the bond-length-dependent transfer integral

$$t_{n+1,n} = t_0 - \alpha\,(u_{n+1}-u_n).$$

Here u_n is the displacement of the n-th CH unit along the chain axis, K the σ spring constant, M the CH-unit mass, α the electron–lattice coupling. (π–π Coulomb interactions are absorbed into screened t_0, α.)

## Perfectly dimerized chain and the Peierls gap

For u_n = (−1)ⁿu the hopping alternates t_0 ± t_1 with **t_1 = 2αu**. Diagonalizing (reduced zone −π/2a ≤ k ≤ π/2a):

$$E_k = \pm\sqrt{\varepsilon_k^2 + \Delta_k^2},\qquad \varepsilon_k = -2t_0\cos ka,\quad \Delta_k = 4\alpha u\,\sin ka.$$

The gap opens at the Fermi points k = ±π/2a:

$$\boxed{E_g = 2\Delta = 4t_1 = 8\alpha u.}$$

Ground-state energy per chain, with z = t_1/t_0 = 2αu/t_0 and E the complete elliptic integral,

$$E_0(u) = -\frac{4Nt_0}{\pi}\,E(1-z^2) + \frac{NKt_0^2 z^2}{2\alpha^2}.$$

Using E(1−z²) ≅ 1 + ½(ln(4/|z|) − ½)z², the electronic term ∝ +z²ln(1/|z|) beats the elastic z² term as z→0, so **u = 0 is a maximum** (Peierls theorem) and E_0(u) is a symmetric double well with minima at ±u_0. Minimizing with K = 21 eV/Å² and 4t_1 = 1.4 eV gives α ≈ 4.1 eV/Å, u_0 ≈ 0.042 Å (bond change √3 u_0 ≈ 0.073 Å), condensation energy −E_c/N ≈ −0.015 eV.

The perfect-chain density of states (per spin):

$$\rho_0(E) = \frac{N}{\pi}\,\frac{|E|}{\sqrt{(4t_0^2-E^2)(E^2-\Delta^2)}},\quad \Delta\le|E|\le 2t_0,\ \text{else }0.$$

## Soliton: amplitude kink

Staggered order parameter ψ_n = (−1)ⁿu_n; ground states ψ_n = ±u_0. Trial soliton

$$\psi_n = u_0\tanh(n/\ell),$$

a φ⁴-type kink with one variational width ℓ. The formation energy (computed via the Green's-function determinant ΔE = (2/π)∫_{−∞}^{0} Im ln det[1 − G⁰(ω)V̂] dω over the local hopping perturbation, evaluating a soliton–antisoliton pair to cancel chain-end effects) is minimized at

$$\ell \approx 7,\qquad E_s \approx 0.4\ \text{eV}\quad (E_g = 1.4\ \text{eV}),$$

a **diffuse** wall (~14 sites).

## Mid-gap state, charge, and spin

The bipartite chain obeys C⁻¹HC = −H (sublattice/chiral symmetry), forcing a localized state at **E = 0**. Its amplitude solves t_{n+1,n}φ₀(n) + t_{n+1,n+2}φ₀(n+2) = 0, living on one sublattice (φ₀ = 0 on odd sites). Closed form:

$$\boxed{\;\phi_0(n) \;\cong\; \frac{1}{\ell}\,\operatorname{sech}\!\Big(\frac{n}{\ell}\Big)\cos\!\Big(\frac{\pi n}{2}\Big).\;}$$

State conservation ∫Δρ(E)dE = 0 with Δρ(E) = Δρ(−E) means this one mid-gap state is pulled **half from the valence band and half from the conduction band** (nonbonding). The local sum rule 2∫_{−∞}^{−Δ}Δρ_{nn}(E)dE + |φ₀(n)|² = 0 makes the neutral soliton locally and globally charge neutral. Occupying φ₀ with 0, 1, 2 electrons:

$$\boxed{(Q,s) = (+e,0),\ (0,\tfrac12),\ (-e,0).}$$

A **neutral** soliton carries spin ½ (the mobile undoped ESR defect); **charged** solitons are spinless (the doping carriers). [Kramers' theorem is respected because a soliton and its antisoliton together remove one full state.]

## Mass, motion, and doping

Rigid translation ψ_n(t) = u_0 tanh[(na − v_s t)/ℓa] gives ½M_s v_s² = ½M Σ ψ̇_n², and Σ_n sech⁴(n/ℓ) ≈ (4/3)ℓ, so

$$\boxed{M_s = \frac{4}{3\ell}\Big(\frac{u_0}{a}\Big)^2 M \approx 5\,m_e.}$$

Tiny mass ⇒ quantum, highly mobile; the periodic-lattice activation barrier is only ≈ 0.002 eV (nearly free translation to 20–40 K).

Doping channel: injecting a band carrier costs Δ; making a charged soliton costs E_s. Since **E_s ≈ 0.4 eV < Δ = 0.7 eV**, doping proceeds by forming **charged spinless solitons** — conductivity rises with no Curie-susceptibility increase. The charged soliton binds to its dopant by ΔE_I = −(e²/ε)Σ_n|φ₀(n−n_s)|²/[(na)²+d²]^{1/2}, giving E_b ≈ 0.3 eV (ε = 10, d ≈ 2.4 Å), comparable to the dilute-alloy conductivity activation energy.

## Continuum / fractional-charge connection

In the continuum the π electrons near ±k_F become a 1+1D Dirac fermion with a mass set by the order parameter; the soliton is a sign-changing mass kink with one self-conjugate zero mode (the continuum φ₀). The regularized fermion number of a single isolated soliton is **±e/2** — the "half a state from each band" sharpened to a fractional charge when soliton and antisoliton are taken far apart. On the finite lattice the eigenvalues remain integer (Q = 0, ±e); the ½ is the expectation value that becomes a sharp fraction only in the infinite, well-separated limit. (This realizes, in a real material, the Jackiw–Rebbi 1976 fermion fractionalization.)

## Worked evaluation of the final formulas

```python
import numpy as np

# parameters fixed from independent data
t0   = 2.5            # eV  (pi bandwidth W = 4 t0 ~ 10 eV)
Eg   = 1.4            # eV  optical gap = 4 t1
t1   = Eg / 4.0       # eV  -> 0.35
Delta = 2 * t1        # eV  half-gap = 0.70
a    = 1.22           # Ang CH spacing along chain axis
u0   = 0.042          # Ang dimerization amplitude (from minimizing E0(u))
ell  = 7              # soliton half-width (sites), from minimizing E(ell)
M    = 13 * 1836.0    # CH-unit mass in electron masses (13 amu)

# mid-gap state  phi0(n) = (1/ell) sech(n/ell) cos(pi n/2)
def phi0(n, ell):
    return (1.0/ell) * (1.0/np.cosh(n/ell)) * np.cos(0.5*np.pi*n)

n   = np.arange(-60, 61)
phi = phi0(n, ell); phi /= np.sqrt(np.sum(phi**2))
print("weight on odd sites :", round(float(np.sum(phi[n % 2 == 1]**2)), 6))  # ~0

# lattice sum that sets the mass:  sum sech^4(n/ell) -> (4/3) ell
S = float(np.sum((1.0/np.cosh(n/ell))**4))
print("sum sech^4          :", round(S, 3), " vs (4/3)ell =", round(4*ell/3, 3))

# soliton mass  M_s = (4/3 ell)(u0/a)^2 M
M_s = (4.0/(3.0*ell)) * (u0/a)**2 * M
print("soliton mass M_s    :", round(M_s, 2), "m_e")

# doping channel
E_s = 0.42
print("E_s < Delta ?       :", E_s, "<", round(Delta, 2), "->", E_s < Delta)

# charge/spin from occupation of the mid-gap level
for occ, label in [(0, "empty"), (1, "single"), (2, "double")]:
    Q = 1 - occ; spin = 0.5 if occ == 1 else 0.0
    print(f"phi0 {label:6s}: Q = {Q:+d} e, s = {spin}")
```

Output: mid-gap state confined to even sites; Σ sech⁴ ≈ (4/3)ℓ; M_s ≈ 5 mₑ; E_s = 0.42 < Δ = 0.70; and (Q,s) = (+e,0),(0,½),(−e,0) for empty/single/double occupancy.
