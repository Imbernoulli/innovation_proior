# The Dirac equation

## Problem

Find a wave equation for a single electron that is consistent with special relativity yet keeps the
probabilistic structure of quantum mechanics — a conserved, **non-negative** probability density. The
direct relativistic equation (Klein–Gordon), obtained by quantizing E² = p²c² + m²c⁴, is *second
order in time*; its conserved density ρ ∝ i(ψ*∂_tψ − ψ∂_tψ*) carries a time derivative and is
sign-indefinite, so it cannot be a probability. It also describes a spinless particle, while the
electron is known to carry spin ½ and a magnetic moment eħ/2mc (g = 2).

## Key idea

Demand an equation that is **first order in ∂/∂t** (so ψ now fixes ψ later and ρ = ψ†ψ ≥ 0) and,
for Lorentz symmetry, **first order in ∇** as well — and require that *squaring* it reproduce
E² = p²c² + m²c⁴. Squaring a linear operator c α·p + βmc² leaves cross terms that vanish only if the
coefficients **anticommute**; they cannot be numbers, they must be matrices forming a Clifford
algebra. The smallest such set is four 4×4 matrices (built from two copies of the Pauli matrices),
so ψ has four components. That structure automatically yields spin ½ and the magnetic moment, and
its unavoidable negative-energy solutions are reinterpreted, via Pauli's exclusion principle, as
antimatter.

## The equation

Free electron, Hamiltonian form:

  iħ ∂ψ/∂t = (c α·p + β mc²) ψ,   p = −iħ∇,

with 4×4 matrices satisfying

  αᵢαⱼ + αⱼαᵢ = 2δᵢⱼ I,   αᵢβ + βαᵢ = 0,   β² = I.

Four-index form, using p₀ = ip₄ and γᵢ = ρ₂σᵢ, γ₄ = ρ₃:

  (i Σ_μ γ_μ p_μ + mc) ψ = 0,   {γ_μ, γ_ν} = 2δ_μν I.

Minkowski form, with γ⁰ = β, γⁱ = βαⁱ, ∂_μ = (1/c ∂_t, ∇):

  (iħ γ^μ ∂_μ − mc) ψ = 0,   {γ^μ, γ^ν} = 2 g^{μν} I,   g = diag(1,−1,−1,−1).

Standard (Dirac–Pauli) representation, in 2×2 blocks:

  β = γ⁰ = [[I, 0],[0, −I]],   αᵢ = [[0, σᵢ],[σᵢ, 0]],   γⁱ = [[0, σᵢ],[−σᵢ, 0]].

## Why it works — the consequences that fall out

1. **Square recovers relativity.** (iħγ^μ∂_μ − mc)(iħγ^ν∂_ν + mc)ψ = −(ħ²γ^μγ^ν∂_μ∂_ν + m²c²)ψ;
   using {γ^μ,γ^ν} = 2g^{μν} this is −(ħ²∂^μ∂_μ + m²c²)ψ = 0, i.e. the Klein–Gordon relation
   E² = p²c² + m²c⁴. Every Dirac solution obeys it.

2. **Positive probability.** ρ = ψ†ψ = Σ|ψₐ|² ≥ 0, with current j = cψ†αψ, obeys
   ∂_tρ + ∇·j = 0. The defect that killed the second-order theory is gone.

3. **Spin ½.** Orbital m = x×p is not conserved (mF − Fm = iħρ₁σ×p), nor is ½ħσ alone, but their
   sum M = m + ½ħσ is conserved — the electron carries intrinsic angular momentum ½ħσ.

4. **Magnetic moment, g = 2.** Minimal coupling p → p + (e/c)A and using
   (σ·u)(σ·v) = (u·v) + iσ·(u×v), with the noncommuting π = p + (e/c)A, gives
   (σ·π)² = π² + (eħ/c)σ·curl A, an extra term −(eħ/2mc)σ·B in the energy: a
   magnetic moment of one Bohr magneton, gyromagnetic ratio 2. Hydrogen fine structure comes out
   correct (Sommerfeld/Pauli–Darwin) with no hand-inserted Thomas factor.

5. **Antimatter.** E = ±√(p²c² + m²c⁴): the negative-energy solutions survive the square root and
   cannot be excluded (quantum perturbations would drive transitions into them). Filling all
   negative-energy states (Pauli exclusion), an *empty* state — a hole — behaves as a particle of
   positive energy, positive charge, and the same mass as the electron: a positive electron, or
   anti-electron. Electron–anti-electron annihilation into radiation and pair creation from
   radiation are both contained in the theory, with complete symmetry between positive and negative
   charge.

## Dirac-Pauli algebra check

```python
import numpy as np

I2 = np.eye(2, dtype=complex)
sigma = [np.array([[0,1],[1,0]], dtype=complex),
         np.array([[0,-1j],[1j,0]], dtype=complex),
         np.array([[1,0],[0,-1]], dtype=complex)]
Z = np.zeros((2,2), dtype=complex)

rho1 = np.block([[Z, I2], [I2, Z]])
rho2 = np.block([[Z, -1j*I2], [1j*I2, Z]])
rho3 = np.block([[I2, Z], [Z, -I2]])
sigma4 = [np.block([[s, Z], [Z, s]]) for s in sigma]

alpha = [rho1 @ s for s in sigma4]                   # alpha_i = rho_1 sigma_i
beta  = rho3                                         # alpha_4 = beta = rho_3
gamma_e = [rho2 @ s for s in sigma4] + [rho3]        # gamma_i, gamma_4 with delta algebra
gamma = [beta] + [beta @ a for a in alpha]           # gamma^0, gamma^i with Minkowski algebra
g = np.diag([1, -1, -1, -1]).astype(complex)
I4 = np.eye(4, dtype=complex)
ac = lambda A, B: A @ B + B @ A

# {alpha_i, alpha_j} = 2 delta_ij,  {alpha_i, beta} = 0,  beta^2 = I
for i in range(3):
    assert np.allclose(ac(alpha[i], beta), 0)
    for j in range(3):
        assert np.allclose(ac(alpha[i], alpha[j]), 2*(i == j)*I4)
assert np.allclose(beta @ beta, I4)

# {gamma_mu, gamma_nu} = 2 delta_mu_nu in the four-index notation
for mu in range(4):
    for nu in range(4):
        assert np.allclose(ac(gamma_e[mu], gamma_e[nu]), 2*(mu == nu)*I4)

# {gamma^mu, gamma^nu} = 2 g^{mu nu} in the standard Minkowski notation
for mu in range(4):
    for nu in range(4):
        assert np.allclose(ac(gamma[mu], gamma[nu]), 2*g[mu, nu]*I4)
print("Dirac/Clifford algebra verified")
```
