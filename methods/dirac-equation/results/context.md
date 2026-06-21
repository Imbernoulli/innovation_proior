# Context: a relativistic wave equation for the electron

## Research question

By 1927 quantum mechanics is a complete and astonishingly successful theory — *as long as the
particle moves slowly*. Schrödinger's equation reproduces the hydrogen spectrum, the harmonic
oscillator, scattering, the whole non-relativistic world. But it is built on the non-relativistic
energy relation E = p²/2m, so it applies only when a particle's speed is well below c. The
outstanding problem is to write down a wave equation for a single electron that is consistent with
special relativity — one that treats space and time on the equal footing Einstein demands, yet
still fits inside the probabilistic framework that makes ordinary quantum mechanics work.

Two demands have to be met simultaneously. First, the equation must be **Lorentz invariant**: it
must hold in every inertial frame, so its solutions in one frame map to solutions in another.
Second, it must support the **statistical interpretation** of quantum mechanics: there must be a
conserved, **non-negative** probability density ρ(x,t) with ∫ρ d³x = 1 for all time, so that |ψ|²
can be read as "the probability of finding the electron here." The setting also includes a known
empirical fact: the electron carries an intrinsic two-valued degree of freedom.

## Background

**Non-relativistic quantum mechanics and its probability current.** Schrödinger's equation
iħ ∂ψ/∂t = (−ħ²/2m ∇² + V)ψ is *first order in time*. Born's interpretation takes ρ = |ψ|² = ψ*ψ
as a probability density. Manipulating the equation and its complex conjugate gives a continuity
equation ∂ρ/∂t + ∇·j = 0 with j = −(iħ/2m)(ψ*∇ψ − ψ∇ψ*). Because ρ = |ψ|² is a modulus squared it
is **manifestly non-negative**, and the continuity equation guarantees ∫|ψ|² d³x is conserved.
This is tied to the equation being first order in ∂/∂t: the state ψ at one instant determines ψ at
every later instant.

**Special relativity's energy–momentum relation.** For a free relativistic particle the energy and
momentum satisfy E² = p²c² + m²c⁴ (equivalently, with the relativistic Hamiltonian for a point
electron in a field with potentials A₀, A, the relation (W/c + (e/c)A₀)² = (p + (e/c)A)² + m²c²).
This is quadratic in E. An equation built straight from it by the substitution E → iħ ∂/∂t,
p → −iħ∇ inherits that quadratic, second-order-in-time structure.

**The electron's spin and magnetic moment — known empirically.** Spectroscopy (the anomalous
Zeeman effect, the doublet fine structure of alkali spectra) had forced Goudsmit and Uhlenbeck to
endow the electron with an intrinsic angular momentum of ½ħ and a magnetic moment of one Bohr
magneton, eħ/2mc — twice the value a classical charged spinning top of that angular momentum would
have (a gyromagnetic ratio g = 2). Pauli (1927) encoded this two-valuedness by promoting the wave
function to a two-component object and introducing the matrices
σ₁ = [[0,1],[1,0]], σ₂ = [[0,−i],[i,0]], σ₃ = [[1,0],[0,−1]],
which obey σᵣ² = 1 and σᵣσₛ + σₛσᵣ = 0 for r ≠ s. Pauli's equation couples spin to the magnetic
field by a term inserted to match experiment, and the strength of the spin–orbit coupling (the
"Thomas factor") is likewise set from spectroscopy. Pauli's theory is non-relativistic.

**The second-order relativistic equation.** The natural first relativistic try is to apply
E → iħ∂/∂t, p → −iħ∇ directly to E² = p²c² + m²c⁴. This gives a second-order-in-time equation. The
density that can be built from its solutions to satisfy a continuity equation involves a *time
derivative* of ψ, ρ ∝ i(ψ*∂ψ/∂t − ψ∂ψ*/∂t). The relation E² = p²c² + m²c⁴ also admits
E = ±√(p²c² + m²c⁴), so the equation refers equally to charge +e and −e and carries solutions of
both signs of energy.

## Baselines

**The Klein–Gordon equation (Gordon 1926, Klein 1927).** The direct second-order relativistic wave
equation. For a free particle, [(iħ/c ∂/∂t)² − (−iħ∇)² − m²c²]ψ = 0, i.e. each solution satisfies
E² = p²c² + m²c⁴. Core idea: quantize the relativistic dispersion relation directly. Its conserved
density is ρ ∝ i(ψ*∂ψ/∂t − ψ∂ψ*/∂t); it is second order in time; it describes a spin-0 object; and
it carries solutions of both signs of energy. It is a relativistically invariant template.

**Pauli's two-component spin theory (1927).** Adds the electron's spin to non-relativistic quantum
mechanics by making ψ a two-component spinor acted on by the σ matrices, with a term −(eħ/2mc)σ·B
coupling spin to the field. Core idea: a two-valued internal degree of freedom carried by 2×2
matrices. It is non-relativistic, and the spin–orbit (Thomas) factor is set from spectroscopy. The
σ matrices obey the algebra σᵣ² = 1, σᵣσₛ + σₛσᵣ = 0 for r ≠ s; at 2×2 there are exactly three of
them with this property.

**Schrödinger's equation (1926).** The non-relativistic baseline: first order in time,
positive-definite ρ = |ψ|², conserved probability. A relativistic equation must reduce to it in the
slow-motion limit.

## Evaluation settings

The natural yardsticks already exist before any new equation is written. (1) **The hydrogen
spectrum**, in particular the *fine structure* — the small splittings of the levels, measured
spectroscopically, that the non-relativistic Schrödinger theory misses and that Sommerfeld's old
quantum theory and the Pauli–Darwin spin treatment fit to order α⁴mc² (α the fine-structure
constant). A relativistic electron equation must reproduce these splittings. (2) **The electron's
gyromagnetic ratio and magnetic moment**: g = 2, μ = eħ/2mc, known from the anomalous Zeeman
effect. (3) **The non-relativistic limit**: the equation must collapse to the Schrödinger/Pauli
description when v ≪ c. (4) **Lorentz invariance** itself: the equation and its physical predictions
must be the same in every inertial frame. (5) **A consistent probability interpretation**: a
conserved, non-negative ρ with ∫ρ d³x = 1.

## Code framework

The mathematical setup that exists before the equation is found consists of the relativistic
energy–momentum relation, the operator substitutions, the empirically known Pauli spin matrices,
and an empty slot where the as-yet-unknown form of the relativistic wave equation will go.

```python
import numpy as np

hbar, c, m = 1.0, 1.0, 1.0   # natural-ish units, kept symbolic in the math

# Operator substitution rules from quantum mechanics:
#   E  ->  i*hbar * d/dt
#   p  -> -i*hbar * grad
# Special-relativity energy-momentum relation:
#   E**2 = (p*c)**2 + (m*c**2)**2

# Pauli's 2x2 spin matrices from the non-relativistic spin theory:
sigma1 = np.array([[0, 1],  [1, 0]],  dtype=complex)
sigma2 = np.array([[0,-1j], [1j,0]],  dtype=complex)
sigma3 = np.array([[1, 0],  [0,-1]],  dtype=complex)
# They satisfy: sigma_r**2 = I  and  sigma_r sigma_s + sigma_s sigma_r = 0  (r != s)

def relativistic_wave_equation(psi):
    # TODO: write down a single-electron wave equation that is consistent with
    #       special relativity and admits a conserved, non-negative probability
    #       density rho with continuity d_t rho + div j = 0, reducing to the
    #       Schrodinger/Pauli description for v << c.
    pass
```
