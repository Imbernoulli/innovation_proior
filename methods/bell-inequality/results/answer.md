# Bell's inequality and the CHSH form — the EPR question made experimental

## The problem

Einstein, Podolsky and Rosen (1935) argued that quantum mechanics is incomplete: for an entangled pair, a measurement on one particle predicts a property of the distant partner with certainty without disturbing it, so (granting local causality and their reality criterion) that property must be predetermined, though the wave function does not contain it. They proposed completing quantum mechanics with hidden variables that restore determinism and locality. von Neumann's 1932 theorem was widely taken to forbid any hidden variables, but it forbids only the wrong class: its load-bearing axiom — additivity of expectation values over *non-commuting* observables — is a peculiar property of quantum ensembles with no claim on individual hidden states (a sum of non-commuting observables is measured by a *distinct* experiment, so the values need not add). With the no-go disarmed, the open question is whether a *local* hidden-variable theory can reproduce quantum mechanics. The Bell inequality answers it: no — and the disagreement is experimentally measurable.

## The key idea

Model a local hidden-variable theory with full generality: a shared variable λ ~ ρ(λ), and two *local* outcome functions A(**a**, λ) = ±1 and B(**b**, λ) = ±1, each depending only on its own analyzer setting (**a** blind to **b**, **b** blind to **a**) — this factorized locality is the entire physical content; determinism itself is *inferred* from locality plus the perfect anticorrelation of the singlet, not assumed. Such a theory's correlations are forced to satisfy an inequality. Quantum mechanics predicts a violation. Locality + realism is therefore falsifiable.

## The quantum correlation (singlet)

For |ψ⟩ = (1/√2)(|↑↓⟩ − |↓↑⟩), using ⟨ψ| σ₁^i σ₂^j |ψ⟩ = −δ_ij:

  E(**a**, **b**) = ⟨ψ|(σ₁·**a**)(σ₂·**b**)|ψ⟩ = −**a**·**b** = −cos θ,

with θ the angle between settings. Perfect anticorrelation E(**a**,**a**) = −1; E = 0 at θ = π/2. The cosine is **stationary at its minimum**, which is what local correlations cannot match.

## Bell's inequality (1964 form, three settings)

Local form E(**a**,**b**) = ∫ dλ ρ(λ) A(**a**,λ) B(**b**,λ). Perfect anticorrelation E(**a**,**a**) = −1 forces B(**a**,λ) = −A(**a**,λ) a.e., so E(**a**,**b**) = −∫ dλ ρ A(**a**,λ)A(**b**,λ). Then, using A(**b**,λ)² = 1,

  E(**a**,**b**) − E(**a**,**c**) = −∫ dλ ρ A(**a**,λ)A(**b**,λ)[1 − A(**b**,λ)A(**c**,λ)],

and since |A(**a**)A(**b**)| = 1 and [1 − A(**b**)A(**c**)] ≥ 0,

  **|E(**a**,**b**) − E(**a**,**c**)| ≤ 1 + E(**b**,**c**).**

QM violates it: at **a**·**c** = 0, **a**·**b** = **b**·**c** = 1/√2, the left side is |−1/√2 − 0| = 0.707 but the right side is 1 − 1/√2 = 0.293. (Local correlations are kinked at the perfect-correlation point; the quantum cosine is smooth there.)

## The CHSH inequality (1969 form, four settings, no perfect correlation)

To avoid relying on exact perfect anticorrelation, drop the B = −A elimination and assume only |A| ≤ 1, |B| ≤ 1. With two settings per side,

  S = E(**a**,**b**) − E(**a**,**b′**) + E(**a′**,**b**) + E(**a′**,**b′**)
    = ∫ dλ ρ { A(**a**)[B(**b**)−B(**b′**)] + A(**a′**)[B(**b**)+B(**b′**)] }.

For each λ one of [B(**b**)±B(**b′**)] is 0 and the other is ±2, so the integrand is ±2 pointwise (and ≤ 2 in magnitude in general); hence

  **|S| = |E(**a**,**b**) − E(**a**,**b′**) + E(**a′**,**b**) + E(**a′**,**b′**)| ≤ 2.**

## The quantum violation: 2√2

With E = −cos θ and coplanar settings **a** = 0°, **a′** = 90°, **b** = 45°, **b′** = 135° (each adjacent pair 45° apart):

  E(**a**,**b**) = −cos45° = −1/√2,  E(**a**,**b′**) = −cos135° = +1/√2 because cos135° = −1/√2,
  E(**a′**,**b**) = −cos45° = −1/√2,  E(**a′**,**b′**) = −cos45° = −1/√2,

  S = (−1/√2) − (+1/√2) + (−1/√2) + (−1/√2) = −2√2,  |S| = 2√2 ≈ 2.828 > 2.

The √2 margin is against the local ceiling itself, before any particular hidden-variable model is specified. Detector losses still have to be handled honestly in the measured outcomes or counting rates; the point of the four-setting form is that the derivation no longer needs exact perfect anticorrelation. Because it needs only the locality factorization P(A,B|a,b,λ) = P(A|a,λ)P(B|b,λ) and bounded outcomes — no spin, no particles — it is a test of local realism itself, decidable by setting two analyzers at the right angles and counting coincidences.

## Numerical check

```python
import numpy as np

def E(theta_deg):
    # quantum singlet correlation E(a,b) = -cos(angle between settings)
    return -np.cos(np.radians(theta_deg))

# Bell 1964 three-setting inequality:  |E(a,b) - E(a,c)| <= 1 + E(b,c)
ab, ac, bc = 45, 90, 45          # relative angles
lhs = abs(E(ab) - E(ac))
rhs = 1 + E(bc)
print(f"Bell-1964: |E(ab)-E(ac)|={lhs:.3f}  1+E(bc)={rhs:.3f}  violated={lhs > rhs}")
# -> 0.707 > 0.293, violated

# CHSH four-setting inequality:  |E(a,b) - E(a,b') + E(a',b) + E(a',b')| <= 2
a, ap, b, bp = 0, 90, 45, 135    # absolute settings (deg)
S = E(a-b) - E(a-bp) + E(ap-b) + E(ap-bp)
print(f"CHSH: S={S:.4f}  |S|={abs(S):.4f}  classical_bound=2  quantum=2*sqrt2={2*np.sqrt(2):.4f}")
# -> |S| = 2.8284 > 2

```
