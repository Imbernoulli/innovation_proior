# The Froissart bound: σ_tot grows at most like ln²s

## Problem

Does relativistic scattering theory put a ceiling on how fast a total cross section can grow with energy? Through the optical theorem this is the same as asking how fast the forward amplitude can grow, which — for the Mandelstam double-dispersion representation — is the same as asking how many subtractions the representation needs. The goal: derive a rigorous upper bound on σ_tot(s) as s → ∞ from analyticity (the representation, with a *finite* number of subtractions) and unitarity (conservation of probability) alone.

## Key idea

Work with the s-channel absorptive part A(s,t) = Im_s f(s,t) of the c.m. amplitude f (normalized so dσ/dΩ = |f|²), whose partial-wave coefficients Im a_l(s) are **positive** and **capped at 1** by unitarity. Angular analyticity (the Lehmann ellipse, whose real semi-axis is set by the nearest t-channel singularity t_0 = 4m_π²) forces the positive Legendre series for A to have partial waves that decay exponentially in l. Crossing that decay against the unitarity cap shows that only L ∝ √s · ln s partial waves are significant. Summing them at the forward point through the optical theorem gives σ_tot ≲ (4π/k²)·L² — in which the energy factors cancel, leaving the square of the logarithm. The physical reading is that a finite-range force can grow only a logarithmically expanding interaction disk, whose area is ∝ ln²s.

## The bound

For two-body elastic scattering at squared c.m. energy s, as s → ∞,
  σ_tot(s) ≤ (4π / t_0) · ln²(s/s_0) = (π / m_π²) · ln²(s/s_0),
where t_0 = 4m_π² is the two-pion threshold (the nearest t-channel singularity), s_0 is a scale that makes the logarithm dimensionless, and the coefficient carries the subtraction-count factor (with a finite number n of subtractions, 4π/t_0 is multiplied by (n−1)²; n = 2 gives the quoted coefficient). Equivalently, for the relativistic Mandelstam-representation amplitude F (normalized so dσ/dΩ = |F/√s|²) this reads |F(s,t)| ≲ s ln²s at the forward (or backward) angle, and |F(s,t)| ≲ s^{3/4} ln²s at a fixed angle in the physical region.

## Derivation

1. **Inputs.** (i) Mandelstam representation with a finite number of subtractions ⇒ polynomial boundedness of the absorptive part at fixed t: |A(s,t)| ≤ C(t) s^N for t in a neighbourhood of 0. (ii) Unitarity ⇒ S_l = 1 + 2i a_l with |S_l| ≤ 1, i.e. Im a_l(s) ≥ |a_l(s)|² ≥ 0 and 0 ≤ Im a_l ≤ 1. (iii) Lehmann analyticity ⇒ at fixed physical s, A(s, cos θ) is analytic inside an ellipse with foci ±1 and real semi-axis z_0(s) = 1 + t_0/2k², t_0 = 4m_π².

2. **Positive Legendre series.** A(s, t) = (1/k) Σ_l (2l+1) Im a_l(s) P_l(z), z = cos θ. Positivity of Im a_l and analyticity inside the ellipse of semi-axis z_0 force the coefficients to decay at least as fast as the boundary growth of P_l. Using P_l(x) ∼ (x + √(x²−1))^l for x>1 (Laplace integral representation P_l(x) = (1/π)∫_0^π (x+cos φ √(x²−1))^l dφ), with z_0 = 1+ε, ε = t_0/2k², so z_0+√(z_0²−1) ≈ 1+√(2ε):
   Im a_l(s) ≲ s^N · exp(−l √(2ε)) = s^N · exp(−l √(t_0)/k).

3. **Counting significant waves.** Each Im a_l ≤ 1 (unitarity) and ≲ s^N exp(−l√(t_0)/k) (analyticity). The crossover is at s^N exp(−L√(t_0)/k) ≈ 1:
   L ≈ (k/√(t_0)) N ln s ≈ (N/2)(√s/√(t_0)) ln s   (k ≈ √s/2).
   Waves with l ≲ L are O(1); waves with l ≫ L are exponentially negligible.

4. **Optical theorem at the forward point.** P_l(1) = 1, so
   σ_tot(s) = (4π/k²) Σ_l (2l+1) Im a_l(s) ≲ (4π/k²) Σ_{l=0}^{L} (2l+1) = (4π/k²)(L+1)² ≈ (4π/k²) L².
   The exponentially suppressed tail does not change the scaling. Substituting L and k²≈s/4:
   σ_tot(s) ≲ (4π/(s/4))·(N²/4)(s/t_0) ln²s = (4π N²/t_0) ln²s.
   The s cancels; the logarithm survives, squared.

5. **Coefficient.** With t_0 = 4m_π², 4π/t_0 = π/m_π². The subtraction count enters as a constant (N², refined to (n−1)² in a sharper truncation; n=2 by Phragmén–Lindelöf), never in the exponent of the logarithm — so the ln²s law is independent of the field-theoretic details and only the prefactor remembers them:
   σ_tot(s) ≤ (4π/t_0) ln²(s/s_0) = (π/m_π²) ln²(s/s_0).

## Semiclassical reading

Partial wave l ↔ impact parameter b = l/k, so the retained block reaches b_max = L/k ≈ (N/√(t_0)) ln s — a logarithmically growing interaction radius. The cross section is of order the area of that disk, σ ∼ π b_max² ∼ (π N²/t_0) ln²s, matching the rigorous ceiling's form and 1/t_0 scale up to an O(1) geometric factor (the rigorous bound's coefficient is the larger 4π/t_0). Why logarithmic? For an absorptive Yukawa exchange of mass κ, the interaction strength at impact parameter a is ∼ g e^{−κa}; scattering is essentially complete out to a_max = (1/κ) ln|g|, giving σ ≈ π a_max² = (π/κ²) ln²|g|. If the effective coupling grows like a power of energy, ln|g| ∝ ln s, and σ ∝ (π/κ²) ln²s. The exponential fall-off of a finite-range force (κ = m_π, the lightest exchange) is what turns power-law coupling growth into logarithmic radius growth, and area gives the square. A lighter exchange would mean longer range, smaller t_0, and a larger ceiling; the lightness of the pion sets the coefficient.

## Worked numerical illustration

```python
import numpy as np

# pion-mass units; equal-mass elastic scattering
m_pi = 1.0
t0   = 4 * m_pi**2          # two-pion threshold = nearest t-channel singularity
def k2(s):                  # c.m. momentum squared
    return (s - 4*m_pi**2) / 4.0

def decay_rate(s):
    # Im a_l <~ poly(s) * exp(-l * rate);  rate = sqrt(t0)/k from z0 = 1 + t0/(2 k^2)
    return np.sqrt(t0) / np.sqrt(k2(s))

N_sub = 2                    # finite number of subtractions
def num_significant_partial_waves(s):
    return N_sub * np.log(s) / decay_rate(s)        # L ~ (k/sqrt(t0)) N ln s

def sigma_tot_ceiling_from_waves(s):
    # optical theorem; each retained wave capped by unitarity (Im a_l <= 1)
    L = int(num_significant_partial_waves(s))
    l = np.arange(L + 1)
    return (4*np.pi / k2(s)) * np.sum(2*l + 1)       # = (4 pi / k^2)(L+1)^2

def cross_section_ceiling(s, s0=1.0):
    return (4*np.pi / t0) * (N_sub**2) * np.log(s/s0)**2   # = (pi/m_pi^2) N^2 ln^2 s (t0=4)

for s in [1e2, 1e4, 1e6, 1e8]:
    summed = sigma_tot_ceiling_from_waves(s)
    closed = cross_section_ceiling(s)
    b_max  = num_significant_partial_waves(s) / np.sqrt(k2(s))
    print(f"s={s:.0e}  L~{num_significant_partial_waves(s):8.1f}  "
          f"wave-sum->{summed:10.2f}  (4pi/t0)N^2 ln^2->{closed:10.2f}  b_max={b_max:6.3f}")
# the wave-sum ceiling and the closed-form ceiling both grow like ln^2 s and agree
# up to the O(1) truncation constant; b_max grows only logarithmically.
```

Running this confirms both ceilings scale as ln²s with the same coefficient (4π/t₀)·N² (= 4π ≈ 12.57 in pion-mass units, since t₀ = 4 and N = 2), and that the interaction radius b_max grows logarithmically — the area of a logarithmically expanding disk.
