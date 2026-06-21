I will derive the universal spectral law of cavity radiation and show why it forces energy itself to come in discrete quanta. The method I am presenting is Planck's law of black-body radiation, or more precisely the Planck quantization hypothesis that underwrites it.

Start with what experiment had made unavoidable by the end of the nineteenth century. A cavity with walls held at a fixed temperature emits radiation whose spectral energy density, energy per unit volume per unit frequency, is a universal function of frequency and temperature alone. Kirchhoff proved this universality, so the problem of finding the spectrum is the problem of finding a single function of two variables. Stefan and Boltzmann added that the total energy density scales as the fourth power of temperature, so whatever the spectral function is, its integral over all frequencies must be finite and proportional to T to the fourth. Wien then used thermodynamics together with the Doppler effect of radiation reflected from a moving piston to show that the spectrum cannot depend on frequency and temperature independently; it must take the form u(ν,T)=ν³ f(ν/T). Only the ratio ν/T can appear. That scaling law is a severe constraint. Finally, the shape of the spectrum was partially known. Wien proposed u∝ν³ e^{-aν/T}, and it fit Paschen's short-wavelength infrared data well. But Rubens and Kurlbaum pushed measurements into the far infrared using residual rays from fluorite and rock salt, and there Wien's exponential falloff failed. In that long-wavelength, high-temperature regime the intensity grew linearly with T instead. No single classical formula reconciled the short-wave exponential with the long-wave linear behavior and a finite total energy.

The clean way to attack the problem is to reduce it to a single resonator. Model the cavity wall as a collection of Hertzian oscillators, one for each frequency, exchanging energy with the radiation field until equilibrium is reached. Because Kirchhoff's law says the answer cannot depend on the material, we are free to choose this convenient idealization. A standard electromagnetic result connects the mean energy U of a resonator at frequency ν to the surrounding spectral density: u=(8πν²/c³)U. So if I can find U(ν,T), I can multiply by the mode factor to obtain u(ν,T). And because thermodynamics gives 1/T=dS/dU, finding U reduces to finding the entropy S(U) of one resonator as a function of its energy. The whole problem funnels into the curve S(U).

It helps to work not with S itself but with its second derivative. In stable equilibrium entropy is concave, so d²S/dU² is negative. Define R as the positive reciprocal R=-1/(d²S/dU²). The two experimental regimes then translate into simple statements about R. Wien's law gives U∝ν e^{-aν/T}; solving for 1/T and differentiating once more yields d²S/dU²=-1/(aνU), so R∝U. The long-wave linear behavior means U∝T, which gives 1/T=dS/dU∝1/U and hence d²S/dU²∝-1/U², so R∝U². Thus small energy says R is linear in U, large energy says R is quadratic in U. The simplest law containing both limits is R=U(U+β)/α, equivalently d²S/dU²=-α/[U(U+β)]. Integrating and imposing Wien scaling gives U=β/(e^{β/(αT)}-1) and therefore u=c₁ν³/(e^{c₂ν/T}-1). This expression does interpolate between the two regimes, but it is only an interpolation. It gives no reason why the entropy should have that particular form.

To find a reason, I need to compute the entropy from counting. Boltzmann's principle S=k log W says entropy measures the logarithm of the number of microscopic ways a macrostate can be realized. Consider N resonators of frequency ν with total energy U_N=NU and total entropy S_N=NS. If energy is continuous, there is no countable number of ways to distribute it; the set of partitions is uncountable and W is undefined. The only way to rescue counting is for the total energy to be composed of a whole number P of identical finite elements ε, so that U_N=Pε. Then distributing P indistinguishable energy elements among N distinguishable resonators is a finite combinatorial problem. The number of arrangements is W=(N+P-1)!/((N-1)!P!). The entropy is therefore S_N=k log W, and for large N and P Stirling's approximation gives S_N=k[(N+P)log(N+P)-N log N-P log P]. Dividing by N and substituting P/N=U/ε yields the per-resonator entropy S=k[(1+U/ε)log(1+U/ε)-(U/ε)log(U/ε)]. This entropy is not guessed; it is derived from counting arrangements of discrete energy elements.

The element ε is still free, but Wien's displacement law fixes it. The displacement law implies that the resonator entropy can depend on U and ν only through the combination U/ν. The counted entropy depends on U only through U/ε. For these to be the same function, ε must be proportional to ν. Write ε=hν, where h is a new universal constant with units of energy times time. This is the quantization hypothesis: the energy of a resonator of frequency ν is restricted to integer multiples of hν. There is no other way to satisfy both Boltzmann counting and Wien scaling.

With ε=hν, the entropy becomes S=k[(1+U/(hν))log(1+U/(hν))-(U/(hν))log(U/(hν))]. Differentiating with respect to U gives dS/dU=(k/(hν))log(1+hν/U). Setting this equal to 1/T and solving for U produces U=hν/(e^{hν/kT}-1). Feeding this through the resonator bridge gives the spectral energy density u(ν,T)=(8πhν³/c³)/(e^{hν/kT}-1). This is Planck's law.

The same expression now follows from counting rather than interpolation, and it unifies the two classical limits automatically. When hν is much smaller than kT, expand the exponential to first order; the denominator becomes hν/kT, so U tends to kT and u tends to 8πν²kT/c³, the classical equipartition or Rayleigh-Jeans law. In this limit the energy elements are so small compared to thermal energy that the lumpiness is invisible. When hν is much larger than kT, the exponential dominates and u tends to (8πhν³/c³)e^{-hν/kT}, Wien's law. Because high-frequency modes cost a whole quantum hν that greatly exceeds kT, they are exponentially suppressed, and the integral over all frequencies is finite and scales as T⁴, restoring Stefan-Boltzmann. The divergence of classical equipartition is cured by discreteness.

The price is a finite quantum of action h. As long as h is imagined to be arbitrarily small, the classical continuum reappears and equipartition returns. But the data require h to be a definite nonzero constant. A smallest unit of action has no place in a physics of continuous variation, so the derivation implies that continuity itself is broken at the atomic scale. The second universal constant k, Boltzmann's constant, also emerges from the counting, linking entropy to molecular mass scales. The result is a single law of nature: energy is counted, not poured.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import h, k, c
from scipy.integrate import quad

# Spectral energy density of black-body radiation per unit frequency (Planck)
def planck_nu(nu, T):
    return (8.0 * np.pi * h * nu**3 / c**3) / (np.exp(h * nu / (k * T)) - 1.0)

def wien_nu(nu, T):
    # Short-wavelength limit coefficient chosen to match Planck prefactor
    return (8.0 * np.pi * h * nu**3 / c**3) * np.exp(-h * nu / (k * T))

def rayleigh_jeans_nu(nu, T):
    return 8.0 * np.pi * nu**2 * k * T / c**3

nu = np.linspace(1e11, 1.5e15, 2000)
T = 5000.0

# Verify limiting behavior
rj_ok = np.allclose(planck_nu(nu[:50], T), rayleigh_jeans_nu(nu[:50], T), rtol=0.05)
wien_ok = np.allclose(planck_nu(nu[-200:], T), wien_nu(nu[-200:], T), rtol=0.05)
print("Rayleigh-Jeans low-frequency match:", rj_ok)
print("Wien high-frequency match:", wien_ok)

# Verify Stefan-Boltzmann scaling by numerical integration
def total_density(T):
    val, _ = quad(planck_nu, 1e8, 1e16, args=(T,), limit=200)
    return val

T_vals = np.array([3000.0, 4000.0, 5000.0, 6000.0])
densities = np.array([total_density(T) for T in T_vals])
log_T = np.log(T_vals)
log_rho = np.log(densities)
power = np.polyfit(log_T, log_rho, 1)[0]
print(f"Numerical Stefan-Boltzmann exponent: {power:.3f}")
print(f"Total energy densities (J/m^3): {densities}")

# Theoretical Stefan-Boltzmann constant for energy density
sigma_energy = 8.0 * np.pi**5 * k**4 / (15.0 * h**3 * c**3)
print(f"Theoretical rho = a T^4 with a = {sigma_energy:.3e}")
```
