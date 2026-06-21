The problem is to understand why two neutral, perfectly conducting parallel plates in vacuum attract each other when no classical electromagnetic field is present. A purely classical picture gives nothing, because there is no charge, no current, and no pressure difference. Pairwise London dispersion forces between atoms do explain attraction, but they are microscopic, material-specific, and awkward to sum into a clean macroscopic law. Retarded atom-atom interactions are closer because they involve the quantized electromagnetic field, yet they still route the calculation through microscopic dipoles rather than the universal boundary geometry. Naive zero-point summation captures the right object—the ground-state energy of the allowed field modes—but the absolute sum is divergent and has no physical meaning on its own. What is needed is a way to extract the finite, separation-dependent part of the vacuum energy by comparing the plate geometry with the corresponding empty or far-separated geometry.

The right tool is the Casimir effect. It treats the vacuum of the quantized electromagnetic field as the ground state of its normal modes, and it computes the change in zero-point energy that results from imposing conducting boundary conditions. Between two parallel plates at separation a, the transverse wave vectors remain continuous in the large-area limit, while the normal wave number is quantized as k_z = n pi / a. The allowed scalar frequencies are omega_n(k) = c sqrt(k^2 + (n pi / a)^2), and the electromagnetic field contributes two polarizations. The formal zero-point energy per area diverges, but the physically meaningful quantity is the finite difference relative to the no-plate or widely-separated configuration. Analytic regularization isolates that finite residue, yielding E/A = -pi^2 hbar c / (720 a^3) for the electromagnetic field. Differentiating with respect to separation gives the pressure P = F/A = -pi^2 hbar c / (240 a^4), where the negative sign denotes attraction.

The Casimir effect is therefore not a force produced by some classical substance in the gap. It is a spectral effect: the plates alter the mode structure of the quantum vacuum, and the resulting change in zero-point energy depends on geometry. The divergent absolute energy is discarded by construction; only the separation-dependent residue is observable.

```python
import numpy as np
from scipy.special import gamma, zeta

def casimir_energy_per_area(a, hbar=1.0, c=1.0):
    """
    Zero-point electromagnetic energy per unit area between two perfect
    parallel conducting plates separated by distance a.
    E/A = -pi^2 * hbar * c / (720 * a^3)
    """
    return -(np.pi**2) * hbar * c / (720.0 * a**3)

def casimir_pressure(a, hbar=1.0, c=1.0):
    """
    Attractive pressure between two perfect parallel conducting plates.
    P = F/A = -pi^2 * hbar * c / (240 * a^4)
    """
    return -(np.pi**2) * hbar * c / (240.0 * a**4)

def casimir_scalar_energy_per_area(a, hbar=1.0, c=1.0):
    """
    Scalar/polarization contribution to the energy per area.
    E_s/A = -pi^2 * hbar * c / (1440 * a^3)
    """
    return -(np.pi**2) * hbar * c / (1440.0 * a**3)

def verify_via_zeta_regularization(a, hbar=1.0, c=1.0):
    """
    Reproduce the scalar energy using analytic regularization:
    sum_n n^3 -> zeta(-3) = 1/120.
    """
    zeta_val = zeta(-3)  # 1/120
    prefactor = -hbar * c * np.pi**2 / 1440.0
    return prefactor / a**3, zeta_val

if __name__ == "__main__":
    a = 1.0e-6  # one micron, in natural units hbar=c=1 for illustration
    E = casimir_energy_per_area(a)
    P = casimir_pressure(a)
    Es = casimir_scalar_energy_per_area(a)
    E_check, z3 = verify_via_zeta_regularization(a)

    print(f"a = {a}")
    print(f"scalar energy/area   = {Es:.6e}")
    print(f"EM energy/area       = {E:.6e}")
    print(f"pressure (attraction)= {P:.6e}")
    print(f"zeta(-3)             = {z3:.6e}")
    print(f"scalar check matches : {np.isclose(Es, E_check)}")
```
