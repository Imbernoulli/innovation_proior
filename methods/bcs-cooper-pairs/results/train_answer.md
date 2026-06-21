The problem is to explain superconductivity from the quantum mechanics of electrons and ions. The phenomenology is clear: zero DC resistance below a critical temperature, the Meissner effect, a second-order transition, an electronic specific heat that decays exponentially at low temperature, and an isotope effect showing that phonons are involved. The central obstacle is one of scale. The condensation energy is of order ten-millionths of an electron volt per electron, while any honest calculation of the total electron-phonon energy carries an uncertainty of order one electron volt per electron. You cannot compute the whole energy and hope the superconducting phase appears as a small remainder. The only viable strategy is to isolate exactly the correlation that differs between the normal and superconducting phases and treat that piece exactly, letting everything common cancel in the difference.

Existing approaches mostly miss this discipline. Fröhlich's and Bardeen's self-energy theories compute how a single electron is dressed by phonons. But nearly all of that self-energy is already present in the normal state and changes little at the transition; Schafroth showed that the Meissner effect cannot be obtained from the Fröhlich Hamiltonian in any finite order of perturbation theory. The localized-pair Bose-condensation picture of Schafroth, Blatt, and Butler is closer in spirit, but Pippard's coherence length shows that the pairs are enormous and massively overlapping, so they cannot be treated as a dilute gas of composite bosons. What is needed is a true many-body instability of the Fermi sea itself, not a self-energy shift and not a pre-formed molecular picture.

The method is BCS theory, named for Bardeen, Cooper, and Schrieffer. It starts from the residual interaction between quasiparticles near the Fermi surface. Because of the isotope effect, that residual interaction is the phonon-mediated attraction that remains after the linear electron-phonon coupling is eliminated. In a thin shell around the Fermi surface, where the electronic energy transfer is smaller than the phonon energy, this attraction can dominate the screened Coulomb repulsion and be net attractive.

The first step is the Cooper instability. Add two electrons above a frozen Fermi sea, with opposite spins and total momentum zero. Even for an arbitrarily weak attraction, they bind into a pair with energy below twice the Fermi energy. The reason is that the filled sea supplies a finite density of states right at the threshold, making the pairing integral logarithmically divergent. There is no threshold attraction strength. The binding energy is proportional to an essential singularity in the coupling, exp(-2/N(0)V), which is why perturbation theory can never find it. This already tells us the normal ground state is unstable.

The second step builds the full many-body ground state. Because the pairs overlap millions of times, one cannot stack independent two-electron pairs. Instead, BCS writes a variational state as a product of amplitudes for every momentum pair. Each pair state has an amplitude to be occupied and an amplitude to be empty, and the state is constrained so that if a state with wave vector k and spin up is occupied, then the time-reversed state with wave vector -k and spin down is also occupied. This coherent pairing makes all attractive scattering matrix elements reinforce. Minimizing the expectation value of the reduced Hamiltonian smears the Fermi step over an energy scale epsilon_0, and epsilon_0 is determined self-consistently by the BCS gap equation.

The consequences match all five facts. The quasiparticle dispersion E_k = sqrt(epsilon_k^2 + epsilon_0^2) has a minimum energy epsilon_0, so the single-particle spectrum has a gap of width 2 epsilon_0, explaining the exponential specific heat. The condensation energy scales as (hbar omega_D)^2 exp(-2/N(0)V), giving the right tiny magnitude and the M^{-1/2} isotope dependence through the phonon frequency. The non-bosonic nature of the pairs, encoded in b_k^2 = 0, locks the condensate phase rigid over macroscopic distances and produces the Meissner effect and persistent currents. At finite temperature the same gap equation with a tanh factor gives a critical temperature kT_c = 1.14 hbar omega_D exp(-1/N(0)V), and the transition is second order because the order parameter epsilon_0 goes continuously to zero. The ratio 2 epsilon_0(0)/kT_c is about 3.5, a parameter-free universal prediction.

```python
import numpy as np
from scipy import integrate, optimize

# Energies in units of the phonon (Debye) cutoff hbar*omega_D = 1.
N0V = 0.3   # dimensionless coupling N(0)*V (weak coupling: < 1)

# --- Cooper instability: two electrons added above a frozen Fermi sea ---
# Eigenvalue condition: 1 = N0V * int_0^1 dxi / (2*xi - E_rel), with E_rel < 0.
E_rel = optimize.brentq(
    lambda E: N0V * integrate.quad(lambda xi: 1.0 / (2.0 * xi - E), 0.0, 1.0)[0] - 1.0,
    -10.0, -1e-12)
print("pair energy relative to 2E_F:", E_rel)
print("  weak-coupling form -2*exp(-2/N0V):", -2.0 * np.exp(-2.0 / N0V))

# --- BCS gap equation: self-consistent order parameter epsilon_0 ---
# 1/(N0V) = int_0^1 dxi / sqrt(xi^2 + eps0^2) = arcsinh(1/eps0)
# Closed form: eps0 = 1 / sinh(1/N0V)
eps0 = optimize.brentq(
    lambda D: integrate.quad(lambda xi: 1.0 / np.sqrt(xi**2 + D**2), 0.0, 1.0)[0] - 1.0 / N0V,
    1e-12, 10.0)
print("gap eps0:", eps0)
print("  closed form 1/sinh(1/N0V):", 1.0 / np.sinh(1.0 / N0V))
print("  weak-coupling form 2*exp(-1/N0V):", 2.0 * np.exp(-1.0 / N0V))

# --- Finite-temperature gap equation and critical temperature ---
# 1/(N0V) = int_0^1 dxi/sqrt(xi^2+D^2) * tanh(sqrt(xi^2+D^2)/(2*kT))
# At Tc the gap goes to zero, giving 1/(N0V) = int_0^1 (dxi/xi) tanh(xi/(2*kTc))
kTc = optimize.brentq(
    lambda kT: integrate.quad(lambda xi: np.tanh(xi / (2.0 * kT)) / xi, 1e-9, 1.0)[0] - 1.0 / N0V,
    1e-6, 1.0)
print("kTc:", kTc)
print("  weak-coupling 1.14*exp(-1/N0V):", 1.14 * np.exp(-1.0 / N0V))
print("  gap-to-Tc ratio 2*eps0/kTc:", 2.0 * eps0 / kTc)
```
