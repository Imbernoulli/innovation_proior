In classical electrodynamics a charged particle feels only the Lorentz force from local electric and magnetic fields, so if both fields vanish along every possible path there should be no observable effect. Potentials are treated as convenient mathematical bookkeeping because a gauge change leaves the fields unchanged. Quantum mechanics changes the story because the charged-particle Hamiltonian couples directly to the scalar and vector potentials, not to the fields themselves, and interference makes relative phase observable. Existing local field-only reasoning and naive gauge-removal arguments fail because they assume the potential can be removed globally; they miss that a field-free region can be multiply connected, so curl-free vector potentials can still carry nontrivial holonomy around excluded flux.

The key insight is that an observable is not the value of the potential at any point, but the gauge-invariant closed-loop phase accumulated by the wavefunction. When a coherent beam is split into two arms that travel around opposite sides of a shielded magnetic flux tube and then recombine, the relative phase is the line integral of the vector potential around the closed contour. By Stokes' theorem this equals the enclosed magnetic flux, even though the electron wavefunction never enters the region where the magnetic field is nonzero. The effect is periodic in the flux quantum and disappears only when the closed-loop phase is an integer multiple of 2π. A single unsplit beam sees only an overall unobservable phase, and a simply connected field-free patch would allow complete gauge removal, but the excluded flux core makes the global topology matter.

The method is the Aharonov-Bohm effect. It identifies the physical observable as the gauge-invariant holonomy

exp[i (q/ℏc) ∮ A · dr]

for the magnetic case, or the difference of time-integrated scalar potentials for the electric case. In the magnetic geometry, place a long thin solenoid or confined flux tube between two arms of an electron interferometer so that the magnetic field is nonzero only in a region where the electron wavefunction has negligible support. Outside the solenoid B = 0, but the vector potential cannot be written as the gradient of a single-valued function over the whole accessible region. Each arm is described by a locally gauge-removed wavefunction with an open-path phase, and when the arms recombine the endpoint phases of a single-valued gauge function cancel, leaving only the loop integral. That loop integral is the magnetic flux Φ through the solenoid, so the interference phase shift is Δθ = qΦ/(ℏc) in Gaussian units, or qΦ/ℏ in SI. The predicted period is the flux quantum Φ0 = h/|q|.

The same logic applies to the electric Aharonov-Bohm effect. Two wave packets pass through separate conducting tubes, and each tube potential is raised and lowered only while the packet is fully inside. The electron remains in a region of zero electric field, but the scalar potential contributes a branch-dependent phase θj = −(q/ℏ) ∫ φj(t) dt. Recombination yields Δθ_electric = −(q/ℏ)[∫φ1 dt − ∫φ2 dt]. Again the observable is a relative phase, not a local force, and a single branch would see only an unobservable overall phase. The quantum interference experiment therefore reads potential information that classical field-only dynamics discards.

The Aharonov-Bohm effect shows that quantum mechanics needs more than local field strengths in the accessible region: a flat connection on a multiply connected space can have nontrivial holonomy, and that holonomy is measurable through interference. The vector potential is not itself a directly measurable local quantity; rather, its gauge-equivalence class around noncontractible loops carries physical information.

```python
import numpy as np

def magnetic_aharonov_bohm_phase(flux, charge=-1.0, hbar=1.0, c=1.0):
    """
    Magnetic Aharonov-Bohm phase for a charge q enclosing flux Phi.
    Gaussian units: phase = q * Phi / (hbar * c)
    SI units: set c=1 and use phase = q * Phi / hbar.
    """
    return charge * flux / (hbar * c)

def electric_aharonov_bohm_phase(phi1, t, phi2=None, charge=-1.0, hbar=1.0):
    """
    Electric Aharonov-Bohm phase difference between two potential histories.
    phi1, phi2: arrays of scalar potential vs time on the two branches.
    Returns -(q/hbar) * (integral of phi1 dt - integral of phi2 dt).
    """
    dt = t[1] - t[0]
    theta1 = np.trapezoid(phi1, t)
    if phi2 is None:
        phi2 = np.zeros_like(t)
    theta2 = np.trapezoid(phi2, t)
    return -(charge / hbar) * (theta1 - theta2)

def flux_quantum(charge, hbar=1.0, c=1.0):
    """Flux quantum h*c/|q| (Gaussian) or h/|q| (SI with c=1)."""
    return 2 * np.pi * hbar * c / abs(charge)

# Example: electron (q=-e) around a solenoid with Phi = 0.3 flux quanta
e_charge = -1.0
flux = 0.3 * flux_quantum(e_charge)
phase = magnetic_aharonov_bohm_phase(flux, charge=e_charge)
print("Magnetic Aharonov-Bohm phase:", phase)

# Example: electric effect with a potential pulse on branch 1 only
t = np.linspace(0, 10, 1000)
phi1 = np.where((t > 3) & (t < 7), 1.0, 0.0)
phi2 = np.zeros_like(t)
e_phase = electric_aharonov_bohm_phase(phi1, t, phi2, charge=e_charge)
print("Electric Aharonov-Bohm phase difference:", e_phase)
```