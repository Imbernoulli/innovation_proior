After special relativity, gravitation is in a strange state. Newton's law of gravity says one mass feels another's pull instantly, across any distance, with no reference to time. That cannot survive the abolition of absolute simultaneity and the speed limit c. The most obvious repair is to keep the Newtonian scalar potential and make it Lorentz-covariant, turning Poisson's equation into a wave equation. But that road fails immediately: because inertia now depends on a body's energy content, the relativistic scalar theory makes the acceleration of a falling body depend on its horizontal velocity and on its internal energy. That directly contradicts the oldest and best-tested fact in the subject, that all bodies fall with the same acceleration. Galileo argued it, Newton verified it with pendulums of many materials, and Eötvös has since pinned the equality of inertial and gravitational mass to about one part in a hundred million. Any theory that breaks universality of free fall is dead on arrival. A different starting point is needed.

The way forward is to stop treating the equality of inertial and gravitational mass as a coincidence and treat it as the central structural clue. If every body falls with the same acceleration, then a freely falling observer cannot detect the field by any local mechanical experiment: a released stone, coin, or drop of water stays beside him or drifts uniformly, because everything shares the same acceleration. A uniformly accelerated frame in empty space produces exactly the same phenomenology: the floor rushes up to meet every released object, and every object falls alike regardless of material. The two situations are mechanically indistinguishable. Extending this beyond mechanics to all physical processes gives the equivalence principle: a homogeneous gravitational field of strength γ is physically equivalent, in a small enough region, to a uniformly accelerated frame with acceleration γ. This is local, because real gravitational fields converge and cannot be mimicked globally by one uniform acceleration, and it is heuristic, meaning gravitational effects can be computed in the accelerated gravity-free frame using special relativity and then read off in the static gravitational frame.

From this single assumption, three consequences follow before any field equation is written. First, energy must gravitate. Send a packet of radiant energy E downward through a gravitational potential difference Φ. In the equivalent accelerated frame, the lower receiver is moving toward the source by the time the light arrives, so the first-order Doppler boost gives received energy E(1 + Φ/c²). A closed cycle comparing this with the work needed to raise a body that has absorbed the energy shows the body's gravitational mass increases by E/c², exactly equal to the inertial mass special relativity already assigns to energy. Second, the same Doppler argument applies to frequency: light received lower in the field has frequency ν = ν₀(1 + Φ/c²). Since wave crests cannot be created or destroyed in a steady stream, the only consistent interpretation is that clocks run at different rates at different potentials; a clock at potential Φ has rate multiplied by (1 + Φ/c²) relative to a clock at the origin. Solar spectral lines are therefore redshifted by Δν/ν = Φ/c² ≈ 2×10⁻⁶. Third, because the speed of light timed by a single distant clock varies with potential as c = c₀(1 + Φ/c²), a transverse potential gradient rotates a wavefront by Huygens' construction, bending light toward the mass. For a ray grazing a mass M at impact parameter Δ, the bending angle is α = 2GM/(c²Δ), giving about 0.83 arcseconds at the Sun's limb.

These results change the conceptual picture of gravity. A gravitational field is no longer primarily a force pulling bodies; it is a feature of space and time that governs the rates of clocks, the speed of light, and hence the trajectories of both matter and light. The equivalence principle is the bridge that lets special relativity, which by itself cannot speak about gravity, generate definite predictions about gravitational effects on energy, time, and light.

```python
G   = 6.674e-11      # gravitational constant, N m^2 / kg^2
c   = 2.998e8        # speed of light, m/s
M_sun = 1.989e30     # kg
R_sun = 6.957e8      # m   (impact parameter for a grazing ray)
ARCSEC_PER_RAD = 206265.0
SOLAR_PHI_DEPTH_OVER_C2 = 2.0e-6  # rounded G M_sun / (R_sun c^2)

def gravitational_potential(r, M):
    """Newtonian potential Phi(r) = -G M / r."""
    return -G * M / r

def fractional_frequency_shift(Phi):
    """
    Signed fractional frequency shift from nu = nu0 (1 + Phi/c^2).
    A clock at potential Phi ticks at rate factor (1 + Phi/c^2).
    """
    return Phi / c**2

def light_deflection(M, impact_parameter):
    """
    Bending angle for a ray grazing mass M at impact parameter b,
    derived from c = c0 (1 + Phi/c^2) and Huygens' wavelet construction:
    alpha = (1/c^2) * integral(dPhi/dn) ds = 2 G M / (c^2 b).
    """
    b = impact_parameter
    return 2.0 * G * M / (c**2 * b)

# Solar surface gravitational potential
Phi_sun = -SOLAR_PHI_DEPTH_OVER_C2 * c**2

# Redshift of solar spectral lines relative to Earth
solar_redshift = -fractional_frequency_shift(Phi_sun)
print("solar line redshift  Delta_nu/nu =", solar_redshift)  # ~2e-6

# Grazing-ray light deflection at the Sun
alpha_rad = light_deflection(M_sun, R_sun)
alpha_arcsec = alpha_rad * ARCSEC_PER_RAD
print("light deflection (rad)    =", alpha_rad)      # ~4e-6
print("light deflection (arcsec) =", alpha_arcsec)   # ~0.83
```
