# The Equivalence Principle, and what it predicts before any field equations

## The problem

After special relativity (1905), gravitation is both incompatible and incomplete. Newton's instantaneous
force G m M / r² cannot survive the abolition of absolute simultaneity, and the relativity principle
itself extends only to inertial frames, leaving acceleration and gravity outside it. Standing apart is an
exactly verified fact with no explanation: all bodies fall with the same acceleration, i.e. inertial mass
equals gravitational mass (Galileo; Newton's pendulums; Eötvös to ~1 part in 10⁸). The most direct
relativistic patch — a Lorentz-covariant scalar potential — fails, because the energy-dependence of
inertia makes a body's fall acceleration depend on its velocity and internal energy, breaking
universality of free fall.

## The key idea

Take the equality of inertial and gravitational mass not as a coincidence but as the central clue. Then a
freely falling observer feels no gravity: every nearby body, whatever its material, shares his
acceleration, so relative to him all free bodies are at rest or in uniform motion, and no local
experiment can reveal the field. Conversely, a uniformly accelerated frame in gravity-free space
reproduces, for all bodies alike, exactly a homogeneous gravitational field.

**Equivalence principle (local).** A homogeneous gravitational field of strength γ is physically
equivalent to a uniformly accelerated reference frame with acceleration γ — for *all* physical processes,
not just mechanics — in a region small enough that the real field is uniform. This makes mᵢ = m_g a
matter of course, extends the relativity principle to accelerated frames, and is *heuristic*: any
gravitational effect may be computed in the accelerated gravity-free frame K′ (where special relativity
holds at each instant) and read off in the static gravitational frame K.

Equivalence holds only locally: a real field converges toward its source and cannot be globally replaced
by one uniform acceleration, so every result below is to first order in a small region.

## Three predictions, derived

Setup: two stations S₂ (high) and S₁ (low, a height h below) on the field axis; potential difference
γh ≡ Φ. Replace K by the upward-accelerating gravity-free frame K′; judge from the inertial frame
momentarily comoving at emission. Light falls the distance h in time h/c, during which the receiver
acquires speed v = γh/c toward the source.

**1. Energy gravitates.** By first-order Doppler, energy received E₁ = E₂(1 + v/c) = E₂(1 + γh/c²) → in
K: E₁ = E₂ + E₂Φ/c². A five-step cycle — emit E downward, lower a body of mass M, add E to it, raise
the heavier body M′, then remove E at the top — gives the gravitational mass gained by a body absorbing energy E:
  **M′ − M = E/c²**, equal to the inertial mass of that energy.

**2. Gravitational time dilation / redshift.** The same Doppler step on frequency: ν₁ = ν₂(1 + γh/c²) →
  **ν = ν₀(1 + Φ/c²),  so  Δν/ν = Φ/c².**
A steady stream of light cannot change its crest-count in transit, so the only consistent reading is that
a clock at potential Φ has its rate multiplied by (1 + Φ/c²) relative to the origin's clock — higher
potential means faster, deeper in the well means slower. Solar lines, emitted where Φ < 0 if Earth is the
zero of potential, reach Earth redshifted by |Φ|/c² = G M_⊙/(R_⊙ c²) ≈ 2×10⁻⁶; equivalently, if Φ denotes
the positive potential rise from the solar surface to Earth, Δν/ν = Φ/c².

**3. Light deflection.** Because clocks vary with Φ, the speed of light timed by a single (origin) clock
varies with place:
  **c = c₀(1 + Φ/c²).**
By Huygens, a transverse gradient of c rotates the wavefront toward smaller c (toward the mass), giving a
signed deflection per path length
  **dθ/ds = −(1/c) dc/dn = −(1/c²) ∂Φ/∂n.**
For n increasing outward from the mass, the positive bending magnitude is
  **α = (1/c²) ∫ (∂Φ/∂n) ds.**
For a ray grazing mass M at impact parameter Δ, with Φ = −GM/r, the transverse integral is 2GM/Δ, so
  **α = 2GM/(c²Δ).**
At the Sun's limb (M_⊙, R_⊙): α ≈ 4×10⁻⁶ rad ≈ **0.83 arcsec**, a star's apparent position pushed
outward — testable against the star field of a total solar eclipse.

Together: energy carries gravitational mass, clocks run at rates set by gravitational potential, and light
bends along that potential, all before any gravitational field equation is specified.

## Numerical check

```python
G   = 6.674e-11      # N m^2 / kg^2
c   = 2.998e8        # m/s
M_sun = 1.989e30     # kg
R_sun = 6.957e8      # m  (impact parameter of a grazing ray)
ARCSEC_PER_RAD = 206265.0
SOLAR_PHI_DEPTH_OVER_C2 = 2.0e-6  # rounded G M_sun / (R_sun c^2) for the solar estimate

def gravitational_potential(r, M):
    return -G * M / r                          # Phi = -G M / r

def fractional_frequency_shift(Phi):
    return Phi / c**2                           # signed (nu - nu0) / nu0 = Phi / c^2

def light_deflection(M, impact_parameter):
    b = impact_parameter                        # integral b/(x^2+b^2)^(3/2) dx = 2/b
    return 2.0 * G * M / (c**2 * b)             # alpha = 2 G M / (c^2 b)

Phi_sun = -SOLAR_PHI_DEPTH_OVER_C2 * c**2
solar_redshift = -fractional_frequency_shift(Phi_sun)
print("solar redshift Delta_nu/nu =", solar_redshift)              # ~2e-6
alpha = 2.0 * SOLAR_PHI_DEPTH_OVER_C2                              # = 2 G M_sun / (c^2 R_sun)
print("deflection (rad)    =", alpha)                              # ~4e-6
print("deflection (arcsec) =", alpha * ARCSEC_PER_RAD)             # ~0.83
```
