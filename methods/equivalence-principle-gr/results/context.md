# Context: gravitation after special relativity, and the puzzle of free fall

## Research question

Special relativity (1905) reorganized space and time around one principle — the laws of physics take
the same form in every inertial frame, and light travels at the same speed c in all of them — and one
corollary that proved as deep as the principle: a body's inertia grows with its energy content,
ΔE = Δm·c². Three features of the situation set the agenda.

First: **Newtonian gravity and the new kinematics.** Newton's force F = G m M / r² depends on the
instantaneous separation r and mentions no time; relativity has abolished absolute simultaneity and
forbidden signals faster than light. A law of gravitation framed within the new kinematics is the
standing task.

Second: **the relativity principle covers uniform motion only.** It equates inertial frames to one
another but says nothing about accelerated frames; acceleration, and gravitation with it, sit outside
the principle. A frame at rest in a gravitational field and a frame accelerating through empty space are
treated as separate situations.

Third: **all bodies fall with the same acceleration.** This is among the oldest and best-tested facts
in physics: the mass that resists acceleration equals, for every substance, the mass that responds to
gravity.

The question is how to frame gravitation within relativity, in a way that engages this universal equality
of fall and says something definite and testable about how gravity acts on matter, on energy, and on
light.

## Background

**The relativity principle and the inertia of energy (1905).** In an inertial frame, intervals and
energies transform by the Lorentz transformation. To first order in v/c the relevant facts are: a source
of light seen by a receiver moving toward it at speed v has its frequency and its energy raised by the
factor (1 + v/c) (the first-order Doppler effect, for energy as well as frequency, since a light pulse's
energy and frequency scale together); and a quantity of energy E carries inertial mass E/c². These are
among the most securely established consequences of the relativity principle.

**Galileo's universality of free fall.** That all bodies, whatever their material, fall together is not
folklore but a reasoned result. Galileo's argument in *Two New Sciences* (1638): suppose a heavy body
fell faster than a light one. Tie them together. The slow one should retard the fast one, so the pair
falls slower than the heavy body alone — yet the pair is heavier than either, so it should fall faster.
Contradiction. The only consistent conclusion is that bodies of the same material fall at equal speed,
and experiment extends this across materials. Stated dynamically: all bodies undergo the same
acceleration in a given gravitational field.

**Newton's equality of inertial and gravitational mass.** Newton tied this to the structure of his
mechanics. The same symbol m appears in two laws: in F = m a it is *inertial* mass, the resistance to
being accelerated; in F = G m M / r² it is *gravitational* mass, the charge that couples to gravity.
That these are equal is why the acceleration g = G M / r² is independent of the falling body's m. Newton
checked it directly — pendulums of "gold, silver, lead, glass, sand, common salt, wood, water and wheat"
(Principia, Book III, Proposition 6) all keep the same period, showing weight proportional to quantity of
matter to high accuracy. He noted the contrast with magnetism, where the attracting power is *not*
proportional to the mass. In Newton's mechanics the equality mᵢ = m_g enters as an empirical fact.

**Eötvös's precision test.** By the early twentieth century this equality was a recognized prize problem.
The Göttingen Academy offered the Beneke Prize (1906) for establishing the equality of inertial and
gravitational mass by experiment and theory; Baron Roland Eötvös, using a torsion balance comparing the
gravitational pull and the centrifugal effect of Earth's rotation on different materials, had confirmed
mᵢ = m_g to roughly one part in 10⁸–10⁹. So the universality of free fall is not approximate folklore but
an experimentally exact law — the most precisely verified fact then bearing on gravitation.

**Frames, clocks and rigid rods.** The available way to give physical meaning to a frame is a lattice of
rigid measuring rods carrying identical clocks. A uniformly accelerated frame is described by imagining
such a rigid frame given a constant acceleration γ; at any instant one compares it to the inertial frame
momentarily moving with it, where the ordinary special-relativistic facts hold.

**Relativistic scalar gravity.** One route to a relativistic gravity keeps the Newtonian gravitational
potential Φ — a single scalar field with ∇²Φ = 4πGρ (Poisson) — and upgrades the equation to be Lorentz
covariant by adding a second time derivative, while adapting the point-mass equation of motion to special
relativity. Because inertial mass depends on a body's energy content, the law of motion ties a body's
vertical fall acceleration to its horizontal velocity and to its internal energy.

## Baselines

**Newtonian gravitation.** Core: instantaneous force G m M / r², potential Φ with ∇²Φ = 4πGρ; predicts
Kepler's orbits, tides, the return of comets. The equality mᵢ = m_g is taken as an empirical fact.

**Special relativity as it stood (1905).** Core: Lorentz invariance of the laws of physics among inertial
frames, constancy of c, inertia of energy E/c², first-order Doppler (1 + v/c). It applies among inertial
frames.

**Relativistic scalar-potential gravity (the obvious extension).** Core: retain the scalar Φ, make its
field equation Lorentz covariant (a wave equation sourced by mass–energy), adapt the equation of motion.
In it, the acceleration of fall is tied to the body's velocity and internal energy.

**Newtonian "corpuscular" deflection of light.** One can ask, within Newton's framework, by how much a
fast corpuscle skimming a mass M at distance Δ is deflected — a finite bending follows from treating
light as a projectile in the 1/r² field.

## Evaluation settings

The natural yardsticks are astronomical and spectroscopic, all available before any new theory:

- **Solar limb light deflection at a total eclipse.** During totality, stars whose lines of sight pass
  close to the Sun's limb become visible; their apparent positions can be compared with their positions
  measured months later when the Sun is elsewhere. The relevant geometry: a ray grazing the Sun
  (impact parameter ≈ the solar radius R_⊙ ≈ 6.96×10⁸ m, mass M_⊙ ≈ 1.99×10³⁰ kg). The measurable is an
  angular shift of order a second of arc, at the edge of contemporary astrometric precision.
- **Jupiter's limb** offers a far smaller analogous deflection — a fallback target.
- **Solar spectral lines.** High-resolution spectroscopy of Fraunhofer lines in sunlight versus the same
  lines from a terrestrial source: the quantity of interest is a fractional wavelength shift toward the
  red of order 10⁻⁶. Such fine line displacements toward the red had already been reported (L. F. Jewell,
  1897; Ch. Fabry and H. Boisson, 1909), attributed to pressure in the absorbing solar layer, alongside
  other effects (pressure, temperature broadening and shifting line centres) present in the spectra.

Constants for the order-of-magnitude estimates: c ≈ 3×10⁸ m/s, G ≈ 6.67×10⁻¹¹ N·m²/kg².

## Code framework

Gravitation here is handled by a chain of physical arguments, not an algorithm; no library or training loop is involved.
The one place a few lines of computation are natural is a final numerical check — turning whatever
expressions the argument supplies for the solar frequency shift and the grazing-ray deflection into
numbers. The scaffold is just the constants, the Newtonian potential used for the weak solar field, and
placeholders for the two quantities the argument must determine.

```python
G   = 6.674e-11      # gravitational constant, N m^2 / kg^2
c   = 2.998e8        # speed of light, m/s
M_sun = 1.989e30     # kg
R_sun = 6.957e8      # m   (impact parameter for a grazing ray)
ARCSEC_PER_RAD = 206265.0

def gravitational_potential(r, M):
    """Newtonian potential Phi(r) = -G M / r."""
    return -G * M / r

def fractional_frequency_shift(Phi):
    """Placeholder for the predicted fractional line shift at potential difference Phi."""
    pass  # TODO

def light_deflection(M, impact_parameter):
    """Placeholder for the predicted bending angle of a grazing ray."""
    pass  # TODO
```
