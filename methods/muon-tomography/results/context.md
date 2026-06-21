# Context: seeing inside dense objects with naturally occurring radiation (circa 2003)

## Research question

The concrete problem is to look inside a large, dense, optically opaque object — a sealed cargo
container, a truck, a shielded cask — and decide whether it hides a lump of high-`Z` material
(uranium, plutonium, tungsten, lead) without opening it, without an artificial radiation source,
and in a usable amount of time. X-rays and gamma rays are absorbed by a few centimetres of steel;
a uranium core wrapped in lead shielding is essentially invisible to them. Active interrogation
with high-energy beams penetrates better but needs an accelerator, dose control, and a one-sided
source/detector geometry that a closed container does not afford.

What is freely available, everywhere on Earth's surface, is a steady rain of highly penetrating
charged particles — and the question is whether their behaviour as they cross an object can be
turned into a three-dimensional, `Z`-sensitive image.

## Background

**Cosmic-ray muons as a free, penetrating probe.** Primary cosmic rays (mostly protons) strike
the upper atmosphere and produce hadronic cascades; charged pions and kaons decay to muons
(`pi -> mu nu`). Muons are heavy charged leptons (≈207 electron masses, mean life 2.2 µs) that
reach the ground in abundance. The surface spectrum is well measured: the energy distribution is
nearly flat below ~1 GeV and falls as `p^-2.7` above ~10 GeV, with a mean energy of 3–4 GeV; the
angular distribution peaks at the zenith and falls roughly as `cos^2(theta)`. The integrated rate
is about `10^4 m^-2 min^-1` for a horizontal detector — about one muon through a fingernail-sized
area per minute, ≈160 Hz over a square metre. Two facts about this flux matter. First, GeV muons
are *penetrating*: a several-GeV muon loses energy only by ionization at roughly `2 MeV` per
`g cm^-2`, so it punches through metres of rock or steel where photons cannot. Second, the rate is
*low* — low enough to record and reconstruct muons one at a time, yet high enough to accumulate
useful statistics over a target in minutes-to-hours.

**Two ways a muon carries information about the matter it crosses.** (i) *Energy loss / range.* A
muon slows by ionization; very dense or thick material can range it out entirely. Measuring the
*attenuation* of the muon flux gives an absorption radiograph. (ii) *Multiple Coulomb scattering
(MCS).* As the muon crosses material it is deflected by many small-angle elastic scatters off
nuclei (Rutherford scattering, `dsigma/dOmega ~ Z^2 / sin^4(theta/2)`). The cumulative effect is a
small net deflection and a small lateral displacement. By the central-limit theorem the *central*
≈98% of the projected scattering-angle distribution is Gaussian, with zero mean and a width given
by the Highland / Lynch–Dahl formula,

```
theta_0 = (13.6 MeV / (beta c p)) * z * sqrt(L / X0) * [1 + 0.038 * ln(L z^2 / (X0 beta^2))]
```

where `p`, `beta c` are the particle momentum and velocity, `z` its charge (1 for a muon),
`L` the thickness, and `X0` the material's **radiation length**. For singly-charged relativistic
muons (`beta ≈ 1`) the logarithmic term is a slow correction and the width is well approximated by
`theta_0 ≈ (15 MeV / p) sqrt(L / X0)`. Radiation length is the load-bearing material quantity: it
is the characteristic depth for electromagnetic interactions, and it *decreases sharply with `Z`*
— measured as a depth, `X0` is ≈36.1 cm for water, 10.7 cm for concrete, 1.76 cm for iron, 0.56 cm
for lead, 0.32 cm for uranium. So per centimetre, high-`Z` material scatters muons dramatically
more. Concretely, the projected-angle RMS for a 3 GeV muon crossing 10 cm is ≈2.6 mrad in water,
11.9 mrad in iron, 21.1 mrad in lead, 28.0 mrad in uranium — an order of magnitude of contrast
between ordinary cargo and a high-`Z` core.

The muon momentum is not measured by tracking alone, so the width must be evaluated at some
nominal momentum `p0` (≈3 GeV); a slow and a fast muon crossing the same material then give
different deflections, contributing extra spread.

**Why scattering, not absorption, is the sensitive channel.** Scattering is about as `Z`-sensitive
as energy loss, but the milliradian deflection angle is *measurable from millimetre-level position
measurements* using tracking stations above and below the object — no magnetic spectrometer or
precise energy measurement is needed. Absorption radiography yields a line-integral summed along
each muon path.

**Scattering also displaces.** In either projected detector view, a muon crossing thickness `L`
emerges not only with a net projected deflection `Delta_theta` but also shifted sideways by a lateral
displacement `Delta_x` from where its incoming track would have placed it. Both are observable from
the same in/out track measurement.

**Tomography.** Reconstructing a spatial distribution of some material property from many rays that
cross it from different directions is classical tomography. For deterministic line-integral signals
(X-ray CT), one discretizes the volume into voxels with unknown values `f_j`, writes each ray's
signal as a raysum `s_i = sum_j w_ij f_j` (`w_ij` = path length of ray `i` in voxel `j`), and solves
the linear system `s = W f` — by filtered back-projection, or iteratively by Algebraic
Reconstruction Techniques (ART), or by statistical maximum-likelihood expectation-maximization as
developed for emission tomography. This machinery assumes each ray contributes a deterministic
line-integral signal; the muon datum arrives as an in/out track pair and the small random deflection
between the legs.

## Baselines

**Cosmic-ray absorption / range radiography (George 1955; Alvarez et al. 1970).** The first use of
cosmic-ray muons to image a large object. E. P. George measured the rock overburden above a tunnel
by comparing the muon flux inside to that outside. Luis Alvarez and collaborators radiographed the
Pyramid of Chephren at Giza by placing muon counters in a chamber beneath it and measuring the
*differential attenuation* of the downward muon flux versus direction, searching for hidden
chambers (they found none). The method is a transmission count: where there is less material along
a given direction, more muons survive.

**Active (man-made source) radiography — X-ray / gamma cargo scanners.** High-energy X-ray cargo
scanners use an accelerator source in a one-sided transmission geometry to image the interior of
containers. They establish the *operational target* — scan a container for dense contraband — and
demonstrate what an imaging instrument must accomplish.

**Classical tomographic reconstruction (ART; ML-EM for emission tomography).** ART and ML-EM are
the established machinery for inverting many rays into a voxel image. They presuppose a
deterministic raysum model `s_i = sum_j w_ij f_j`: each ray's measured number is a linear functional
of the voxel values, and the established inversions take that linear raysum as their starting point.

## Evaluation settings

The natural apparatus is two tracking stations (e.g. drift/proportional tubes or wire chambers)
bracketing the inspection volume — one station above, one below, each with enough position planes to
fit a straight track in two projected views — so that the incoming and outgoing positions and
direction angles are determined for every muon. The object volume is discretized into voxels (cells
of order ~1 cm and up). The probe is the natural,
unmodified cosmic-ray muon flux (`~1 cm^-2 min^-1`, mean 3–4 GeV, `cos^2` zenith dependence). The
operationally meaningful figure of merit is the ability to *segregate* low-, medium-, and high-`Z`
material (the strong `Z`-dependence of the per-centimetre scattering is the discriminant) and to do so
within a tolerable exposure (minutes for detection, hours for centimetre-scale imaging). Test
geometries are blocks of known materials (water/plastic, aluminium, iron, concrete, copper,
tungsten, lead, uranium) placed in the volume, with reconstructions compared against the known
layout.

## Code framework

The usable primitives are vector geometry, a voxel grid, projected track angles, line traversal
through voxels, a constrained optimizer, and the scattering physics (the Highland width) that
relates a material's radiation length to the expected projected scatter.

```python
import numpy as np
from scipy.optimize import minimize

P0_MEV = 3000.0  # nominal muon momentum (3 GeV/c); momentum is not measured by tracking alone

def highland_msq_per_length(X0_cm, p0_mev=P0_MEV):
    """Highland width squared per unit length for a material of radiation length X0,
    at nominal momentum: (15/p0)^2 / X0  (mean-square projected scatter per cm)."""
    return (15.0 / p0_mev) ** 2 / X0_cm

MSQ_AIR = highland_msq_per_length(X0_cm=3.04e4)  # X0(air) ~ 304 m

class VoxelGrid:
    """A discretization of the inspection volume into cells."""
    def __init__(self, lo, hi, n):
        self.lo, self.hi, self.n = np.asarray(lo, float), np.asarray(hi, float), np.asarray(n, int)
        self.size = (self.hi - self.lo) / self.n
        self.L = float(np.mean(self.size))
    def index(self, pt):
        idx = ((np.asarray(pt) - self.lo) / self.size).astype(int)
        if np.any(idx < 0) or np.any(idx >= self.n):
            return None
        return tuple(idx)

def line_voxels(p0, p1, grid):
    """Cells crossed by a straight segment through the inspection volume."""
    pass

def projected_angles(v):
    """Track angles in the two transverse projected views."""
    pass

def reconstruct(muons, grid):
    """Turn a set of measured in/out track pairs into a 3D voxel image."""
    pass
```
