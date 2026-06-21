# Context: elementary excitations of a bond-alternated conjugated polymer chain

## Research question

Trans-polyacetylene, (CH)_x, is the simplest linear conjugated polymer: a chain of CH units, each contributing one π electron (a p_z orbital perpendicular to the chain) on top of the σ framework that holds the backbone together. At the lowest energy the chain is *dimerized* — the C–C bonds alternate "short/long" (the chemist's "double/single"), so the carbons displace into a period-doubled pattern rather than sitting on a uniform lattice. There are two ways to alternate (short-long-short-… versus long-short-long-…), and they are mirror images with exactly the same energy.

A set of measurements on undoped and lightly doped material present a set of phenomena:

- Undoped trans-(CH)_x carries a dilute population (order a few hundred ppm) of paramagnetic defects with an electron-spin-resonance line that is **narrow** (g = 2.00263, ΔH ≈ 1.65 Oe at room temperature) and **Lorentzian**, persisting essentially unchanged down to 10 K. Narrow Lorentzian shape and motional narrowing indicate a **highly mobile, neutral, spin-½** entity that exists *without* any dopant.
- On doping with acceptors (AsF₅, iodine) or donors, the electrical conductivity rises over many orders of magnitude, yet the Curie-law spin susceptibility does **not** rise in proportion — there is an anomalously *small* magnetic contribution.
- The infrared absorption shows extra weight (a feature near ~0.1 eV in lightly doped samples) and the magnitude and temperature dependence of the thermopower in lightly doped material show features not present in a simple shifted-Fermi-level semiconductor.

The question is what microscopic theory of the dimerized chain accounts for these observations.

## Background

**π electrons on a dimerized chain; the σ framework as an elastic medium.** The four valence electrons of each carbon: three go into sp² σ bonds (two backbone C–C, one C–H), and the fourth into the π system. The σ–σ* gap is large (~10 eV), far above any energy scale of interest here (≤ 0.5 eV), so the σ electrons never get excited — they act only as a stiff elastic spring that resists displacing the carbons. Expanding the σ bonding energy to second order about the undimerized geometry gives a harmonic restoring term ½K Σ_n (u_{n+1} − u_n)², with K an effective spring constant and u_n the displacement of the n-th CH unit along the chain axis. The π electrons are treated in a tight-binding (Hückel) approximation: one orbital per site, nearest-neighbour hopping t. Because the transfer integral depends on the C–C separation, t is *modulated* by the displacements; for the small displacements at issue a linear dependence t_{n+1,n} = t_0 − α(u_{n+1} − u_n) is accurate, where α is the electron–lattice coupling. The bare π bandwidth is W = 4t_0 ≈ 10 eV.

**The Peierls instability.** A uniform half-filled one-dimensional tight-binding band (one π electron per site, Fermi points at ±k_F = ±π/2a) is unstable: a periodic lattice distortion that doubles the unit cell opens a gap exactly at the Fermi level, pushing all the filled states down in energy. Peierls (1955) showed the electronic energy gain from opening such a gap always beats the elastic cost of the distortion in 1D, so the uniform chain is never the ground state — it spontaneously dimerizes. The dimerized chain is therefore a small-gap semiconductor (gap of order 1.4 eV in (CH)_x). Crucially, the dimerization can lock in with either sign of the alternation, giving **two degenerate ground states**.

**Bond alternation in long polyenes.** That long conjugated chains alternate (rather than equalize) their bonds — and the trade-off between the σ-bond elastic energy and the π-electron energy that fixes the alternation amplitude — was established for finite polyenes (Longuet-Higgins & Salem 1959; Salem, *The Molecular Orbital Theory of Conjugated Systems*, 1966). The same competition, taken to the infinite chain, is the Peierls instability above.

**Topological excitations of a degenerate-vacuum system.** When a system has two (or more) degenerate ground states, a natural class of excitations is a spatial region that interpolates between them — a domain wall separating a stretch in one ground state from a stretch in the other. Such walls are robust because the two sides are pinned to distinct minima; they cannot be removed by a small local deformation. Whether such a wall in the dimerized chain is a genuine low-lying, localizable excitation, and what its quantum numbers are, is open.

**Fermions in a topologically nontrivial background (field theory).** In relativistic field theory, a Dirac fermion coupled to a scalar field that takes a kink (domain-wall) profile was shown by Jackiw and Rebbi (1976) to have a striking consequence: the Dirac operator acquires a **normalizable zero-energy mode** localized on the kink, and the fermion number of the ground state becomes **fractional, ±½**. The mechanism is a conjugation (charge-conjugation/sublattice) symmetry C with C⁻¹HC = −H, which pairs every positive-energy state with a negative-energy partner; an unpaired self-conjugate zero mode then forces the filled-sea fermion number to shift by half a unit. This was a field-theory result with, at the time, no experimentally accessible realization.

**Continuum soliton pictures for polyacetylene.** Rice (1979) discussed charged π-phase kinks in lightly doped polyacetylene using a continuum description, raising the possibility that the doping carriers are charged domain-wall-like objects rather than band electrons/holes. Continuum (Ginzburg–Landau-type and Dirac-type) treatments of the dimerized chain were also being developed. A continuum treatment must decide whether the relevant soliton is a **phase** soliton of a charge-density wave (a sine-Gordon-like π phase slip, which would carry charge) or an **amplitude** distortion of the dimerization order parameter — a distinction with direct consequences for the soliton's charge and spin.

## Baselines

**Rigid band / shifted-Fermi-level semiconductor picture of doping.** The textbook account of doping a semiconductor: a donor adds an electron to the conduction band, an acceptor a hole to the valence band, with no structural change to the host. Applied to dimerized (CH)_x this predicts that each added carrier costs the band-edge energy Δ (half the gap) and carries spin ½ (an electron in the conduction band, or a hole in the valence band, is a spin-½ object).

**Mobile-bond-alternation-defect interpretation of the ESR (Goldberg et al. 1976; Snow et al.).** The narrow undoped ESR line was attributed to a bond-alternation domain wall quenched into the polymer during cis→trans isomerization, and the lightly-doped behaviour to charged domain walls (Weinberger et al.; Grant & Batra).

**Sine-Gordon / charge-density-wave phase soliton (Rice; Bishop, Krumhansl, Trullinger 1976).** Treat the dimerization as a charge-density wave and its low-energy excitations as phase solitons satisfying a sine-Gordon equation; a π phase shift interchanges the bond pattern and (in this reading) carries charge ±e.

**Field-theoretic fractional fermion number (Jackiw–Rebbi 1976).** The kink-bound zero mode and ±½ fermion number result lives in a relativistic continuum field theory with a *postulated* scalar kink background, where the Dirac operator acquires a normalizable zero-energy mode localized on the kink and the fermion number of the ground state becomes fractional, ±½.

## Evaluation settings

The natural yardsticks (pre-existing facts and instruments, no results stated):

- **Material:** undoped and lightly doped trans-(CH)_x films (Shirakawa-type), with controlled doping by acceptors (AsF₅, I₂) and donors.
- **Measured quantities the theory must connect to:** the ESR g-factor and linewidth (g = 2.00263, ΔH ≈ 1.65 Oe) and its temperature dependence down to ~10 K; the Curie-law spin susceptibility vs dopant concentration; the d.c. conductivity vs doping; the measured low-temperature activation energy for conductivity in the dilute alloy (≈ 0.3 eV); infrared absorption spectra (including the ~0.1 eV feature and the C–C stretch modes near 1370 cm⁻¹); thermopower.
- **Model parameters to be fixed from independent data:** the π bandwidth W = 4t_0 ≈ 10 eV; the optical gap E_g ≈ 1.4 eV; the σ spring constant K ≈ 21 eV/Å² (from vibrational/quantum-chemical estimates); the CH-unit spacing a ≈ 1.22 Å along the chain axis; the macroscopic dielectric constant ε ≈ 10 and a CH unit mass M.
- **Theoretical targets:** the dimerization amplitude u_0 and the resulting bond-length change; the ground-state condensation energy per site; for any proposed excitation, its formation energy, spatial width, effective mass, activation energy for motion, charge, spin, and the resulting doping mechanism (which excitation is energetically preferred over a band electron/hole).

## Code framework

A small numerical scaffold for a one-dimensional tight-binding chain with displacement-dependent hopping, written in pre-method terms. The contribution will be the choice of displacement field that defines the excitation and the quantities computed from it; that slot is left empty.

```python
import numpy as np

# ---- model parameters (fixed from independent data) ----
t0  = 2.5     # eV, bare hopping; pi bandwidth W = 4 t0 ~ 10 eV
t1  = 0.35    # eV, dimerization hopping shift at the minimum (E_g = 4 t1 ~ 1.4 eV)
K   = 21.0    # eV / Ang^2, sigma-bond spring constant
a   = 1.22    # Ang, CH-unit spacing along chain axis
# alpha (electron-lattice coupling) and u0 (dimerization amplitude) are fixed
# by minimizing the ground-state energy; see below.

def hopping(u, n):
    """Displacement-dependent transfer integral t_{n+1,n} = t0 - alpha*(u[n+1]-u[n])."""
    # linear electron-lattice coupling; alpha set after u0 is known
    raise NotImplementedError

def build_hamiltonian(u, N):
    """Nearest-neighbour pi-electron Hamiltonian on N sites for a displacement set {u_n}."""
    H = np.zeros((N, N))
    # off-diagonals from hopping(u, n); diagonal = 0 (one orbital per site)
    raise NotImplementedError
    return H

def electronic_energy(u, N):
    """Sum of occupied (negative-energy / valence) single-particle eigenvalues."""
    H = build_hamiltonian(u, N)
    evals = np.linalg.eigvalsh(H)
    return evals[evals < 0].sum() * 1.0      # half-filling: lower half occupied (x2 for spin handled by caller)

def elastic_energy(u):
    """sigma-bond elastic energy  (1/2) K sum_n (u_{n+1}-u_n)^2 ."""
    du = np.diff(u)
    return 0.5 * K * np.sum(du**2)

def total_energy(u, N):
    return 2.0 * electronic_energy(u, N) + elastic_energy(u)   # factor 2 for spin

# ---- the displacement field defining the excitation, and what we read off it ----
def displacement_field(N):
    """The configuration {u_n} we will study."""
    # TODO: the displacement pattern we will design here, and its free parameter(s)
    raise NotImplementedError

def excitation_observables(N):
    """Energy, spatial extent, mass, and electronic quantum numbers of the configuration."""
    # TODO: the quantities we will compute from {u_n} and the electronic spectrum
    raise NotImplementedError
```
