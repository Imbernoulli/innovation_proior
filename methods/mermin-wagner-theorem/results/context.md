# Context: low-dimensional order in isotropic spin lattices

## Research question

Does a magnet ordered by short-range, rotationally-symmetric exchange actually develop a
nonzero spontaneous magnetization when it lives in one or two spatial dimensions, at a finite
temperature? In three dimensions the answer is plainly yes — real ferromagnets and
antiferromagnets order below their Curie/Néel temperatures. The question is whether the same
microscopic Hamiltonian, laid out on a chain or a plane instead of a bulk crystal, still orders.

The stakes are conceptual and practical. Conceptually, "spontaneous magnetization" means the
Hamiltonian has a continuous rotational symmetry in spin space that the *state* does not share:
the spins pick a direction the energy does not prefer. Whether thermal agitation can or cannot
prevent that symmetry from breaking, as a function of the lattice dimension, is a sharp yes/no
question about the foundations of phase transitions. Practically, thin films and layered
materials are effectively two-dimensional, and one wants to know whether they can be intrinsic
magnets at all.

## Background

**The Heisenberg model.** The standard microscopic description of an insulating magnet is the
isotropic Heisenberg Hamiltonian

  H = − Σ_{ij} J_{ij} **S**_i · **S**_j ,

with quantum spin operators **S**_i obeying the angular-momentum algebra
[S_i^z, S_j^±] = ±ℏ δ_{ij} S_i^±, [S_i^+, S_j^-] = 2ℏ δ_{ij} S_i^z, and
**S**_i·**S**_j = ½(S_i^+ S_j^- + S_i^- S_j^+) + S_i^z S_j^z. The dot product is rotationally
invariant, so H commutes with every component of the total spin **S** = Σ_i **S**_i; the model
has a full continuous SO(3) symmetry. The sign of J_{ij} sets the order: J_{ij} > 0 favors
parallel neighbors (ferromagnet), J_{ij} < 0 favors antiparallel neighbors (antiferromagnet,
i.e. two interpenetrating sublattices with opposite magnetization, described by a staggered order
parameter). "Isotropic" means no preferred spin axis; "short-range" means the J_{ij} fall off
fast enough that the second moment Σ_j J_{ij}(R_i − R_j)² of the exchange is finite.

**Spin waves (Bloch, 1930).** The first quantitative theory of the temperature dependence of the
magnetization treats the low-lying excitations of the aligned ferromagnet as quantized spin waves
(magnons). For the Heisenberg ferromagnet the magnon dispersion is *quadratic* and *gapless* at
long wavelength,

  E(**k**) ≈ D k²   as k → 0,

a direct consequence of the rotational symmetry: tilting all spins together costs no energy, so
the uniform mode (k=0) is a zero-energy mode and the cost grows only as k². Each thermally excited
magnon reduces the magnetization, and summing over modes with Bose occupation gives the reduction

  ΔM(T) ∼ ∫₀^∞ N(E) / (e^{E/k_B T} − 1) dE,

where N(E) is the magnon density of states. From E ∼ k² in d dimensions, the volume element
k^{d-1}dk converts to N(E) ∼ E^{(d−2)/2}. In d = 3, N(E) ∼ E^{1/2}, the integral converges, and
one recovers Bloch's celebrated low-temperature result that the magnetization falls as
M(0) − const·T^{3/2}. In **d = 2** the density of states is constant, N(E) ∼ E⁰, so

  ΔM(T) ∼ T ∫₀ dx/(e^x − 1) ∼ T ∫₀ dx/x ,

which **diverges logarithmically at the lower limit** (using e^x − 1 ≈ x for small x); in d = 1
it diverges even faster. Read literally, the reduction of the magnetization is infinite for any
T > 0 — the thermal population of long-wavelength magnons is unbounded because gapless quadratic
modes are "infinitely easy to excite" in low dimensions. Bloch already noted that short-range
exchange below three dimensions seemed to forbid long-range order.

**Crystalline analogues (Peierls; Landau).** A parallel and older argument concerns positional
order: in a two-dimensional crystal the mean-square displacement of an atom from its lattice site,
summed over the long-wavelength phonons (also gapless, with ω ∼ k), diverges, suggesting that
genuine long-range crystalline order cannot survive in two dimensions at finite temperature.

**Bogoliubov's rigorous machinery (early 1960s).** Two tools from many-body theory are available.
First, the concept of **quasi-averages**: in a system with a continuous symmetry, the
naive thermal average of a non-invariant operator vanishes (by symmetry, no direction is
preferred), so spontaneous order must be defined as a quasi-average — add an infinitesimal
symmetry-breaking field νH′, take the thermodynamic limit, *then* let ν → 0; the two limits need
not commute, and a nonzero result signals broken symmetry. Second, the **Bogoliubov inequality**,
a rigorous relation between two essentially arbitrary operators A, C and any Hamiltonian H, valid
in thermal equilibrium at temperature T = 1/(k_B β):

  ½ β ⟨{A, A⁺}⟩ · ⟨[[C, H], C⁺]⟩  ≥  |⟨[A, C]⟩|² .

It is proved from a positive-semidefinite scalar product on operators and the Schwarz inequality;
nothing about it assumes order. Its power is entirely in the choice of A and C.

**The superfluid precedent (Hohenberg).** Applied to a Bose system, the Bogoliubov inequality
shows there is no Bose–Einstein condensation and no conventional superfluid or superconducting
long-range order in one or two dimensions at finite temperature: with the operators built from the
Bose field operator and the density, the inequality forces the condensate fraction to vanish in
d ≤ 2. This is a demonstration that a rigorous low-dimensional no-go statement can come
out of the Bogoliubov inequality — for a particle system with U(1) symmetry.

## Baselines

**Bloch spin-wave theory (1930).** Quantizes deviations from the fully aligned state into
non-interacting magnons with E(**k**) ∼ D k², computes the magnetization via the Bose-occupied
density of states, and yields M(T) = M(0) − const·T^{3/2} in three dimensions.

**Mean-field / Weiss molecular-field theory.** Replaces the neighbor interaction by an
average internal field, giving a finite ordering temperature T_c ∝ z J (z = coordination number)
in every dimension.

**Peierls/Landau fluctuation arguments for 2D crystals.** Sum the displacement fluctuations over
gapless phonon modes and find a logarithmic divergence in two dimensions, suggesting no positional
long-range order.

**Bogoliubov-inequality applications to Bose/Fermi systems (Hohenberg line).** Uses the rigorous
inequality with operators built from the field operator and the density to bound the condensate
order parameter, excluding superfluid/superconducting order in d ≤ 2. The operators, order
parameter, and relevant commutators for a magnet with SO(3) spin-rotation symmetry and the spin
algebra are distinct from the particle-system U(1) case, and the spin algebra introduces a sum
rule that has no direct particle-number analogue.

## Evaluation settings

The natural yardstick is purely theoretical: the isotropic spin-S Heisenberg Hamiltonian on a
d-dimensional Bravais lattice (d = 1, 2, 3) with finite-range exchange J_{ij}, optionally with an
explicit ordering field b coupling to Σ_i e^{−i**K**·**R**_i} S_i^z (the wave vector **K** = 0
selecting ferromagnetic order, **K** at a zone boundary selecting staggered/antiferromagnetic
order). The figure of merit is the spontaneous magnetization (or staggered magnetization)
m = m(T, b) and its behavior in the double limit b → 0 after N → ∞, at fixed T > 0. The exact
spin facts available are textbook: the commutation relations of the spin operators, the fixed
spin length **S**_i² = ℏ² S(S+1) per site, and the finiteness of the exchange second moment
Σ_j |J_{ij}|(R_i − R_j)² for short-range couplings. A self-consistent answer should reproduce
ordinary three-dimensional
behavior (order permitted) and isolate exactly what changes at and below two dimensions, with the
result depending only on the dimension, the temperature, the spin, and the exchange range — not on
the choice of **K**, so that ferromagnetic and antiferromagnetic order are treated on the same
footing.

## Code framework

The "code" here is the analytical scaffold the derivation will be filled into: the exact algebra
that is known before the result exists, with the load-bearing inequality and the final integral
left as stubs. A small numerical routine can later make the key divergence visible. Everything
below is a known primitive (the spin algebra, the Bose factor, momentum integration) except the
two marked slots.

```python
import numpy as np
_trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy renamed trapz -> trapezoid

# --- known: the angular-momentum algebra of quantum spins ---
# [S^z, S^pm] = ± hbar S^pm ;  [S^+, S^-] = 2 hbar S^z
# S_i . S_j = (1/2)(S_i^+ S_j^- + S_i^- S_j^+) + S_i^z S_j^z

# --- known: thermodynamic averages at inverse temperature beta = 1/(kB T) ---
def bose(E, beta):
    return 1.0 / (np.expm1(beta * E))         # Bose occupation 1/(e^{beta E}-1)

# --- known: momentum-space sum -> integral over the Brillouin zone ---
# (1/N) sum_k  ->  V_cell/(2 pi)^d  int d^d k    (volume element k^{d-1} dk in d dims)
def momentum_integral(integrand, d, kmax, n=20000):
    k = np.linspace(1e-9, kmax, n)
    surface = {1: 2.0, 2: 2*np.pi, 3: 4*np.pi}[d]   # surface of unit d-sphere * appropriate factor
    return _trapz(surface * k**(d-1) * integrand(k), k)

# --- the relation that bounds the order parameter (TO DERIVE) ---
def order_parameter_bound(T, b, d, S, Jbar):
    """Upper bound on the (staggered) magnetization at temperature T and field b.
    TODO: the rigorous inequality we will derive, and the momentum integral it produces."""
    raise NotImplementedError

# --- behavior of that bound as the symmetry-breaking field is removed ---
def spontaneous_order(T, d, S, Jbar):
    """TODO: evaluate lim_{b->0} of order_parameter_bound; report whether it forces m -> 0."""
    raise NotImplementedError
```
