# Context: a symmetry that hides in the vacuum, not in the laws

## Research question

The laws we write down for elementary particles are symmetric — they have to be, because symmetry is what gives us conservation laws and what tames the infinities of quantum field theory. Yet the world is full of states that are manifestly *less* symmetric than the laws that govern them. A cold magnet picks a direction; a crystal picks a lattice orientation; a superconductor behaves as though particle number were not conserved. In each case the underlying interaction is perfectly isotropic or perfectly gauge-symmetric, but the actual realized state is not.

The precise question is this: **can the equations of motion be exactly invariant under a continuous symmetry group while the lowest-energy state — the vacuum, the ground state — fails to be invariant?** And if so, *what does that cost?* Is there an observable price for hiding a symmetry in the ground state rather than in the Lagrangian?

This is not idle. By 1960 there are two sharp pressures. First, the strong interactions show approximate symmetries (isospin, and a near-symmetry under chirality) that are *almost but not exactly* respected — and the pion is suspiciously light, far lighter than any other hadron, as though it were trying to be massless. Second, the BCS theory of superconductivity has just succeeded spectacularly using a ground state that does not manifestly conserve electron number — and it is unclear whether that theory even respects gauge invariance, or whether its predictions (the Meissner effect) can be trusted. A mechanism by which a symmetric theory produces an asymmetric ground state would speak to both. What a solution must achieve: a Lagrangian one can keep exactly symmetric, a ground state that is not, and a *clear accounting* of the excitation spectrum that results — in particular, whether such a thing is even allowed in a relativistic quantum field theory.

## Background

**Mass is the curvature of the potential.** In Lagrangian field theory a free scalar field has `L = ½(∂φ)² − ½m²φ²`; the coefficient of the quadratic term *is* the mass-squared. More generally, given a potential `V(φ)`, the mass-squared of small oscillations about a configuration `φ₀` is the second derivative `V''(φ₀)` — the local curvature of the potential. A particle is massless precisely when the potential is flat in its direction: there is no restoring force, so a zero-momentum displacement does not oscillate. This single fact — *mass = curvature of V at the vacuum* — is the lever everything below turns on. A field whose potential is flat along some direction is, in that direction, a massless particle.

**The Landau order parameter and the Mexican-hat free energy.** Landau's theory of continuous phase transitions (1937) already contains the shape that matters. A continuous transition is a point where the symmetry of the body drops to a subgroup; it is described by an *order parameter* η that is zero in the symmetric phase and nonzero in the broken phase. Expanding the thermodynamic potential in powers of η, keeping only invariants, gives `Φ(η) = Φ₀ + A η² + B η⁴ + …`, with `B > 0` and `A ≈ a(T − Tc)`. Above Tc, `A > 0` and the minimum is at η = 0 (symmetric). **Below Tc, `A < 0`, and the minimum moves to `η² = −A/2B ≠ 0`.** For a *single real* order parameter this is a double well; for a *complex* order parameter (two real components) with a symmetry `η → e^{iα}η`, the potential `A|η|² + B|η|⁴` with `A<0` is a surface of revolution with a circular trough — the "Mexican hat" / wine-bottle shape, a whole *circle* of degenerate minima. Landau's framework is classical thermodynamics, and the same algebra — quadratic coefficient changing sign, minimum sliding off the origin onto a degenerate set — is the template for a field-theory vacuum.

**The Heisenberg ferromagnet and spin waves.** The exchange Hamiltonian `H = −J Σ Sᵢ·Sⱼ` is exactly rotationally invariant: it prefers neighboring spins parallel but cares nothing for the *absolute* direction. Yet the ground state is fully aligned along *some* axis — rotational symmetry is broken by the state, not the Hamiltonian. The low-lying excitations are **spin waves (magnons):** long-wavelength, slow twists of the local magnetization direction. Crucially, twisting *all* spins by the same angle costs *no* energy (it is a symmetry operation), so a spin wave of wavevector k → 0 costs energy → 0: the magnon dispersion has no gap, ω(k) → 0 as k → 0. A broken rotational symmetry comes with a *gapless* mode whose softness is forced by the symmetry. This is the concrete physical exemplar of "broken continuous symmetry ⇒ a soft mode."

**BCS, the gap, and the gauge worry.** In the BCS ground state (1957) electrons near the Fermi surface bind into zero-momentum opposite-spin Cooper pairs and condense. The quasiparticle excitation spectrum is `E_k = √(ε_k² + ε₀²)`, with an energy gap `ε₀` — there is a minimum energy `ε₀` to make a single excitation. The condensate has a definite phase, and the state does not have definite electron number: in the language of symmetries, the U(1) particle-number / electromagnetic gauge symmetry is broken by the ground state. This is the first microscopic theory built explicitly on a non-invariant ground state — and it carries an unresolved worry: the BCS ansatz and the Bogoliubov–Valatin quasiparticles (each a superposition of electron and hole, carrying no definite charge) do not manifestly respect gauge invariance, so it is not obvious the theory's electromagnetic predictions (the Meissner effect, the response to a vector potential) are even consistent. Resolving that worry — showing charge conservation survives — is an open thread.

**The analogy waiting to be drawn.** The Bogoliubov–Valatin equations for the BCS quasiparticle, `E ψ_{p,+} = ε ψ_{p,+} + Δ ψ†_{−p,−}`, `E = √(ε² + Δ²)`, have the *same algebraic form* as the Dirac equation, with the gap Δ playing the role of a mass. A self-energy that opens a gap in a superconductor looks formally like a self-energy that generates a *mass* for a fermion. If the gap arises from a symmetric interaction acting on a non-invariant ground state, then so might a particle's mass.

## Baselines

**Explicit symmetry breaking (the old way).** The standard way to describe a broken symmetry, before any of this, is to put a non-symmetric term into the equations of motion *by hand* — an external field that picks a direction, a bare mass that violates the symmetry. This works descriptively but explains nothing: the asymmetry is an input, not an output. It cannot account for *why* a symmetry that is exact in the interaction should appear broken, and it has no predictive content about the resulting spectrum. A genuine mechanism would produce the asymmetry from a symmetric Lagrangian, with *no* asymmetric input.

**Perturbation theory around `φ = 0`.** The default quantization of a scalar field expands around the field configuration `φ = 0`, treating each normal mode as a harmonic oscillator. This is correct when `φ = 0` is the minimum of the potential, i.e. when `m² > 0`. But when the quadratic coefficient is *negative* — `V = −μ²|φ|² + λ|φ|⁴` with `μ² > 0` — the point `φ = 0` is a local *maximum*, not a minimum. Expanding there gives a tachyonic `m² < 0` and the perturbation series is built on an unstable, wrong vacuum. The usual response in the literature is simply to declare that "the theory with `m² < 0` does not exist." The gap this leaves: it never asks whether a *different*, non-perturbative ground state exists when `m² < 0`, and what its excitation spectrum is.

**System-specific mean-field theories.** Weiss's molecular-field theory of magnets and van der Waals's theory of fluids each describe a particular broken-symmetry transition with a model tailored to that system. They get the qualitative behavior but offer no *general* statement: no reason why unrelated systems share critical behavior, and no transferable account of what happens to the spectrum of excitations when a continuous symmetry breaks. Landau's order-parameter expansion unifies the thermodynamics but stops at classical free energy; it does not address the *quantum field theory* question of which particles are massive and which massless.

**BCS as the one microscopic success — with the gauge thread open.** BCS is the standout: a real, quantitative, microscopic theory built on a non-invariant ground state, predicting the gap, the specific heat, and the Meissner effect. Its gap leaves: (i) the gauge-invariance / charge-conservation worry above, unresolved at the time; and (ii) no statement of *generality* — it is a theory of electrons and phonons, not a statement about what any continuous symmetry hidden in a non-invariant ground state must produce. Whether the BCS mechanism is a special trick or an instance of a universal field-theoretic phenomenon is exactly what is unsettled.

## Evaluation settings

The phenomena against which any theory of "symmetry hidden in the ground state" would be measured:

- **The ferromagnet's magnon spectrum.** Inelastic-neutron and spin-wave measurements give the magnon dispersion of a Heisenberg ferromagnet; the relevant qualitative fact is that it is *gapless*, ω → 0 as k → 0. Any candidate mechanism for broken continuous symmetry must reproduce a gapless mode here.
- **The pion and the hadron spectrum.** The pion mass (far below the nucleon and every other hadron), and the pattern of strong-interaction near-symmetries. A mechanism that ties a light/massless boson to a broken continuous symmetry must confront *why the pion is so light* relative to the scale set by the nucleon mass.
- **Weak-decay couplings of nucleon and pion** — the measured vector and axial-vector couplings `g_V`, `g_A`, the pion decay constant `g_π`, and the relation `g_π ≈ 2M g_A G` (Goldberger–Treiman, 1958) connecting pion–nucleon coupling to the axial coupling and the nucleon mass M. These constrain whether the pion behaves as the soft boson of an approximately broken (chiral) symmetry.
- **The superconducting gap and the Meissner effect.** The measured energy gap `2ε₀` (exponential low-T specific heat) and the expulsion of magnetic field (London penetration depth). A field-theoretic account of broken gauge symmetry must be consistent with these — and must show charge conservation is intact.

## Code framework

Generic scalar-field-theory bookkeeping starts from a Lagrangian with a potential: find the lowest-energy constant field configuration, expand the fields about it, and read off the masses of the small oscillations as the curvature of the potential. A symbolic skeleton:

```python
import sympy as sp

# --- Generic scalar-field-theory primitives ---------------------------------
# A scalar field theory is fixed by a potential V(fields). Masses of the small
# oscillations about a constant background are the second derivatives of V there
# (mass = curvature of the potential).

def potential(fields, params):
    """The scalar potential V as a symbolic expression in the field components.
    The specific V and its parameters are the input."""
    raise NotImplementedError  # TODO: supply V(fields; params)

def find_vacuum(V, fields):
    """Constant field configuration(s) minimizing V: solve dV/dfields = 0 and
    keep the minima. Returns the vacuum value(s)."""
    # TODO: stationary points of V, select minima, return vacuum configuration
    pass

def expand_about_vacuum(V, fields, vacuum):
    """Shift each field by its vacuum value and Taylor-expand V. The constant and
    linear terms are dropped (linear vanishes at a stationary point); the
    quadratic terms carry the spectrum."""
    # TODO: substitute fields -> vacuum + fluctuation, expand, collect quadratics
    pass

def spectrum(V_expanded, fluctuations):
    """Mass-squared of each fluctuation = its diagonal quadratic coefficient
    (the Hessian of V at the vacuum). A flat direction gives m^2 = 0."""
    # TODO: Hessian of V at the vacuum -> eigenvalues = mass^2 of each mode
    pass

# Workflow:
#   choose V, solve for the stable constant background, expand around it, and
#   diagonalize the Hessian to obtain the small-oscillation masses.
```
