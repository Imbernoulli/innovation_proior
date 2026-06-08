# Context — a general theory of continuous phase transitions

## Research question

By the mid-1930s a whole class of phase transitions sits without any general theory. These are the transitions that proceed *without latent heat* — the body's state changes continuously through the transition point, yet at that point something abrupt happens to its response functions. The specific heat, for instance, jumps or spikes. The prototypes are the Curie point of a ferromagnet (magnetization appearing as the metal is cooled through a critical temperature), the order–disorder transition in alloys such as β-brass, the transition in ammonium chloride, ferroelectric transitions, the λ-transition in liquid helium discovered in 1932, and the symmetry changes between crystal modifications.

The precise question: is there a *single* theory — independent of the microscopic details of magnets versus alloys versus crystals — that says what all these continuous transitions have in common, predicts how the relevant quantities behave as the transition is approached, and explains *why* a continuous transition must look the way it does (a continuously vanishing something, a jump in the specific heat) rather than being merely catalogued after the fact? A solution would have to identify the right variable to describe the transition, fix what controls its appearance, and yield the near-critical behaviour of measurable quantities from that structure alone.

## Background

**Phases, transitions, and what "continuous" means.** A phase transition is a change from one behaviour of matter to another. The familiar first-order transitions — melting, boiling — involve a discontinuous jump in a thermodynamic variable (volume, entropy) and absorb or release latent heat; the specific heat diverges at the transition because a finite amount of heat enters at fixed temperature. The continuous transitions are different: no latent heat, no jump in volume or entropy, and instead a *finite* anomaly in a second derivative of the free energy.

**Ehrenfest's classification (1933).** Following the 1932 discovery of the helium λ-transition (Keesom and coworkers, Leiden, who saw the specific-heat curve spike into a λ-shape), Ehrenfest organised transitions by *which derivative of the free energy first becomes discontinuous*. If the first derivatives of the free energy (entropy, volume) jump, the transition is "first order" and carries latent heat. If the first derivatives are continuous but the second derivatives (specific heat, compressibility, expansivity) jump, it is "second order". This is a clean taxonomy, but it is only descriptive: it labels a transition after you have measured it. It does not say what variable governs the transition, why the second derivative should jump, or what small quantity controls the approach to the critical temperature.

**Weiss molecular-field theory of ferromagnetism (1907).** Pierre Weiss explained spontaneous magnetization by postulating that each atomic moment feels, besides the external field H, an internal "molecular field" proportional to the magnetization itself — an effective field H + λM. Self-consistency, M = (moment) · tanh[(H + λM)/kT], then has a nonzero solution for M below a critical temperature even at H = 0: the magnet orders spontaneously. Above the critical temperature the susceptibility follows the Curie–Weiss law χ = C/(T − θ), diverging as T approaches θ from above (a sharpening of Curie's earlier χ = C/T for a paramagnet). This is a genuine theory of a continuous transition — it produces an order appearing continuously and a diverging response — but it is wedded to the magnetic problem and to a specific microscopic ansatz (the mean field replacing the true interaction). It says nothing about alloys, crystals, or helium.

**Van der Waals and the liquid–gas critical point (1873).** The van der Waals equation of state, with its excluded-volume and attraction corrections, has a critical point at which the liquid–gas distinction disappears. Approached along the coexistence curve, the density difference between liquid and gas vanishes; the theory predicts it vanishes as (T_c − T)^{1/2}, and the law of corresponding states makes the behaviour near the critical point look universal across fluids. This is the one continuous transition that was thoroughly understood — but again it is a specific model of a specific system. A key qualitative fact comes with it: because both liquid and gas are isotropic, *they have the same symmetry*, which is exactly why one can pass continuously from liquid to gas by going around the critical point, with no sharp transition at all.

**The common thread nobody had drawn.** Weiss (magnets) and van der Waals (fluids) are both, in modern language, *mean-field* theories: each replaces the real fluctuating environment of a degree of freedom by an averaged effective field. Each independently produces a continuously appearing order and a power-law approach to the critical point — and, strikingly, the *same* powers (the density difference and the magnetization both vanish like (T_c − T)^{1/2}; both susceptibilities diverge like (T − T_c)^{−1}). That two unrelated systems share the same near-critical behaviour is a clue that the behaviour is not in the microscopic details. But no model-independent statement of this existed; each result was derived inside its own model.

**The decisive observation about symmetry.** Liquid and gas share a symmetry, so their transition can be a continuous *crossover* with a skirtable critical point. A crystal, by contrast, either has a given element of symmetry or it does not — there is no intermediate amount of a symmetry. So a transition that *changes the symmetry* of a body (crystal-to-crystal, disordered-to-ordered) cannot be skirted: at the transition point the symmetry must change abruptly even though the state changes continuously. Symmetry groups are discrete, and that discreteness is the lever the whole theory will turn on.

## Baselines

**Ehrenfest classification.** Core idea: sort transitions by the order of the lowest discontinuous free-energy derivative. Math: first order ⇒ jump in S = −∂G/∂T and V = ∂G/∂p (latent heat L = T ΔS ≠ 0); second order ⇒ S, V continuous but C_p = −T ∂²G/∂T², κ, α discontinuous. Gap: purely descriptive. It supplies no governing variable, no prediction of the size or even the existence of the jump, and no notion of a small parameter near the critical temperature. It cannot tell you *why* helium's specific heat spikes or *why* a ferromagnet orders.

**Weiss molecular-field theory.** Core idea: replace interactions by a self-consistent internal field H + λM. Math: M = M_sat · tanh[μ(H + λM)/kT]; expanding for small M at H = 0 gives M(λμ²/kT − 1) ≈ (cubic), so a nonzero M appears below T_c = λμ²/k, and above T_c, χ = C/(T − T_c). Gap: it is a magnet-specific microscopic model. The "order" is the magnetization; there is no abstraction that would carry the same machinery to a binary alloy or a structural transition, and the result is tied to the particular tanh self-consistency rather than to a general principle.

**Van der Waals theory of fluids.** Core idea: (p + a/v²)(v − b) = RT, an effective single-particle equation of state. Math: at the critical point ∂p/∂v = ∂²p/∂v² = 0; expanding about it gives (ρ_l − ρ_g) ∝ (T_c − T)^{1/2}, an isothermal compressibility diverging as (T − T_c)^{−1}, and a critical isotherm p − p_c ∝ (ρ − ρ_c)³. Gap: a one-system model. It happens to produce exactly the same exponents as Weiss, but the coincidence is left unexplained, and the framework does not transfer to symmetry-changing transitions where there is no obvious "equation of state".

**The microscopic-model route in general.** One could imagine attacking each continuous transition by building and solving its own statistical-mechanical model (a lattice of spins, a lattice gas, an order–disorder alloy model). Math: write the partition function and find where its free energy is non-analytic. Gap: intractable in general (the relevant lattice models had no exact solution at the time), and even when solvable it gives one system at a time, never the common structure. What is missing is a theory that throws the microscopic detail away on purpose and keeps only what is shared.

## Evaluation settings

The natural objects against which any such theory would be checked are the measured signatures of continuous transitions, all available before the theory:

- **Ferromagnets (Fe, Ni, Co)** at their Curie points: spontaneous magnetization M(T) appearing below T_c, and the susceptibility χ(T) above T_c (the Curie–Weiss regime). Metric: the temperature dependence (exponent) of M and χ near T_c.
- **Order–disorder alloys** such as β-brass (CuZn): the long-range order parameter as a function of temperature through the critical point, and the specific-heat anomaly.
- **Structural / ferroelectric transitions** (e.g. in ammonium chloride, Rochelle-salt-type ferroelectrics): the symmetry change between crystal modifications and the associated specific-heat behaviour.
- **Liquid helium λ-transition**: the λ-shaped specific-heat curve at T_λ.
- **Liquid–gas critical point**: density difference and compressibility along the coexistence curve — the one case where a continuous-transition theory (van der Waals) already had quantitative grip.

The yardsticks are thermodynamic: the temperature dependence of the order quantity, the form (jump vs. spike) and magnitude of the specific-heat anomaly, and the divergence of the susceptibility/compressibility — all measured as the critical temperature is approached.

## Code framework

To keep the thermodynamic scaffold concrete, the symbolic slot below starts with ordinary calculus and the relations between a free energy and its derivatives. What remains blank is the *form of the free energy as a function of the ordering variable* — that form is the entire content to be discovered.

```python
import sympy as sp

T, Tc = sp.symbols('T T_c', positive=True)
eta, h = sp.symbols('eta h', real=True)     # ordering variable; conjugate field

# --- Thermodynamic primitives that already exist ---
def entropy(Phi):           # S = -dPhi/dT
    return -sp.diff(Phi, T)

def specific_heat(Phi):     # C = -T d^2 Phi / dT^2
    return -T * sp.diff(Phi, T, 2)

def susceptibility(eta_eq): # chi = d(eta_eq)/dh at h->0
    return sp.diff(eta_eq, h)

# --- The slot the theory must fill ---
def free_energy(eta, T, h):
    """The thermodynamic potential as a function of the ordering variable.
    Its functional form near the transition is exactly what is unknown.
    # TODO: what variable plays the role of `eta`, and what terms in it are allowed?
    """
    raise NotImplementedError

def equilibrium_order_parameter(T, h):
    """Minimise free_energy over eta -> eta_eq(T, h).
    # TODO: solve dPhi/deta = 0 and select the stable branch.
    """
    raise NotImplementedError

def transition_temperature():
    """The locus in (p,T) where the ordered phase first appears.
    # TODO: what condition on the free-energy coefficients marks it?
    """
    raise NotImplementedError
```
