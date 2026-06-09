# Context: superconductivity on the eve of a microscopic theory

## Research question

Superconductivity has resisted a microscopic explanation for nearly half a century. The phenomenology is rich and, by the mid-1950s, well measured: below a critical temperature $T_c$ a metal loses all DC resistance (Kamerlingh Onnes, 1911); it expels magnetic flux from its bulk, the Meissner–Ochsenfeld effect (1933), so it is a perfect *diamagnet*, not merely a perfect conductor; the transition at $T_c$ in zero field is second-order (no latent heat, a jump in specific heat); the electronic specific heat at low temperature falls off as $\exp(-T_0/T)$, the signature of an *energy gap* for exciting individual electrons; and the transition temperature depends on the isotopic mass of the ions, $T_c M^{1/2} \approx \text{const}$ (the isotope effect, 1950).

The precise problem is to write down, from the quantum mechanics of electrons and ions, a state of the metal that reproduces all five of these facts at once — and in particular to explain the energy gap and the second-order transition. The obstacle that has defeated every attempt is one of scale: the condensation energy that distinguishes the superconducting state from the normal state is tiny, of order $N(0)(kT_c)^2 \sim 10^{-8}$ eV per electron, while the uncertainty in the *total* energy of the interacting electron–phonon system, even in the normal state, is of order $1$ eV per electron — eight orders of magnitude larger. A solution must isolate exactly the part of the electron correlations that is peculiar to the superconducting phase and treat *that* accurately, letting the enormous common part cancel in the difference between the two phases. Any theory that computes the total energy and hopes to see superconductivity emerge as a small residue is hopeless.

## Background

The normal metal is described by the Sommerfeld–Bloch independent-particle model: conduction electrons occupy single-particle Bloch states of energy $\epsilon(\mathbf{k})$ labelled by wave vector and spin, and in the ground state every state below the Fermi energy $\mathcal{E}_F$ is filled, every state above is empty — a sharp filled Fermi sphere. Correlations from the Coulomb interaction and from the electron–lattice (phonon) interaction are left out of this model.

Landau's theory of the Fermi liquid puts the independent-particle picture on a firmer footing: as long as turning on the interactions does not produce a discontinuous change, the low-energy excitations of the fully interacting normal metal stand in one-to-one correspondence with those of a free Fermi gas — they are *quasiparticles* with a renormalized effective mass and a decay rate that vanishes at the Fermi surface. So the sharp Fermi surface survives the interactions, and it is the *residual* interaction between quasiparticles, not the bulk of the correlation energy, that must be responsible for superconductivity.

On the phenomenological side, several deep ideas are already on the table. Gorter and Casimir's two-fluid model (1934) splits the electrons into a "superfluid" fraction that grows from zero at $T_c$ to unity at $T=0$ and an interpenetrating "normal" fraction, and reproduces the thermodynamics. Fritz and Heinz London (1935) propose that the superfluid current is fixed locally by the vector potential, $\mathbf{j}_s \propto -\mathbf{A}$, from which the Meissner effect and a penetration depth $\lambda \sim 10^{-6}$ cm follow; Fritz London argues that this requires a "rigidity" of the wavefunction — that the superconducting wavefunction is barely changed by an applied field — and, in his 1950 book, that superconductivity is a quantum phenomenon on a macroscopic scale, "a kind of solidification or condensation of the average momentum distribution," from which he even predicts flux quantization. Ginzburg and Landau (1950) extend the London theory with a complex order parameter $\psi(\mathbf{r})$ and a free-energy functional $a|\psi|^2 + \tfrac{b}{2}|\psi|^4 + \dots$, excellent near $T_c$. Pippard (1953), from penetration experiments, introduces a *coherence length* $\xi_0 \sim 10^{-4}$ cm: a disturbance of the superconductor at one point influences the superfluid out to a distance $\xi_0$, and the London relation must be generalized to a non-local form. It has been shown that Pippard's non-local electrodynamics would follow from a model with an energy gap.

The decisive empirical clue to the *mechanism* is the isotope effect, found in 1950 independently by Maxwell and by Reynolds, Serin, Wright and Nesbitt: $T_c$ scales with ionic mass as $M^{-1/2}$. Since the ion mass enters only through the lattice vibrations, the lattice — the phonons — must be involved. In the same year Fröhlich independently predicts on theoretical grounds that the electron–phonon interaction drives superconductivity.

A combination of two facts sharply restricts which electrons matter. Pippard's coherence length, fed through the uncertainty principle, gives the momentum spread of the states that build the superconducting wavefunction: $\Delta x \sim \xi_0 \Rightarrow \Delta p \sim \hbar/\xi_0 \sim 10^{-4} p_F$. So only electronic states within about $10^{-4} p_F$ of the Fermi surface — about $10^{-4}$ of the electrons — are significantly involved, and they have their energies lowered by of order $kT_c$. This is consistent with the observed condensation energy: about $10^{-4}kT_c$ per electron, equivalently of order $N(0)(kT_c)^2$ in density-of-states units. So the active degrees of freedom are confined to electrons in a thin shell at the Fermi surface.

The microscopic origin of the attraction is also in hand. Eliminating the linear electron–phonon coupling by a canonical transformation (Fröhlich; then Bardeen and Pines, including Coulomb screening through the Bohm–Pines collective model) leaves a true electron–electron interaction mediated by virtual phonon exchange. Its matrix element for scattering a pair from $(\mathbf{k},\mathbf{k}')$ near the Fermi surface is

$$ \frac{2\hbar\omega_\kappa |M_\kappa|^2}{(\epsilon_\mathbf{k}-\epsilon_{\mathbf{k}+\kappa})^2 - (\hbar\omega_\kappa)^2}, $$

which is *negative* — attractive — whenever the electronic energy difference is less than the phonon energy, $|\epsilon_\mathbf{k}-\epsilon_{\mathbf{k}+\kappa}| < \hbar\omega_\kappa$. Opposed to it is the repulsive screened Coulomb interaction $\sim 4\pi e^2/(\kappa^2 + \kappa_s^2)$. The net interaction for states in the thin shell $|\epsilon| < \hbar\omega$ can be a *net attraction* when the phonon term dominates.

## Baselines

These are the microscopic attempts that exist and the precise way each falls short.

**Fröhlich's self-energy theory (1950, 1952).** Starting from the Fröhlich Hamiltonian (electrons + phonons, Coulomb dropped), a perturbation-theoretic treatment of the electron self-energy in the phonon field. It correctly gives the isotope-effect mass dependence of the critical field $H_0$, but it does *not* yield a phase with superconducting properties, and the energy difference it assigns between "normal" and "superconducting" configurations is far too large — orders of magnitude above the observed condensation energy. It is built on the self-energy of single electrons, not on the true interaction between them.

**Bardeen's variational self-energy theory (1950–51).** A variational rather than perturbative attack, but again based on the electron self-energy in the phonon field, with energy gaps at the Fermi surface arising from dynamic lattice interactions. It runs into the same difficulty: nearly all of the self-energy is already present in the normal state and changes little at the transition. Schafroth showed that, starting from the Fröhlich Hamiltonian, the Meissner effect cannot be obtained in *any* order of perturbation theory; and a diagrammatic treatment correct to order $(m/M)^{1/2}$ gives no gap and no instability. The lesson: a genuine many-body interaction *between* the electrons is required, and the effect being sought is invisible to perturbation theory in the coupling.

**Heisenberg–Koppe theory.** Based on the long-wavelength Coulomb interaction producing density fluctuations that localize a small fraction of the electrons. It does not isolate the small phonon-driven correlation that the isotope effect points to.

**Schafroth / Blatt–Butler localized-pair Bose condensation.** The idea that electrons bind into pairs which, behaving as bosons, undergo a Bose–Einstein condensation. Attractive in spirit, because a pair of fermions is a boson and Bose condensation already explains superfluid $^4$He. The gap it leaves: it pictures tightly bound, spatially *localized* pairs forming a dilute Bose gas. But the coherence length says the pairs are enormous, $\sim 10^{-4}$ cm; with $\sim 10^{-4}$ of the electrons involved, the mean spacing between condensed electrons is $\sim 10^{-6}$ cm, so within the volume of one pair lie the centers of $\sim (10^{-4}/10^{-6})^3 \sim 10^6$ other pairs. The pairs overlap massively — this is not a dilute gas of bosons, and the dilute-Bose-condensation treatment does not apply in that regime.

## Evaluation settings

The natural yardsticks are the measured properties of real superconductors. The quantities a microscopic theory would be tested against: the critical temperature $T_c$ and its isotope dependence $T_c M^{1/2}$; the electronic specific heat versus temperature, in particular its exponential decay $\exp(-T_0/T)$ at low $T$ and the size of the jump $\Delta C$ at $T_c$ on a reduced-temperature scale $T/T_c$; the energy gap $2\Delta(T)$ and its temperature dependence, measured through electromagnetic and acoustic absorption, ultrasonic attenuation, and nuclear-spin relaxation through $T_c$; the penetration depth $\lambda$ and the coherence length $\xi_0$; and the critical field $H_c(T)$ via the condensation free energy $H_c^2/8\pi = F_n - F_s$. Candidate clean materials for quantitative comparison include single crystals of weakly coupled superconductors such as tin and vanadium. Dimensionless combinations — for example the ratio of the zero-temperature gap to $kT_c$, and the specific-heat-jump ratio — are especially demanding because they cannot be adjusted by changing the overall energy scale.

## Code framework

The computational scaffold is a generic harness for electrons in a thin shell about the Fermi surface. It has the density of states at the Fermi level $N(0)$, the energy variable $\xi$ measured from $\mathcal{E}_F$, a Debye/phonon cutoff $\hbar\omega_D$ defining the shell, a root-finder, and a numerical integrator. The interaction used inside the shell, and what is built from it, are left open.

```python
import numpy as np
from scipy import integrate, optimize

# --- shared primitives ---
HBAR_OMEGA_D = 1.0        # phonon (Debye) energy cutoff defining the shell |xi| < hbar*omega_D
N0 = 0.3                  # density of states per spin per unit volume at the Fermi surface

def shell(xi):
    """States within an energy hbar*omega_D of the Fermi surface (xi measured from E_F)."""
    return np.abs(xi) < HBAR_OMEGA_D

def effective_interaction(xi, xip):
    """Net electron-electron matrix element inside the shell.
    Phonon-mediated attraction vs screened Coulomb repulsion lives here.
    # TODO: choose the shell interaction.
    """
    pass

# --- open slot ---

def ground_state(N0, hbar_omega_D, V):
    """Characterize the ground state of the electrons in the shell under the
    effective interaction, and any energy scale it sets.
    # TODO: fill in.
    """
    pass
```
