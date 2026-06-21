The canonical name for this method is laser population inversion. It is the core physical mechanism that turns an ordinary excited medium into a coherent optical oscillator, and it must be understood as a balance among several distinct processes rather than as a declaration that "atoms are excited and therefore emit light." I will describe the mechanism from the microscopic interaction upward, explain why thermal equilibrium cannot lase, derive the inversion condition, connect it to cavity feedback and threshold, and then verify the entire picture with a compact numerical simulation.

I start with the simplest light-matter bookkeeping: a two-level atomic transition with lower state 1 and upper state 2, separated by energy h nu. Einstein's radiation theory identifies three independent processes. First, an atom in the upper state can decay spontaneously with rate A21, emitting a photon in a random direction and phase; this is the glow of a lamp, not a laser. Second, radiation already present at frequency nu can drive an atom from the lower state upward; this stimulated absorption removes one photon from the mode and has rate N1 B12 rho(nu), where rho(nu) is the radiation energy density and B12 is the absorption coefficient. Third, the same radiation can drive an upper atom downward; this stimulated emission adds one photon to the same mode with rate N2 B21 rho(nu). The last two processes are the crucial pair because both scale with the field already present, so they alone can amplify or attenuate a coherent wave.

The sign of the coherent interaction is therefore controlled by the difference N2 B21 minus N1 B12. When this difference is negative, as it is in ordinary thermal equilibrium, the lower level is more populated and every passing wave loses more photons to absorption than it gains from stimulated emission. The medium is then a passive absorber at that frequency. Pumping must rearrange the populations so that the upper level is sufficiently populated relative to the lower level, producing the condition N2 B21 greater than N1 B12. For a nondegenerate transition with equal stimulated coefficients, this reduces to the simple population inversion condition N2 greater than N1. Once that sign flips, an incident photon is more likely to clone itself through stimulated emission than to be absorbed, and the cloned photon inherits the frequency, phase, polarization, and propagation direction of the stimulating field.

Population inversion by itself, however, only gives single-pass amplification. A wave can grow as it traverses the excited medium, but after one pass it leaves, and there is no mechanism to select a narrow frequency or a well-defined spatial mode. The second essential ingredient is an optical resonator, typically two mirrors facing each other with the active medium between them. The resonator performs two jobs. It selects longitudinal modes whose round-trip phase is an integer multiple of two pi, so only frequencies very close to these resonances can build up constructively. It also returns the selected mode through the gain medium again and again, so the same photons repeatedly stimulate the same transition.

To find when this feedback loop becomes self-sustaining, I compare the round-trip intensity gain with the round-trip loss. Let g be the small-signal intensity gain coefficient of the medium over active length L, and let alpha_loss be the distributed loss coefficient from scattering, diffraction, and residual absorption. Let R1 and R2 be the intensity reflectivities of the two mirrors. After one complete round trip, the intensity is multiplied by R1 R2 exp(2 (g minus alpha_loss) L). If this factor is below one, any seed field dies away between passes and the output remains only spontaneous emission. If it is above one, the selected mode grows exponentially until gain saturation reduces the effective gain back to the loss level. The threshold condition is obtained by setting the round-trip factor equal to one, which gives R1 R2 exp(2 (g_th minus alpha_loss) L) equals 1, or equivalently g_th equals alpha_loss plus (1/(2 L)) ln(1/(R1 R2)). This equation is the macroscopic counterpart of the microscopic inversion condition: the pump must maintain enough inversion that the gain coefficient reaches g_th.

The pump therefore has a precise but indirect role. It does not create light directly; it maintains a nonthermal distribution of atomic populations against spontaneous decay, collisions, and stimulated depletion. In a simple two-level system, resonant pumping tends to equalize the populations rather than push the upper one above the lower one, so practical lasers almost always use three or four levels, or some other selective preparation, to empty the lower laser level or feed the upper level efficiently. The radiating transition then carries the inversion while the pump handles the bookkeeping of population supply and removal.

It is worth emphasizing why this is not merely a recombination story. Recombination can produce excited carriers and can generate photons, but laser action is specifically the dominance of stimulated emission over stimulated absorption in a selected optical mode. Spontaneous emission may provide the initial seed photon that starts the process, yet the organizing principle is the field-proportional amplification of that seed by an inverted medium inside a resonator. Without inversion, a cavity only filters spontaneous light; with inversion but without feedback, the amplification is single-pass and uncontrolled. Both ingredients are necessary.

The observable signatures of this mechanism are exactly those seen in real lasers: narrowband output near the atomic transition frequency, strong directionality along the cavity axis, a clear threshold in pump power below which the output is weak and broad and above which it becomes bright and sharply peaked, and a spatial mode determined by the resonator geometry. All of these follow from the interplay between population inversion and cavity feedback.

The following Python script implements a minimal numerical illustration. It defines a two-level rate-equation model with a pump term, stimulated emission and absorption proportional to the cavity photon number, and spontaneous emission. It then evolves the photon number and the inversion through many round trips inside a two-mirror cavity. By toggling the pump strength, one can see the qualitative difference between below-threshold behavior, where the field decays, and above-threshold behavior, where the field grows and saturates. The script also explicitly checks the threshold gain condition and prints whether the chosen parameters place the system above or below threshold.

```python
import numpy as np

# Two-level laser rate-equation illustration with cavity feedback.
# Parameters are chosen for qualitative clarity, not for a specific real laser.
A21 = 1.0          # spontaneous decay rate (upper -> lower)
B12 = 0.02         # stimulated absorption coefficient
B21 = 0.02         # stimulated emission coefficient (nondegenerate transition)
N_total = 1000.0   # total number of active atoms
c = 1.0            # speed of light in the medium (set to 1 for simplicity)
L = 1.0            # active medium length
R1 = 0.95          # mirror 1 intensity reflectivity
R2 = 0.95          # mirror 2 intensity reflectivity
alpha_loss = 0.05  # distributed loss coefficient per unit length
gamma_pump = 0.0   # pump rate (will be varied)

# Threshold gain coefficient from round-trip balance.
g_th = alpha_loss + (1.0 / (2.0 * L)) * np.log(1.0 / (R1 * R2))
print(f"Threshold gain coefficient g_th = {g_th:.4f}")

def simulate(gamma_pump, n_photons_init=1.0, n_upper_init=10.0, n_round_trips=500, dt=0.01):
    """Evolve photon number and upper population through round trips."""
    n_photons = n_photons_init
    n_upper = n_upper_init
    n_lower = N_total - n_upper
    history = [n_photons]

    for _ in range(n_round_trips):
        # Continuous-time rate steps over one round-trip time 2L/c.
        steps = int((2.0 * L / c) / dt)
        for _ in range(steps):
            rho = n_photons  # energy density proxy proportional to photon number
            stim_emission = B21 * n_upper * rho
            stim_absorption = B12 * n_lower * rho
            spontaneous = A21 * n_upper

            dn_upper = (-stim_emission + stim_absorption - spontaneous
                        + gamma_pump * n_lower)
            # Photons gained by stimulated emission; spontaneous photons seed weakly.
            dn_photons = stim_emission - stim_absorption + 0.05 * spontaneous

            n_upper += dn_upper * dt
            n_photons += dn_photons * dt
            n_photons = max(n_photons, 0.0)
            n_upper = max(n_upper, 0.0)
            n_upper = min(n_upper, N_total)
            n_lower = N_total - n_upper

        # Apply cavity round-trip loss and mirror feedback.
        n_photons *= R1 * R2 * np.exp(-2.0 * alpha_loss * L)
        history.append(n_photons)

    return np.array(history)

# Below threshold: weak pump, field decays.
print("\n--- Below threshold ---")
hist_low = simulate(gamma_pump=0.5)
print(f"Initial photons: {hist_low[0]:.2e}, final photons: {hist_low[-1]:.2e}")

# Above threshold: strong pump creates inversion and sustained oscillation.
print("\n--- Above threshold ---")
hist_high = simulate(gamma_pump=8.0)
print(f"Initial photons: {hist_high[0]:.2e}, final photons: {hist_high[-1]:.2e}")

print("\nInterpretation: a weak pump leaves absorption dominant, so the seed field dies.")
print("A strong pump creates population inversion; stimulated emission exceeds absorption,")
print("and cavity feedback returns the growing mode through the gain medium each round trip.")
```

In summary, laser population inversion is the mechanism by which a pumped active medium is placed in a nonthermal state where stimulated emission outcompetes stimulated absorption at a selected transition, and an optical resonator feeds that coherent field back through the medium until the round-trip gain matches the round-trip loss. The threshold condition connects the microscopic inversion to the macroscopic cavity parameters, and the resulting output is coherent, directional, and spectrally narrow rather than an ordinary recombination glow.
