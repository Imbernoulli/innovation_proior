The thing that has bothered me about chemical kinetics is that we all write $k = A\,\exp(-E_a/RT)$ and act as though the job is half finished because the exponential is honest — only the small Boltzmann tail of encounters carries enough energy to clear the barrier, fine. But the pre-exponential factor $A$ is a scandal. It sits out front carrying all the units, it swings over many orders of magnitude from one reaction to the next, and nobody can compute it from the molecules themselves. Collision theory at least tried: model the reactants as hard spheres, count how often two of them hit hard enough along the line of centers, and $A$ becomes a collision frequency $\sigma\langle u_{\text{rel}}\rangle$ with $\langle u_{\text{rel}}\rangle = (8k_BT/\pi\mu)^{1/2}$. For atom-plus-atom that is roughly right. But the moment real molecules are involved the measured rate falls far below that collision frequency, and we patch the gap with a "steric factor" $p \le 1$ read off after the fact as $p = k_{\text{obs}}/k_C$, with no way to predict it. That $p$ is the whole mystery in disguise. Marcelin had the better picture — a reaction as the passage of the system's representative point across a dividing surface, with an activation free energy $k \propto \exp(-\Delta^\ddagger G/RT)$ out front — but he never wrote down the constant of proportionality. And Pelzer and Wigner, with Eyring and Polanyi's potential energy surface for $\mathrm{H}+\mathrm{H_2}$ in hand, computed the flux of a system over the col directly, but as a hard case-by-case dynamical calculation, not a closed formula. The common gap is plain: collision theory has an absolute $A$ but no molecular structure; Marcelin has the structure and the entropy but no computable constant; the trajectory calculation has the dynamics at the col but no general expression. What I want is the one thing none of them gives — a closed formula that turns "here is the potential surface, here are the molecules" into a number for $k$, using the full internal structure of the reactants and of the configuration at the pass.

I propose what I will call transition state theory (activated-complex theory). The starting observation is that on the potential surface a reaction is not a billiard collision at all; it is the representative point of the whole system climbing from the reactant valley to the col — the saddle point, a maximum along the path through the pass and a minimum across it — and sliding down into the product valley. The col is the bottleneck, so I place a dividing surface right there, perpendicular to the path, and ask for the forward flux across it. Flux is a dynamical quantity, and I cannot do general dynamics, so I make two assumptions that convert it into equilibrium statistics, which I *can* do. The first is quasi-equilibrium: the systems sitting on the dividing surface are Boltzmann-distributed, in thermal equilibrium with the reactants behind them. This is defensible because in a gas the reactants are battered by collisions and thermalize far faster than the rare reactive event, so the few systems that find themselves at the pass are drawn from the ordinary thermal population — the reaction is a slow leak off the top of an otherwise-equilibrated distribution. The second is no recrossing: a system that crosses the surface forward keeps going and becomes product, plausible because the descent into the product valley is steep and the fragments fly apart. With these, the forward flux *is* the rate and the population feeding it is the equilibrium population.

Granting that, I treat the configurations on the dividing surface as a genuine species — the activated complex — in equilibrium with the reactants: $C^\ddagger/(C_A C_B) = K^\ddagger = (F^\ddagger/(F_A F_B))\exp(-E_0/k_BT)$, where the $F$ are complete partition functions per unit volume and $E_0$ is the height of the pass above the reactants, referenced to the reactant zero-point energy. That referencing is not bookkeeping pedantry — the zero-point energy at the pass differs from that of the reactants, and that very difference is what will make $\mathrm{H}$ and $\mathrm{D}$ react at different rates. This gives the standing population of complexes, but a population is not a rate; I still need the frequency at which each complex tips over into products. Here the activated complex stops being an ordinary molecule. An ordinary molecule has $3N-6$ (or $3N-5$ if linear) bound vibrational modes, all real oscillators captured by $q_{\text{vib}}$. But this configuration sits at a saddle: along the reaction path the curvature is the wrong sign. That direction is not a vibration — it is the crossing motion itself. So of the complex's $3N$ degrees of freedom, $3N-1$ are honest bound modes (overall translation and rotation plus the remaining real vibrations) that go into $F^\ddagger$ as usual, and one — the reaction coordinate — must be handled separately as motion.

What makes the method work is that this one special degree of freedom contributes reciprocally to the population and to the crossing rate, so the way I model it does not survive into the answer. Take it first as a vibration gone soft, $\nu \to 0$, since the curvature along the path vanishes at the top of the pass. Its partition function is $q^* = \lim_{\nu\to 0} 1/(1-e^{-h\nu/k_BT})$; expanding $e^{-h\nu/k_BT}\approx 1 - h\nu/k_BT$ gives $q^* = k_BT/h\nu$. The decomposition rate of one complex is just $\nu$, the frequency at which it rattles toward the top. Factoring $F^\ddagger = q^* F^\ddagger_{\!\perp} = (k_BT/h\nu)F^\ddagger_{\!\perp}$ and forming $k = \nu \cdot K^\ddagger_{\text{full}} C_A C_B / (C_A C_B)$, the $\nu$ appears once multiplying (the attempt rate) and once dividing (inside $q^*$), and they annihilate:
$$k = \frac{k_BT}{h}\cdot\frac{F^\ddagger_{\!\perp}}{F_A F_B}\,\exp\!\left(-\frac{E_0}{k_BT}\right).$$
The fictional frequency leaves no trace, and the universal group $k_BT/h\approx 6.25\times 10^{12}\ \mathrm{s^{-1}}$ — the same one Herzfeld stumbled on in diatomic dissociation, never given general meaning — is left standing as an attempt frequency with no molecular input at all. I distrusted a result that drops out of a self-cancelling fiction, so I re-derived the crossing as honest free translation across a slab of width $\delta$ at the pass: $q_{\text{tr}} = (2\pi m^* k_BT)^{1/2}\delta/h$, with the mean forward speed $\langle v\rangle = (k_BT/2\pi m^*)^{1/2}$ carrying the crucial factor of one half (only half the complexes in the slab move forward, so the relevant quantity is the forward flux over the total population, not the mean speed of the forward half — forgetting that half would double the rate). The crossing rate is $\langle v\rangle/\delta$, and $(\langle v\rangle/\delta)\cdot q_{\text{tr}} = k_BT/h$ exactly, with both the arbitrary slab width $\delta$ and the ill-defined effective mass $m^*$ cancelling. A third route, the cleanest, writes it as a forward-momentum flux integral, $\int_0^\infty (p/m^*)\,e^{-p^2/2m^* k_BT}\,dp / [h(2\pi m^* k_BT)^{1/2}] = k_BT/[h(2\pi m^* k_BT)^{1/2}]$, where the $(2\pi m^* k_BT)^{1/2}$ cancels the one-dimensional translational density of states that the same coordinate contributed to $F^\ddagger$, again leaving $k_BT/h$. Three different fictions, one answer: the reaction coordinate's contribution to counting complexes and its contribution to crossing are reciprocal, which is why the formula never has to say what the reaction coordinate "is."

With $Q^\ddagger$ now meaning the activated-complex partition function with the reaction-coordinate mode removed, the rate constant is
$$k = \kappa\cdot\frac{k_BT}{h}\cdot\frac{Q^\ddagger}{Q_A Q_B}\cdot\exp\!\left(-\frac{E_0}{k_BT}\right),$$
where I prepend a transmission coefficient $\kappa \le 1$ (occasionally $>1$ when tunnelling helps) to absorb everything the two assumptions miss — recrossing, depletion of the pass population, tunnelling *through* the barrier (mass-sensitive and temperature-independent, $\sim\exp[-2(2m(V_0-E)/\hbar^2)^{1/2}a]$, so it matters for light atoms at low $T$), and hops between electronic surfaces — with $\kappa=1$ as the honest default. Two checks confirm the formula. First, it must reduce to collision theory where collision theory is right. For structureless atoms $A$, $B$, the reactant partition functions are pure translation and the complex is translation $\times$ rotation with no leftover vibration; substituting $q_{\text{trans}}\propto m^{3/2}$ and $q_{\text{rot}} = 8\pi^2 I k_BT/(\sigma h^2)$ with $I = \mu d^2$, the mass powers collapse to $\mu^{-1/2}$ and the temperature powers to $(k_BT)^{1/2}$, reproducing exactly $\pi d^2\langle u_{\text{rel}}\rangle\exp(-E_0/k_BT)$ with $\langle u_{\text{rel}}\rangle = (8k_BT/\pi\mu)^{1/2}$. Collision theory was the structureless special case all along. Second, the steric factor now has meaning: when real molecules meet at the pass, free rotations and relative translations they had as separated reactants get tied up into the complex, so $Q^\ddagger/(Q_A Q_B)$ is smaller than for bare fragments, and that deficit is $\exp(\Delta S^\ddagger/R) < 1$ — precisely the empirical $p$, now computable from geometries and frequencies.

The same formula has a thermodynamic face useful in the laboratory. Since $K^\ddagger = \exp(-\Delta G^\ddagger/RT)$ and $\Delta G^\ddagger = \Delta H^\ddagger - T\Delta S^\ddagger$,
$$k = \kappa\cdot\frac{k_BT}{h}\cdot\exp\!\left(-\frac{\Delta G^\ddagger}{RT}\right) = \kappa\cdot\frac{k_BT}{h}\cdot\exp\!\left(\frac{\Delta S^\ddagger}{R}\right)\exp\!\left(-\frac{\Delta H^\ddagger}{RT}\right),$$
times a standard-state factor $(c^\ominus)^{1-m}$ for molecularity $m$. Measuring $k$ over several temperatures hands back $\Delta H^\ddagger$ and $\Delta S^\ddagger$ separately. Matching to Arrhenius, the activation energy is $E_a = \Delta H^\ddagger + RT$ (unimolecular or condensed) or $\Delta H^\ddagger + 2RT$ (bimolecular gas), and the pre-exponential factor is $A = e\cdot(k_BT/h)\exp(\Delta S^\ddagger/R)(c^\ominus)^{1-m}$ or $e^2\cdot(k_BT/h)\exp(\Delta S^\ddagger/R)(c^\ominus)^{1-m}$ respectively — so the empirical $A$ is no fudge but the universal frequency $k_BT/h$ modulated by the activation entropy. And the kinetic isotope effect comes for free from the very $E_0$-referencing that $K^\ddagger$ required: since a mode's zero-point energy is $\tfrac12 h\nu$ with $\nu\propto\mu^{-1/2}$, replacing $\mathrm{H}$ by $\mathrm{D}$ lowers the (stiff) reactant zero-point energy more than the looser complex's, raising $E_0$ for $\mathrm{D}$, and if the bond is fully broken at the pass, $k_H/k_D = \exp[(E_0(D)-E_0(H))/k_BT]$.

```python
import math

kB = 1.380649e-23     # J/K
h  = 6.62607015e-34   # J s
NA = 6.02214076e23    # 1/mol
c  = 2.99792458e10    # cm/s  (wavenumber -> frequency)

def q_trans(m, T):                       # translation, per unit volume
    return (2.0 * math.pi * m * kB * T / h**2) ** 1.5

def q_rot_linear(I, T, sigma=1):         # linear rotor
    return 8.0 * math.pi**2 * I * kB * T / (sigma * h**2)

def q_vib_mode(nu, T):                    # one harmonic mode
    return 1.0 / (1.0 - math.exp(-h * nu / (kB * T)))

def crossing_frequency(T):               # universal attempt frequency kB*T/h
    return kB * T / h

def eyring_k(Q_dagger, Q_A, Q_B, E0, T, kappa=1.0):
    return kappa * crossing_frequency(T) * (Q_dagger / (Q_A * Q_B)) \
        * math.exp(-E0 / (kB * T))

def eyring_k_thermo(dG, T, kappa=1.0, c_std=None, molecularity=1):
    R = kB * NA
    k = kappa * crossing_frequency(T) * math.exp(-dG / (R * T))
    if c_std is not None:
        k *= c_std ** (1 - molecularity)
    return k

def collision_rate(mu, sigma_cs, E0, T):
    mean_u_rel = math.sqrt(8 * kB * T / (math.pi * mu))
    return sigma_cs * mean_u_rel * math.exp(-E0 / (kB * T))

if __name__ == "__main__":
    T = 300.0
    amu = 1.66053907e-27
    mH, mH2, mC = 1.008*amu, 2.016*amu, 3.024*amu   # H, H2, linear H..H..H complex
    I_H2, I_C = 4.6e-48, 2.7e-47                     # moments of inertia (kg m^2)
    nu_H2 = 4400.0 * c                               # H2 stretch
    nus_C = [w*c for w in (2000.0, 900.0, 900.0)]    # bound modes at the pass
    E0 = 9.7 * 4184.0 / NA                           # ~9.7 kcal/mol, J/molecule

    Q_A = q_trans(mH, T)
    Q_B = q_trans(mH2, T) * q_rot_linear(I_H2, T, sigma=2) * q_vib_mode(nu_H2, T)
    Q_d = q_trans(mC, T) * q_rot_linear(I_C, T, sigma=2)
    for nu in nus_C:                                 # reaction coordinate omitted
        Q_d *= q_vib_mode(nu, T)

    print("kB*T/h        =", crossing_frequency(T), "s^-1")  # ~6.25e12
    print("k (TST)       =", eyring_k(Q_d, Q_A, Q_B, E0, T))
    mu = mH*mH2/(mH+mH2)
    print("k (collision) =", collision_rate(mu, math.pi*(2.5e-10)**2, E0, T))
```
