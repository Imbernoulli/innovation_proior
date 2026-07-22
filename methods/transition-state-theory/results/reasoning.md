The thing that keeps nagging me is the pre-exponential factor. Everyone writes k = A exp(−E_a/RT) and pretends the job is half done because we understand the exponential — only the high-energy tail of the Boltzmann distribution can get over the barrier, fine, that part is honest. But A is a scandal. It sits in front carrying all the units, it swings over many orders of magnitude from one reaction to the next, and nobody can compute it from the molecules. Collision theory at least tried: count how often two hard spheres hit each other hard enough, A becomes a collision frequency σ⟨u_rel⟩, and for atom-plus-atom that's roughly right. But then for almost everything with real molecules in it, the measured rate comes in far below that collision frequency, and we paper over the gap with a "steric factor" p ≤ 1 that we just read off as p = k_obs/k_C. That p is the whole mystery wearing a disguise. I want to compute it. And the only thing I can think of that knows about the *shape* of molecules — their rotations, their vibrations, the way atoms are arranged when a bond is half-broken — is statistical mechanics. Collision theory threw all of that away the moment it called a molecule a sphere.

So let me start from what I can actually compute and see how close it gets me to a rate. Equilibrium I can do. For any chemical equilibrium I write the chemical potential of each species as μ = −k_B T ln(q/N), set the reactant and product potentials equal, and out drops the equilibrium constant as a ratio of partition functions, K = (q_P q_Q …)/(q_A q_B …), with all the q referenced to a common zero of energy. The partition functions themselves I have in closed form: q_trans = (2π m k_B T/h²)^{3/2}, q_rot = 8π² I k_B T/(σ h²) for a linear rotor, q_vib = 1/(1−e^{−hν/k_B T}) per mode. So *equilibrium* is solved — I can get the relative populations of any two molecular species from their masses, geometries, and frequencies. The trouble is that equilibrium is silent about speed. K tells me the ratio of products to reactants at the end; it tells me nothing about how fast we get there.

But hold on. What *is* a rate, physically, on the surface I've been handed? Eyring and Polanyi built the energy as a function of the internuclear distances, and the landscape has two valleys — reactants, products — with a col between them, a mountain pass. Pelzer and Wigner stared at exactly this and said: watch the system go over the col, and the rate is the flux through the pass. That's the picture. A reaction is not a "collision" in the billiard sense; it is the representative point of the whole system climbing to the pass and sliding down the other side. The rate is how many representative points cross the dividing surface at the top, per unit time, in the forward direction. Marcelin already had this instinct — a dividing surface in phase space, an activation free energy in front — but he never wrote down the constant of proportionality. The pass is the bottleneck. So let me put a dividing surface right at the col, perpendicular to the path through it, and ask for the flux across it.

Now the genuinely hard thing about a flux is that flux is a dynamical quantity — it wants the full nonequilibrium trajectory of how systems pile up at the pass and pour over. I can't do general dynamics. So let me see what I'd have to assume to turn this into something I *can* do, which is equilibrium statistics. Here's the gamble: suppose the population of systems sitting right at the dividing surface is the same as it would be if everything were in thermal equilibrium — that the systems on the pass are Boltzmann-distributed, in equilibrium with the reactants behind them. Is that crazy? In a gas at decent pressure the reactants are getting battered by collisions constantly, thermalizing far faster than the rare event of actually reacting; so the molecules that happen to find themselves up at the pass are drawn from the ordinary thermal distribution. The reaction is a slow leak off the top of an otherwise-equilibrated population. If I also assume that once a system crosses the pass moving forward it keeps going and doesn't come sloshing back — no recrossing, which is plausible because the descent into the product valley is steep and the products fly apart — then the forward flux across the surface *is* the reaction rate, and the population feeding it is the equilibrium population. That converts the dynamics into statistics. I'll have to remember I made two assumptions there and account later for the fact that they can fail; for now, run with them.

So I treat the configurations sitting on that dividing surface as a *species*. Call it the activated complex — it's a particular arrangement of all the atoms, the one at the pass. And by the equilibrium gamble it's in equilibrium with the reactants:

    C‡ / (C_A C_B) = K‡ = (F‡ / (F_A F_B)) exp(−E₀/k_B T),

where F are the complete partition functions per unit volume and E₀ is the height of the pass above the reactants, referenced to the reactant zero-point energy (I must reference everything to one zero of energy or the K is meaningless — and that reference is not a triviality, because the zero-point energy at the pass differs from that of the reactants, and that difference is exactly what will make H and D react at different rates later). Good. This gives me how many activated complexes are sitting up there at any instant, for given reactant concentrations. It's just an equilibrium calculation, which I can do.

But a concentration of complexes is not a rate. I have the *population* on the pass; I still need the *frequency* with which each one tips over into products. Rate = (number of complexes in transit) × (rate at which one of them crosses). And here is where the activated complex stops being an ordinary molecule. An ordinary molecule has 3N − 6 bound vibrational modes (3N − 5 if linear), all of them little oscillators with real frequencies, all of them captured by q_vib. But this configuration sits at a *saddle*: a maximum along the path through the pass and a minimum across it. Along the direction of the path the curvature of the energy is the wrong sign — it's a hilltop, not a valley. That direction is not a vibration at all. It's the very motion that carries the system over the pass. So of the configuration's 3N degrees of freedom, one of them — the reaction coordinate — is special: it is not a bound mode, it is the crossing motion itself. The other 3N − 1 are ordinary bound degrees of freedom — the overall translation and rotation of the complex plus its remaining real vibrations — that I can stuff into F‡ as usual. The crossing one I have to handle separately, as motion, not as an oscillator.

Let me try the most literal thing first: treat the crossing degree of freedom as if it were still a vibration, but a very loose one — a mode whose restoring force has gone soft, frequency ν → 0, because at the very top of the pass the curvature along the path is heading to zero (and beyond, to negative). Its vibrational partition function in that limit is

    q* = lim_{ν→0} 1/(1 − e^{−hν/k_B T}).

Expand the exponential for small hν/k_B T: e^{−hν/k_B T} ≈ 1 − hν/k_B T, so 1 − e^{−hν/k_B T} ≈ hν/k_B T, and

    q* = k_B T / hν.

Interesting — there's k_B T/h appearing again, the same group Herzfeld stumbled on years ago when he was doing diatomic dissociation. It dangled there with no general meaning. Let me see if it earns one here. If I factor this loose mode out of the full F‡, I can write F‡ = q* · F‡_⧧ = (k_B T/hν) F‡_⧧, where F‡_⧧ is the partition function of the complex with the reaction-coordinate mode *removed*, i.e. only the 3N − 1 honest bound modes left.

Now the crossing rate. If I'm modeling that soft mode as a vibration of frequency ν, then ν *is* the frequency at which the complex rattles in the reaction-coordinate direction — which is to say, the frequency at which it attempts to go over the top. So the rate at which one complex decomposes into products is just ν (times, eventually, the fraction of attempts that succeed, but take that as 1 for now). So:

    rate = ν · C‡ = ν · K‡_full · C_A C_B,

where K‡_full uses the *full* F‡ including the soft mode. Pull it apart:

    k = ν · (F‡/(F_A F_B)) exp(−E₀/k_B T) = ν · (k_B T/hν) · (F‡_⧧/(F_A F_B)) exp(−E₀/k_B T).

The ν cancels. It just cancels — the frequency I picked for the soft mode appears once in the attempt rate (multiplying) and once inside q* = k_B T/hν (dividing), and they annihilate. What's left is

    k = (k_B T/h) · (F‡_⧧/(F_A F_B)) exp(−E₀/k_B T).

That cancellation is the whole game. It means the answer does not depend on the frequency I assigned to the unbound mode — which is a relief, because that frequency was fictional; the mode isn't a real oscillator, I invented ν as a crutch. The crutch leaves no trace. And k_B T/h is left standing alone as a *universal* attempt frequency, the same ≈ 6.2 × 10¹² s⁻¹ for every reaction, with no molecular input at all. Herzfeld's group has just become a law.

I'm suspicious of a result that drops out of a fictional frequency cancelling itself, though. Let me re-derive the crossing the honest way, as translation across the pass, with no pretend vibration, and check I get the same thing. Picture the dividing surface as having a small thickness δ along the reaction coordinate — a thin slab at the top of the pass, and I'll call any complex inside it "in transit." The motion along that one coordinate is just one-dimensional free translation of a particle of effective mass m* (the reduced mass for the reaction-coordinate motion). One-dimensional translation in a length δ has partition function

    q_tr = (2π m* k_B T)^{1/2} δ / h.

That replaces the soft-vibration q* as the partition function of the crossing degree of freedom; so now F‡ = q_tr · F‡_⧧. How fast does a complex leave the slab? It has some forward velocity v along the coordinate and leaves in time δ/v, i.e. at rate v/δ. I should average over the thermal distribution of velocities. My first instinct is to take the mean speed of the forward-moving complexes, ∫₀^∞ v e^{−m*v²/2k_B T} dv / ∫₀^∞ e^{−m*v²/2k_B T} dv — numerator (k_B T/m*), denominator (π k_B T/2m*)^{1/2}, giving (2 k_B T/π m*)^{1/2}.

Hm — wait, that's not the quantity I want. The slab holds *all* complexes, forward- and backward-moving, but the population q_tr I'm about to multiply by counts both halves too; so the crossing rate per complex-in-the-slab is not the mean speed of the forward half, it's the forward *flux* divided by the total population, which carries an extra factor of one half (only half are heading out the front, and they leave at their own speeds). So the right factor is ⟨v⟩ = ½·(2 k_B T/π m*)^{1/2} = (k_B T/2π m*)^{1/2}. Forgetting that half would double the rate. With the flux factor correct, the crossing rate of one complex is ⟨v⟩/δ, and the rate constant is

    k = (⟨v⟩/δ) · (F‡/(F_A F_B)) exp(−E₀/k_B T) = (⟨v⟩/δ) · q_tr · (F‡_⧧/(F_A F_B)) exp(−E₀/k_B T).

Plug in q_tr = (2π m* k_B T)^{1/2} δ/h and the mean forward speed ⟨v⟩ = (k_B T/2π m*)^{1/2}:

    (⟨v⟩/δ) · q_tr = (k_B T/2π m*)^{1/2} · (1/δ) · (2π m* k_B T)^{1/2} δ / h
                   = (k_B T/2π m*)^{1/2} (2π m* k_B T)^{1/2} / h.

Multiply the two square roots: (k_B T/2π m*)·(2π m* k_B T) = (k_B T)², so the product of the roots is k_B T, and

    (⟨v⟩/δ) · q_tr = k_B T / h.

The same universal factor — and look what cancelled: the slab thickness δ (it was arbitrary; it had *better* cancel, since the location and width of my dividing surface is a bookkeeping choice and the physics can't depend on it) and the effective mass m* (which I never had a clean way to define anyway). Both crutches gone. So

    k = (k_B T/h) · (F‡_⧧/(F_A F_B)) exp(−E₀/k_B T),

identical to the loose-vibration route. Two completely different fictions — a soft oscillator on one hand, a thin slab of free translation on the other — give the same k_B T/h. That's the kind of agreement that tells me the factor is real and the fictions were just scaffolding.

Let me make sure I can state the crossing as a clean flux integral, because that's the form that doesn't hide any of the averaging. I want the concentration of complexes that have their reaction-coordinate momentum p in [p, p+dp] sitting per unit length along the coordinate; then I multiply by the velocity p/m* (that's how fast such a complex leaves), and I sum only over forward momenta p = 0 to ∞ (backward-movers aren't crossing forward). The momentum part of the canonical density for the one coordinate is e^{−p²/2m* k_B T}, and the one-dimensional phase-space measure is dp/h (one quantum state per h of phase-space area), so the concentration per unit length with momentum in [p,p+dp] is C‡_⧧ · e^{−p²/2m* k_B T} dp / [h · Z₁], where Z₁ = ∫_{−∞}^∞ e^{−p²/2m* k_B T} dp = (2π m* k_B T)^{1/2} is the momentum normalization and C‡_⧧ is the concentration of complexes with the reaction coordinate handled by F‡_⧧. Then

    rate per complex-density = ∫₀^∞ (p/m*) · e^{−p²/2m* k_B T} dp / [h · (2π m* k_B T)^{1/2}].

Do the integral: ∫₀^∞ (p/m*) e^{−p²/2m* k_B T} dp = (1/m*) · [m* k_B T] = k_B T (substitute u = p²/2m* k_B T, the integral of p e^{−p²/2m* k_B T} dp is m* k_B T). So the whole thing is

    k_B T / [h (2π m* k_B T)^{1/2}],

and when I reassemble C‡_⧧ from the full equilibrium constant, the factor (2π m* k_B T)^{1/2} that the reaction-coordinate translation contributed to F‡ cancels the (2π m* k_B T)^{1/2} in the denominator here, leaving — again — k_B T/h out front. Three ways in, one answer. The reaction-coordinate degree of freedom contributes a one-dimensional translational density of states (2π m* k_B T)^{1/2}/h to the equilibrium population, and the forward flux divides exactly that back out and replaces it with the mean forward momentum flux k_B T, net result k_B T/h. The mass m* is along for the ride and never survives. This is why the formula can be written without ever specifying what the reaction coordinate "is" in detail: whatever it is, its contribution to the count of complexes and its contribution to the crossing rate are reciprocal.

So the rate constant is

    k = (k_B T/h) · (F‡/(F_A F_B)) · exp(−E₀/k_B T),

with the understanding that F‡ now means the partition function of the activated complex *with the reaction coordinate omitted* — 3N − 1 degrees of freedom, all bound. I'll keep writing F‡ for it from here, since the omission is part of the definition of the activated complex's partition function.

Now I owe an honest accounting of the two assumptions I leaned on. The no-recrossing one and the equilibrium-population one can both fail: a system might cross the pass and slosh back, or the population at the pass might be depleted below equilibrium if reaction is fast compared to thermalization, or the system might tunnel *through* the barrier rather than go over (matters for light atoms at low T, since the tunnelling probability exp[−2(2m(V₀−E)/ħ²)^{1/2} a] is mass-sensitive and temperature-independent), or it might hop between electronic surfaces. Every one of these is a multiplicative correction to the count of "crossings that actually become products." So I'll prepend a transmission coefficient κ ≤ 1 (occasionally > 1 when tunnelling helps) that absorbs all of it, and set κ = 1 as the default honest baseline:

    k = κ · (k_B T/h) · (F‡/(F_A F_B)) · exp(−E₀/k_B T).

Let me now check that this thing actually reduces to collision theory in the case collision theory is supposed to be right — atom plus atom — because if it doesn't, I've made an error somewhere. Take A and B to be structureless atoms; the activated complex is then a diatomic-like configuration. F_A and F_B are pure translation, F_A = q_trans(m_A), F_B = q_trans(m_B). The complex, with its reaction coordinate removed, has translation of the whole thing plus rotation (a two-atom-like object can rotate) and no leftover vibration once the one bond-breaking mode is the reaction coordinate: F‡ = q_trans(M) · q_rot. Write out the translational ratio: q_trans(M)/(q_trans(m_A) q_trans(m_B)) where M = m_A + m_B. Each q_trans ∝ (mass)^{3/2}, so the ratio is (M^{3/2})/((m_A m_B)^{3/2}) × (h²/2π k_B T)^{3/2} = (1/μ)^{3/2}(h²/2π k_B T)^{3/2} with μ = m_A m_B/M the reduced mass (since M/(m_A m_B) = 1/μ). And q_rot = 8π² I k_B T/(σ h²) with I = μ d² for a separation d at the pass. Assemble k = κ(k_B T/h)·(1/μ)^{3/2}(h²/2π k_B T)^{3/2}·(8π² μ d² k_B T/σ h²)·exp(−E₀/k_B T). Collect the powers: the (1/μ)^{3/2} times μ gives μ^{−1/2}; the (h²/2π k_B T)^{3/2} times (k_B T/h²) times the leading k_B T/h, with the 8π²/σ, all collapse — tracking the k_B T powers: (k_B T)^{1}·(k_B T)^{−3/2}·(k_B T)^{1} = (k_B T)^{1/2}, and h powers: h^{−1}·h^{3}·h^{−2} = h^{0}. So out front I get something ∝ (k_B T)^{1/2} μ^{−1/2} times a length² — exactly the structure of σ⟨u_rel⟩ with ⟨u_rel⟩ = (8 k_B T/π μ)^{1/2} and σ = π d². It lands on the collision-theory rate. So when the molecules have no internal structure, my formula *is* collision theory; collision theory was the special case all along.

And now the steric factor falls out with meaning. The difference between my k and the bare collision rate is the ratio of partition functions for the *internal* structure that real molecules have and atoms don't. When two real molecules come together at the pass, modes that were free in the separated reactants — free rotations, free relative translations — get tied up into the complex; the complex has fewer or stiffer such modes than the reactants did, so F‡/(F_A F_B) is *smaller* than it would be for structureless fragments. That deficit is the steric factor. It is exp(ΔS‡/R): forming the complex loses entropy (ties up motions), ΔS‡ < 0, and exp(ΔS‡/R) < 1, which is precisely the empirical p ≤ 1 that collision theory had to insert by hand. I can now compute it from the geometries and frequencies of the reactants and the pass.

That entropy remark suggests a second, more useful face of the formula. The ratio F‡/(F_A F_B) exp(−E₀/k_B T) is just the equilibrium constant K‡ for forming the complex, and any equilibrium constant can be written through a free energy: K‡ = exp(−ΔG‡/RT) on a per-mole basis (with whatever standard-state convention I pick). So

    k = κ · (k_B T/h) · exp(−ΔG‡/RT),

and splitting ΔG‡ = ΔH‡ − TΔS‡,

    k = κ · (k_B T/h) · exp(ΔS‡/R) · exp(−ΔH‡/RT).

This is the form an experimentalist can use without ever building a partition function: measure k at several temperatures, and the slope and intercept hand back ΔH‡ and ΔS‡ separately. Compare with Arrhenius, k = A exp(−E_a/RT): the activation energy is E_a = ΔH‡ + RT for a unimolecular or condensed-phase reaction (because k_B T/h itself carries a factor of T, d ln k/dT picks up an extra RT; for a bimolecular gas reaction the translational partition functions add another, giving E_a = ΔH‡ + 2RT), and the pre-exponential factor is A = e·(k_B T/h)·exp(ΔS‡/R) (or e²· for bimolecular gas), with a standard-state factor (c⊖)^{1−m} for molecularity m to keep the units honest. The empirical A is no longer a fudge — it is the universal frequency k_B T/h modulated by the activation entropy. The steric factor and the absolute rate both come from molecular structure now.

One more consequence worth seeing before I write this down: the isotope effect. E₀ is referenced to the zero-point energy of the reactants. The zero-point energy of a mode is ½hν and ν ∝ μ^{−1/2} in the reduced mass, so replacing H by D lowers the reactant zero-point energy more than it lowers the (usually looser) complex's, raising E₀ for D and slowing the reaction — and if the bond to the isotope is fully broken at the pass, the whole difference is just the reactant zero-point energy difference, k_H/k_D = exp[(E₀(D) − E₀(H))/k_B T] in the obvious sign. The same E₀-referencing discipline that I needed just to make K‡ well-defined turns out to predict the kinetic isotope effect for free.

Let me put the method down as something I can evaluate. I need the partition-function primitives (translation, linear-rotor rotation, harmonic vibration), the barrier height E₀, and the molecular data — masses, moments of inertia, vibrational frequencies — of the reactants and of the configuration at the pass (the latter with its reaction-coordinate mode left out). Then the rate is the assembly k = κ(k_B T/h)(F‡/(F_A F_B))exp(−E₀/k_B T). I'll also keep the collision-theory expression alongside, to confirm the structureless limit.

```python
import math

kB = 1.380649e-23     # J/K
h  = 6.62607015e-34   # J s
NA = 6.02214076e23    # 1/mol
c  = 2.99792458e10    # cm/s   (wavenumber -> frequency)

# --- partition-function primitives, per unit volume, common energy zero ---
def q_trans(m, T):
    return (2.0 * math.pi * m * kB * T / h**2) ** 1.5

def q_rot_linear(I, T, sigma=1):
    return 8.0 * math.pi**2 * I * kB * T / (sigma * h**2)

def q_vib_mode(nu, T):                       # one harmonic mode, frequency nu (s^-1)
    return 1.0 / (1.0 - math.exp(-h * nu / (kB * T)))

# --- the universal attempt frequency that survives the cancellation ---
def crossing_frequency(T):
    return kB * T / h                        # ~6.25e12 s^-1 at 300 K

# --- the rate model: population on the pass times kB*T/h ---
def eyring_k(Q_dagger, Q_A, Q_B, E0, T, kappa=1.0):
    # Q_dagger = activated complex with the reaction-coordinate mode removed
    return kappa * crossing_frequency(T) * (Q_dagger / (Q_A * Q_B)) \
        * math.exp(-E0 / (kB * T))

def eyring_k_thermo(dG, T, kappa=1.0, c_std=None, molecularity=1):
    R = kB * NA
    k = kappa * crossing_frequency(T) * math.exp(-dG / (R * T))
    if c_std is not None:
        k *= c_std ** (1 - molecularity)     # standard-state factor (c0)^(1-m)
    return k

# --- collision-theory limit, to confirm the structureless special case ---
def collision_rate(mu, sigma_cs, E0, T):
    mean_u_rel = math.sqrt(8 * kB * T / (math.pi * mu))
    return sigma_cs * mean_u_rel * math.exp(-E0 / (kB * T))

if __name__ == "__main__":
    T = 300.0
    amu = 1.66053907e-27
    mH, mH2, mC = 1.008*amu, 2.016*amu, 3.024*amu       # H, H2, linear H..H..H
    I_H2, I_C = 4.6e-48, 2.7e-47                         # moments of inertia
    nu_H2 = 4400.0 * c                                   # H2 stretch
    nus_C = [w*c for w in (2000.0, 900.0, 900.0)]        # bound modes at the pass
    E0 = 9.7 * 4184.0 / NA                               # ~9.7 kcal/mol, J/molecule

    Q_A = q_trans(mH, T)                                 # atom: translation only
    Q_B = q_trans(mH2, T) * q_rot_linear(I_H2, T, sigma=2) * q_vib_mode(nu_H2, T)
    Q_d = q_trans(mC, T) * q_rot_linear(I_C, T, sigma=2)
    for nu in nus_C:                                     # reaction coordinate omitted
        Q_d *= q_vib_mode(nu, T)

    print("kB*T/h =", crossing_frequency(T), "s^-1")
    print("k (TST)       =", eyring_k(Q_d, Q_A, Q_B, E0, T))
    mu = mH*mH2/(mH+mH2)
    print("k (collision) =", collision_rate(mu, math.pi*(2.5e-10)**2, E0, T))
```
