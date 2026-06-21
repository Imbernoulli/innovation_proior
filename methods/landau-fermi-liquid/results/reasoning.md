Let me start from the thing that actually bothers me. Liquid helium-3, at a few millikelvin, has a
heat capacity that goes linearly with temperature. That is the unmistakable fingerprint of a
degenerate ideal Fermi gas — only the bottom shell of fermions, within about k_BT of the Fermi
surface, can be excited, so the heat capacity is proportional to T. And yet helium-3 is a *liquid*.
The atoms interact strongly enough to condense; the interaction energy is not small compared to the
kinetic energy. By every right the ideal-gas picture should be useless here. But the *form* of the
law survives. Only the *slope* is off — measured, the effective mass that fits the slope is several
times the bare atomic mass. The same story plays out for the conduction electrons in an ordinary
metal: a tiny linear-in-T electronic heat capacity, Fermi-gas in shape, wrong in magnitude, even
though the Coulomb repulsion between electrons is comparable to their kinetic energy.

So something deep is going on. A strongly interacting system is mimicking a non-interacting one,
qualitatively, while quantitatively renormalizing the constants. I want to understand what survives
and what gets renormalized, and I want to do it without pretending I can solve the many-body
Schrödinger equation, because I cannot.

The honest difficulty is this: in an ideal gas the total energy is a sum of fixed single-particle
energies, E = Σ ε_p n_p with ε_p = p²/2m, and the cost of adding one more fermion is just ε_p,
independent of who else is around. That additivity is exactly the thing that dies when the
interaction is of order one. Add a fermion, and it pushes on, and is pushed by, every other
fermion; the energy it costs depends on the entire configuration. There is no fixed band energy to
write down. Hartree–Fock tries to rescue this by dressing each particle in the averaged field of
the rest, but that is only the mean field — it throws away the residual interaction between the
dressed objects, and it shows: for the electron gas the Hartree–Fock single-particle velocity
diverges logarithmically right at the Fermi surface, which is nonsense for a real metal. The
medium's response to a moving particle is not in the mean field.

Let me not reach for the Hamiltonian at all. What do I actually have control over? Not the exact
eigenstates. What I have control over is the *low-energy* structure, the things living near the
Fermi surface, because that is what every low-temperature measurement sees. So let me reason from
the spectrum, the way one reasons about a Bose liquid like He II — there the phonon/roton spectrum,
not the microscopic Hamiltonian, dictates superfluidity. A Fermi liquid should be governed by the
*type* of its spectrum too, just a Fermi type rather than a Bose type.

Let me gamble on this. Imagine I turn the interaction on slowly, gradually, starting
from the ideal gas. As I do this, I assume the *classification* of the energy levels does not
change — each state of the gas goes over continuously into a state of the liquid carrying the same
quantum numbers. This is a strong assumption, but it is a natural one: the conserved quantities
(total momentum, total spin, particle number) cannot jump under a slow, smooth switching, and as
long as the system does not undergo an actual change of state — does not become superfluid, does
not open a gap — there is no reason for the level structure to reorganize. In fact, the assumption
carries its own warning label: the place where it must fail is exactly where the spectrum
qualitatively changes, i.e. the onset of superfluidity. That is fine; that is where I would *want*
the theory to break.

If the classification is preserved, then to each elementary excitation of the gas — a single
particle of definite momentum sitting above the filled sea, or a hole below it — there corresponds
an elementary excitation of the liquid, of the same definite momentum. I will call these
quasiparticles. They obey Fermi statistics, because they are in one-to-one correspondence with the
gas excitations, and their number coincides with the number of real particles. A quasiparticle is,
in a real sense, a real particle dragging along the self-consistent field of all its neighbours.

Now I have to be careful that these objects make sense as *particles* at all, since they are
dressed and could in principle decay. Take one quasiparticle of energy ε just above the Fermi
surface. It can scatter off the sea, creating a particle–hole pair: it drops in momentum, a hole
opens at some k′ below the surface, and an electron at k′ gets kicked above. Energy and momentum
conservation, together with the Pauli principle, pin down the available final states. The hole at
k′ must lie within an energy (ε − ε_F) of the surface — otherwise there is no room for it — and the
promoted electron likewise. So *two* phase-space factors, each of order (ε − ε_F), squeeze the
decay rate: it goes as (ε − ε_F)², which I can write as ω² for an excitation of frequency ω above
the surface. The lifetime τ ~ 1/ω². Compare that to the oscillation period of the excitation's own
wavefunction, which is ~1/ω. The ratio τ/T_osc ~ (1/ω²)/(1/ω) = 1/ω → ∞ as ω → 0. So the
excitation oscillates many, many times before it decays; right at the Fermi surface it is
infinitely sharp. Quasiparticles are legitimate, well-defined entities precisely *because* the
Pauli principle starves their decay near the surface. (At finite temperature the same counting
gives a mutual collision rate ∝ T², so a relaxation time τ ~ 1/T² that diverges as T → 0. Hold on
to that number; it will matter.)

Good. So I will describe the liquid by the distribution function n(p) of quasiparticles, or rather
by its *deviation* δn from the ground-state Fermi step, because at low temperature that is all that
is nonzero — only a thin shell within ~k_BT of the surface is disturbed. Everything I can measure is
controlled by that shell.

Now, what is the energy? This is where I cannot copy the gas. Because the dressing of one
quasiparticle depends on the surroundings, I am not allowed to write E = Σ ε_p n_p with a fixed ε_p.
The energy is a *functional* of the whole distribution, E[n]. The thing I can write cleanly is its
*variation*. If I change the distribution by δn, the energy changes by

    δE = ∫ ε(p) δn(p) dτ,     dτ = d³p / (2πℏ)³.

The quantity ε(p) defined this way — the first functional derivative ε = δE/δn — is the change in
the system's energy when I add a single quasiparticle of momentum p to the *actual* distribution.
It is the Hamiltonian function of the added quasiparticle. Notice what this buys me: ε is no longer
a fixed band energy; it depends on n, which depends on temperature and on whatever distortion I
apply. That is the whole content. (The particles carry spin, and spin is quantum mechanical, so I
should really let ε be a matrix in spin and write the spin spur, δE = Sp_σ ∫ ε δn dτ; for an
isotropic equilibrium liquid with no external field, ε cannot depend on spin operators, and I take
spin ½.)

What is the equilibrium distribution? The entropy is still combinatorial — it counts ways to
distribute quasiparticles over states, and since they are fermions,

    S = − Sp_σ ∫ { n ln n + (1−n) ln(1−n) } dτ.

Maximize S at fixed total number δN = Sp_σ∫δn dτ = 0 and fixed energy δE = Sp_σ∫ε δn dτ = 0. The
variation gives ln[(1−n)/n] − (ε − μ)/θ = 0, i.e.

    n(ε) = [ e^{(ε − μ)/θ} + 1 ]^{−1}.

This is the Fermi distribution again — but now ε is itself a functional of n, so ε depends on
temperature too. Near equilibrium and near the surface I can linearize the dispersion. The natural
parameter is the slope of ε at the limiting momentum p₀:

    m* = p / (∂ε/∂p) |_{p = p₀}.

This defines the effective mass. And immediately the heat capacity comes out gas-like with m → m*:
the low-T specific heat is the Sommerfeld formula with the density of states evaluated at the
*renormalized* slope. That already explains the helium-3 puzzle qualitatively — the law is linear
because it only knows about the Fermi surface; the slope is renormalized because ε's slope is m*,
not m. But I have not yet computed *anything* about the interaction. m* is so far just a name for
the measured slope.

A change δn does not only add energy ε δn; it also shifts
ε *itself*, because ε is a functional of n. So there is a second variation, and it is exactly the
interaction between quasiparticles:

    δε(p) = Sp_{σ'} ∫ f(p, p') δn(p') dτ',     f(p, p') = δ²E / δn(p) δn(p').

This f is the heart of the matter. It is the second functional derivative of the energy — symmetric
in p and p', and it depends on the spins. It tells me how the presence of one quasiparticle changes
the energy of another. Writing the energy to second order in the distortion:

    E − E₀ = Σ (ε_p − μ) δn_p + ½ Σ f_{p,p'} δn_p δn_{p'} + O(δ³).

The first term, with the m* slope, gives the gas-like part — the heat capacity. The *new* physics
of the liquid — the mass renormalization, the compressibility, the susceptibility, and (I will
find) collective modes — lives entirely in the cross term ½ f δn δn. And since |δn| is small (the
distortion is a thin shell of relative size δ around the surface), going to third order is O(δ³)
and negligible. Two terms is exactly right.

Now I get a big simplification for free. Because δn is nonzero only on the Fermi surface, I only
ever need f with both p and p' *on* the surface. Their magnitudes are both p₀, so f can depend only
on the *angle* between p and p', call it θ (or χ), plus the spins. What spin structure is allowed?
For spin ½ with time-reversal and reflection symmetry, the only invariants are a scalar and the
product σ·σ'. So

    f(p, p') = f^s(θ) + (σ·σ') f^a(θ).

The symmetric part f^s and the antisymmetric (exchange) part f^a are each just a function of one
angle. And a function of one angle on a sphere is naturally expanded in Legendre polynomials:

    f^{s,a}(θ) = Σ_l f_l^{s,a} P_l(cos θ).

I will make these dimensionless by pulling out the density of states at the Fermi surface,
D(ε_F) = V m* p₀ / π²ℏ³ (the gas density of states with m → m*), and define the Landau parameters

    F_l^{s,a} = D(ε_F) f_l^{s,a}.

These are pure numbers, comparable across systems. I expect — though I will check — that different
physical responses pick out different l: a uniform compression is an l = 0 distortion, a current is
l = 1, and so on. (There is one more thing to notice about f. The number of quasiparticle scattering
events per unit time can be written with a transition amplitude F(p₁,p₂; p₁',p₂'); f is nothing but
that amplitude for forward scattering, −F at zero angle. Its imaginary part is the actual scattering
cross-section, but for real collisions near the surface that is heavily suppressed, ∝ T², so the
real part — my f — is what controls the static and collective response.)

Now let me actually *compute* a renormalization, to prove f earns its keep. The cleanest one is the
effective mass, and the cleanest handle is Galilean invariance, because helium-3 is a neutral
liquid with no lattice. In a Galilean-invariant liquid, the momentum density must equal the mass
flow — the total momentum carried by the quasiparticles must be the real mass current. The velocity
of a quasiparticle is ∂ε/∂p, and since the number of quasiparticles equals the number of real
particles,

    Sp ∫ p n dτ  =  Sp ∫ m (∂ε/∂p) n dτ.

Left side: total momentum. Right side: bare mass times the particle current. This must hold for any
distribution, so it must hold under variation. Vary both sides in n. The left side gives Sp∫ p δn dτ.
The right side is subtle: varying n changes ∂ε/∂p directly through δn, but it *also* changes ε
itself through f, so ∂ε/∂p picks up a piece from δε. Carefully,

    Sp ∫ (p/m) δn dτ = Sp ∫ (∂ε/∂p) δn dτ + Sp_σ Sp_{σ'} ∫ (∂f/∂p) n' δn dτ dτ'.

Integrate the last term by parts in p' (it is the derivative of n' that I want), rename indices, and
since δn is arbitrary the integrands must match:

    p/m = ∂ε/∂p − Sp_{σ'} ∫ f (∂n'/∂p') dτ'.

At T = 0, ∂n'/∂p' = −(p'/p') δ(p' − p₀): the derivative of the step is a delta on the surface,
pointing radially. So the integral collapses onto the Fermi surface. Take p on the surface too,
along the z-axis, let θ be the angle between p and p'; then p̂·p̂' = cos θ, and ∂ε/∂p|_{p₀} = p₀/m*.
Doing the radial integral against the delta,

    p₀/m = p₀/m* + (1/2) (V p₀ / (2πℏ)³) Sp_σ Sp_{σ'} ∫ f cos θ dΩ,

the ½ correcting for the spin sum. Equivalently, dividing through by p₀ and writing it as

    1/m = 1/m* + (p₀ / 2(2πℏ)³) Sp_σ Sp_{σ'} ∫ f cos θ dΩ.

Now sum over both spins; only the symmetric part f^s survives the spin trace, and the cos θ kernel
projects out exactly the l = 1 Legendre component (since P₁ = cos θ and the polynomials are
orthogonal). Putting in the dimensionless F and the orthogonality ∫ P_l P_{l'} = 2δ_{ll'}/(2l+1),
the cos θ integral pulls out F_1^s/3, and I get the clean statement

    m*/m = 1 + F_1^s / 3.

There it is. The effective mass is *not* a free fit parameter after all — it is fixed by the l = 1
symmetric Landau parameter, which is a property of the forward-scattering interaction. The measured
m*/m ≈ several for helium-3 means F_1^s is of order ten. The interaction has been turned into a
number I can extract from a thermodynamic measurement. (One caveat I must flag to myself: this
Galilean argument is *only* valid where p is the true momentum. For electrons in a metal p is the
crystal quasi-momentum and the lattice carries momentum too, so this particular relation does not
apply there — the lattice is not Galilean-invariant. For helium-3 it is exact.)

Next, the compressibility — the response to squeezing the liquid, which is an l = 0 distortion. The
chemical potential is μ = ε(p₀) = ε₀. The compressibility I want is essentially ∂μ/∂N. By
homogeneity μ depends only on N/V, so ∂μ/∂N = −(V/N) ∂μ/∂V = −(V²/N) ∂p/∂V, and the sound velocity
follows from c² = ∂p/∂(mN/V) = (1/m)(N ∂μ/∂N). So I need δμ when I change the total number by δN.
Two things move. The Fermi level shifts because the surface moves out by δp₀ — that is the
kinematic ∂ε₀/∂p₀ piece. *And* ε shifts because n changed, through f — the interaction feedback:

    δμ = (1/2) Sp_σ ∫ f δn' dτ' + (∂ε₀/∂p₀) δp₀.

Adding δN particles inflates the Fermi sphere; the extra particles all sit in a thin shell at p₀, so
δn' is appreciable only near p₀, and δN = 8π p₀² δp₀ V / (2πℏ)³ (the surface area of the Fermi
sphere times the radial shift, with the spin factor). The f-integral therefore also localizes to
the surface; carrying it out and combining with the kinematic term gives

    ∂μ/∂N = Sp_σ Sp_{σ'} ∫ f do / 16π V + (2πℏ)³ / (8π p₀ m* V).

The first term is the interaction stiffening; the second is the m*-renormalized free piece. Now
multiply through and assemble c² = (1/m)(N ∂μ/∂N). Using N/V and the F-normalization, the angular
integral of f with no cos θ weight projects out the l = 0 component F_0^s, while the m* already
carries F_1^s. The result is

    c² = p₀²/3m*² + (1/6m)(p₀/2πℏ)³ Sp_{σσ'} ∫ f (1 − cos θ) do,

which, written in the dimensionless parameters, is

    c_s = (v_F* / √3) √[ (1 + F_0^s)(1 + F_1^s/3) ],     and equivalently   n² κ = D(ε_F) / (1 + F_0^s),

with v_F* = p₀/m*. The compressibility is the gas value, renormalized by 1/(1 + F_0^s) from the
interaction and by m* from the dispersion. And I notice something immediately: if F_0^s could reach
−1, the compressibility would diverge and then go negative — the liquid would be unstable against
collapse. So *stability* itself imposes 1 + F_0^s > 0. I will come back to that as a general
principle.

Now the magnetic susceptibility, which is an l = 0 distortion in the *spin* (antisymmetric) channel.
Put on a field H. The free-particle energy shifts by −β(σ·H), with β the magnetic moment. But the
field also distorts the distribution, and the spin part of f feeds back, because a spin
polarization changes the energy of the other spins. So the *actual* shift δε is self-consistent:

    δε = −β(σ·H) + Sp_{σ'} ∫ f δn' dτ'.

Write f = φ + ψ(σ·σ') and look for δε = −γ(σ·H), an effective moment γ to be found. Since δn is, to
first order in H, just (∂n/∂ε) δε, and only the spin (ψ) part contributes to the polarization,

    γ = β + (1/2) ∫ ψ (∂n/∂ε) γ' dτ'.

The ∂n/∂ε is a delta on the surface, so this is an algebraic self-consistency on the surface:

    γ = β − (1/2) ψ̄₀ γ (∂τ/∂ε)₀,

with ψ̄₀ the angle-averaged l = 0 spin parameter at the surface. Solve for γ and assemble the
susceptibility χ H = β Sp∫ n σ dτ. The result is

    1/χ = β^{−2} { (2π² k² / 3α) + ψ̄₀ },

where α is the linear-heat-capacity coefficient — i.e. the susceptibility is the Pauli value with
*two* renormalizations, the m* in the density of states and a Stoner-like enhancement
1/(1 + F_0^a) from the spin part:

    χ / χ₀ = (1 + F_1^s/3) / (1 + F_0^a).

For helium-3 the analysis of the data says ψ̄₀ is *negative*, about −2/3 of the heat-capacity term —
i.e. F_0^a < 0, an exchange interaction that *enhances* the susceptibility. This is striking: it
says that in the liquid, unlike a classical gas, there is no simple relation between the heat
capacity and the susceptibility — the exchange interaction has driven them apart. The gas has them
locked together; the liquid does not.

Let me step back and notice the pattern that has emerged. Every static response is the gas answer,
renormalized by m* (i.e. by F_1^s) and by one more Landau parameter chosen by the symmetry of the
distortion: compression → F_0^s, spin → F_0^a, current → F_1^s. And each "1 + F" denominator that
shows up is exactly a stability boundary: if it hits zero, the corresponding response diverges and
the uniform liquid becomes unstable to that particular deformation of the Fermi surface. The
general statement must be that the quadratic energy ½ f δn δn has to be positive for *every*
possible surface deformation. Decomposing a deformation into Legendre components, the energy of the
l-th channel carries a factor (1 + F_l^{s,a}/(2l+1)), so stability of the spherical Fermi surface
requires

    1 + F_l^{s,a} / (2l+1) > 0   for every l,

in both the symmetric and antisymmetric channels. (The effective-mass relation m*/m = 1 + F_1^s/3
makes the l = 1 case vivid: if F_1^s < −3 the effective mass goes negative and excitations across
the surface lower the energy — the surface spontaneously deforms.) When one of these is violated
the liquid undergoes a spontaneous change of shape of its Fermi surface — exactly the kind of
"change of state" where my adiabatic-continuity assumption was always going to break.

So far I have done statics. What about dynamics, oscillations? For that I need a
transport equation for n. In the quasi-classical regime, n obeys a Boltzmann-like equation, but I
have to be careful about what "the energy" is, because ε depends on position through n(r):

    ∂n/∂t + (∂n/∂r)·(∂ε/∂p) − (∂n/∂p)·(∂ε/∂r) = I(n),

where I(n) is the collision integral. The crucial structural point is that ∂ε/∂r is nonzero even
with no external field, purely because ε is a functional of n and n varies in space — the
quasiparticles feel a self-consistent force −∂ε/∂r = −∂/∂r ∫ f δn(r) dτ' from the others. This is
the molecular field, and it is what will let the system oscillate on its own.

Ordinary (first) sound needs collisions to keep each fluid element in
local thermodynamic equilibrium; its propagation requires ωτ ≪ 1. But I just argued τ ~ 1/T². As
T → 0, τ → ∞, the collision integral I → 0, the mean free path diverges, and the viscosity diverges
— ordinary sound is *increasingly absorbed* and cannot propagate at absolute zero. So either there
is no sound at T = 0, or there is a *different* mode that does not need collisions. Let me look for
the collisionless mode.

Set I = 0 (the T = 0 limit, or equivalently ωτ ≫ 1). Linearize: write n = n₀ + δn(p) and ε = ε₀ + δε,
keeping first order. The equilibrium n₀ and ε₀ are uniform, so their spatial derivatives drop, and

    ∂δn/∂t + (∂n₀/∂p)·(∂δε/∂r) + (∂ε₀/∂p)·(∂δn/∂r) = 0.

Take a plane wave, δn, δε ∝ e^{i(k·r − ωt)}. With v = ∂ε₀/∂p the quasiparticle velocity, and using
∂n₀/∂p = (∂n₀/∂ε) v,

    (k·v − ω) δn = (k·v)(∂n₀/∂ε) δε.

Now δε is not independent — it is the molecular field generated by δn through f:

    δε = ∫ F(χ) ν' do'/4π,

where I have introduced F(p,p') = Sp_{σ'} f · 4π p² dp / (2πℏ)³ dε integrated over the magnitude (so
F depends only on the angle χ between p and p'), and ν(p̂) = ∫ δn dε is the angular function that
measures the displacement of the Fermi surface in the direction p̂. Because ∂n₀/∂ε = −δ(ε − ε_F) is
a delta on the surface, only the surface displacement ν matters, and everything lives on the
surface. The physical picture is concrete: δn ∝ ∂n₀/∂ε times something means the disturbance is a
*deformation of the shape* of the Fermi surface, and ν(p̂) is how far the surface moves outward in
direction p̂.

Substitute and integrate over the energy. Let k define the polar axis; let θ, φ be the direction of
the quasiparticle momentum (and the velocity v points along it). Introduce the phase velocity
u = ω/k and the dimensionless ratio η = u/v. Then k·v − ω = kv(cos θ − η), and the equation becomes

    (η − cos θ) ν(θ, φ) = cos θ ∫ F(χ) ν(θ', φ') do'/4π.

This is a self-consistent eigenvalue equation for the oscillation: the molecular field on the right,
generated by the deformation everywhere on the surface, drives the deformation cos θ/(η − cos θ) on
the left. It is a propagating mode that needs *no collisions* — the restoring force is the
interaction f itself. This is the new sound. Let me call it zero sound.

Solve the simplest case: F(χ) = F₀, a constant (only l = 0 interaction). Then the integral on the
right does not depend on θ', φ', so the right side is cos θ × const, and

    ν = const · cos θ / (η − cos θ).

Plug back into the self-consistency. Substituting ν into the right-hand integral and demanding
consistency,

    (F₀/4π) ∫₀^π [ cos θ / (η − cos θ) ] 2π sin θ dθ = 1.

Do the integral. Let x = cos θ:

    (F₀/2) ∫_{−1}^{1} x/(η − x) dx = 1.

Now x/(η − x) = −1 + η/(η − x), so

    ∫_{−1}^{1} x/(η − x) dx = −2 + η ∫_{−1}^{1} dx/(η − x) = −2 + η [ −ln(η − x) ]_{−1}^{1}
                            = −2 + η ln[(η + 1)/(η − 1)].

So the self-consistency is (F₀/2){ −2 + η ln[(η+1)/(η−1)] } = 1, i.e.

    (η/2) ln[(η + 1)/(η − 1)] − 1 = 1/F₀.

Define φ(η) ≡ (η/2) ln[(η + 1)/(η − 1)] − 1. The zero-sound dispersion is simply

    φ(η) = 1/F₀.

This is beautiful and it tells me everything qualitatively. For a real, undamped wave I need η > 1,
i.e. u > v — the wave must outrun the fastest quasiparticle. If u < v, some quasiparticles move in
phase with the wave and absorb it (the logarithm becomes complex for η < 1, signalling damping).
The function φ(η) is positive and decreases monotonically from +∞ (as η → 1⁺, the log blows up) to 0
(as η → ∞). So a solution with η > 1 exists *only* when 1/F₀ > 0, i.e. F₀ > 0 — a repulsive l = 0
interaction. Zero sound propagates undamped only if the forward interaction is repulsive (for this
symmetric mode). Two limits make it quantitative. For strong coupling F₀ → ∞: expand the log for
large η, ln[(η+1)/(η−1)] = 2/η + 2/3η³ + …, so φ(η) ≈ 1/3η², and the dispersion gives
1/3η² = 1/F₀, i.e. η = √(F₀/3) — the zero-sound speed grows as √F₀. For weak coupling F₀ → 0⁺: η
must approach 1 from above, and one finds η − 1 ~ exp(−2 − 2/F₀), an exponentially small excess of
u over v — the mode barely escapes the quasiparticle continuum. That weak-coupling result is more
general than the constant-F₀ assumption: a nearly ideal Fermi gas corresponds to small F, η near 1,
ν concentrated at small angles, and one recovers the same form with F₀ → F(0).

Let me compare zero sound to ordinary sound, because I want to be sure it is genuinely different and
not a relabelling. Ordinary sound is a rigid translation of the whole Fermi sphere — the
displacement is ν ∝ cos θ, the surface moves bodily without changing shape. Zero sound has
ν ∝ cos θ/(η − cos θ), which is *not* proportional to cos θ: with η just above 1 the denominator is
small near θ = 0, so the surface bulges strongly forward and flattens behind — the Fermi surface
changes *shape*, elongated along the propagation direction. Different mode, different angular
structure, different speed. And the speeds are close but not equal: setting m* ≈ m and dropping F
in the compressibility result gives the gas first-sound speed c² ≈ p₀²/3m*² = v²/3, i.e. c ≈ v/√3,
the *ordinary* sound velocity; whereas zero sound for a nearly ideal gas has η ≈ 1, i.e. u ≈ v > c.
So zero sound is faster than ordinary sound. Two distinct sounds in the same liquid, at the two ends
of the ωτ scale — ordinary sound when ωτ ≪ 1 (collisions maintain local equilibrium), zero sound
when ωτ ≫ 1 (collisionless), with strong absorption and no clean mode in the crossover ωτ ~ 1.

I can push the angular structure further. If F is not constant but has an l = 1 part too,
F(χ) = F₀ + F₁ cos χ, then besides the symmetric (m = 0) zero sound there are *asymmetric* solutions
with an azimuthal factor e^{±iφ} — ν ∝ sin θ cos θ/(η − cos θ) e^{iφ}. Substituting and doing the
azimuthal integral (using the addition theorem cos χ = cos θ cos θ' + sin θ sin θ' cos(φ − φ')) gives
a self-consistency

    ∫₀^π sin³θ cos θ/(η − cos θ) dθ = 4/F₁.

The left side decreases monotonically with η and is maximal at η = 1, where the integral evaluates to
2/3; so 4/F₁ ≤ 2/3, which requires F₁ ≥ 6. An asymmetric zero-sound mode exists only if F₁ > 6.
Different angular momentum channels of zero sound, each with its own threshold on the corresponding
Landau parameter.

And the same machinery gives spin waves. If I keep the spin structure, the relevant kernel is not F
but its spin partner. Writing the full angular interaction as K = ½ F(χ) + ½ G(χ)(σ·σ'), an
oscillation that is a wave in the *spin* density rather than the charge density satisfies the very
same integral equation with F replaced by G/4. So if G has the right sign and magnitude there are
propagating spin waves — collective oscillations of the magnetization — by exactly the analysis I
just did. (For helium-3 the available data give a negative mean G, which would forbid the symmetric
spin-zero-sound, but the structure is there.)

Let me put real numbers on helium-3, because the whole point was to connect to it. From the
effective mass relation and the compressibility relation I can read F₀ and F₁ off the measured m*
and ordinary-sound speed:

    F₁/3 = m*/m − 1,        F₀ = 3 m m* c²/p₀² − 1.

From the low-temperature heat capacity the effective mass is about m* ≈ 1.43 m for the conditions in
question (the slope says the carriers are heavy). From Walters–Fairbank's compressibility the
ordinary sound speed is c ≈ 195 m/s, and the density fixes p₀/ℏ ≈ 0.76 × 10⁸ cm⁻¹. Plugging in, I
get F₀ ≈ 5.4 and F₁ ≈ 1.3. Then the zero-sound dispersion φ(η) = 1/F₀ with F₀ = 5.4 gives η ≈ 1.83,
so the zero-sound speed is u = η v = 1.83 p₀/m* ≈ 206 m/s — slightly above the ordinary-sound speed,
exactly as the theory demands. A concrete, falsifiable prediction: a second, faster sound at the
collisionless (low-temperature, high-frequency) end, with no analogue in hydrodynamics. (And since
the energy spectrum then automatically contains a phonon-like "Bose branch" ε = u p from these zero-
sound quanta, one should in principle add its contribution to the thermodynamics — though it scales
as a higher power of T, T³ in the heat capacity, than the T-linear quasiparticle part, so it is a
small correction at the lowest temperatures.)

One more set of relations I should record for completeness, because they make the kinetic equation a
proper conservation theory and pin down the fluxes. Multiply the kinetic equation by p_i and
integrate over phase space; collisions conserve momentum so the right side vanishes, and after
integrating by parts I get a continuity equation for momentum,

    ∂/∂t Sp∫ p_i n dτ + ∂Π_{ik}/∂x_k = 0,    with the momentum-flux tensor
    Π_{ik} = Sp∫ p_i (∂ε/∂p_k) n dτ + δ_{ik} [ Sp∫ ε n dτ − E ].

The second, isotropic piece appears precisely because ε is a functional of n: when I integrate
∫ p_i ∂/∂p_k(n ∂ε/∂x_k) and use the fact that ∂ε/∂x_i comes from the n-dependence, the rearrangement
forces a δ_{ik}[Sp∫ ε n dτ − E] term — the pressure-like contribution from the field energy. Doing
the same with ε instead of p_i gives energy conservation,

    ∂E/∂t + div Q = 0,    Q = Sp∫ n ε (∂ε/∂p) dτ,

the energy flow being quasiparticles carrying energy ε at velocity ∂ε/∂p. These confirm that the
quasiparticle description is internally consistent as a hydrodynamics: the deviations of Π and Q
from equilibrium come both from the direct change δn and from the induced change in ε through its
functional dependence on n.

Let me trace the whole causal chain once, to see that it hangs together. A strongly interacting
degenerate Fermi system mimics a gas because, by adiabatic continuity, its low-lying excitations are
in one-to-one correspondence with gas excitations — quasiparticles, fermionic, equal in number to
the real particles, long-lived near the surface because Pauli starves their decay (rate ∝ ω²). Since
the dressing depends on the surroundings, the energy is a *functional* of the distribution; its
first variation defines the quasiparticle energy ε with renormalized slope m*, and its second
variation defines the interaction function f(p,p'), which on the Fermi surface reduces to two
functions of angle, f^s and f^a, Legendre-expanded into the dimensionless Landau parameters F_l. The
gas thermodynamics returns with m → m*; every interaction effect is one Landau parameter dictated by
symmetry — m*/m = 1 + F_1^s/3 from Galilean invariance, the compressibility renormalized by
1/(1 + F_0^s), the susceptibility by 1/(1 + F_0^a) — and each 1 + F denominator is a stability
boundary, 1 + F_l/(2l+1) > 0. Finally, because ε depends on position through n, the quasiparticles
feel a self-consistent molecular field; the collisionless kinetic equation then supports a brand-new
propagating mode, zero sound, a self-sustained oscillation of the *shape* of the Fermi surface,
faster than ordinary sound, undamped only for repulsive F₀, governed by

    (η/2) ln[(η + 1)/(η − 1)] − 1 = 1/F₀,

with asymmetric modes and spin waves emerging from the same equation in higher angular-momentum and
spin channels. For helium-3 the measured m* and ordinary-sound speed give F₀ ≈ 5.4, F₁ ≈ 1.3, and
predict a zero-sound velocity of about 206 m/s — a new wave waiting to be looked for.
