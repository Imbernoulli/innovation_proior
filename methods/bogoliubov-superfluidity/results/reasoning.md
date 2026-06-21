Helium below the λ-point flows through a capillary as if it had no viscosity, and I want to understand that from the molecules. The temptation everyone has felt is to say: it is a Bose gas, it condenses, the condensate is the frictionless fluid. London said something like that, Tisza built a whole two-fluid hydrodynamics on it, and it organizes the phenomenology beautifully. But there is one objection I cannot talk my way out of, and it decides what kind of theory is even allowed.

Take the degenerate *ideal* Bose gas, every particle piled into the zero-momentum state, and ask whether it can carry a current without friction. It cannot. Picture the condensate drifting past an obstacle, or simply a foreign particle ploughing through it. Nothing stops a condensate molecule from being knocked into an excited momentum state by a collision — the excitation costs energy f²/2m, which can be made as small as I like by choosing |f| small, since f²/2m → 0 continuously. So for *any* relative velocity, however tiny, there is an arbitrarily-low-energy single-particle excitation into which the moving object can dump momentum. There is no energetic floor protecting the flow. The ideal-gas spectrum, the bare f²/2m, is precisely what dooms it. So whatever superfluidity is, it is not a property the *ideal* condensate has; the condensate alone is not enough. The friction-or-not question lives entirely in the *shape of the excitation spectrum*, and the ideal gas has the wrong shape.

Landau saw this and answered it by quantizing the hydrodynamics and *asserting* the spectrum: phonons ε = c p at small momentum, rotons ε = Δ + p²/2µ at larger momentum. Then his criterion is clean. A body moving at velocity V through the liquid can create one excitation of momentum p and energy ε(p) only if both energy and momentum balance; in the rest frame of the liquid the body loses kinetic energy V·p and that must cover ε(p), so creation is possible only when V ≥ ε(p)/p. Hence no excitation can be made — and no momentum lost, no friction — as long as
   V < V_c = min_p ε(p)/p.
For a linear branch ε = c p this minimum is c itself, strictly positive; for the roton branch it is √(2Δ/µ); either way V_c > 0. A finite critical velocity, frictionless flow below it. The mechanism is the existence of that strictly positive minimum of ε(p)/p. And now I see exactly why the ideal gas fails when I run the same criterion on *its* spectrum: min_p (p²/2m)/p = min_p (p/2m) = 0. The free-particle spectrum has critical velocity zero. Same criterion, opposite verdict, and the only difference is the spectrum.

So the whole problem reduces to one thing: produce a many-boson system whose *low-momentum* excitation energy goes like |p| (linear, phonon-like) instead of like p²/2m. If I can derive a linear branch from molecules, Landau's criterion hands me superfluidity for free. Landau postulated that branch. I want it to come out of the equations. And the ideal gas does not give it — its spectrum is rigidly f²/2m — so the linear branch, if it exists, has to be *made by the interaction*. That settles the starting point: a non-ideal Bose gas, weak repulsion between molecules. Weak, because the moment the interaction is strong I have a hopeless N-body problem; weak, because then I can carry the interaction as a small parameter and hope to keep only the leading effect. I will pay for "weak" later — it means I am throwing away the hard impenetrable core of real molecules, since I am not letting Φ(r) blow up at small r — but let me get the mechanism first and worry about the hard core afterward.

Write the Hamiltonian for N identical bosons in volume V with a pair potential,
   H = Σ_i T(p_i) + Σ_{i<j} Φ(|q_i − q_j|),   T(p) = p²/2m,
and second-quantize it in the plane-wave basis φ_f(q) = V^{-1/2} e^{i(f·q)/ℏ}, so that a_f, a_f^+ create and destroy a molecule of momentum f with [a_f, a_{f'}^+] = δ_{f,f'}. The field operator is Ψ(q) = Σ_f a_f φ_f(q). The kinetic term is diagonal, Σ_f T(f) a_f^+ a_f. The potential, written through the Fourier amplitude v(f) = ∫ Φ(|q|) e^{−i(f·q)/ℏ} dq (which depends only on |f| by radial symmetry), couples four operators with momentum conservation. This quartic interaction is the obstruction: it is not diagonal, it mixes momenta, and I cannot solve it as it stands. I need an approximation that exploits the one thing that is special about this system.

The special thing is the condensate. In the weakly excited gas near zero temperature, the overwhelming majority of molecules sit in the f = 0 state, so N_0 = a_0^+ a_0 is of order N — macroscopic, enormous. Now look at the commutator that is supposed to make a_0 an operator rather than a number: a_0 a_0^+ − a_0^+ a_0 = 1. But a_0 itself, acting on the condensate, pulls down a factor of order √N_0 ~ √N. So the *non-commutativity* of a_0 and a_0^+ is a relative correction of order 1/N_0 — utterly negligible against the operators themselves. That is the lever. I will treat a_0 and a_0^+ as ordinary c-numbers, both equal to √N_0, neglecting that they do not commute. (Dirac did essentially this for a macroscopically occupied mode.) Concretely, split the field into its condensate part and the rest: Ψ = a_0/√V + ϑ, where ϑ = (1/√V) Σ_{f≠0} a_f e^{i(f·q)/ℏ} collects the depleted, excited molecules, and a_0/√V is a number of order √(N_0/V). Treat ϑ as a "first-order correction," small because the depletion is small.

Why is this the right move and not, say, ordinary perturbation theory in Φ? Because perturbation theory in the *operators* keeps the quartic term quartic — four a's — and never closes. The c-number trick is what *lowers the degree*: every factor of a_0 or a_0^+ becomes a number √N_0, so the surviving operators in the interaction are the f≠0 ones, and the more condensate legs a term has, the lower its operator order. Keep only the terms with the *most* condensate legs (highest power of √N_0) consistent with each operator structure, drop everything with three or more excited operators — those are higher order in the depletion (N−N_0)/N ≪ 1 — and the quartic interaction collapses to something *quadratic* in the excited operators. A quadratic Hamiltonian I can hope to diagonalize. That is the entire strategic bet: macroscopic occupation buys me a quadratic problem.

Let me carry it through on the equations of motion rather than the Hamiltonian directly, which is cleaner here. Keeping the leading condensate factors and dropping second-and-higher powers of the correction ϑ, the equation for the excited field is, schematically,
   iℏ ∂ϑ/∂t = −(ℏ²/2m) Δϑ + (N_0/V) Φ_0 ϑ + (N_0/V) ∫Φ(|q−q'|) ϑ(q') dq' + (a_0²/V) ∫Φ(|q−q'|) ϑ^+(q') dq',
where Φ_0 = ∫Φ(|q|) dq = v(0), together with iℏ ∂a_0/∂t = (N_0/V) Φ_0 a_0. That last equation says the condensate amplitude just rotates with a phase, a_0 = e^{−i E_0 t/ℏ} b with E_0 = (N_0/V) Φ_0; and the local term (N_0/V) Φ_0 ϑ = E_0 ϑ in the excited equation carries that *same* energy E_0, so de-phasing the excited amplitudes by the identical e^{−i E_0 t/ℏ} — set b_f for the de-phased amplitudes — cancels exactly that piece. Fourier-transform the excited field, ϑ = (1/√V) Σ_{f≠0} b_f e^{i(f·q)/ℏ}; the two integrals become multiplications by v(f), and the local E_0 = (N_0/V)v(0) term is gone with the phase, leaving the kinetic T(f) plus the integral's (N_0/V) v(f) on the diagonal. The equation of motion splits into one equation per momentum f:
   iℏ ∂b_f/∂t = { T(f) + (N_0/V) v(f) } b_f + (N_0/V) v(f) b_{−f}^+.
The right-hand side does not just contain b_f. It contains b_{−f}^+ — the *creation* operator for the opposite momentum. The interaction, fed by the condensate, takes two molecules out of the condensate and puts them into momenta +f and −f, or the reverse. It creates and destroys *pairs* of opposite momentum. So the equation for b_f is coupled to the equation for b_{−f}^+, and writing that companion equation,
   −iℏ ∂b_{−f}^+/∂t = (N_0/V) v(f) b_f + { T(f) + (N_0/V) v(f) } b_{−f}^+,
I have a closed 2×2 linear system mixing b_f with b_{−f}^+. The off-diagonal pieces, the (N_0/V) v(f) terms multiplying the *other* operator, are the anomalous pair terms. They are not in the ideal gas. They are made entirely by the interaction acting through the condensate, and they are precisely what is going to bend the spectrum away from f²/2m. So I should not try to get rid of them by hand — they are the mechanism. I should solve the 2×2 system honestly.

Two coupled linear ODEs with constant coefficients; the solutions are combinations of e^{±iE(f)t/ℏ}, and E(f) is set by the eigenvalues of the 2×2 matrix
   M = [ T+α ,  α ; −α , −(T+α) ],   α ≡ (N_0/V) v(f),
where the asymmetry in signs comes from b_{−f}^+ being a creation operator (the conjugated equation carries the minus). Its eigenvalues satisfy E² = (T+α)² − α² = T² + 2αT. So
   E(f) = √[ T(f)² + 2 T(f) (N_0/V) v(f) ] = √[ 2 T(f) (N_0/V) v(f) + T²(f) ].
There it is. Stare at this. For the *ideal* gas, α = 0 and E = T = f²/2m, the bare spectrum, and I already know that gives no superfluidity. With the interaction on, the cross term 2 T α is added *under the square root*, and that term changes the small-f behavior completely. At small f, T(f) = f²/2m → 0 like f², so the T² term is order f⁴ and the cross term 2Tα is order f² (with α → (N_0/V) v(0) constant). The cross term dominates:
   E(f) ≈ √[ 2 T(f) (N_0/V) v(0) ] = √[ (f²/m) (N_0/V) v(0) ] = √[ (N_0/V) v(0)/m ] · |f|.
*Linear in |f|.* The square root of (a constant times f²) is a constant times |f|. The interaction has converted the quadratic free-particle dispersion into a *linear* one at small momentum. This is a phonon — sound — and it appeared from the algebra, not from a postulate. The slope is a velocity; call it c. Landau's whole edifice asked for exactly this branch, and here it falls out of c-numbering the condensate and solving a 2×2 system.

I want to make sure that slope really is the speed of sound, because if it is, the link to thermodynamics is a strong internal check. The coefficient is c = √[ (N_0/V) v(0)/m ]. To leading order N_0 ≈ N, so (N_0/V) ≈ N/V = 1/v with v = V/N the volume per molecule, and c² = v(0)/(m v). Now compute the pressure at T = 0 from the leading energy. At absolute zero the free energy is the mean energy, whose main term (the condensate self-interaction) is E = (N²/2V) ∫Φ dq = (N²/2V) v(0). Then P = −∂E/∂V = (N²/2V²) v(0). With mass density ρ = Nm/V, this is P = ρ² v(0)/(2m²), so ∂P/∂ρ = ρ v(0)/m² = (Nm/V) v(0)/m² = (N/V) v(0)/m = v(0)/(mv). Therefore
   c² = v(0)/(mv) = ∂P/∂ρ,
which is exactly the hydrodynamic sound speed. The linear branch *is* sound, with the right thermodynamic slope. The small-momentum quasiparticle is a phonon, full stop — and I did not have to assume it.

But the square root only makes sense if what is under it is non-negative, and that is not automatic. E(f)² = T² + 2T(N_0/V)v(f). At large f, T dominates and this is safely positive whatever v(f) does. At small f, the sign is governed by the cross term, i.e. by the sign of v(f) → v(0). If v(0) = ∫Φ(|q|) dq > 0 — net repulsion — the radicand is positive for all f and E(f) is real, oscillatory, a genuine excitation energy. If v(0) < 0 — net attraction — then for small enough f the radicand goes negative, E(f) is imaginary, and the solutions of the 2×2 system are real exponentials e^{±|E|t/ℏ}: one of them *grows without bound in time*. The amplitude b_f of a low-momentum mode blows up. That is not an excitation; that is the condensate tearing itself apart. The states with small N_f are unstable. So the theory only describes a stable superfluid when
   v(0) = ∫ Φ(|q|) dq > 0,
net repulsion. And notice what this condition *is*, physically: I just showed P = (N²/2V²) v(0), so ∂P/∂ρ = v(0)/(mv) > 0 exactly when v(0) > 0. The condition that my approximation be stable is identical to the condition of thermodynamic stability of the gas at absolute zero, ∂P/∂ρ > 0 — a fluid with negative compressibility collapses. The mathematics refusing to give a real spectrum and the physics refusing to hold the gas together are the same statement. That coincidence makes me trust the calculation. I will restrict to repulsive potentials, v(0) > 0, from here on. (For the long-wavelength frequency, incidentally, if I form ω = E/ℏ and take ℏ→0 at fixed f/ℏ = k, I land on the classical dispersion of density waves — another consistency check that the linear branch is sound.)

Let me also check the *other* end. For sufficiently large momenta, v(f) → 0 (any smooth potential's Fourier transform dies at high momentum), so E(f) → √(T²) = T(f) = f²/2m. The quasiparticle at large momentum is just a bare molecule again — the interaction has stopped mattering at that scale. So the spectrum interpolates: linear phonon c|f| at small f, smoothly rising to the free-molecule kinetic energy f²/2m at large f. Expanding the large-f branch, E(f) = f²/2m + v(f)/v + …, the kinetic energy plus a small interaction correction. One continuous curve from sound to free particle — and crucially there is no second branch, no separate "roton" species in this dilute model; phonons and individual-molecule excitations are the *same* curve at its two ends. There is nothing here forcing me to split the quasiparticles into two kinds.

So I have the spectrum from the equations of motion. But I have been a little cavalier — I solved for the time dependence, but I have not actually exhibited the *operators* that diagonalize the Hamiltonian, the genuinely independent excitations whose number is conserved by the free dynamics. The 2×2 system mixing b_f and b_{−f}^+ tells me the right normal modes are some linear combination of a *destruction* operator at +f and a *creation* operator at −f. So I look for new operators of the form
   ξ_f = (b_f − L_f b_{−f}^+)/√(1 − |L_f|²),   ξ_f^+ = (b_f^+ − L_f b_{−f})/√(1 − |L_f|²),
with a single number L_f to be fixed and the √(1−|L_f|²) put there for normalization, whose purpose I will see in a second. Why a creation operator mixed into a destruction operator? Because the offending term in the dynamics is exactly a b_f–b_{−f}^+ coupling; only a transformation that itself mixes creation and annihilation can rotate it away. An ordinary number-conserving rotation among the a_f's cannot touch it — it conserves particle number, while the pair term manifestly does not (it changes the number by two). The mixing of b and b^+ is not optional; it is dictated by the structure of the coupling.

Now I have two demands on L_f, and they had better be compatible. First, the new operators must still be honest bosons — the transformation has to be *canonical*, preserving the commutators [ξ_f, ξ_{f'}^+] = δ_{f,f'} and [ξ_f, ξ_{f'}] = 0 — otherwise "number of quasiparticles" is meaningless. Compute: with the b's canonical and L_f real,
   [ξ_f, ξ_f^+] = ( [b_f, b_f^+] − L_f² [b_{−f}^+, b_{−f}] )/(1 − |L_f|²) = (1 − L_f²)/(1 − |L_f|²) = 1,
so the normalization factor √(1−|L_f|²) is *exactly* what makes the transform canonical, for *any* real L_f with |L_f| < 1. Good — that is why it is there, and it costs me nothing yet. Second, L_f must be the specific value that *kills the anomalous term*, leaving the Hamiltonian diagonal in the ξ's. Inverting the transform,
   b_f = (ξ_f + L_f ξ_{−f}^+)/√(1 − |L_f|²),   b_f^+ = (ξ_f^+ + L_f ξ_{−f})/√(1 − |L_f|²),
and substituting into the equations of motion, the cross terms cancel precisely when
   L_f = (V/(N_0 v(f))) · { E(f) − T(f) − (N_0/V) v(f) },
and then the equations decouple into
   iℏ ∂ξ_f/∂t = E(f) ξ_f,   −iℏ ∂ξ_f^+/∂t = E(f) ξ_f^+.
Pure harmonic motion at frequency E(f)/ℏ, no mixing left. The same E(f) as before, reassuringly. Let me record the weights this implies, because they tell me what a quasiparticle *is*:
   |L_f|² = [ (N_0/V) v(f) / ( E(f) + T(f) + (N_0/V) v(f) ) ]²,
   1 − |L_f|² = 2 E(f) / ( E(f) + T(f) + (N_0/V) v(f) ).
At large f, (N_0/V)v(f) → 0, so L_f → 0: the quasiparticle is just the bare molecule, ξ_f ≈ b_f, as the spectrum already told me. At small f, E(f) → 0 and L_f → 1: the quasiparticle is a near-equal superposition of creating a +f molecule and destroying a −f one — a collective, paired object, *not* a single molecule. That is the precise sense in which the phonon "cannot be identified with an individual molecule": it is a coherent mixture of a particle and a hole drawn out of the condensate. (Equivalently, writing b_f = u_f ξ_f + v_f ξ_{−f}^+ with u_f² − v_f² = 1, the same content: u_f² = (T+α+E)/2E, v_f² = (T+α−E)/2E, both diverging like 1/|f| as f→0 — strong particle–hole mixing in the phonon — and u_f→1, v_f→0 at large f.)

Now assemble the Hamiltonian itself in the new operators, to confirm it is genuinely a free gas of these excitations and to read off the ground-state energy. Kinetic part: H_kin = Σ_f T(f) a_f^+ a_f = Σ_f T(f) b_f^+ b_f (the de-phasing does not change it). Potential part: expand Ψ^+Ψ^+ΨΨ keeping the condensate amplitude a_0/(V√V) wherever possible and ϑ for the rest, and drop all terms third order and higher in ϑ (the same depletion-is-small approximation). What survives is
   H_pot = Φ_0 { (1/2)(N_0²/V) + (N_0/V) Σ_{f≠0} b_f^+ b_f }
           + (1/2)(1/V) Σ_{f≠0} v(f) [ b_f^+ b_{−f}^+ + b_f b_{−f} ] · N_0
           + (N_0/V) Σ_{f≠0} v(f) b_f^+ b_f.
Use the number identity Σ_{f≠0} b_f^+ b_f = Σ_{f≠0} N_f = N − N_0, so that to the order I am working (N_0/V)(N−N_0) + (1/2)(N_0²/V) ≈ (1/2)(N²/V) — collecting the condensate self-energy and the leading exchange term into (1/2)(N²/V)Φ_0. Then
   H = (1/2)(N²/V) Φ_0 + Σ_{f≠0} T(f) b_f^+ b_f
       + (N_0/V) Σ_{f≠0} v(f) b_f^+ b_f
       + (1/2)(N_0/V) Σ_{f≠0} v(f) [ b_f^+ b_{−f}^+ + b_f b_{−f} ].
This is exactly the quadratic-with-pair-terms form the 2×2 system was the dynamics of. Replace the b's by the ξ's using the inverse transform; the diagonal and anomalous pieces recombine, the off-diagonal cancels by the choice of L_f, and I am left with
   H = H_0 + Σ_{f≠0} E(f) n_f,   n_f = ξ_f^+ ξ_f,
   H_0 = (1/2)(N²/V) Φ_0 + (1/2) Σ_{f≠0} [ E(f) − T(f) − (N_0/V) v(f) ]
       = (1/2)(N²/V) Φ_0 + V/(2(2πℏ)³) ∫ [ E(f) − T(f) − (N_0/V) v(f) ] df.
The total energy is a ground-state constant H_0 plus a sum of independent quanta, each of energy E(f), occupation n_f. The quasiparticles do not interact — to this order — and obey Bose statistics, since the ξ's are canonical bosons. So the weakly excited non-ideal Bose gas *is* a perfect gas of these elementary excitations. The interaction between them would only show up if I kept the cubic and higher ϑ terms I threw away; those would let quasiparticles scatter and equilibrate, but they are higher order in the depletion, so to leading order the gas is free. The H_0 beyond the classical (1/2)(N²/V)Φ_0 is the zero-point energy of the quasiparticle vacuum — the quantum correction to the ground-state energy, the price of the pair admixture. (For a contact potential the integral needs the bare v(0) to be re-expressed through the true scattering amplitude before it converges, but that is a refinement of the same expression, not a change of mechanism.)

The transform also costs the ground state its purity. Even at absolute zero, the ground state is the ξ-vacuum, not the bare-molecule vacuum, so it contains a finite population of excited *molecules*: the b-modes have nonzero occupation in the ξ-vacuum. The momentum distribution of molecules at T = 0 comes out
   W(f) = C δ(f) + (smooth piece) with 1 − C = (1/N) · V/(2πℏ)³ ∫ { [E(f)+T(f)+(N_0/V)v(f)]/(2E(f)) − 1 } df > 0,
so only a fraction C < 1 of the molecules sit exactly at zero momentum; the rest are smeared over the whole spectrum even in the ground state. This is depletion, and it is the self-consistency knob: my whole expansion assumed (N−N_0)/N = 1−C ≪ 1, and this integral is what 1−C actually is, so the interaction must be weak enough to keep it small. Estimating it for Φ(r) = Φ_m F(r/r_0) of range r_0, the integral scales as η^{3/2} with η ≡ (N_0/V) v(0)·(m r_0²/ℏ²) the dimensionless ratio of interaction to kinetic scale, so 1−C ~ η^{3/2} stays small precisely when η ≪ 1 — when the interaction energy scale is small against ℏ²/(m r_0²). That is the dimensionless small parameter I promised at the outset to identify, now pinned down by demanding the depletion be small. In the dilute limit it is cleaner to say (N−N_0)/N = (8/3√π)√(n a³), small precisely when n a³ ≪ 1 — diluteness controls the expansion.

Now close the loop back to where I started — friction. I have a free Bose gas of quasiparticles with energy E(f). Put this gas into uniform motion: let the assembly of quasiparticles drift with velocity u relative to the condensate. In equilibrium the occupation of mode f is the Bose function in the frame moving with u,
   n̄_f = [ exp( (E(f) − f·u)/Θ ) − 1 ]^{-1},   Θ = temperature modulus,
because boosting the gas shifts each excitation's energy by −f·u (Galilean shift of a momentum-f excitation). Occupation numbers must be non-negative, which forces the exponent to be positive for every mode: E(f) − f·u > 0 for all f ≠ 0, i.e. E(f) > f·u ≥ |f| |u| cosθ, whose tightest case is alignment, giving E(f) > |f| |u|, i.e.
   |u| < min_{f≠0} E(f)/|f|.
So the quasiparticle gas can sustain a *finite* drift relative to the condensate — equivalently the condensate can move through the quasiparticle gas — only up to the velocity min E(f)/|f|, and below that there is a stationary, equilibrium relative motion with *no excitations being created spontaneously*, hence no friction. That is superfluidity, and it is now Landau's criterion with *my* spectrum plugged in. Evaluate the minimum. The ratio E(f)/|f| is continuous and positive; at |f| → 0 it tends to c > 0 (the phonon slope), and at large |f| it grows like |f|/2m. A continuous positive function that starts at c and rises to infinity has a strictly positive minimum. Therefore
   V_c = min_{f≠0} E(f)/|f| > 0,
a genuine, finite critical velocity, derived end to end from the molecular Hamiltonian. And the contrast with the ideal gas is now airtight: had E(f) stayed f²/2m, the ratio E/|f| = |f|/2m would tend to 0 as f→0, the minimum would be 0, and there would be no superfluidity — which is exactly the ideal-gas failure I opened with. The interaction made the small-f branch linear, and the linear branch is what makes min E/|f| positive. The phonon *is* the superfluidity.

Let me restate the chain in one breath, because every link mattered. An ideal Bose condensate cannot be a superfluid: its free-particle spectrum has min ε/p = 0, so flow at any speed can shed momentum into low-energy single-particle excitations. The cure must come from interaction, so take a weakly repulsive Bose gas. Its condensate is macroscopically occupied, so c-number the f=0 operators; the quartic interaction then collapses to a quadratic Hamiltonian whose only non-trivial feature is anomalous pair terms b_f b_{−f} — created by the interaction acting on the condensate — which couple +f to −f. Solving that 2×2 system gives the excitation energy E(f) = √[T(f)² + 2T(f)(N_0/V)v(f)], whose small-f limit, when v(0) > 0 (net repulsion, equivalently ∂P/∂ρ > 0, thermodynamic stability), is linear: a phonon E = c|f| with c = √(∂P/∂ρ) the sound speed; whose large-f limit is the bare molecule f²/2m. A canonical transformation mixing b_f with b_{−f}^+ — forced to mix creation and annihilation because the pair term doesn't conserve number, normalized by √(1−|L_f|²) to stay canonical — diagonalizes H into a free Bose gas of these quasiparticles plus a zero-point ground-state energy, with a finite ground-state depletion that controls the expansion. Feed E(f) into the energy–momentum argument for a drifting quasiparticle gas and the critical velocity is V_c = min_f E(f)/|f|, strictly positive because the spectrum is linear at small f. Superfluidity, derived from the molecules.

```python
import numpy as np

# hbar = m = 1.  Contact (dilute) limit: v(f) -> g = v(0) = 4*pi*a, constant.
# n0 = condensate density.  Sound speed c = sqrt(g*n0); healing length xi = 1/sqrt(2*g*n0).

def kinetic(k):
    "T(f) = f^2 / 2m, the bare molecule kinetic energy."
    return 0.5 * k * k

def excitation_energy(k, g, n0):
    "E(f) = sqrt( 2 T(f) (N0/V) v(f) + T(f)^2 ): the 2x2 eigenvalue."
    T = kinetic(k)
    return np.sqrt(T * (T + 2.0 * g * n0))   # = sqrt(T^2 + 2 T g n0)

def transform_weights(k, g, n0):
    "u^2 - v^2 = 1 (canonical).  u->1,v->0 at large k; both ~1/k (phonon) at small k."
    T = kinetic(k); E = excitation_energy(k, g, n0)
    u2 = (T + g * n0 + E) / (2.0 * E)
    v2 = (T + g * n0 - E) / (2.0 * E)
    return u2, v2

def sound_speed(g, n0):
    "c = sqrt(g n0) = sqrt(dP/drho): the linear small-k slope of E(f)."
    return np.sqrt(g * n0)

def critical_velocity(g, n0, kmax=50.0, npts=200000):
    "Landau: V_c = min_f E(f)/|f| > 0 because the small-f branch is linear."
    k = np.linspace(kmax / npts, kmax, npts)
    return np.min(excitation_energy(k, g, n0) / k)

def quantum_depletion(g, n0):
    "(N - N0)/N = (8/3 sqrt(pi)) sqrt(n a^3): the ground-state depletion (a=g/4pi)."
    a = g / (4.0 * np.pi)
    return (8.0 / (3.0 * np.sqrt(np.pi))) * np.sqrt(n0 * a ** 3)

if __name__ == "__main__":
    g, n0 = 1.0, 0.01
    print("c  =", sound_speed(g, n0))
    print("Vc =", critical_velocity(g, n0), " (equals c)")
    print("depletion =", quantum_depletion(g, n0))
```
