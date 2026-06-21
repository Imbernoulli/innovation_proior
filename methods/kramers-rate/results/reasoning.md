Let me start from what actually bothers me. The Arrhenius law, k = ν·exp(−E_b/k_BT), fits everything — the exponential is the cost of reaching the barrier, and that part I believe. It is the prefactor ν that I cannot account for. The equilibrium-flux recipe — take the saddle on the potential surface, assume the activated complex there is in thermal equilibrium with the reactants, count the one-way flux of phase points crossing the saddle toward products — gives a clean number for ν out of pure statistical mechanics. And it works, often beautifully. But I keep tripping over the word *assume*. It assumes two things, and both are statements about dynamics, while the recipe itself uses only equilibrium. First, that a system which crosses the saddle moving toward products actually ends up as products — that it never turns around and comes back. Second, that the population sitting at the saddle stays at its equilibrium value, that the reactants keep delivering fresh activated systems fast enough to replace the ones leaking away. People write a factor κ ≤ 1 in front to cover the slippage, but nobody can compute κ, and nobody can tell me when κ = 1.

Both of those assumptions are about the *medium* — the solvent, the gas of colliding molecules, the other vibrational modes of the molecule, the internal friction of the nuclear droplet in the fission problem. So the prefactor cannot really be a property of the well and the saddle alone. It must depend on how strongly the reacting coordinate is coupled to its surroundings. And there is the lever I want: a model where that coupling is a single tunable number. Dial it from nothing to enormous, watch ν, and the two failure modes should reveal themselves — one at weak coupling (the supply runs dry), one at strong coupling (the back-and-forth at the saddle). If I can do that, I will know exactly when the equilibrium recipe is right and what it is missing when it is wrong.

So: one particle, one coordinate q, mass 1, in an external field K(q) = −U'(q), and on top of that an irregular force X(t) from a medium held at temperature T. I'll measure temperature in energy units, k_B = 1, to keep the algebra clean. The equations of motion are
  ṗ = K(q) + X(t),   q̇ = p.
That's a Langevin pair. Now — do I want to chase individual noisy trajectories? No. A trajectory is hopeless to control, and what I want, the *rate*, is a statistical quantity. I want the density ρ(p,q,t) of an ensemble of such particles in phase space, and an equation for how it flows. The escape rate will be a steady current of this density over the barrier.

Let me build that equation honestly from the noise. Pick a coarse-graining time τ — short enough that the velocity barely changes over it, long enough that the random force at the end of τ has forgotten its value at the start. Over τ the random force delivers an impulse
  B_τ = ∫ X(t') dt'   (integrated over the interval τ),
a random number whose distribution I'll call φ_τ(B; p,q). Its moments matter. The first non-vanishing dependence on τ should be linear: I'll write the n-th moment as
  ⟨B_τ^n⟩ = μ_n · τ.
Why linear in τ? For the second moment and higher there's a subtlety I should respect. ⟨B_τ^n⟩ is an n-fold time integral of ⟨X(t₁)X(t₂)…X(t_n)⟩ over a little cube of side τ. If the forces at well-separated times are independent, most of that cube contributes nothing; the only piece that survives is the thin tube along the diagonal t₁ = t₂ = … = t_n, where the times sit within a correlation time of each other, and that tube has volume ∝ τ. So yes, each moment picks up a term ∝ τ, and I keep that leading term. I won't assume a priori which μ_n vanish — I'll let physics decide.

Now propagate the density one step. The density at (p₁,q₁) at time t+τ comes from where things were at time t. In time τ, position advances by p·τ and momentum by K·τ plus the random impulse B. So a particle now at (p₁,q₁) was, a step ago, at momentum p₂ = p₁ − Kτ − B and position q₂ = q₁ − p₁τ (drift the position back along the streamline). Write
  ρ(p₁,q₁,t+τ) = ∫ ρ(p₂, q₂, t)·φ_τ(B; p₂, q₂) dB,
with p₂ = (p₁ − Kτ) − B. Expand to first order in τ and to as many powers of B as I need (B itself is order √τ, so B² is order τ — I have to keep up to B²):
  ρ + ∂ρ/∂t·τ + ∂ρ/∂p·Kτ + ∂ρ/∂q·pτ = ∫ [ρφ − B·∂/∂p(ρφ) + (B²/2)·∂²/∂p²(ρφ) − …] dB.
Integrate over B using the moments: ∫φ dB = 1, ∫Bφ dB = μ₁τ, ∫B²φ dB = μ₂τ, and so on. Divide by τ. The streaming terms move to the left, and what's left on the right is the medium's action:
  ∂ρ/∂t = −K(q)∂ρ/∂p − p∂ρ/∂q − ∂/∂p(μ₁ρ) + ½∂²/∂p²(μ₂ρ) − …   (★)
This is the Fokker–Planck equation in phase space. The first two terms are just the Liouville/Gibbs streaming of the ensemble along its mechanical trajectories; the rest are the corrections from the Brownian forces. The current has a q-component p·ρ and a p-component K·ρ + μ₁ρ − ½∂/∂p(μ₂ρ) − … . Good — that's a continuity equation, exactly what a flux-over-population rate will need.

I still have to pin down the μ_n. Here's the constraint I trust completely: the medium is at temperature T, so the *equilibrium* of (★) had better be the Boltzmann distribution
  ρ_B = exp(−(½p² + U(q))/T),   K = −∂U/∂q,
and it had better be stationary. Plug ρ_B in and demand ∂ρ_B/∂t = 0. The streaming terms −K∂ρ_B/∂p − p∂ρ_B/∂q cancel by themselves (Boltzmann is a function of the energy, and streaming conserves energy). So the bracket of Brownian terms must vanish on its own:
  ∂/∂p[ −μ₁ ρ_B + ½∂/∂p(μ₂ ρ_B) − (1/6)∂²/∂p²(μ₃ ρ_B) + … ] = 0.
Carrying out the p-derivatives on ρ_B (each ∂/∂p brings down a factor −p/T) and dividing out ρ_B, I get a relation of the form
  −μ₁ − (p/T)·½μ₂ + ½∂μ₂/∂p − … = F(q,T)·(something),
and stationarity demands the whole thing vanish. Now physical symmetry: μ₁, μ₃, … are odd in p (a drag and its odd corrections), μ₂, μ₄, … are even (a diffusion). The simplest closure that makes the bracket vanish identically is
  μ₁ = −η p,   μ₂ = 2ηT,   μ₃ = μ₄ = … = 0.
Check: −(−ηp) − (p/T)·½·(2ηT) = ηp − ηp = 0. ✓. So the drag is −ηp and the momentum-diffusion is 2ηT — and notice I did not *postulate* the relation between them; stationarity of Boltzmann *forced* μ₂ = 2ηT once μ₁ = −ηp. That's the fluctuation–dissipation relation falling out as a consistency condition: the same η that drags energy out must, through the noise, feed it back at the rate set by T, or equilibrium would drift. (Could there be richer friction laws — μ₁ = −ηp − ζp³, μ₂ = 2ηT + 6ζT², with μ₃ not even vanishing? Stationarity allows them. But I know of no physical situation that needs them, so I'll take the simple Einstein closure.) With η taken independent of q, (★) becomes
  ∂ρ/∂t = −K(q)∂ρ/∂p − p∂ρ/∂q + η·∂/∂p( pρ + T·∂ρ/∂p ).   (FPE)
This is the object. η is my knob. Everything now is solving this for a quasi-stationary escape current and watching the answer as η runs from tiny to huge.

Let me get my bearings on the rate definition first, so I know what I'm computing. The barrier E_b ≫ T — escape is rare. So inside the well A the system thermalizes long before any appreciable number has escaped; I can treat A as a quasi-infinite Boltzmann reservoir slowly leaking over the saddle C to the product side B, with B kept empty (absorb anything that arrives). Then there's a steady current w of probability over the barrier, fed from the reservoir, and the rate is just
  r = w / n_A,
the flux over the population of A. That's the flux-over-population picture. Fine. Now solve the FPE for w in the two extremes of η, where I expect it to collapse to something one-dimensional.

Start with large η. The friction is so strong it dominates the external force. Then the velocity gets slammed into a local Maxwell distribution almost instantly — after a time of order 1/η — at every q. So
  ρ(q,p,t) ≈ σ(q,t)·exp(−p²/2T),
with σ the spatial density, and what's left is slow diffusion of σ in q. To extract its equation, rewrite the FPE by completing the friction operator. Group the terms as
  ∂ρ/∂t = η·∂/∂p( pρ + T∂ρ/∂p − (K/η)ρ + (T/η)∂ρ/∂q ) − ∂/∂q( (K/η)ρ − (T/η)∂ρ/∂q ).
Now the trick: integrate both sides along a line of constant q + p/η (these are the directions the strong friction sweeps the density along on the fast timescale), from p = −∞ to +∞. Call the integral of ρ along that line σ(q₀). The first big group is a total p-derivative, so it integrates to zero at the limits. What survives is
  ∂σ/∂t ≈ −∂/∂q₀( (K/η)σ(q₀) − (T/η)∂σ(q₀)/∂q₀ ).   (Smol)
The Smoluchowski equation — overdamped diffusion of position with diffusion constant D = T/η. (This is legitimate only when the velocity range that matters, |p| ≲ √T, corresponds to a spread in q of order √T/η that is small compared to the scale over which K and σ vary — which is exactly the condition for the Smoluchowski limit to apply in the first place. Consistent.) The stationary current is
  w = (K/η)σ − (T/η)∂σ/∂q = const,
and since this is −(T/η)·exp(−U/T)·∂/∂q(σ·exp(U/T)), I can integrate it between A and B:
  w = T·[σ·exp(U/T)]_A^B / ( η·∫_A^B exp(U/T) dq ).   (curr-L)
Good — this is the high-friction tool. Hold onto it.

Now the opposite, small η. The friction is feeble; over one oscillation in the well it barely changes the energy. So the fast variable is the *phase* of the oscillation and the slow variable is the *energy*. The right reduction isn't diffusion in q at all — it's diffusion in energy. Let I(E) = ∮ p dq be the action (the area enclosed by the orbit of energy E), and average the FPE over one constant-energy ring. The streaming terms −K∂ρ/∂p − p∂ρ/∂q vanish on the average over a ring, because in the absence of friction the orbit conserves E and just goes around. The friction term survives. Using ∂q/∂p relations on a ring and ⟨p²⟩ = I·ω where ω = dE/dI is the orbital frequency (the action-integral over one period divided by the period), the averaged equation is
  ∂ρ/∂t = η·∂/∂I( Iρ + TI·∂ρ/∂E ).   (En)
That's diffusion along the energy/action coordinate, with the diffusion measured by η·T·I and the geometry by ω = dE/dI. Its stationary current is
  w = −η( Iρ + TI·∂ρ/∂E ) = −ηT·I·exp(−E/T)·∂/∂E( ρ·exp(E/T) ),
which integrates between two energies to
  w = ηT·[ρ·exp(E/T)]_A^B / ∫_A^B (1/I)·exp(E/T) dE.   (curr-S)
This is the low-friction tool. Now assemble the rates.

Now put in the barrier on the high-friction side. Take U smooth with a single barrier — well at A, maximum at C, downhill to B — and use (curr-L). The integral ∫_A^B exp(U/T) dq is overwhelmingly dominated by the neighborhood of the top C, where U is largest. Near C write
  U ≈ E_b − ½(2πω′)²(q − q_c)²,
a parabola inverted, with 2πω′ the angular frequency of that inverted (unstable) mode — I'm writing the *ordinary* frequencies ω, ω′, so the angular frequencies carry the 2π. Then
  ∫ exp(U/T) dq ≈ exp(E_b/T)·∫ exp(−(2πω′)²(q−q_c)²/2T) dq = exp(E_b/T)·(1/ω′)·√(T/2π).
For the population, near A write U ≈ ½(2πω)²q² (well frequency ω) and σ ≈ σ_A·exp(−U/T), so
  n_A = ∫ σ_A·exp(−(2πω)²q²/2T) dq = (σ_A/ω)·√(T/2π).
The current is w = (T/η)·σ_A·exp(0)/(integral) — with [σexp(U/T)]_A = σ_A at A (U=0 there) and ≈0 at B — so w = (T/η)·σ_A / [exp(E_b/T)·(1/ω′)√(T/2π)]. Divide by n_A:
  r = w/n_A = (2π·ω·ω′/η)·exp(−E_b/T).   (R-high)
There it is — the high-friction rate. It is the equilibrium-flux prefactor *cut down by a factor ∝ 1/η*. The stronger the friction, the slower the escape. That's the recrossing failure made quantitative: in a thick medium the coordinate crawls over the top and is dragged back again and again, and the net forward current scales as 1/η. Note (R-high) bears no resemblance to anything the equilibrium recipe could produce, because η is not in the recipe at all.

Now I want more than the high-friction limit — I want a formula good across the whole range, not just η → ∞. And I notice something: in (curr-L) and the Smoluchowski reduction I *assumed* large η to drop the velocity. To get the intermediate behavior I must go back to the full FPE near the barrier and solve it *without* throwing away the velocity. Let me try, because the resistance to escape is localized right at C — that's where σ·exp(U/T) plunges from its reservoir value to zero — so I only need the FPE in a small neighborhood of C, where U is the inverted parabola. That makes the equation quadratic-coefficient, and maybe exactly solvable.

Set q' = q − q_c, and use U ≈ E_b − ½(2πω′)²q'². The stationary FPE near C is
  0 = −(2πω′)²q'·∂ρ/∂p − p·∂ρ/∂q' + η·∂/∂p( pρ + T·∂ρ/∂p ).
(The sign: K = −∂U/∂q' = +(2πω′)²q', and the streaming p-term is −K∂ρ/∂p = −(2πω′)²q'∂ρ/∂p.) Strip out the equilibrium part by writing
  ρ = ζ · exp(−(p² − (2πω′)²q'²)/2T),
which is exp(−H/T) with the local barrier Hamiltonian H = ½p² − ½(2πω′)²q'². The constant ζ = const reproduces thermal equilibrium and carries no net current, so the interesting physics is in a *non-constant* ζ. Substituting and simplifying — every place the exponential's derivative would appear cancels against the streaming, leaving
  0 = −(2πω′)²q'·∂ζ/∂p − p·∂ζ/∂q' − ηp·∂ζ/∂p + ηT·∂²ζ/∂p².   (Z)

Two independent variables, second order — not obviously tractable. Let me guess that ζ depends on p and q' only through *one* linear combination,
  ζ = ζ(u),   u = p − a q',
for some constant a I'll fix. Why a linear combination? Because the equilibrium solution ζ = const is the trivial member of this family, and the next-simplest non-equilibrium correction should be organized along the single special direction the barrier dynamics picks out — the unstable normal mode through the saddle. Collapsing to one variable will turn (Z) into an ODE if it works. Compute the derivatives: ∂ζ/∂p = ζ', ∂ζ/∂q' = −a ζ', ∂²ζ/∂p² = ζ''. Insert:
  0 = −(2πω′)²q'·ζ' − p·(−a ζ') − ηp·ζ' + ηT·ζ''
    = [ a p − (2πω′)²q' − ηp ]·ζ' + ηT·ζ''
    = [ (a−η)p − (2πω′)²q' ]·ζ' + ηT·ζ''.   (Z2)
For this to be a function of u = p − aq' alone, the bracket must be proportional to u, i.e.
  (a−η)p − (2πω′)²q' = (a−η)·(p − a q').
Matching the q' coefficient: −(2πω′)² = (a−η)·(−a), so
  (2πω′)² = a(a − η),   ⇒   a = η/2 ± √( η²/4 + (2πω′)² ).   (a-cond)
Beautiful — it closes. With that a, (Z2) becomes
  0 = (a−η)·u·ζ' + ηT·ζ'',
a first-order ODE in ζ':
  ζ''/ζ' = −(a−η)·u/(ηT),   ⇒   ζ' ∝ exp( −(a−η)u²/2ηT ),
and so, besides the constant solution,
  ζ(u) = K·∫ exp( −(a−η)u²/2ηT ) du.   (Z3)
This is a Gaussian integral — an error function. For it to be a sensible, bounded, *decaying* solution I need (a−η) > 0, i.e. the **upper** sign in (a-cond): a = η/2 + √(η²/4 + (2πω′)²), so a − η = −η/2 + √(η²/4 + (2πω′)²) > 0. With that sign, ζ runs from a constant to zero as u sweeps from −∞ to +∞ — which is exactly the boundary condition I want: on the reactant side (well to the left of C) ρ approaches thermal equilibrium (ζ → const), and on the product side (right of C) ρ → 0 because B is kept empty. The other root would give a diverging, unphysical ζ. So the boundary conditions *select* the unstable-mode root; the math and the physics agree on which sign.

Now read off the numbers. Take the lower limit of the integral to −∞ (no particles to the right of C). Well to the left of C, u → −∞ and ζ → its full value: ∫_{−∞}^{∞} exp(−(a−η)u²/2ηT) du = √(2πηT/(a−η)), so deep in the well
  ζ → K·√(2πηT/(a−η)),
and the density near A is ρ ≈ K·√(2πηT/(a−η))·exp(−(p²+(2πω)²q²)/2T) (the well's own parabola). The number of particles caught near A:
  n_A = K·√(2πηT/(a−η))·∫∫ exp(−(p²+(2πω)²q²)/2T) dp dq = K·√(2πηT/(a−η))·(2πT/2πω)
      = K·√(2πηT/(a−η))·(T/ω).
The current through C is the integral of p·ρ over p at q' = 0:
  w = ∫ dp · p·ρ(0,p) = K·∫ dp · p · exp(−p²/2T)·∫_{−∞}^{p} exp(−(a−η)p''²/2ηT) dp''.
Doing the double Gaussian (integrate by parts, or recognize it as a standard form): the two Gaussians combine to give
  w = K·T·√(2πηT/a).
(The key is that the inner error-function integral and the p·exp(−p²/2T) weight produce a single Gaussian of width set by the *combination* a, not a−η — that's the a in the denominator.) Now the rate:
  r = w/n_A = [K·T·√(2πηT/a)] / [K·(T/ω)·√(2πηT/(a−η))]
            = ω·√((a−η)/a)·exp(−E_b/T).
Wait — where did the exp(−E_b/T) come from? n_A was computed with the well bottom as the energy zero, but ζ near A carried the reservoir value at energy 0, while the current w was evaluated at C at energy E_b; tracking the Boltzmann factor through ρ = ζ·exp(−H/T) with H measured from the well puts exactly one exp(−E_b/T) between the populated well and the saddle. So
  r = ω·√((a−η)/a)·exp(−E_b/T).
Substitute a and a−η from (a-cond), upper sign. With a = η/2 + R and a−η = R − η/2 where R = √(η²/4 + (2πω′)²), the ratio
  √((a−η)/a) = √( (R − η/2)/(R + η/2) ).
Multiply numerator and denominator inside by (R − η/2): (R−η/2)²/(R²−η²/4) = (R−η/2)²/(2πω′)². So √((a−η)/a) = (R − η/2)/(2πω′). Hence
  r = (ω/2πω′)·( √(η²/4 + (2πω′)²) − η/2 )·exp(−E_b/T).   (R-full)
This is the formula I was after — valid for any η near a parabolic barrier. Let me check its limits.
 — η/2 ≫ 2πω′ (strong friction): √(η²/4 + (2πω′)²) − η/2 = (η/2)√(1 + 4(2πω′)²/η²) − η/2 ≈ (η/2)(1 + 2(2πω′)²/η²) − η/2 = (2πω′)²/η. So r → (ω/2πω′)·(2πω′)²/η = 2π·ω·ω′/η · exp(−E_b/T) — exactly (R-high). ✓.
 — η/2 ≪ 2πω′ (weak friction, still spatial-diffusion picture): √(η²/4 + (2πω′)²) − η/2 → 2πω′, so
  r → ω·exp(−E_b/T).   (R-TST)
And ω·exp(−E_b/T) is *precisely the equilibrium-flux value*. Let me confirm that independently: the equilibrium recipe counts the one-way flux over C. With the Boltzmann–Gibbs density ρ₀ = exp(−E/T), the flux from left to right through C is ∫₀^∞ p·ρ₀ dp at q=q_c = exp(−E_b/T)·∫₀^∞ p·exp(−p²/2T) dp = T·exp(−E_b/T). The population caught near A is the double integral over the well, T/ω. So r_eq = T·exp(−E_b/T)/(T/ω) = ω·exp(−E_b/T). ✓. So (R-full) interpolates smoothly: it *is* the equilibrium value at small-to-moderate friction and falls as 1/η at large friction. The transmission factor — the ratio of the true rate to the equilibrium prefactor — is
  κ = r/r_eq = (1/2πω′)·( √(η²/4 + (2πω′)²) − η/2 ) = √(1 + (η/4πω′)²) − η/4πω′,
i.e. in angular frequencies ω_b = 2πω′, κ = √(1 + (η/2ω_b)²) − η/2ω_b. Exactly 1 at η → 0, and ω_b/η at η → ∞. Lovely.

But wait. I've been sloppy in calling the small-η limit of (R-full) "the equilibrium value, period." (R-full) was derived assuming the velocity has time to do its thing near the barrier — it's still a *spatial*-diffusion-type solution. As η → 0 it says the rate plateaus at the equilibrium value and stays there. That cannot be the whole story, because I argued at the very start that *very* weak coupling must *starve* the reaction: if the medium barely talks to the coordinate, it cannot deliver energy E_b fast enough to keep the saddle supplied. The plateau in (R-full) must break down at sufficiently small η. So (R-full) is right coming down from large η through the plateau, but it misses the energy-supply bottleneck at the bottom. For that I need the energy-diffusion tool (curr-S), not this one.

Back to the very-weak-friction side, with (curr-S). Now the rate is throttled by diffusion in energy up to the barrier energy E_b. Assume a system that does manage to leave near C essentially never comes back — so set ρ·exp(E/T) ≈ 0 at the top, and take the lower limit "near A" to mean energy of order T (integrating literally from E = 0 makes ∫(1/I)exp(E/T)dE diverge at I = 0, which just reflects that the deep well is irrelevant to the bottleneck). Then
  w ≈ ηT·ρ_A / ∫_{~T}^{E_b} (1/I)·exp(E/T) dE.
The integral is dominated by E near E_b. Pull I out at its barrier value I_c = I(E_b) and substitute E = E_b − (E_b − E):
  ∫ (1/I)·exp(E/T) dE ≈ (1/I_c)·exp(E_b/T)·∫₀^∞ exp(−(E_b−E)/T) d(E_b−E) = (T/I_c)·exp(E_b/T).
So w ≈ ηT·ρ_A·(I_c/T)·exp(−E_b/T) = η·ρ_A·I_c·exp(−E_b/T). The population near A is n_A = ρ_A·T/ω (the well's phase-space volume in this normalization). Hence
  r = w/n_A = η·(I_c·ω/T)·exp(−E_b/T).
For a near-harmonic well I(E) ≈ E/ω, so the barrier action I_c ≈ E_b/ω, giving I_c·ω ≈ E_b and
  r ≈ η·(E_b/T)·exp(−E_b/T).   (R-low)
Now the rate is *proportional to η*: it rises linearly from zero as the friction is turned up. This is the energy-supply failure made quantitative — at vanishing coupling the medium cannot pump energy E_b into the mode, so the escape is rate-limited by that slow energy diffusion, and turning up η speeds it up. (Note the subtlety the spatial picture never saw: it's the action *at the barrier energy* — including any anharmonicity of the well out near E_b — that sets the weak-friction rate, not just the bottom-of-well curvature.)

So the full picture as η runs:
 — η very small: r ≈ η·(E_b/T)·exp(−E_b/T), rising linearly (energy diffusion, supply-limited).
 — η intermediate: r ≈ ω·exp(−E_b/T), the equilibrium-flux value — a *plateau*. This is where both deliveries are adequate and recrossing is negligible.
 — η large: r ≈ (2π·ω·ω′/η)·exp(−E_b/T), falling as 1/η (spatial diffusion, recrossing-limited).
The rate-versus-friction curve rises, plateaus at the equilibrium value, then falls. The equilibrium-flux prefactor is not "the answer" — it is the *top of the curve*, the best you can ever do, achieved only in a window of medium coupling. Outside that window it overestimates the rate, for two opposite reasons. That answers the question I started with: the recipe is reliable in a band of friction, roughly from η of order ω·T/E_b up to η of order ω′ (the *ordinary* barrier frequency — about a fifth of the angular barrier frequency 2πω′, since κ has fallen only to about 0.9 by η ≈ 1.2ω′) — for the illustrative case E_b/T = 10, ω′ = ω, that's a wide band where it's good to about 10%.

Now — an honest difficulty. (R-full) covers the plateau and the high-friction side (it came from the spatial picture). (R-low) covers the rising weak-friction side (it came from the energy picture). They overlap *only* because E_b/T is large: both reduce to the equilibrium plateau value over a stretch of moderate friction, so the curve is continuous. But the energy picture (R-low) was derived assuming η small enough that the energy changes little over an oscillation — its own validity needs η small compared to ω. Aperiodic damping sets in around η ≈ 4πω; for friction approaching that, the energy is no longer slow and (R-low) cannot be trusted. The spatial picture (R-full) was derived assuming the velocity equilibrates near the barrier, which is the *opposite* regime. There is a region between them — right around the peak of the turnover, where the friction is neither small nor large compared to the barrier frequency — that *neither* derivation reaches. I have not found a trustworthy way to bridge it: to extend (R-low) up to η-values not small compared to 4πω would require solving the full phase-space diffusion equation through the saddle with energy and phase both varying, which I cannot do in closed form here. The cleanest hope would be the exactly-parabolic well, where the fundamental diffusion equation might be solved with the right boundary conditions; but I leave that open. The result I can stand behind is the pair of closed forms with their domains, and the turnover picture they jointly paint — and, crucially, the fact that for large E_b/T the gap between the two valid regimes is harmless: even at very small η the plateau formula and the energy-diffusion formula already agree with each other and with the equilibrium value, so the only genuinely uncertain region is a narrow neighborhood of the peak that doesn't change the qualitative story.

Let me make the turnover concrete with a small computation — it's the cleanest way to see the plateau and the two slopes, and to check that my two formulas join up. Mass 1, k_B = 1; take ω′ = ω = 1 (so ω_b = 2π) and E_b/T = 10, the illustrative case. Sweep η and evaluate the spatial-diffusion rate (R-full) and the energy-diffusion rate (R-low), each as a ratio to the equilibrium plateau ω·exp(−E_b/T), and read off where they cross and where the true rate (the lower of the two branches, since whichever bottleneck is worse controls the rate) peaks.

```python
import numpy as np

# units: mass = 1, k_B = 1.  omega0, omegab are ANGULAR frequencies.
def k_eq(omega0, Eb, T):
    # equilibrium one-way flux over the saddle: the plateau / best-case prefactor
    return (omega0 / (2.0 * np.pi)) * np.exp(-Eb / T)

def transmission_factor(omegab, eta):
    # kappa = lambda_+/omegab, lambda_+ = -eta/2 + sqrt(omegab^2 + (eta/2)^2)
    #       = sqrt(1 + (eta/2/omegab)^2) - eta/(2*omegab)
    lam_plus = -eta/2.0 + np.sqrt(omegab**2 + (eta/2.0)**2)
    return lam_plus / omegab

def k_spatial(omega0, omegab, eta, Eb, T):
    # plateau -> 1/eta side: k = kappa * k_eq;  high-eta -> omega0*omegab/(2pi*eta) e^{-Eb/T}
    return transmission_factor(omegab, eta) * k_eq(omega0, Eb, T)

def k_energy(omega0, eta, Eb, T, Ib):
    # supply-limited weak-friction side: k = eta * Ib/T * (omega0/2pi) e^{-Eb/T}, ~ eta
    return eta * (Ib / T) * (omega0 / (2.0 * np.pi)) * np.exp(-Eb / T)

omega0 = omegab = 2*np.pi      # ordinary frequency omega = omega' = 1
T, Eb = 1.0, 10.0
f = omega0 / (2.0*np.pi)       # ORDINARY well frequency (= 1 here)
Ib = Eb / f                    # near-harmonic barrier action I_c = E_b/omega (omega ordinary)

keq = k_eq(omega0, Eb, T)
etas = np.logspace(-3, 3, 400) * omegab     # sweep eta around the barrier frequency
ks = k_spatial(omega0, omegab, etas, Eb, T)
ke = k_energy(omega0, etas, Eb, T, Ib)
true_rate = np.minimum(ks, ke)              # the worse bottleneck wins
i = int(np.argmax(true_rate))
print("plateau k_eq =", keq)
print("turnover peak at eta/omegab =", etas[i]/omegab, " k_peak/k_eq =", true_rate[i]/keq)
# limits: kappa(eta->0)=1 (equilibrium plateau); kappa(eta->inf)=omegab/eta (1/eta)
assert abs(transmission_factor(omegab, 1e-4*omegab) - 1.0) < 1e-3
assert abs(transmission_factor(omegab, 1e4*omegab) - 1.0/1e4) / (1.0/1e4) < 1e-3
```

Running it through in my head against the closed forms: at η/ω_b ≪ 1 the spatial branch sits at the plateau (κ → 1) while the energy branch rises linearly and is the smaller — so the true rate is the rising energy branch; they cross where η·(E_b/T) ≈ ω, i.e. η ≈ ω·T/E_b, which for E_b/T = 10 is η ≈ 0.1·ω — the peak sits just below the plateau right there. For η/ω_b ≫ 1 the spatial branch falls as 1/η and is the smaller — recrossing-limited. The two slopes and the plateau between them are the turnover. The equilibrium prefactor is the ceiling, touched only in the middle band.

To recap the causal chain: I doubted the equilibrium-flux prefactor because its two load-bearing assumptions — no recrossing, and a saddle kept at equilibrium population — are dynamical, while the recipe is purely equilibrium; so the prefactor must depend on the system's coupling to its medium. I modeled that coupling by one friction coefficient η in a Langevin equation, and turned the dynamics into a Fokker–Planck flow of the phase-space density, fixing the noise strength by demanding the Boltzmann distribution be stationary (which handed me the fluctuation–dissipation relation). Treating escape as a quasi-stationary flux over the barrier divided by the well population, I solved the flow in two reductions: strong friction collapses it to spatial diffusion, weak friction to energy diffusion. Strong friction gives a rate falling as 1/η (recrossing); an exact parabolic-barrier solution, organized along the unstable mode, gives the transmission factor κ = √(1 + (η/2ω_b)²) − η/2ω_b that interpolates from the equilibrium value at small friction down to the 1/η law at large friction; weak friction gives a rate rising linearly in η (energy starvation). Together they show the escape rate rising, plateauing at the equilibrium-flux value, then falling — so the equilibrium recipe is the best-case ceiling, valid only in a band of medium coupling, and I can now name the band and the corrections on either side. The one piece I cannot close is the bridge directly over the turnover peak for friction comparable to the barrier frequency; for a high barrier that gap is narrow and the two valid branches already agree where they should.
