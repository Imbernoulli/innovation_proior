Let me start from the thing that keeps nagging me, which is the magnet. The exchange interaction between spins, `H = −J Σ Sᵢ·Sⱼ`, doesn't care which way the spins point — rotate them all together and the energy is identical. The interaction has the full rotational symmetry of empty space. And yet a cold ferromagnet is magnetized along *a* direction; the realized state has picked one, and the symmetry of the laws is nowhere to be seen in the state. The symmetry didn't go anywhere — every other direction is an equally good ground state, sitting right there at the same energy. The state just had to choose. That's the whole puzzle in miniature, and I want to know if it can happen in a relativistic field theory, and if it does, what it costs.

So let me hold onto the structure of the magnet, because it's telling me something I shouldn't lose. There's a continuum of ground states, all degenerate, all related to each other by the symmetry operation. Pick one. Now ask: what are the cheap excitations? If I twist the magnetization slightly here and slightly there — a long, slow ripple in the direction of the spins — how much energy does that cost? Here's the key: if I twist *every* spin by the same angle, that's just a global rotation, and by the symmetry it costs *exactly nothing*. So a twist that varies very slowly in space, that is locally almost a uniform rotation, must cost *almost* nothing. The cost has to vanish as the wavelength goes to infinity. The excitation energy ω(k) → 0 as k → 0. There is no gap. The broken rotational symmetry has *forced* a soft, gapless mode into existence — the spin wave. Twisting is free in the long-wavelength limit *because* twisting everything uniformly is a symmetry.

I want to feel why that's inevitable and not an accident of magnets. The reason is purely geometric. The degenerate ground states form a *connected set* — a circle, a sphere, whatever the broken symmetry sweeps out. Moving along that set costs no energy by definition; it's the flat valley floor of the energy landscape. A long-wavelength excitation is exactly "move along the valley floor, but let the position along the floor vary slowly across space." The only energy cost is gradient energy from the slow spatial variation, and that scales away as k → 0. So *any* broken continuous symmetry should come with a gapless mode living in the flat valley direction. That's the intuition I want to make exact.

Now let me try to build the smallest possible field theory that does this, because magnets are full of lattice complications I don't need. Forget spins. Take a single scalar field with a potential and ask whether the field-theory version of "the ground state breaks the symmetry" can be made to happen on paper, and then read the spectrum straight off the Lagrangian.

Mass first, because mass is the thing I'm trying to track. For a scalar field with `L = ½(∂φ)² − ½m²φ²`, the mass-squared is just the coefficient of the quadratic term. More generally, if I have a potential `V(φ)` and I expand about a configuration `φ₀`, the mass-squared of the small oscillation is `V''(φ₀)` — the curvature of the potential at the point I'm sitting at. A particle is massless exactly when the potential is *flat* in its direction: no restoring force, no oscillation, no rest energy. That's the bridge. "Flat valley direction of the energy landscape" from the magnet, and "massless particle" from field theory, are the *same statement*: `V'' = 0` along the broken-symmetry direction. If I can build a field theory whose ground state sits in a degenerate valley, the field component pointing along the valley will be massless, automatically.

Let me try the obvious thing and watch it break, to make sure I understand the obstruction. Take a real scalar with the standard `V = ½ m² φ² + (stuff)` and quantize around `φ = 0` as usual. Fine if `m² > 0`. But suppose `m² < 0`. Now `φ = 0` is a *maximum* of the potential, not a minimum. If I blindly expand there I get a tachyonic `m² < 0` and a perturbation series built on an unstable vacuum — nonsense. The literature's reflex is to say "the theory with `m² < 0` doesn't exist, throw it out." And that's where I want to stop and push, because I don't think the theory doesn't exist — I think the *expansion point* is wrong. The negative mass-squared isn't a sickness of the theory; it's a signal that `φ = 0` is the wrong vacuum. The field wants to roll downhill to somewhere else. The wrong move is to quantize about the hilltop. The right move is to find where the field actually sits and quantize *there*.

So make the potential bounded below — I need a positive quartic to catch the field as it rolls down. Real scalar, with the reflection symmetry `φ → −φ`:

  `V(φ) = −½ μ² φ² + ¼ λ φ⁴`,  with `μ² > 0`, `λ > 0`.

The quadratic coefficient is negative (that's the `m² < 0` situation written honestly), the quartic catches it. Where are the minima? `V'(φ) = −μ² φ + λ φ³ = φ(−μ² + λ φ²) = 0`, so either `φ = 0` (that's the maximum) or `φ² = μ²/λ`, i.e. `φ = ±v` with `v = μ/√λ`. Two minima, symmetric about the origin, at the bottom of a double well. Check the curvature there: `V''(φ) = −μ² + 3λφ²`, and at `φ² = μ²/λ` this is `−μ² + 3μ² = 2μ² > 0`. Good, a genuine minimum, and the oscillation about it has

  `m²_oscillation = V''(v) = 2μ² > 0`.

A perfectly healthy massive particle. Now expand the field about a *real* vacuum: write `φ = v + σ` where σ is the fluctuation. Then `(□ + 2μ²) σ = 0` for the small oscillation — mass-squared `2μ²`, positive, no tachyon. The disease is cured the moment I expand about the true minimum instead of the hilltop.

And look what happened to the symmetry. The Lagrangian had `φ → −φ`. The vacuum `φ = +v` does *not* respect it — `φ = +v` maps to `φ = −v`, the *other* vacuum. By choosing one of the two minima I have thrown away the reflection symmetry. The symmetry of the laws is still there in the sense that `±v` are degenerate, but the state I built on top of `+v` knows nothing of `−v`; they're orthogonal, with a superselection rule between them — you can't superpose them into a symmetric combination in any physical way. The symmetry is *spontaneously broken* by the vacuum, with no asymmetric term anywhere in the Lagrangian. That's the mechanism, stripped to its bones. A symmetric Lagrangian, a non-symmetric ground state, and the asymmetry is an *output*, not an input.

But this discrete case gave me a *massive* particle, `m² = 2μ²`, and no massless mode at all. That's right and it's the point: the broken symmetry here is *discrete* (just `φ → −φ`), so there's no valley to walk along — only two isolated points. No continuous family of degenerate vacua means no flat direction means no massless mode. The double well has no valley floor, just two pits. So if I want the magnet's gapless spin wave, I need the symmetry to be *continuous*, so the degenerate vacua form a connected curve I can slide along.

Make the field complex. One complex scalar `φ` is two real fields. Give it a phase symmetry `φ → e^{iα} φ` — a continuous U(1), the analog of "rotate all the spins." The invariant potential is a function of `|φ|²` only:

  `V(φ) = −μ² |φ|² + λ |φ|⁴`,  `μ² > 0`, `λ > 0`.

Now find the vacuum. `V` depends only on `|φ|²`; let `r = |φ|`. `V(r) = −μ² r² + λ r⁴`, `dV/dr = −2μ² r + 4λ r³ = 2r(−μ² + 2λ r²) = 0`, so `r² = μ²/2λ`. Define

  `v² ≡ μ²/2λ`,  so the minima are at `|φ| = v`.

And here is the thing the discrete case couldn't give me: the condition is `|φ| = v` — the *modulus* is fixed, the *phase* is completely free. The set of minima is an entire **circle** of radius `v` in the complex plane. The potential is a surface of revolution with a circular trough at the bottom — the wine-bottle, the Mexican hat: a bump at the origin (`φ = 0` is the maximum) surrounded by a ring of degenerate minima. Every point on that circle is an equally good vacuum, and they are all related to each other by the symmetry `φ → e^{iα}φ`. The vacuum must pick one phase. Picking it breaks the U(1), exactly as the magnet picks a direction.

So pick one. The phase is arbitrary; take it to be zero, so the vacuum is `⟨φ⟩ = v`, real and positive. Now I expand about this point and — this is the whole game — I have to expand in the *right variables*, because the two directions off the vacuum are not equivalent. One direction climbs *up the side* of the hat (radially, changing `|φ|`); the other runs *around the trough* (angularly, changing the phase at fixed `|φ|`). The radial direction has the steep restoring force of the hat's wall. The angular direction is the flat valley floor — no restoring force at all, because moving along it is exactly the symmetry operation, and the symmetry says it costs nothing. I already know, from the magnet, that the angular mode must be massless. Let me now make the Lagrangian say it.

Write the fluctuation in Cartesian components about the chosen vacuum:

  `φ(x) = v + (σ(x) + i π(x))/√2`,

where `σ` and `π` are two real fields, `σ` the radial fluctuation (real part, along the vacuum direction) and `π` the imaginary part (perpendicular, tangent to the circle at the vacuum). The `1/√2` is so the kinetic terms come out canonically normalized — let me check that immediately. `∂φ = (∂σ + i ∂π)/√2`, so

  `∂φ* ∂φ = ½[(∂σ)² + (∂π)²]`,

which is exactly two canonically normalized real scalars, `½(∂σ)² + ½(∂π)²`. Good, the `1/√2` was the right choice; without it I'd carry stray factors into the masses.

Now the potential. Compute `|φ|²` first:

  `|φ|² = (v + σ/√2)² + (π/√2)² = v² + √2 v σ + ½(σ² + π²)`.

Let me write `u ≡ |φ|²` and note that the potential is most cleanly written using the minimum. Since `μ² = 2λv²`,

  `V = −μ² u + λ u² = −2λv² u + λ u² = λ(u − v²)² − λ v⁴`.

The `−λv⁴` is a constant (vacuum energy), drop it. So `V = λ(u − v²)²`, and

  `u − v² = √2 v σ + ½(σ² + π²)`.

Therefore

  `V = λ\big(√2 v σ + ½(σ² + π²)\big)²`.

I only need the quadratic part to read off the masses (the linear part must vanish — it does: there's no term linear in σ or π alone, because the bracket starts at `√2 v σ` but it's *squared*, so the lowest term is order `σ²`; the field sits at a stationary point, as it must). Expand the square and keep terms of second order in the fields:

  `V_quad = λ (√2 v σ)² = λ · 2v² · σ² = 2λ v² σ²`.

And `μ² = 2λv²`, so

  `V_quad = μ² σ²`.

Read off the mass. For a canonically normalized real scalar the mass term is `½ m² σ²`, so `½ m²_σ = μ²`, giving

  `m²_σ = 2 μ².`

The radial mode is massive, mass-squared `2μ²` — same as the discrete double well, which makes sense: radially, the Mexican hat *is* a double well through any diameter, and the curvature of the wall is the same.

Now look for the `π` mass. Where does `π` appear in `V_quad`? I need a term `π²` in the *quadratic* part of `V`. But `V = λ(u − v²)²`, and `π` enters `u − v²` only through the `½(σ² + π²)` piece — it appears at *second* order in the bracket. Squaring that piece gives a pure `π⁴` term, not a `π²` term. The mixed term `2·(√2 v σ)·(½ π²) = √2 v σ π²` is cubic, not a mass. There is *no* standalone `π²` term in `V` at all. So

  `m²_π = 0.`

The angular field is *exactly* massless. Not approximately, not "light" — the quadratic term is identically absent, and the absence is protected: `V` depends on `φ` only through `|φ|²`, and the phase direction along the circle is a symmetry direction. In these Cartesian variables there are still interactions involving `π`, like the cubic `σπ²` and the quartic `π⁴`, because a straight line in the tangent direction eventually leaves the circular valley. What is protected is the curvature at the vacuum: no standalone `π²` term can appear. The radial mode is massive, the angular mode is massless, and the masslessness is not a fluke of this expansion — it's the symmetry, expressed.

Let me say out loud *why* this had to come out this way, because the algebra can hide it. The field `π` is the direction *around the circle of vacua*. Displacing `φ` along `π` is, to leading order, just changing its phase — it slides the field from one degenerate vacuum to a neighboring degenerate vacuum. The potential is by construction *constant* along that circle. A constant potential has zero curvature. Zero curvature is zero mass. The massless particle *is* the field excitation that walks along the valley of degenerate vacua. It is the field-theory incarnation of the magnet's spin wave: long-wavelength rotations of the phase, which in the strict long-wavelength limit are a global symmetry operation and therefore cost no energy. When all the `φ(x)` rotate in phase together there is no gain in energy, because of the symmetry — so the cost of an almost-uniform phase rotation goes to zero as the wavelength grows, which is exactly a gapless dispersion, which is exactly a massless particle. The radial mode `σ` measures *how far up the wall* you've climbed and so feels the restoring force; the angular mode `π` measures *where around the trough* you are and feels nothing.

So I can write the new Lagrangian in these variables. With `V = λ(u−v²)²` expanded,

  `L = ½(∂σ)² + ½(∂π)² − μ² σ² − [cubic and quartic interactions of σ, π] + const`,

— a massive real scalar `σ` of mass `√(2)μ`, a *massless* real scalar `π`, and their interactions. The original U(1) symmetry hasn't vanished; it has rearranged itself. It no longer acts as a simple phase on a single complex field sitting at the origin; around the chosen vacuum its infinitesimal action starts with a shift of the massless field,

  `δπ = √2 v α + α σ`,  `δσ = −απ`,

so at the vacuum `π → π + √2 v α` to leading order. A standalone `π²` mass term could not respect that shift. In exact angular coordinates the potential is independent of the angle; in these Cartesian variables the same fact appears as zero curvature in the tangent direction, with only higher interactions allowed. The continuous symmetry didn't disappear when the vacuum broke it — it turned into the statement that one particle is exactly massless.

I should make sure this isn't a special feature of one complex field. Count. The vacuum manifold here is the circle `|φ| = v`, which is *one-dimensional* — one flat direction — and I got *one* massless boson. Suppose instead I had `N` real fields with an `O(N)` symmetry and a potential depending only on `Σφᵢ²`, minimized on the *sphere* `Σφᵢ² = v²`. The sphere in `N`-space has dimension `N − 1`. Expanding about a chosen point on it: one direction is radial (off the sphere, up the wall — massive), and the remaining `N − 1` directions are *tangent to the sphere* (along the degenerate valley — flat — massless). So `N − 1` massless bosons, one for each independent flat direction of the vacuum manifold. The number of massless bosons equals the dimension of the manifold of degenerate vacua, which equals the number of symmetry generators that *move* the vacuum — the *broken* generators. One massless spinless boson per spontaneously broken continuous generator. That's the general law, and now I see it's just "count the tangent directions of the valley floor."

This is a strong enough pattern that I want to prove it doesn't depend on the classical Mexican-hat picture — that it's a theorem about quantum field theory, holding for *any* relativistic local theory whatever its detailed dynamics. Let me see if I can get the masslessness purely from the symmetry and Lorentz invariance, without solving a model.

The continuous symmetry, being a symmetry of the action, gives me by Noether a conserved current `j_μ` with `∂^μ j_μ = 0`, and a charge `Q = ∫ d³x j₀` that generates the symmetry: `δφ = i[Q, φ]`. "The vacuum breaks the symmetry" means the symmetry variation of some field has a nonzero vacuum expectation value — there is a field `φ` (the order parameter, here the one that got the VEV) with

  `⟨0| δφ |0⟩ = ⟨0| i[Q, φ] |0⟩ ≠ 0.`

If the vacuum were invariant, `Q|0⟩ = 0`, this would vanish. It doesn't, so `Q|0⟩ ≠ 0` — the charge does not annihilate the vacuum; acting with it moves you along the degenerate set. Now consider the two-point function of the current with this field,

  `J_μ(x) ≡ ⟨0| j_μ(x) φ(0) |0⟩.`

Lorentz covariance pins its form. `φ` is a scalar; `j_μ` carries one vector index; the only Lorentz-covariant object built from the single coordinate `x` that I can write down is a derivative of a Lorentz-invariant function:

  `J_μ(x) = ∂_μ J(x)`,  with `J(x)` a function of the invariant `x²` only.

Now impose current conservation, `∂^μ j_μ = 0`, which means `∂^μ J_μ(x) = 0`:

  `0 = ∂^μ ∂_μ J(x) = □ J(x).`

So `J` satisfies the *massless* wave equation. In momentum space, `□J = 0` means `J̃_μ(k) = λ k_μ δ(k²)` for some constant `λ` — the Fourier transform is supported entirely on the light cone `k² = 0`. And `λ` cannot be zero: integrating `J₀` over space reconstructs `⟨[Q, φ]⟩`, which I assumed nonzero, so `λ ≠ 0` is forced by the symmetry-breaking condition `⟨δφ⟩ ≠ 0`. But `J_μ` is a sum over intermediate physical states inserted between `j_μ` and `φ`; a contribution supported at `k² = 0` means there is a physical one-particle state with `k² = 0` — a *massless* particle — that couples to both the broken current and the order-parameter field. The current creates it out of the vacuum; it is spinless because it saturates a relation involving a scalar `φ`. There is no escape: if `⟨δφ⟩ ≠ 0`, the spectral support at `k² = 0` is nonzero, and a massless boson exists. The classical valley-floor argument and this spectral argument are the same fact seen two ways — the `δ(k²)` is the long-wavelength softness made exact.

That's Goldstone's theorem, and it's completely general: *whenever a continuous symmetry of a relativistic local Lagrangian is spontaneously broken by the vacuum, the spectrum contains a massless spinless boson for each broken generator.* The Mexican-hat model is just the simplest place to watch it happen; the spectral proof shows it doesn't depend on the model. The assumption I leaned on hard is Lorentz covariance of `J_μ`, which let me write `J_μ = ∂_μ J`; within a manifestly Lorentz-invariant local theory, the massless boson is unavoidable.

Now I want to come at the same mechanism from the fermion side, because that's where it actually has to earn its keep, and where I first got worried. The trigger was the BCS theory of superconductivity. Their ansatz for the ground state is bold and it works, but it does not have a definite number of electrons — the condensate has a definite *phase* instead, and phase and number are conjugate. So the BCS ground state breaks the U(1) of electron-number / electromagnetic gauge symmetry. That nags me: the Bogoliubov–Valatin quasiparticles are superpositions of an electron and a hole, carrying no definite charge, and I cannot see how to trust the electromagnetic predictions — the Meissner effect — if charge conservation seems to be violated by the very excitations I'm computing with. The continuity equation for the BCS charge density doesn't naively hold. Something has to restore it, and an individual quasiparticle cannot be the whole answer.

The missing current has to come from the condensate itself. The same interaction that produces the gapped BCS ground state *also* leaves a collective variable behind: the condensate phase. Let that phase oscillate slowly in space and time, and I get a collective excitation that carries the missing piece of the current. It is the Goldstone boson of the broken U(1): a gapless phase oscillation of the condensate, contributing to the charge-current and patching the continuity equation back together. Charge conservation isn't violated; it is restored by the collective mode that the broken symmetry guarantees must exist. The masslessness I derived abstractly is the thing that saves gauge invariance in the superconductor.

Now the leap I keep circling back to. The Bogoliubov–Valatin equations for the quasiparticle,

  `E ψ_{p,+} = ε_p ψ_{p,+} + Δ ψ†_{−p,−}`,  `E = √(ε_p² + Δ²)`,

have the *same algebraic form as the Dirac equation*. The gap `Δ` sits exactly where a *mass* would sit. A self-energy that opens an energy gap in a superconductor is formally a self-energy that generates a *mass* for a fermion. So what if a particle's mass is not a parameter in the Lagrangian at all, but a *gap* — generated dynamically by a symmetric interaction acting on a non-invariant vacuum, just as in BCS? Start with a Lagrangian of *massless* fermions, give it a symmetry that a mass term would violate, and let the interaction break that symmetry spontaneously by generating `⟨ψ̄ψ⟩ ≠ 0` — a fermion condensate, the analog of the Cooper-pair condensate. The fermion then acquires a mass `M` dynamically, the way the quasiparticle acquired the gap `Δ`.

Which symmetry does a mass term break? A Dirac mass term `M ψ̄ψ` is *not* invariant under chiral rotations `ψ → e^{iγ₅α}ψ` — under chirality, `ψ̄ψ → cos(2α)ψ̄ψ + i sin(2α) ψ̄γ₅ψ`, so `M ψ̄ψ` is forbidden if I demand chiral symmetry. So: take a chirally symmetric Lagrangian of massless fermions with a four-fermion interaction (the analog of the attractive electron-electron interaction), invariant under both number `ψ → e^{iα}ψ` and chirality `ψ → e^{iγ₅α}ψ`:

  `L = −ψ̄ γ^μ ∂_μ ψ + g\big[(ψ̄ψ)² − (ψ̄ γ₅ ψ)²\big]`.

Look for a self-consistent solution where the interaction generates a self-energy `⟨ψ̄ψ⟩ ≠ 0`, i.e. a mass. The condition that a nonzero mass `M` solves the self-consistency — the *gap equation* — comes out, with a momentum cutoff `Λ`, as

  `1 = \frac{2g\Lambda²}{π²}\left[1 − \frac{M²}{Λ²}\ln\!\left(1 + \frac{Λ²}{M²}\right)\right]`,

which has a nontrivial solution `M ≠ 0` once the coupling `g` is strong enough — exactly the structure of the BCS gap equation, a nonzero solution appearing non-perturbatively. The fermion gets a mass `M ~ 2g⟨ψ̄ψ⟩` out of a Lagrangian that had no mass in it. Chiral symmetry has been spontaneously broken by the vacuum condensate `⟨ψ̄ψ⟩`.

And now Goldstone's theorem demands its price: the broken continuous chiral symmetry *must* produce a massless spinless boson. Where is it? It has to be a bound state of the fermions — a pseudoscalar (because chirality is what's broken, the would-be massless mode is the `ψ̄γ₅ψ` channel). Solve for the bound states of fermion–antifermion pairs in this theory and indeed: the pseudoscalar `0⁻` channel, `ψ̄γ₅ψ`, has a bound state at *exactly zero mass* — the Goldstone boson — while the scalar `0⁺` channel, `ψ̄ψ`, has a bound state at mass `2M`. The masslessness of the `0⁻` is not put in; it falls out, forced by the same theorem, and the `0⁺` sitting at `2M` is the radial "amplitude" partner — the analog of my massive `σ`, twice the fermion mass because it takes a pair to climb the wall. A massive fermion *and* a massless pseudoscalar, both born from a massless symmetric Lagrangian.

Now I look at the real world and the identification is irresistible. The pion is a pseudoscalar, and it is *anomalously light* — far lighter than the nucleon, as though it were trying to be massless. If chirality is a symmetry of the strong interaction that is spontaneously broken by a fermion condensate, then the nucleon mass is the dynamically generated gap, and *the pion is the Goldstone boson of broken chiral symmetry*. It isn't exactly massless because chirality isn't an exact symmetry — a small explicit chiral-symmetry-breaking term lifts the Goldstone boson to the small but nonzero pion mass. The smallness of the pion mass relative to the nucleon mass is then *explained*: it's the small explicit breaking on top of the spontaneous one. And the relations between axial and vector couplings — that the axial current is approximately conserved, that `g_π ≈ 2M g_A G` (Goldberger–Treiman) — are exactly what you'd get if the pion were the soft boson of an almost-exact spontaneously broken chiral symmetry. The whole package hangs together: nucleon mass as a BCS-like gap, pion as the Nambu–Goldstone boson, the soft-pion relations as consequences.

Let me also be honest about what *kind* of solution this is, because it cost me the most grief. These broken-symmetry vacua are genuinely *new* solutions, invisible to ordinary perturbation theory around `φ = 0` (or around the massless fermion). They appear precisely when the naive perturbative solution has a negative boson mass-squared, or equivalently when the perturbative vacuum is unstable — the signal that says "expand somewhere else." The condensate and the gap are non-analytic in the coupling, which is why no finite order of perturbation theory ever sees them. The right picture is: there's a family of vacua, you build a complete tower of particle states on each, a superselection rule separates the towers, and you live in one of them. The symmetry isn't gone — it relates the towers — but inside your tower it's realized not by a symmetric vacuum but by the existence of an exactly massless boson.

Let me close the loop by re-deriving the cleanest statement once more, since the scalar model is where it's transparent. Continuous global symmetry `φ → e^{iα}φ`; symmetric potential `V = −μ²|φ|² + λ|φ|⁴` with `μ², λ > 0`; degenerate minima on the circle `|φ| = v`, `v² = μ²/2λ`; the vacuum must pick a phase, breaking the symmetry; expand `φ = v + (σ + iπ)/√2`; the potential's curvature is `2μ²` in the radial direction (giving the massive `σ`, `m²_σ = 2μ²`) and *exactly zero* in the angular direction (giving the massless `π`, `m²_π = 0`); the masslessness is the symmetry surviving as a shift on `π`; and a current-conservation-plus-Lorentz-covariance argument elevates this from a model to a theorem — one massless spinless boson per broken continuous generator. The same mechanism with fermions turns a mass into a dynamically generated gap and a massless pseudoscalar into the Goldstone boson, with the pion as its physical realization.

Here is the mechanism worked symbolically, exactly the calculation above — find the vacuum, expand about it, read the two masses off the curvature:

```python
import sympy as sp

mu2, lam = sp.symbols('mu2 lambda', positive=True)   # mu2 = mu^2 > 0,  lambda > 0
sigma, pi = sp.symbols('sigma pi', real=True)         # radial and angular fluctuations

# --- vacuum: a CIRCLE of degenerate minima -----------------------------------
r = sp.symbols('r', positive=True)                    # r = |phi|
V_r = -mu2*r**2 + lam*r**4                             # V depends only on |phi|^2
v = sp.solve(sp.diff(V_r, r), r)                       # stationary points in r
v = [s for s in v if s != 0][0]                        # the nonzero minimum
#   v = sqrt(mu2/(2*lambda))  ==>  v**2 = mu^2 / (2 lambda);  the PHASE is free.

# --- expand phi about ONE chosen point on the circle (phase = 0) --------------
# phi = v + (sigma + i pi)/sqrt(2);  read off |phi|^2:
phi_re = v + sigma/sp.sqrt(2)
phi_im = pi/sp.sqrt(2)
mod2   = phi_re**2 + phi_im**2                         # |phi|^2

V = -mu2*mod2 + lam*mod2**2                            # potential in fluctuations
V = sp.expand(V)

# --- masses = curvature (Hessian) of V at the vacuum (sigma = pi = 0) ---------
# (v already carries v^2 = mu^2/(2 lambda), so the masses come out in mu^2)
m2_sigma = sp.simplify(sp.diff(V, sigma, 2).subs({sigma:0, pi:0}))  # radial mode
m2_pi    = sp.simplify(sp.diff(V, pi,    2).subs({sigma:0, pi:0}))  # angular mode

print("m_sigma^2 =", m2_sigma)   # -> 2*mu2   : radial mode is MASSIVE
print("m_pi^2    =", m2_pi)       # -> 0       : angular Goldstone mode is MASSLESS
# The angular direction is the flat valley of degenerate vacua: zero curvature,
# zero mass. That massless boson is the Goldstone boson of the broken U(1) --
# one for each broken continuous generator.
```
