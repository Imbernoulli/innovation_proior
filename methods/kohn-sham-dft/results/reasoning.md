Let me start from the thing that actually hurts. I want ground-state energies and densities for real systems — molecules, solids, alloys — at chemical accuracy. The honest object is the many-electron wavefunction Ψ(r₁,…,r_N), and the honest method is Rayleigh–Ritz: E = min_Ψ ⟨Ψ|H|Ψ⟩ with H = −½Σ∇_i² + Σv(r_i) + ½Σ_{i≠j}1/|r_i−r_j|. For H₂ this works beautifully; with a dozen-odd variational parameters per coordinate you nail the dissociation energy and bond length. So far so good. But let me count what happens as the system grows.

To pin Ψ to fractional accuracy O(10⁻⁴) in the energy I need maybe O(10⁻²) in Ψ, which means something like p parameters per continuous variable with p somewhere from 3 to 10. Ψ has 3N continuous variables. So the number of parameters is M ≈ p^{3N}. That exponent is the whole story. Invert it: the largest N I can handle is N ≈ (1/3)·log M / log p. Be wildly optimistic, M = 10⁹, p = 3 — I get N ≈ 6. Be clever, push to N ≈ 20. Turn it around: for N = 100, taking p = 3, I'd need M = 3^{300} ≈ 10^{150} parameters, and I'd be minimizing in a space of 10^{150} dimensions. I cannot imagine a computer that does that. And even if I could compute such a Ψ, I couldn't *record* it: q^{3N} bits with q ≈ 3 and N = 10⁴ is more bits than there are baryons in the universe. So past N of order ten or twenty, Ψ is not really a usable object at all. There's an exponential wall and I keep slamming into it. The wall is not a software problem; it's the dimensionality of configuration space.

So the wavefunction is the wrong variable for big systems. What's the right one? The quantities I actually care about — the total energy E, the density n(r), the pair correlation — depend on only a handful of variables. The density especially: n(r) = N∫|Ψ(r,r₂,…)|²dr₂…dr_N is a function of three variables no matter how big N is. If I could write the ground-state energy as a functional of n(r) and minimize *that*, the exponential wall would simply be gone — I'd be minimizing over a function of three variables instead of 3N.

This isn't a new wish. There's a very old, very crude version of it sitting on the table: take the kinetic, exchange, and electrostatic energies of a *uniform* electron gas, evaluate their densities at the *local* n(r), and integrate over space. That gives a closed energy functional E[n] = C_F∫n^{5/3} + ∫v n + ½∬n n′/|r−r′| (plus a Dirac −C_x∫n^{4/3} for exchange), and minimizing over n at fixed N yields a single equation for the density. It's appealing because it's pure density. But I know its track record: it gives rough total energies for atoms and then fails at exactly the thing I want — it predicts no chemical binding whatsoever. Assemble a molecule and the dissociated atoms are always lower. No bonds. So as a *quantitative* tool it's dead.

Now, why is it dead? Let me locate the failure, because the failure is the clue. The electrostatic term is just classical electrostatics — fine. The external term ∫v n is exact. So the suspects are the kinetic term C_F∫n^{5/3} and the local exchange. And there's a second data point right next to it: the self-consistent single-particle theory where I instead compute the kinetic energy from *orbitals* — (−½∇² + v_H)φ_j = ε_jφ_j with v_H the nucleus-plus-mean-field potential and n = Σ|φ_j|² — binds atoms much better and even shows the shell structure. The *only* structural difference between that and the density-only theory is how the kinetic energy is handled: a Laplacian acting on orbitals versus a local n^{5/3}. So the kinetic energy is where the local approximation is fatal. The interaction terms I might be able to get away with locally; the kinetic energy I cannot. Hold onto that — it's going to dictate the whole design.

But before I can even hope to write E as a functional of n, I have to settle a question I've been ducking: is that legitimate? Does the density actually contain enough information? It obviously fixes N by integration. Does it fix the whole Hamiltonian — really the external potential v(r), since the kinetic and interaction operators are universal? In the crude density theory it does: substitute n into the model relations and you recover v up to a constant. For any one-particle system it does. For a uniform gas weakly perturbed by Δv(r), the induced density Δn₁(r) is tied to Δv by the gas susceptibility, and I can invert to get Δv back from Δn₁. Every special case I try says yes. Let me conjecture it in general: the ground-state density of any electronic system, interacting or not, uniquely determines the external potential, hence H, hence everything.

Can I prove it? Suppose not — suppose two potentials v₁(r) and v₂(r), differing by more than a constant, both give the same ground-state density n(r). Call their ground states Ψ₁, Ψ₂ (necessarily different, since they solve different Schrödinger equations) and their energies E₁, E₂. Take Ψ₂ as a trial function for the Hamiltonian H₁ that belongs to v₁. Since Ψ₁ is the (nondegenerate) ground state of H₁, Rayleigh–Ritz gives a strict inequality:
  E₁ < ⟨Ψ₂|H₁|Ψ₂⟩ = ⟨Ψ₂|H₂|Ψ₂⟩ + ⟨Ψ₂|(H₁−H₂)|Ψ₂⟩ = E₂ + ∫[v₁(r) − v₂(r)] n(r) dr,
because H₁ − H₂ is just the multiplicative operator v₁ − v₂, whose expectation in any state with density n is ∫(v₁−v₂)n. Now do the symmetric thing — Ψ₁ as a trial function for H₂:
  E₂ ≤ ⟨Ψ₁|H₂|Ψ₁⟩ = E₁ + ∫[v₂(r) − v₁(r)] n(r) dr.
(I write ≤ here because I'm not separately assuming Ψ₂ is nondegenerate; ≤ is all I need.) The two density integrals are exact negatives of each other. Add the inequalities and they cancel:
  E₁ + E₂ < E₁ + E₂.
That's a contradiction. So no two potentials differing by more than a constant can share a ground-state density. The density fixes v up to a constant, hence fixes H and N, hence fixes Ψ and every observable. The map from n to "the whole system" is well defined. Good — now "energy as a functional of n" is not a hope, it's a theorem. The crude density theory was *assuming* this; I've now earned it.

So define, for any density n that comes from some ground state, the functional
  F[n] = ⟨Ψ[n]| T + U |Ψ[n]⟩,  T = −½Σ∇², U = ½Σ_{i≠j}1/|r_i−r_j|,
where Ψ[n] is the ground state that produces n. This F is *universal* — it has no v in it; the same F works for atoms, molecules, crystals. Then for a given external potential the energy functional is
  E_v[n] = ∫v(r) n(r) dr + F[n],
and by the variational principle E_v[n] ≥ E₀ for any admissible trial density, with equality at the true ground-state density. (One can phrase the same thing as a constrained search: minimize ⟨Ψ|T+U|Ψ⟩ over all Ψ that yield a fixed n, then minimize over n — but the upshot is the same universal F[n] and the same minimum principle.) This is the formal exactification of the density-only dream. The 3N-dimensional minimization over Ψ has become a 3-dimensional minimization over n(r).

And right here is the catch that keeps it from being a free lunch. F[n] is defined but unknown, and unpacking the definition sends me straight back to 3N-dimensional wavefunctions. So the whole game is now: find a good, computable approximation to F[n]. If I'm careless I'll just rebuild Thomas–Fermi — approximate T[n] by the local gas value C_F∫n^{5/3}, approximate U by the classical Hartree term, and I'm back to no binding. That's the wall again, in a new costume. The lesson from before is loud: do not model the kinetic energy by a local density functional. That single term is what killed the cheap theory.

So what do I do with the kinetic energy instead? Let me look at what made the orbital-based single-particle theory work. There, the kinetic energy is computed *exactly* for that theory's wavefunction — it's just Σ_j⟨φ_j|−½∇²|φ_j⟩, the sum of orbital kinetic energies, with the orbitals carrying the Laplacian honestly. The Laplacian *is* the thing the local n^{5/3} fails to capture. I want that honesty.

I have an interacting system with density n. I cannot compute its true kinetic energy T[n] cheaply. But there is a *different* system — a fictitious system of *non-interacting* electrons — that I could in principle tune to have the *same* density n. For that non-interacting system, the kinetic energy is easy and exact: solve single-particle Schrödinger equations, fill the lowest states, and add up the orbital kinetic energies. Call that quantity T_s[n] — the kinetic energy of non-interacting electrons with density n. It is an *implicit* functional of n (it's an explicit functional of the orbitals, and the orbitals are fixed by n), but it is exact for what it is, and crucially it carries the Laplacian.

Now T_s[n] is not the true kinetic energy T[n] of the interacting system — the interacting electrons correlate their motion and have a slightly different kinetic energy. So write T[n] = T_s[n] + T_c[n], where T_c[n] = T[n] − T_s[n] is whatever's left, a kinetic correlation piece. The point is that T_c is *small* compared to T_s, whereas in Thomas–Fermi the *entire* kinetic energy was being mangled. I've taken the big, badly-modeled object and computed the dominant part of it exactly.

Let me reorganize F[n] around this. Split off the two pieces I can handle exactly or classically — the non-interacting kinetic energy T_s[n] and the classical Hartree energy U_H[n] = ½∬n n′/|r−r′| — and shovel everything else into a single remainder:
  F[n] = T_s[n] + U_H[n] + E_xc[n],
which *defines*
  E_xc[n] ≡ (T[n] − T_s[n]) + (U[n] − U_H[n]) = T_c[n] + (U[n] − U_H[n]).
So E_xc — the exchange-correlation energy — is the kinetic correlation plus the part of the electron–electron interaction beyond the classical mean field, i.e. exchange (from antisymmetry) and Coulomb correlation. This decomposition is still exact; I've only named things. But now the unknown is isolated into E_xc, which is a *small* fraction of the total energy, and the two big terms are under control. That's the structural bet: model a small term crudely rather than a large term crudely.

The energy functional is therefore
  E_v[n] = ∫v n dr + T_s[n] + ½∬ n(r)n(r′)/|r−r′| dr dr′ + E_xc[n].
Now minimize. The Euler–Lagrange condition, holding ∫n = N fixed with a Lagrange multiplier µ, is
  δT_s/δn(r) + v(r) + ∫ n(r′)/|r−r′| dr′ + δE_xc/δn(r) = µ.
Define the last three (density-dependent) pieces as a single effective potential:
  v_eff(r) = v(r) + ∫ n(r′)/|r−r′| dr′ + v_xc(r),  v_xc(r) ≡ δE_xc/δn(r).
Then the stationarity condition reads
  δT_s/δn(r) + v_eff(r) = µ.   (★)

Stare at (★) for a second. Now consider a genuinely *non-interacting* system of electrons sitting in some external potential v_s(r). Its energy functional is just ∫v_s n + T_s[n], and *its* Euler–Lagrange condition is
  δT_s/δn(r) + v_s(r) = µ.   (★★)
Equations (★) and (★★) are *identical in form*. If I choose v_s(r) = v_eff(r), they're literally the same equation, so they have the same minimizing density. That means: the density of my interacting system is reproduced *exactly* by a non-interacting system moving in the potential v_eff. And I already know how to get the ground state of a non-interacting system — I don't fuss with δT_s/δn at all, I just solve its single-particle Schrödinger equation. So the minimizing density is delivered by
  (−½∇² + v_eff(r)) φ_i(r) = ε_i φ_i(r),
  n(r) = Σ_i^{occ} |φ_i(r)|²,
  v_eff(r) = v(r) + ∫ n(r′)/|r−r′| dr′ + v_xc(r),  v_xc = δE_xc/δn,
with the occupied set being the N lowest eigenstates (the Pauli principle / ground state of the auxiliary non-interacting system). And these are self-consistent: v_eff depends on n through the Hartree and xc terms, and n depends on v_eff through the orbitals. Guess n, build v_eff, diagonalize, rebuild n, iterate to convergence.

That's the whole thing, and it's exact in principle — every many-body effect lives in E_xc and v_xc; if I had them exactly, these single-particle equations would give the *exact* interacting density and energy. Two sanity checks. First, drop E_xc and v_xc entirely: v_eff collapses to v + the Hartree potential, and I recover the orbital-based self-consistent equations exactly. So those weren't an ad hoc ansatz after all — they're the no-exchange-no-correlation limit of a formally exact theory, and now I can see precisely what they were the approximation *to*. Second, if I were perverse and approximated even T_s by the local gas value, I'd fall right back to Thomas–Fermi. So this scheme contains both old theories as limits, and the new content is exactly "compute T_s honestly with orbitals, hide only the small remainder in E_xc."

I should also get the total energy in a convenient form, because naively I'd want E = Σε_i, but that double-counts. The eigenvalue sum is Σ_i ε_i = Σ_i⟨φ_i|−½∇² + v_eff|φ_i⟩ = T_s[n] + ∫v_eff n. Now ∫v_eff n = ∫v n + ∬n n′/|r−r′| + ∫v_xc n, where note the Hartree piece here is ∬n n′/|r−r′| = 2U_H (the full double integral, not the ½). So
  T_s[n] = Σ_i ε_i − ∫v n − 2U_H − ∫v_xc n.
Plug into E_v = ∫v n + T_s + U_H + E_xc:
  E = Σ_i ε_i − ∫v n − 2U_H − ∫v_xc n + ∫v n + U_H + E_xc
    = Σ_i ε_i − U_H[n] − ∫ v_xc(r) n(r) dr + E_xc[n]
    = Σ_i ε_i − ½∬ n(r)n(r′)/|r−r′| dr dr′ − ∫ v_xc(r) n(r) dr + E_xc[n].
The band-energy sum, minus the Hartree double-counting, minus the xc double-counting, plus E_xc. Clean.

Now the only thing standing between me and numbers is E_xc[n]. The whole practical fate of this theory rides on whether there's an approximation that's at once simple and accurate. And here's where I get to *use* the structural bet I made. E_xc is small; the big terms are exact. So a crude local model of E_xc might be tolerable in a way that a crude local model of T never was.

What's the simplest possible local model? Borrow from the one solved many-body system I have — the uniform electron gas. For a gas of constant density n, let e_xc(n) be the exchange-correlation energy per electron. Then approximate, for the real inhomogeneous system, by pretending each little volume is locally a piece of uniform gas at the local density:
  E_xc^{LDA}[n] = ∫ e_xc(n(r)) n(r) dr.
This is the local density approximation. The exchange part of e_xc is elementary, because the exchange energy density of the uniform gas is known in closed form. Per volume it's e_x^{hom}(n) = −(3/4)(3/π)^{1/3} n^{4/3}, so
  E_x^{LDA}[n] = −(3/4)(3/π)^{1/3} ∫ n(r)^{4/3} dr,
and the exchange potential is the functional derivative
  v_x^{LDA}(r) = δE_x/δn = −(3/4)(4/3)(3/π)^{1/3} n^{1/3} = −(3/π)^{1/3} n(r)^{1/3}.
The correlation part e_c(n) of the uniform gas is a genuine many-body number — not analytic — but it's a *known* number, available from interpolation formulas (e.g. Wigner's e_c ≈ −0.44/(r_s + 7.8) in atomic units, r_s = (3/4πn)^{1/3} being the radius of the sphere holding one electron) and improvable to high precision later. So E_xc^{LDA} is fully specified with no free parameters once I pick a parameterization of the gas data.

Hold on — look at what the exchange piece just did to the old Slater stand-in. Slater replaced Hartree–Fock exchange by a local potential −3α(3/8π·n)^{1/3} and chose α = 1 by averaging over the *whole* Fermi sphere — an uncontrolled choice. My v_x^{LDA} = −(3/π)^{1/3}n^{1/3}. Let me compare the coefficients. (3/π)^{1/3} versus 3·(3/8π)^{1/3}: their ratio is (3/π)^{1/3} / [3(3/8π)^{1/3}] = (1/3)·(3/π · 8π/3)^{1/3} = (1/3)·8^{1/3} = (1/3)·2 = 2/3. So v_x^{LDA} = (2/3)·v_x^{Slater,α=1}. The variational derivative *pins* α to 2/3. Slater's α was floating because he averaged the potential over the whole Fermi sphere; here the 2/3 falls out of differentiating the energy. The coefficient is no longer a knob — it's forced. That's reassuring: the right thing for energy minimization is the 2/3, and now I know why.

But I'm uneasy about LDA for a deeper reason, and I should face it. The uniform gas is, by construction, a *slowly varying* density. Atoms and molecules are about as far from slowly varying as an electronic system gets — the density swings over an order of magnitude within a Fermi wavelength. By the stated logic LDA has no business working for an argon atom. So why might it work anyway? Let me try to understand E_xc more physically, because if I only have the formula I can't trust it where it has no right to be good.

Go back to what exchange and correlation *are*. Put one electron at r. Because of the Pauli principle (same-spin electrons avoid each other) and Coulomb repulsion (all electrons avoid each other), the presence of that electron *depletes* the density of the other electrons nearby. Define the conditional density g(r,r′) = density of electrons at r′ given one is at r, and the exchange-correlation hole as the depletion,
  n_xc(r,r′) = g(r,r′) − n(r′).
Physically there's exactly one electron's worth of "missing" charge dug out around r — the electron screens itself — so the hole integrates to −1:
  ∫ n_xc(r,r′) dr′ = −1.   (the sum rule)
Now I want to connect E_xc to this hole, and the clean way is a continuous deformation between the non-interacting and the interacting world. Introduce a coupling constant λ ∈ [0,1] that scales the electron–electron interaction, U → λU, and at each λ adjust the external potential to a fixed v_λ(r) so that the density stays equal to the *physical* density n(r) for *all* λ. At λ = 0 this is exactly my non-interacting auxiliary system (same n); at λ = 1 it's the physical interacting system (same n). The whole family shares one density. Let H_λ = T + V_λ + λU with ground state Ψ_λ and energy E_λ. By the Hellmann–Feynman theorem,
  dE_λ/dλ = ⟨Ψ_λ| dH_λ/dλ |Ψ_λ⟩.
Now dH_λ/dλ has two pieces: ∂(λU)/∂λ = U, and ∂V_λ/∂λ from the λ-dependence of the confining potential. But V_λ is a one-body multiplicative operator, so ⟨Ψ_λ|∂V_λ/∂λ|Ψ_λ⟩ = ∫(∂v_λ/∂λ) n_λ dr = ∫(∂v_λ/∂λ) n dr, since n_λ ≡ n is fixed in λ. Integrate the full dE_λ/dλ from 0 to 1: the λU piece gives ∫₀¹⟨U⟩_λ dλ, and the V_λ piece gives ∫(v_1 − v_0) n dr. The latter is just the difference of one-body energies between λ=1 and λ=0 at fixed n; tracking the bookkeeping, the universal interaction-plus-kinetic part assembles as
  F[n] = T_s[n] + ∫₀¹ ⟨U⟩_λ dλ.
Subtract the classical Hartree energy U_H to get the exchange-correlation energy:
  E_xc[n] = ∫₀¹ dλ (⟨U⟩_λ − U_H[n]).
And ⟨U⟩_λ − U_H is exactly the interaction energy of the density with its *hole* at coupling λ: ⟨U⟩_λ = ½∬ [n(r)n(r′) + n(r) n_xc^λ(r,r′)]/|r−r′|, so
  E_xc[n] = ½ ∫ dr ∫ dr′ n(r) n̄_xc(r,r′) / |r − r′|,  n̄_xc(r,r′) = ∫₀¹ n_xc^λ(r,r′) dλ,
the coupling-averaged hole. So E_xc is, formally exactly, the electrostatic interaction between each electron and its own coupling-averaged exchange-correlation hole. And because every λ-hole integrates to −1, so does the average: ∫ n̄_xc(r,r′) dr′ = −1. The sum rule survives the averaging.

Now I can see why LDA works far outside its stated comfort zone. E_xc is an integral of the hole against the Coulomb kernel 1/|r−r′|. That kernel is isotropic, so E_xc depends mainly on the *spherical average* of the hole and on its *normalization* — not on the hole's detailed angular shape. The LDA hole is the hole of the uniform gas, which has exactly the right normalization (it integrates to −1, the same sum rule). So even when the LDA hole has the *wrong shape* for an inhomogeneous system, it has the *right charge*, and the Coulomb integral is forgiving about shape. That's the real reason LDA gives good E_xc for atoms it has no business describing. It also explains the famous error pattern: LDA tends to overestimate the magnitude of exchange and underestimate the magnitude of correlation, and the two errors partly cancel — a cancellation that is *systematic*, not lucky, because it's tied to the exact normalization of the hole. This is also a warning: any "improvement" to LDA that breaks the sum rule should be expected to do *worse*, not better. (Indeed, the most naive thing — a straight gradient expansion of the hole around the uniform gas — breaks the normalization and disappoints; the fruitful improvements are the ones built to *respect* the sum rule.)

Let me make sure I can actually drive these equations on a computer, because the proof of the pudding is a converged density. I'll take the simplest nontrivial setting that still exercises every piece: spin-paired electrons in a one-dimensional well, atomic units, exchange-only LDA (correlation I'll leave out for now — its formula is more tedious than illuminating, and the *structure* is identical: it's just another additive piece of e_xc with its own potential). I need: a kinetic operator, a way to get orbitals and the density, the Hartree and exchange potentials, the self-consistency loop, and the total energy.

For the kinetic energy T_s I represent −½ d²/dx² as a finite-difference matrix on a uniform grid — the second-derivative stencil (−1, 2, −1)/h² gives −d²/dx², so half of it is −½ d²/dx². The matrix is real symmetric, so np.linalg.eigh returns real orthonormal eigenvectors: those are my Kohn–Sham orbitals φ_i and energies ε_i. I normalize each orbital so ∫|φ|²dx = 1 (one electron's worth of capacity), fill the lowest states with occupations 2,2,… by the Aufbau rule (the ground state of the non-interacting auxiliary system), and form n(x) = Σ_n f_n|φ_n(x)|². Exchange is E_x = −(3/4)(3/π)^{1/3}∫n^{4/3}dx with v_x = −(3/π)^{1/3}n^{1/3}, straight from the derivative I did above. For the Hartree term in 1D I have to be a little careful: the bare 1/|x−x′| diverges on the diagonal in one dimension, so I soften it to 1/√((x−x′)²+1) — that's a 1D numerical artifact, not physics; in 3D I'd use the bare Coulomb kernel. Then v_H(x) = ∫n(x′)/√((x−x′)²+1)dx′ and E_H = ½∫n v_H dx. Assemble v_eff = v_ext + v_H + v_x, diagonalize T + diag(v_eff), rebuild the density, mix old and new densities for stability, and iterate. The total energy is the band sum minus the double-counting, exactly the (Σε − U_H − ∫v_xc n + E_xc) bookkeeping I derived, specialized to exchange-only: E = Σ_n f_n ε_n − E_H − ∫n v_x dx + E_x. (I'll trust this by cross-checking it against the energy functional evaluated directly, E = T_s + ∫n v_ext + E_H + E_x with T_s = Σ_n f_n⟨φ_n|−½d²/dx²|φ_n⟩; the two must agree at self-consistency, and they do.)

```python
import numpy as np

def build_kinetic(x):
    # T_s operator: -1/2 d^2/dx^2 via the (-1,2,-1)/h^2 second-difference stencil
    n = len(x); h = x[1] - x[0]
    lap = (np.diag(np.full(n, 2.0))
           + np.diag(np.full(n-1, -1.0), 1)
           + np.diag(np.full(n-1, -1.0), -1)) / h**2
    return 0.5 * lap                              # -1/2 * d2/dx2

def density(psi_gn, occ, x):
    # n(x) = sum_n f_n |phi_n(x)|^2 from L2-normalized KS orbitals
    h = x[1] - x[0]; n = np.zeros_like(x)
    for i, f in enumerate(occ):
        if f:
            psi = psi_gn[:, i]
            psi = psi / np.sqrt(np.sum(psi**2) * h)   # integrate to one electron
            n += f * psi**2
    return n

def exchange_lda(n, x):
    # E_x = -(3/4)(3/pi)^(1/3) int n^(4/3) dx ;  v_x = -(3/pi)^(1/3) n^(1/3)
    h = x[1] - x[0]; c = (3.0/np.pi)**(1.0/3.0)
    E_x = -(3.0/4.0)*c*np.sum(n**(4.0/3.0))*h
    v_x = -c*n**(1.0/3.0)
    return E_x, v_x

def hartree(n, x):
    # classical electrostatics; 1D-softened kernel (bare 1/|x-x'| diverges in 1D)
    h = x[1] - x[0]
    K = 1.0/np.sqrt((x[:, None] - x[None, :])**2 + 1.0)
    v_H = K @ n * h
    return 0.5*np.sum(n*v_H)*h, v_H

def occupations(num_electrons, num_states):
    # fill lowest states, 2 per state (ground state of the auxiliary system)
    occ = np.zeros(num_states); e = num_electrons; i = 0
    while e > 0 and i < num_states:
        occ[i] = min(2, e); e -= occ[i]; i += 1
    return occ

def solve_ks(x, v_ext, num_electrons, iters=200, mix=0.3, tol=1e-8):
    T = build_kinetic(x); h = x[1] - x[0]
    n = np.zeros_like(x)                          # start from n = 0
    occ = occupations(num_electrons, len(x))
    E_old = None
    for _ in range(iters):
        E_x, v_x = exchange_lda(n, x)             # v_xc = delta E_xc / delta n
        E_H, v_H = hartree(n, x)
        v_eff = v_ext + v_H + v_x                 # v_eff = v + v_H + v_xc
        eps, psi_gn = np.linalg.eigh(T + np.diag(v_eff))   # KS single-particle eqn
        n = (1 - mix)*n + mix*density(psi_gn, occ, x)
        band = np.sum(occ*eps[:len(occ)])
        E = band - E_H - np.sum(n*v_x)*h + E_x    # Sum eps - U_H - int v_xc n + E_xc
        if E_old is not None and abs(E - E_old) < tol:
            break
        E_old = E
    return E, eps, n

if __name__ == "__main__":
    x = np.linspace(-8, 8, 401); v_ext = x**2     # harmonic well
    for Ne in (2, 6):
        E, eps, n = solve_ks(x, v_ext, Ne)
        h = x[1] - x[0]
        print(f"N={Ne}: E={E:.5f} Ha, N_check={np.sum(n)*h:.4f}, eps[:3]={np.round(eps[:3],4)}")
```

Running it, the density integrates to exactly the number of electrons and the self-consistency loop converges to a stable energy — the auxiliary non-interacting system is faithfully reproducing the density I asked for, with the kinetic energy carried honestly by the orbitals and only the small exchange-correlation piece approximated.

Let me retrace the causal chain so I'm sure nothing leans on luck. The wavefunction is unusable past N of order ten because of the exponential wall, so I switch to the density, which has three variables for any N. The density legitimately determines the whole system — proved by reductio against Rayleigh–Ritz — so there is a universal functional F[n] and a variational principle for the energy. F[n] is unknown, and the one thing I must *not* do is model its kinetic part locally, because that's exactly what made the cheap density theory predict no bonds. So I compute the dominant kinetic piece exactly by mapping onto a fictitious non-interacting system with the same density: T[n] = T_s[n] + (small remainder), and I fold the small remainder, together with exchange and Coulomb correlation, into a single E_xc[n]. Minimizing the resulting energy functional gives an Euler–Lagrange equation identical in form to that of a non-interacting system in an effective potential v_eff = v + v_H + v_xc, so the interacting density is obtained by solving non-interacting single-particle Schrödinger equations self-consistently — exact in principle, with all many-body physics in E_xc. Finally, E_xc is small enough that a local approximation from the uniform electron gas works; and it works even for wildly inhomogeneous atoms because E_xc is a Coulomb integral against the exchange-correlation hole, which depends mainly on the hole's normalization (the sum rule ∫n_xc dr′ = −1), and the local-gas hole gets that normalization exactly right. Drop E_xc and the scheme is Hartree; model T_s locally and it's Thomas–Fermi; keep both honest and it's a formally exact, computationally cheap route to the ground state.
