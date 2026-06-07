Let me start from the thing that actually bothers me, which is a magnet near its Curie point, or a fluid near its critical point. As I bring the temperature to Tc the correlation length ξ — the distance over which one spin's orientation biases another — grows without bound. Right at Tc it is infinite. And what that means physically is that there are correlated patches of *every* size: atomic, micron, visible, all at once. The steam and water near the critical point are intermixed at all scales; that's why the thing goes milky. So whatever is going on is not a one-scale phenomenon.

That single fact is what defeats every method I know. Perturbation theory wants a small parameter and a single scale to expand around; here there is no single scale. Numerical integration chokes past a handful of variables; a faithful simulation would need a lattice that resolves millimeters across thousands of miles — the number of coupled integrations is astronomical. And the strange part: the partition function Z = Σ_configs exp(−H/kT) is a sum of Boltzmann factors, each analytic in T for every T except 0. A sum of analytic functions is analytic. Yet the magnetization curve has a kink, M ~ (Tc−T)^β; the susceptibility diverges; ξ diverges as (T−Tc)^{−ν}. The non-analyticity is real only in the infinite-volume limit, where the analyticity theorems no longer apply to the infinite sum — but knowing that doesn't tell me *how* the non-analyticity is built. I want a method that produces it.

Look at what Landau does, because it's the standard answer and I want to see exactly where it breaks. Landau says: fix the magnetization density M, then the free energy is analytic in M, and by symmetry (no field, so F can't care about the sign of M) F = V{ R M² + U M⁴ } to fourth order, with R ∝ (T−Tc) and U a positive constant. Minimize over M: above Tc the minimum is M=0; below, 0 = 2RM + 4UM³ gives M² = −R/(2U), so M ∝ (Tc−T)^{1/2}. That's β=1/2. Allow M to vary in space, F = ∫[ (∇M)² + R M² + U M⁴ − BM ]; drop a delta-function field in and the response decays with a length ξ ∝ R^{−1/2} ∝ (T−Tc)^{−1/2}, so ν=1/2 too. Both wrong: experiment and Onsager's exact 2D solution say β≈1/3, ν≈2/3 (and ν=1 in 2D).

Where's the lie? It's the hidden assumption. Landau treats the magnet like hydrodynamics: average out the atomic-scale fluctuations, and what's left, M(x), is a smooth classical field that only wiggles in response to external stimuli. He assumes analyticity *survives* the averaging of space-dependent fluctuations, and pins all the non-analyticity on the final step of minimizing over the one number M. But near Tc that's exactly false — the fluctuations that matter are not atomic, they run all the way out to ξ, on every scale in between. Landau has averaged out the very thing that's causing the trouble. So the assumption "only atomic-scale fluctuations matter" is the bug. I can even see where it's least bad: above four dimensions the long-wavelength fluctuations are harmless and Landau is right; four is the dividing line; just below four their effect is small, proportional to (4−d). That "small just below four" is a thread I should remember to pull.

So the problem, stated honestly: I have to actually carry out the average over thermal fluctuations on *all* size scales, and I can't average them all at once — that's the hopeless coupled integral. What can I do instead?

My own history pushes me here, because I didn't come at this from magnets — I came from quantum field theory, where the same disease appears. In QED the "elementary" electron has structure on all scales down to zero; the bare charge e₀ lives at short distance, the measured charge e at long distance, and the loop integrals run over all momenta and diverge at the top. Gell-Mann and Low taught me to think of a *family* of effective charges e_λ, one for each momentum scale λ, interpolating from e at low λ to e₀ at high λ, obeying λ de_λ/dλ = ψ(e_λ). You determine the coupling at one scale from its value at the neighboring scale, and march. That's the renormalization group, and it's the right instinct: deal with the scales in sequence. But it carries a *single* coupling, the charge, in a fixed one-dimensional space, and it's wedded to perturbative QED. A critical point is strongly coupled and has no obvious single marching parameter. Still, the instinct — one scale at a time — is the thing to keep.

Let me try to make "one scale at a time" concrete in a model I understand, the fixed-source meson theory I cut my teeth on. The trouble with the full theory is the continuum of momenta. So I do something crude: I rub out everything except well-separated momentum slices, |k| around 1, around Λ, around Λ², …, Λⁿ, with Λ a big number. Now the energy scales are wildly different — order Λⁿ for the nth slice — so the natural move is to take the highest slice as the unperturbed Hamiltonian and treat all the lower slices as a perturbation. I do that: solve the top slice, sit in its ground state, and ask what effective Hamiltonian governs the remaining n−1 slices. And the answer is clean — it's the *same* Hamiltonian with one fewer slice, except the meson–nucleon coupling g has been renormalized by a matrix element of that top-slice ground state.

That's the moment it clicks for me. For the first time I have a natural, concrete basis for a renormalization-group analysis: solve and *eliminate one momentum scale* from the problem, and read off how the couplings of the remaining problem changed. It's exactly Dyson's old idea that you should handle the high energies before the low ones, but now it's a transformation I can iterate. I am no longer grasping at straws.

But I want to understand this for a real field theory, the φ⁴ theory — which, I notice, is exactly the Landau–Ginzburg model of a critical point written as a field theory, with the φ⁴ coupling playing the role of U. So the magnet problem and the field-theory problem are the *same* problem. Good. Now, what are the degrees of freedom I should be eliminating? The kinetic term (∇φ)² is diagonal in the Fourier modes φ_k; the interaction φ⁴ is diagonal in φ(x). Neither basis diagonalizes both. So I want a compromise basis: wavefunctions as localized as possible in *both* position and momentum, i.e. occupying minimum volume in phase space — and the uncertainty principle sets that minimum to one unit. Phase space chopped into unit cells. And the momentum-slice work tells me the momentum axis should be marked off logarithmically — shells 1<|k|<2, 2<|k|<4, and so on — because each shell is a "scale." For each shell, translational invariance makes the position-space cells a uniform lattice, finer for higher momentum shells. The upshot, when I push on it: the Hamiltonian has to be cut off at some large momentum to make sense, and once cut off it *is* a lattice theory. To understand field theory I'll have to understand it on a lattice. And a magnet is already a lattice. The two ends meet.

So now I'm at Aspen, I've finally worked through Onsager, I'm convinced these eliminate-a-scale ideas apply to critical phenomena — and I'm told I've been scooped, that I should read Kadanoff. I read it. Kadanoff's picture is beautiful and it's almost mine: near Tc, group the spins into blocks, say 2×2×2 atoms, and let each block act as a single effective moment. Because ξ is huge, the blocks are still small compared to it, so this should be legitimate. He assumes the block moments interact through the *same* nearest-neighbor form as the original spins, just with an effective temperature T_L and field h_L at block size L, with T_{2L}, h_{2L} analytic functions of T_L, h_L, reaching L-independent fixed values at Tc. From that one hypothesis he derives Widom's scaling and the relations among exponents. That's real progress — it's the first time the *form* of scaling has a physical picture under it.

But I see exactly what's missing, and it's the same hole as Gell-Mann–Low: Kadanoff *postulates* that blocking preserves the two-parameter nearest-neighbor form, and he gives no way to *compute* the functions T_L → T_{2L}. It's an assumption about the transformation, not the transformation itself. He can relate exponents to each other but cannot produce a single number from the microscopic model. And Widom, whose scaling form started all this, had no theoretical basis for the homogeneity at all — I remember being puzzled by that when he presented it.

So I have a target now that's sharper than "compute critical exponents." It's: *actually construct* Kadanoff's transformation — the map from the couplings before blocking to the couplings after — from first principles, and then I get everything, because the exponents and the scaling all follow from the transformation.

Let me try to do it the way Kadanoff and Gell-Mann–Low both assume it should be done: derive a transformation for *just* the two couplings T_L, h_L (or just the charge). I try many ways. They all fail. And when I finally do the momentum-slice elimination *honestly* — to all orders, because in real life Λ = 2, not some huge number, so 1/Λ isn't small and I can't truncate the perturbation series — I see *why* they fail. Each time I eliminate a scale, the effective Hamiltonian I generate is not the same two-parameter object. It's an infinitely complicated Hamiltonian with an infinite set of new coupling constants. Eliminate another scale, another infinitely complicated Hamiltonian. The transformation simply does not live in a fixed, low-dimensional space of couplings. Trying to force it into two parameters is fighting the actual structure of the problem — that's the wall every previous attempt, including my own, kept hitting.

The patch is to stop fighting it. Let the couplings proliferate. The question is only whether that's a disaster, and I can check, because for large enough Λ I can control the generated Hamiltonians rigorously: I can prove that the higher-order perturbative corrections have only a small, boundable effect on the effective Hamiltonian, even after arbitrarily many iterations. So an RG transformation that produces arbitrarily many couplings is *not* a disaster. Better: the couplings come with a natural order of importance. In the slice model it's the order in powers of 1/Λ. On a lattice — for Ising-type models — it's *locality*: in any finite region there are only finitely many multi-spin couplings you can write down, and the most localized ones are the most important. So I restate Kadanoff: the nearest-neighbor coupling is the most important coupling because it is the most localized one you can define, but other couplings — next-nearest, four-spin, whatever fits in a 3³ or 4³ region — are generated too, and I keep them in order of importance and truncate the rest. The instant I am liberated from "only two couplings," defining the transformation becomes *easy*. The hard part isn't defining it anymore; it's finding computable approximations to it.

I can now write the machine down cleanly. I work with a Hamiltonian H, or the Landau–Ginzburg free-energy functional F_L, and I do one step of scale elimination.

I integrate out the fluctuations in the shortest length scale — one momentum shell L to L+δL, or in real space, the shortest-wavelength spins. Concretely, I hold the long-wavelength part M_H fixed and average over the fluctuation in one phase-space cell. Expanding the constrained Boltzmann sum and keeping the leading terms, the logarithm of the cell integral feeds back into the parameters: R and U pick up corrections, and the result is a free-energy functional for the remaining longer-wavelength field with changed R_L, U_L and, in general, new couplings. For the Landau–Ginzburg case this gives differential equations for how R_L and U_L run with L, with dU_L/d(ln L) carrying a term proportional to −U_L² and a piece that depends on dimension through the phase-space counting.

Then I put F_{L+δL} back into dimensionless form: measure lengths in units of L, x = Ly, and rescale M, R, and U accordingly, so that the transformation from F_L to F_{L+δL} is identical in form at every step. Now I genuinely have a single map R acting on the space of dimensionless couplings, applied over and over. This repeated transformation is the renormalization group. It is a semigroup, really, because I can coarse-grain but I cannot un-coarse-grain, but the name sticks.

As L → ∞ the dimensionless couplings can approach a fixed point μ* with R(μ*) = μ*. Why is the fixed point everything? Because at a fixed point the system is invariant under the scale change, and a scale-invariant correlation structure has ξ either 0 or ∞. The fixed point I care about is the one with ξ = ∞: that is criticality. And because the fixed point is reached regardless of the atomic-scale details I started with, systems that differ microscopically — a magnet, a fluid, an alloy — flow to the same fixed point and therefore share the same critical behavior. Universality stops being a mystery; it is the flow forgetting where it started.

The fixed point itself sits at Tc; to get exponents I have to ask how the flow leaves it. Linearize R about μ*: a small departure δμ transforms as δμ → A δμ with A the derivative of the map. Diagonalize A; its eigenvalues λ_i tell the story. A direction with λ_i > 1 grows under coarse-graining, so the system is driven away from criticality along it; these are the relevant couplings, and there are only a few of them, including temperature and field. A direction with λ_i < 1 shrinks; it is irrelevant, it decays, and this is why microscopic details do not matter. They are irrelevant directions that the flow washes out. A λ_i = 1 direction is marginal and needs more care, and I notice that the old Gell-Mann–Low, Callan–Symanzik field-theory machinery lives entirely on the marginal case, which is why it sees only logarithms. The relevant eigenvalue along the temperature direction gives the correlation-length exponent: if a length rescales by b each step and the temperature deviation t scales as t → λ_t t with λ_t = b^{y_t}, then since ξ rescales by b while t rescales by λ_t, ξ ∝ t^{−ν} with ν = ln b / ln λ_t. Every exponent comes from an eigenvalue. The transformation Kadanoff postulated, I can now linearize, and that produces the numbers he couldn't.

Now I owe myself an actual computation, not just a formalism, or I've done no better than Kadanoff. Widom asks me to present this in his stat-mech seminar, and I realize I have to put a *computable* example on the board even if it's crude. So I take the Landau–Ginzburg model, apply the phase-space-cell analysis, and ruthlessly simplify it to a nonlinear integral recursion on a function of a single variable — make no demand for accuracy, just preserve the essence of the cell picture. I iterate that recursion on a computer. It has a fixed point; I can extract exponents from it; and I can show (at least partly) that Widom's scaling theory falls out of the fixed-point structure. The recursion formula is the proof of concept that the whole machine runs.

Then the thread I left dangling — "small just below four" — pays off, and it happens while I'm showing Fisher some numbers off the recursion formula. We see it together: the nontrivial fixed point I've been studying collapses onto the trivial (Gaussian) one as d → 4. At exactly four dimensions the φ⁴ coupling is marginal; above four it's irrelevant and Landau's Gaussian fixed point is the stable one (which is *why* mean field is exact above four); below four the Gaussian fixed point goes unstable and a *new*, nontrivial fixed point splits off from it — and its distance from Gaussian is small, of order ε = 4 − d. Small distance means I can reach it perturbatively. The dimension d sits in the recursion formula as a plain parameter, so working out the leading behavior in ε is straightforward; ν comes out as 1/2 + ε/12 + …, giving ν ≈ 0.6 in three dimensions, not 1/2. Non-trivial exponents, computed. And because the controlling parameter is just the smallness of the φ⁴ coefficient, I can drop the crude recursion and do the real Landau–Ginzburg model with a Feynman-diagram expansion, using field-theory training to compute the diagrams and the fixed-point formalism to know what to do with them. The exponents become a systematic expansion in ε.

Let me now make the whole machine completely concrete and *runnable* on the cleanest case I can — the one-dimensional Ising chain by real-space decimation. It's exactly soluble, so it needs no truncation, and it shows every part of the formalism: the coarse-graining, the changed coupling, the rescaling, the fixed points, the free-energy bookkeeping. The chain is H/kT = −K Σ_i σ_i σ_{i+1}, with K = J/kT and σ = ±1.

The decimation step: keep every other spin and sum over the ones in between. Take three consecutive spins, neighbors σ₁ and σ₂ with the spin s between them. The only K-dependence on s is exp[K s (σ₁ + σ₂)], and s = ±1, so summing it,

Σ_{s=±1} exp[K s (σ₁ + σ₂)] = 2 cosh( K (σ₁ + σ₂) ).

I want the result to be a new chain of the *same form*: a constant times exp[K' σ₁ σ₂]. So I demand

2 cosh( K (σ₁ + σ₂) ) = z · exp[K' σ₁ σ₂]

for every configuration of σ₁, σ₂. There are only two distinct cases, σ₁σ₂ = +1 and σ₁σ₂ = −1:

σ₁ = σ₂ (so σ₁+σ₂ = ±2): 2 cosh 2K = z e^{+K'},
σ₁ ≠ σ₂ (so σ₁+σ₂ = 0): 2 cosh 0 = 2 = z e^{−K'}.

Two equations, two unknowns K' and z. Divide them: e^{2K'} = cosh 2K, so K' = ½ ln cosh 2K, and multiplying them, z² = 4 cosh 2K, so z = 2√(cosh 2K). The transformation closes on the
nearest-neighbor form — here no new couplings are even generated, which is the lucky feature of 1D —
and I've computed it explicitly, no postulate. (A tidy check: this is the same as tanh K' = (tanh K)²,
since e^{2K'} = cosh 2K ⇒ tanh K' = (e^{2K'}−1)/(e^{2K'}+1) = (cosh 2K − 1)/(cosh 2K + 1) = tanh² K.)

Now the fixed points of K → K' = ½ ln cosh 2K. Solve K* = ½ ln cosh 2K*. For small K, cosh 2K ≈ 1 + 2K², so K' ≈ ½ ln(1+2K²) ≈ K² — *smaller* than K. So **K* = 0 is a fixed point and it's stable**: any small coupling flows toward 0, with zero correlation length (totally disordered). For large K, cosh 2K ≈ ½ e^{2K}, so K' ≈ ½ ln(½ e^{2K}) = K − ½ ln 2 — also smaller, by a fixed amount. So **K* = ∞ is a fixed point and it's unstable**: T = 0, infinite correlation length, perfect order, but the faintest bit of temperature drives you off it toward the disordered sink. And there is *nothing in between*. Every finite K iterates down to 0. The flow itself is telling me that the one-dimensional chain has no ordering at any finite temperature — no finite-T critical point — which is exactly what Onsager-style exact solutions and the transfer matrix say (in 1D the largest transfer-matrix eigenvalue is analytic for all finite T). The RG didn't just reproduce that; it *explained* it, as the absence of any finite-K fixed point.

I should also track the free energy, because the additive constant z is not garbage — it's where the actual thermodynamics lives. Each decimation removes half the spins and contributes a factor z per removed spin: Z(K) = z^{N/2} · Z'(K') on the chain with N/2 spins. So the free energy per spin φ(K) = (1/N) ln Z obeys

φ(K) = ½ ln z(K) + ½ φ(K'),  with z(K) = 2√(cosh 2K).

Unrolling, φ = Σ_k 2^{−(k+1)} ln z_k + (tail), and once K has flowed to ≈ 0 the remaining chain is free spins contributing ln 2 each. This reconstructs the exact free energy — it had better match the transfer-matrix answer f = −ln(2 cosh K), and it does, to machine precision. So decimation isn't a heuristic here; it's an exact resummation, with the flow giving the structure and the accumulated constants giving the numbers.

```python
"""
Real-space decimation renormalization group for the 1D Ising chain.

H/kT = -K sum_i s_i s_{i+1},  s_i = +-1,  K = J/(k_B T).

One RG step: sum over every other spin (decimate, rescaling factor b=2).
Summing a middle spin s between fixed neighbors a, c:

    sum_{s=+-1} exp[K s (a + c)] = 2 cosh(K (a + c))

We demand this equal  z * exp[K' a c]  for ALL a,c in {+-1}:
    a c = +1 :  2 cosh(2K) = z e^{ K'}
    a c = -1 :  2          = z e^{-K'}
=>  K' = (1/2) ln cosh(2K),     z = 2 sqrt(cosh 2K),   so per decimated spin the
free energy picks up  g' = g + (1/2) ln z  (z accounts for two new-lattice bonds,
hence (1/2) ln z per surviving bond).

Fixed points of K -> K':  K* = 0 (stable, xi=0) and K* = oo (unstable, xi=oo).
Equivalent compact form:  tanh K' = (tanh K)^2.
"""

import math


def rg_step(K):
    """
    One decimation step (b=2). Returns (K', ln_z) where ln_z is the
    partition-function factor PER DECIMATED spin:  Z = z^{N/2} Z'(K').
    z = 2 sqrt(cosh 2K),  K' = (1/2) ln cosh 2K.
    """
    c = math.cosh(2.0 * K)
    K_new = 0.5 * math.log(c)
    ln_z = math.log(2.0 * math.sqrt(c))
    return K_new, ln_z


def flow(K0, n_steps):
    """Iterate the RG map; show the coupling flowing to the K*=0 sink."""
    K = K0
    traj = [(0, K)]
    for i in range(1, n_steps + 1):
        K, _ = rg_step(K)
        traj.append((i, K))
    return traj


def free_energy_per_spin_via_rg(K0, n_steps):
    """
    Free energy per spin f = -(1/N) ln Z (units of kT), built from the
    additive constants generated by the flow.  One step halves the spins:
        ln Z(K)/N = (1/2) ln z(K) + (1/2) * ln Z'(K')/(N/2),
    i.e. with phi(K) = ln Z/N,   phi(K) = (1/2) ln z + (1/2) phi(K').
    Unrolling:  phi = sum_k 2^{-(k+1)} ln z_k + 2^{-n} phi(K_n).
    Once K_n ~ 0 the residual chain is free spins: phi -> ln 2.
    """
    K = K0
    phi = 0.0
    weight = 0.5  # 2^{-(k+1)} for k=0
    for _ in range(n_steps):
        K_new, ln_z = rg_step(K)
        phi += weight * ln_z
        K = K_new
        weight *= 0.5
    phi += 2.0 * weight * math.log(2.0)  # residual: 2^{-n} * ln2 ; weight=2^{-(n+1)}
    return -phi


def free_energy_per_spin_exact(K):
    """Transfer-matrix result for the 1D Ising chain (zero field)."""
    return -math.log(2.0 * math.cosh(K))


if __name__ == "__main__":
    print("RG flow of K (decimation, b=2):  K* = 0 is the stable sink\n")
    for K0 in (0.1, 0.5, 1.0, 2.0, 4.0):
        traj = flow(K0, 8)
        ks = "  ".join(f"{k:.4f}" for _, k in traj)
        print(f"K0={K0:>4}:  {ks}")

    print("\ntanh K' = (tanh K)^2 check:")
    for K in (0.3, 0.7, 1.5):
        Kp, _ = rg_step(K)
        lhs, rhs = math.tanh(Kp), math.tanh(K) ** 2
        print(f"  K={K}:  tanh K'={lhs:.6f}   (tanh K)^2={rhs:.6f}")

    print("\nFree energy per spin: RG (accumulated g) vs exact transfer matrix:")
    for K0 in (0.25, 0.5, 1.0, 2.0):
        f_rg = free_energy_per_spin_via_rg(K0, 30)
        f_ex = free_energy_per_spin_exact(K0)
        print(f"  K={K0}:  f_RG={f_rg:.6f}   f_exact={f_ex:.6f}   diff={abs(f_rg-f_ex):.2e}")
```

The causal chain, start to finish: near a critical point fluctuations live on every length scale, so no single-scale expansion can work and Landau's "average out the atomic fluctuations" assumption fails below four dimensions. The way through is to handle the scales in sequence — integrate out the shortest one, getting an effective description of the rest. Doing that honestly forces a Hamiltonian with arbitrarily many couplings, and the key liberation is to *let* them proliferate (locality orders them, so truncation is controlled) instead of forcing Kadanoff's two-parameter form. Rescaling to dimensionless variables makes the step repeat identically — a flow, the renormalization group — whose fixed points are scale-invariant, hence critical (ξ=∞), hence universal (the flow forgets microscopic detail along its irrelevant directions). Linearizing the flow about the fixed point turns each eigenvalue into a critical exponent, with ν = ln b / ln λ_t. The ε = 4 − d expansion locates a perturbatively reachable nontrivial fixed point just below four dimensions and makes the exponents computable. And the 1D Ising decimation is the whole machine in miniature: K' = ½ ln cosh 2K, fixed points at 0 (stable, disordered) and ∞ (unstable, ordered), no finite-T critical point — with the accumulated free-energy constants reproducing the exact transfer-matrix thermodynamics.
