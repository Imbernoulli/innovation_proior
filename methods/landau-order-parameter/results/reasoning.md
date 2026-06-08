Let me start from what is actually bothering me. There is a whole family of transitions that proceed with no latent heat — the body's state slides continuously through the transition, yet right at the transition point its specific heat does something abrupt. A ferromagnet cooled through its Curie point starts to magnetize; β-brass cooled through its critical temperature orders its copper and zinc onto distinct sublattices; ammonium chloride and the ferroelectrics rearrange; liquid helium shows that λ-shaped spike in the specific heat. These all feel like the same phenomenon wearing different clothes. And yet the only one of them I can actually compute is the liquid–gas critical point, through van der Waals. Everything else I can at best classify.

And classification is all Ehrenfest gives me. His scheme says: look at which derivative of the free energy first jumps. If entropy and volume jump, first order, latent heat. If they stay continuous and it is the specific heat and compressibility that jump, second order. That is tidy bookkeeping, but it is bookkeeping after the measurement. It does not tell me what variable is doing the transitioning, it does not predict whether or how big the specific-heat jump is, and worst of all it gives me no small quantity to expand in as I approach the critical temperature. I want a theory, not a filing cabinet.

The two real theories I have, Weiss and van der Waals, are each shackled to one system. Weiss explains the ferromagnet by inventing an internal molecular field proportional to the magnetization, H + λM, and solving M = M_sat·tanh[μ(H+λM)/kT] self-consistently. Below a critical temperature a nonzero M appears at H = 0 — spontaneous order — and above it the susceptibility goes like C/(T − T_c), the Curie–Weiss law, diverging as you come down to T_c. That is genuinely a theory of a continuous transition. But it is a theory of *magnets*, built on a particular guess for the interaction. Van der Waals does the analogous thing for a fluid: an effective single-particle equation of state, a critical point where the liquid–gas distinction dissolves, and along the coexistence curve the density difference ρ_l − ρ_g vanishing like (T_c − T)^{1/2}, the compressibility blowing up like (T − T_c)^{−1}.

Now here is the thing that stops me. Those two numbers. Weiss's magnetization vanishes as (T_c − T)^{1/2}; van der Waals's density difference vanishes as (T_c − T)^{1/2}. Weiss's susceptibility diverges as (T − T_c)^{−1}; van der Waals's compressibility diverges as (T − T_c)^{−1}. Same powers. These are utterly unrelated systems — spins on iron atoms, molecules in a gas — derived from utterly unrelated models, and they come out with the identical near-critical behavior. If the exponent were buried in the microscopic detail it would differ between the two. It does not. So the behavior near a continuous transition is *not living in the microscopic detail at all*. It is living in something the two share. I want to find that something and write a theory that knows only about it.

So what do a magnet at its Curie point and a fluid at its critical point share? Let me push on the fluid first, because it is the one I understand, and because there is a feature of it that has always been a little strange. The liquid–gas critical point can be *gone around*. You can take a liquid, heat it and compress it on a path that loops over the critical point, and arrive at the gas without ever crossing a sharp transition — the liquid and the gas are continuously connected. Why is that allowed for liquid–gas but not, say, for a crystal turning into a different crystal? Stare at this for a second.

The answer is symmetry. A liquid and a gas have the *same* symmetry — both are isotropic, both look the same in every direction and at every point on average. There is no qualitative, all-or-nothing distinction between them; "liquid" and "gas" differ only quantitatively, in density. Nothing forbids interpolating smoothly from one to the other. But a crystal either possesses a given symmetry element — a particular reflection, a particular rotation, a translation by half a lattice spacing — or it does not. There is no such thing as "half a reflection symmetry." Symmetry is discrete. So if a transition *changes the symmetry* of the body — adds or removes a symmetry element — that change cannot happen gradually. At one temperature the symmetry group is one thing; an instant later it is a different group. The symmetry must jump even while the state of the body deforms continuously.

That reframes everything. A continuous phase transition is exactly a transition where the *state* changes continuously but the *symmetry* changes discontinuously, dropping from a higher-symmetry group to a subgroup of it. Liquid–gas is then not really a symmetry transition at all (same symmetry on both sides), which is precisely why it can be skirted. The Curie point, the alloy ordering, the structural changes — those *are* symmetry transitions, and they cannot be skirted, and that is what makes them sharp.

If symmetry is the organizing principle, then I need a variable that *measures how much symmetry has been broken*. In the symmetric phase it should be exactly zero — the high symmetry is intact. In the less-symmetric phase it should be nonzero, and as I approach the transition from below it should go continuously to zero, because the state itself is changing continuously. Let me build it concretely. Describe the body by its density function ρ(x,y,z), the probability of finding atoms at each point — this object carries the full symmetry of the body. In the symmetric phase the density is ρ₀, invariant under the big group. In the broken phase write ρ = ρ₀ + δρ, where ρ₀ keeps the high symmetry and δρ is the extra piece that lowers it. Near the transition δρ is small, and at the transition it vanishes. δρ is my variable.

But δρ is a whole function, not a number, and it can be many shapes. I want to organize it by how it transforms under the symmetric group. Decompose δρ into the irreducible representations of that group: δρ = Σ over representations n, Σ over components i, of c_i^{(n)} φ_i^{(n)}, where the φ are fixed basis functions and the amplitudes c are the numbers that go to zero at the transition. The point of using irreducible representations is that the group shuffles the components *within* one representation among themselves and never mixes one representation into another — so each representation is an independent, indivisible bundle, and any invariant I can build will be a tidy function of one representation's amplitudes at a time. The amplitude of whichever representation actually switches on at the transition — call its overall magnitude η — is the variable I have been looking for: the order parameter. Zero above, nonzero below, continuous through.

Now what do I *do* with η? The fluctuations near a continuous transition are slow and large, every microscopic model is intractable, and anyway I have just argued the answer should not depend on the microscopic model. So I refuse to compute a partition function. Instead I work with the thermodynamic potential Φ, which at fixed pressure and temperature is minimized at equilibrium, and I ask how it depends on η. Because η is small near the transition, Φ should be expandable in a power series in δρ:

Φ(ρ₀ + δρ) = Φ(ρ₀) + [first order in δρ] + [second order] + [third] + …

Each term here is some integral built from the c's, and now symmetry earns its keep. Φ is a scalar — it cannot change under any symmetry operation of the *symmetric* phase, because in the symmetric phase those operations are exact symmetries of the body. So every term in the expansion must be an invariant of the high-symmetry group. That is a brutal constraint, and it does almost all the work.

Take the first-order term first. It is linear in the c's. But ρ₀ is itself an equilibrium state of the symmetric phase, which means Φ is already at a minimum with respect to δρ when δρ = 0 — there can be no term that grows linearly away from δρ = 0, or else the symmetric phase would not be a minimum and would slide downhill immediately. The linear term must vanish identically. I can also see it from invariance: a single power of a representation's amplitude is generally not an invariant of the group, so it cannot appear anyway. Good — the expansion starts at second order.

The second-order term is quadratic in the c's, and for a single representation the only invariant the group allows is the sum of squares, Σ_i c_i², which is just η² up to its coefficient. So the leading term is A(p,T)·η², with A some smooth function of pressure and temperature inherited from the microscopic details I am refusing to look at. The quartic term is B(p,T)·η⁴ — again, for a single representation, the symmetric fourth-order invariant. So, to the order that matters,

Φ = Φ₀ + A(p,T)·η² + B(p,T)·η⁴ + …

with no term linear in η, and — I will come back to this — possibly a cubic term in between, depending on the representation.

Hold off on the cubic for a moment and look at what A and B already force. In the symmetric phase the equilibrium is η = 0, and for that to be a genuine *minimum* of Φ I need A > 0: a positive coefficient on η² makes η = 0 the bottom of a bowl. In the broken phase, η = 0 is no longer the equilibrium — some nonzero η minimizes Φ instead. The cleanest way for the bottom of the bowl at η = 0 to give way to minima at nonzero ±η, continuously, is for A to pass through zero and go negative: A > 0 makes η = 0 stable; A < 0 turns η = 0 into a local *maximum* and pushes the minima out to nonzero η. So the transition is precisely the point where

A(p,T) = 0.

And for the broken phase to have a stable minimum at all — for Φ to curve back up at large η rather than running off to minus infinity — I need B > 0 right there at the transition. That is the condition that the transition is genuinely a continuous one and not something pathological: A passing through zero with B > 0. In the (p,T) plane the equation A(p,T) = 0 is a curve, the locus of continuous transitions — the Curie line.

Now let me actually find η. Minimize:

∂Φ/∂η = 2A·η + 4B·η³ = 0  ⟹  η(2A + 4B η²) = 0.

Two solutions. Either η = 0, or η² = −A/(2B). When A > 0 the second solution is negative and unphysical, so only η = 0 survives — the symmetric phase. When A < 0 the second solution is positive and real: η² = −A/(2B), a pair of degenerate minima at η = ±√(−A/2B). That is the broken phase, and the body has to pick one of the two — it spontaneously breaks the symmetry, dropping into one well. The η = 0 state, now a maximum, is abandoned.

So the order parameter switches on the instant A goes negative. To get its temperature dependence I only need to know how A behaves near the transition. A(p,T) is smooth and crosses zero at the transition temperature T_c, so to leading order A ≈ a·(T − T_c) with a > 0 (so that A < 0 below T_c, which is where order should appear), and B ≈ b > 0, roughly constant. Then for T just below T_c,

η² = −A/(2B) = a(T_c − T)/(2b),  so  η = √(a/2b)·(T_c − T)^{1/2}.

There it is — the square root. The order parameter rises out of zero as (T_c − T)^{1/2} with exponent one-half. And I derived it from nothing but: a symmetry-respecting expansion, a coefficient that changes sign, and a minimization. No spins, no molecules, no tanh, no equation of state. This is *exactly* the power Weiss got for the magnetization and van der Waals got for the density difference — because the magnetization and the density difference are just two faces of η, and the (T_c − T)^{1/2} was never in their microscopics, it was in this universal structure all along.

Now the specific heat, where Ehrenfest only saw a jump and could not predict it. Take the value of Φ *at* equilibrium and watch it across T_c. In the symmetric phase η = 0 so Φ = Φ₀(p,T), smooth. In the broken phase substitute η² = −A/(2B) back into Φ:

Φ = Φ₀ + A·η² + B·η⁴ = Φ₀ + A·(−A/2B) + B·(A²/4B²) = Φ₀ − A²/(2B) + A²/(4B) = Φ₀ − A²/(4B).

So below T_c the equilibrium potential is lowered by A²/4B — ordering always lowers the free energy, as it must. Now the thermodynamics. The entropy is S = −∂Φ/∂T. The extra piece is −∂/∂T[−A²/4B] = (1/4B)·2A·(∂A/∂T) = A(∂A/∂T)/(2B), holding B roughly constant. At the transition A = 0, so this extra entropy term vanishes right at T_c: the entropy is continuous across the transition, no jump, no latent heat. That is the defining mark of a second-order transition, and it falls straight out. The order parameter carries entropy only quadratically through A, and A is zero at the crossing.

The specific heat is the next derivative, C = −T ∂²Φ/∂T² = T ∂S/∂T. Differentiate the broken-phase entropy excess A(∂A/∂T)/(2B) once more in T. Near T_c, A = a(T − T_c) and ∂A/∂T = a, so the excess entropy is a²(T − T_c)/(2B); its T-derivative is a²/(2B), constant. So the broken phase carries an extra specific heat

ΔC = T·a²/(2B),

evaluated at the transition: ΔC = T_c·a²/(2B). And this term is present only below T_c (only where there is order to contribute); above T_c, η = 0 and there is no such term. So as I cross *up* through T_c, the specific heat drops by T_c a²/(2B) — a finite discontinuity, not a divergence. A jump, downward, of a size the theory predicts in terms of the very coefficients in the expansion. Ehrenfest could only say "second order means the specific heat jumps"; here the jump has a value, T_c a²/2B, and a sign.

Now the susceptibility — the response that diverged in Weiss's law, and I would love to recover that from this structure too. To probe the response I need a field conjugate to η — a magnetic field for a ferromagnet, the appropriate ordering field in general — that couples linearly and biases η. Add it to the potential as −h·η:

Φ = Φ₀ + A·η² + B·η⁴ − h·η.

Minimizing now gives 2A·η + 4B·η³ = h. The susceptibility is χ = (∂η/∂h) at h → 0. Differentiate the equilibrium condition implicitly with respect to h:

2A·(∂η/∂h) + 12B·η²·(∂η/∂h) = 1  ⟹  χ = 1/(2A + 12B η²).

Two regimes. Above T_c, η = 0 in zero field, so χ = 1/(2A) = 1/[2a(T − T_c)]. That diverges as (T − T_c)^{−1} as I come down to the transition. This is the Curie–Weiss law, χ = C/(T − T_c), reproduced — with the Weiss temperature θ identified as T_c itself, and an amplitude C = 1/2a fixed by the expansion coefficient. Weiss's empirical-looking law is just the η = 0 branch of this minimization.

Below T_c the order parameter is nonzero, η² = −A/(2B), so 12B η² = −6A, and

χ = 1/(2A + 12B η²) = 1/(2A − 6A) = 1/(−4A) = 1/[4a(T_c − T)].

So below T_c the susceptibility also diverges as (T_c − T)^{−1}, with the same exponent −1, but with an amplitude smaller by a factor of two — the response is four-times-a versus two-times-a, so the ordered side is exactly half as susceptible as the disordered side at equal distance from T_c. The exponent is one on both sides; that is γ = 1, matching Weiss and matching van der Waals's compressibility, and again it came from the form of the expansion, not from any model.

One more, to pin the behavior down completely: sit exactly at T_c, where A = 0, and ask how η responds to the field. There Φ = Φ₀ + B η⁴ − h η, and minimizing gives 4B η³ = h, so η = (h/4B)^{1/3} ∝ h^{1/3}. On the critical isotherm the order parameter grows as the cube root of the field — exponent δ = 3. Three numbers now, β = 1/2, γ = 1, δ = 3, plus the specific-heat jump (a discontinuity rather than a divergence, which is the "α = 0" statement), and all four are properties of the *form* A η² + B η⁴, completely independent of what a and b actually are. Any two systems whose order parameters break the same symmetry and so admit the same expansion must share these numbers. That is the universality the matching exponents of Weiss and van der Waals were hinting at, now explained: it is the analytic structure of the expansion, nothing else.

Now I have to go back and deal with the thing I deferred — the cubic term. I assumed the expansion ran A η² + B η⁴ with no odd power in between. That is only true if the symmetry forbids a third-order invariant. Whether it does depends on the representation the order parameter belongs to. For a representation with the η → −η symmetry — a ferromagnet, where flipping all spins is a symmetry, so the free energy must be even in the magnetization — every odd power is forbidden, the cubic is absent, and everything above goes through. But for some symmetries the group *does* admit a third-order invariant. Then I must include it:

Φ = Φ₀ + A η² + C η³ + B η⁴ + …,

and now stare at what a cubic term does. I can see the jump without drawing anything. A nonzero stationary point obeys

2A + 3Cη + 4Bη² = 0,

after dividing by η. If this nonzero state coexists with the η = 0 state, its free-energy excess also has to vanish:

A + Cη + Bη² = 0.

Subtract twice the second equation from the first, and the A terms disappear:

Cη + 2Bη² = 0.

The nonzero coexistence value is therefore η_* = −C/(2B), and substituting this back gives A_* = C²/(4B). That is the obstruction in one line. If the cubic coefficient is nonzero, the system jumps to a finite η while A is still positive, before the symmetric state has lost stability at A = 0. Only in the limiting case C → 0 do both the jump and A_* shrink to zero, with η_* of order |A_*|^{1/2}. So a third-order term, whenever the symmetry allows it, *forces the transition to be first order*. A genuinely continuous transition can only occur when the order parameter belongs to a representation with no third-order invariant — when C is forbidden by symmetry, or, on the curve where it is allowed but happens to vanish, only at isolated points. The condition for a continuous transition is therefore not only A = 0 but the *absence of the cubic invariant*; the symmetry of the order parameter decides whether a transition can be continuous at all.

Let me follow the first-order case far enough to read off its small latent heat near the limiting point where the cubic coefficient vanishes. In a stable phase the entropy is S = −(∂Φ/∂T)_η, because the term involving dη/dT is multiplied by ∂Φ/∂η and drops out. For the two less-symmetric branches that differ only by the sign of η, the even A_T η² contribution to the entropy is the same on both sides, while the odd cubic contribution changes sign. The entropy jump is therefore proportional to η³. But near the limiting point the branch value still satisfies η² proportional to |A|, so the latent heat scales as

Q = T ΔS ∝ η³ ∝ |A|^{3/2}.

Since A is linear in the distance from the limiting transition temperature, this is Q ∝ |T − T_c|^{3/2}. So the odd term converts the smooth (T_c − T)^{1/2} emergence into a discontinuous jump, and the latent heat switches off with the three-halves power as the first-order line approaches the continuous endpoint. The presence or absence of one symmetry-forbidden cubic term is the whole difference between a continuous and a discontinuous transition.

There is a second way a transition can fail to be continuous even with no cubic, and I should note it for completeness: if B itself comes out negative. I assumed B > 0 for stability, but in some systems the quartic coefficient is negative; then η⁴ no longer bounds the potential from below and I have to carry the expansion to sixth order, +D η⁶ with D > 0. With A η² − |B| η⁴ + D η⁶, a pair of secondary minima develops at nonzero η while A is still positive, and they overtake the η = 0 minimum at a temperature above where A vanishes — again a first-order transition, this time driven by the sign of the quartic rather than by an odd term. So continuity requires A changing sign *and* B > 0 there *and* no cubic invariant.

One last loose end. I treated η as uniform over the body, but it need not be — it can vary in space, and the transition's fluctuations are exactly slow spatial variations of η. If I let η = η(r), the cheapest way to penalize non-uniformity, consistent with the symmetry, is a term quadratic in the gradient, γ(p,T)(∇η)², with γ > 0 so that the uniform state is favored. This is forced by the same logic: a term linear in ∇η would violate the spatial symmetry, so the leading spatial cost is (∇η)². It barely changes the uniform analysis near T_c — the gradient coefficient stays finite while A → 0 — but it is what lets me ask about the size and energetics of fluctuations, and for a three-dimensional body the resulting fluctuation integral stays finite, so the uniform minimization is self-consistent there. I will keep it in the back of my mind; the core result lives in the uniform expansion.

So here is the whole thing, assembled. The pain was a family of latent-heat-free transitions with no general theory and a suspicious coincidence of critical exponents across unrelated systems. The coincidence said the near-critical behavior is not microscopic. The escape was symmetry: because symmetry is discrete, a continuous transition is a continuous change of state at a point where the symmetry group abruptly drops to a subgroup, and the natural variable is the amplitude η of the symmetry-lowering part of the density — zero in the symmetric phase, nonzero below. Expanding the thermodynamic potential in η and demanding every term be invariant under the high-symmetry group kills the linear term (equilibrium of the symmetric state), generally kills the cubic (when symmetry forbids it), and leaves Φ = Φ₀ + A(p,T)η² + B η⁴ with B > 0. The transition is where the quadratic coefficient A changes sign; A ≈ a(T − T_c). Minimizing gives η² = −A/2B, hence η ∝ (T_c − T)^{1/2}; the equilibrium potential drops by A²/4B, giving a continuous entropy (no latent heat) and a finite downward specific-heat jump T_c a²/2B; adding a conjugate field gives a susceptibility 1/2A above and 1/(−4A) below the transition, both diverging as |T − T_c|^{−1} — the Curie–Weiss law with θ = T_c — and the critical isotherm η ∝ h^{1/3}. All of these exponents depend only on the form of the expansion, so any system with the same order-parameter symmetry shares them. And when symmetry *does* allow a cubic term, coexistence occurs at η_* = −C/(2B) and A_* = C²/(4B), so the order parameter jumps before A reaches zero; near the limiting endpoint the associated latent heat on the neighbouring first-order line goes as |T − T_c|^{3/2}. Everything followed from one move: pick the order parameter, and let the symmetry of its expansion write the physics.

```python
import sympy as sp

T, Tc, a, b, h, eta = sp.symbols('T T_c a b h eta', real=True)
A = a*(T - Tc)
B = b
Phi = A*eta**2 + B*eta**4 - h*eta

dPhi = sp.diff(Phi, eta)
assert sp.expand(dPhi - (2*A*eta + 4*B*eta**3 - h)) == 0

eta2_broken = sp.solve(sp.Eq(2*A + 4*B*eta**2, 0), eta**2)[0]
assert sp.simplify(eta2_broken - a*(Tc - T)/(2*b)) == 0
eta_broken = sp.sqrt(eta2_broken)

Phi_eq_broken = sp.simplify(A*eta2_broken + B*eta2_broken**2)
assert sp.simplify(Phi_eq_broken + A**2/(4*B)) == 0

S_excess = -sp.diff(Phi_eq_broken, T)
assert sp.simplify(S_excess.subs(T, Tc)) == 0
dC_jump = sp.simplify(T*sp.diff(S_excess, T))
assert sp.simplify(dC_jump - T*a**2/(2*b)) == 0

chi_above = 1/(2*A)
chi_below = sp.simplify(1/(2*A + 12*B*eta2_broken))
chi_below_display = 1/(4*a*(Tc - T))
assert sp.simplify(chi_above - 1/(2*a*(T - Tc))) == 0
assert sp.simplify(chi_below - chi_below_display) == 0

eta_crit = sp.Pow(h/(4*B), sp.Rational(1, 3), evaluate=False)

A0, C3 = sp.symbols('A0 C3', real=True)
Phi_cubic = A0*eta**2 + C3*eta**3 + B*eta**4
eta_jump = -C3/(2*B)
A_jump = C3**2/(4*B)
assert sp.simplify(sp.diff(Phi_cubic, eta).subs({eta: eta_jump, A0: A_jump})) == 0
assert sp.simplify(Phi_cubic.subs({eta: eta_jump, A0: A_jump})) == 0

eta_less_symmetric = sp.sqrt(-A0/(2*B))
Q_scale = sp.Pow(-A0/(2*B), sp.Rational(3, 2), evaluate=False)

print("eta^2 =", eta2_broken)
print("eta below =", eta_broken)
print("Phi_eq - Phi0 =", Phi_eq_broken)
print("specific-heat jump =", dC_jump)
print("chi above =", chi_above)
print("chi below =", chi_below_display)
print("critical isotherm eta =", eta_crit)
print("cubic coexistence eta*, A* =", eta_jump, A_jump)
print("latent-heat scaling near endpoint: Q ~", Q_scale)
```
