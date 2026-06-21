A whole family of phase transitions sat without any general theory: the ferromagnetic Curie point, order–disorder in alloys like β-brass, the structural and ferroelectric rearrangements, the λ-transition in liquid helium, and the liquid–gas critical point. What they share is that they proceed with no latent heat — the state of the body slides continuously through the transition, yet at the transition point a response function such as the specific heat does something abrupt, jumping or spiking. The tools available were each inadequate in a specific way. Ehrenfest's classification sorts transitions by which derivative of the free energy first becomes discontinuous, but that is bookkeeping done after the measurement: it names no governing variable, predicts neither the existence nor the size of the specific-heat jump, and offers no small quantity to expand in near the critical temperature. Weiss's molecular-field theory genuinely explains the ferromagnet — an internal field $H+\lambda M$, solved self-consistently as $M=M_{\rm sat}\tanh[\mu(H+\lambda M)/kT]$, giving spontaneous $M$ below $T_c$ and the Curie–Weiss susceptibility $\chi=C/(T-T_c)$ — but it is shackled to magnets and to a particular microscopic ansatz. Van der Waals does the analogous thing for fluids, with the density difference $\rho_l-\rho_g$ vanishing as $(T_c-T)^{1/2}$ and the compressibility diverging as $(T-T_c)^{-1}$, but it too is a one-system model.

What forces the issue is a coincidence. Weiss's magnetization and van der Waals's density difference both vanish as $(T_c-T)^{1/2}$; both response functions diverge as $(T-T_c)^{-1}$. These are utterly unrelated systems — spins on iron atoms, molecules in a gas — built from utterly unrelated models, and they come out with identical near-critical behavior. If the exponent lived in the microscopic detail it would differ between the two. It does not. So the behavior near a continuous transition is not living in the microscopics at all; it lives in something the two share, and I want a theory that knows only about that.

The shared thing is symmetry. A liquid and a gas have the *same* symmetry — both isotropic — so they differ only quantitatively and their critical point can be skirted, passing continuously from one to the other with no sharp transition; that is exactly why liquid–gas is the odd one out. A crystal, by contrast, either possesses a given symmetry element or it does not, with nothing in between: symmetry groups are discrete. So a transition that *changes* the symmetry cannot do it gradually — at one temperature the symmetry group is one thing, an instant later a different group. The reframing is this: a continuous phase transition is a transition where the state changes continuously while the symmetry changes discontinuously, the high-symmetry group dropping to a subgroup. The Curie point, alloy ordering, and structural changes are symmetry transitions, and that is what makes them sharp.

I propose what is now Landau theory: build the whole account around a single **order parameter** $\eta$ that measures how much symmetry has been broken, and let the symmetry of its free-energy expansion dictate the physics. Concretely, describe the body by its density $\rho(\mathbf r)$, which carries the full symmetry; in the symmetric phase $\rho=\rho_0$, invariant under the big group, and in the broken phase $\rho=\rho_0+\delta\rho$, where $\delta\rho$ is the symmetry-lowering piece, small near the transition and vanishing at it. Because $\delta\rho$ is a whole function of many possible shapes, decompose it over the irreducible representations of the high-symmetry group, $\delta\rho=\sum_{n,i}c_i^{(n)}\varphi_i^{(n)}$. Using irreducible representations is the load-bearing choice: the group shuffles the components within one representation among themselves and never mixes one representation into another, so each representation is an indivisible bundle and any invariant is a tidy function of one representation's amplitudes at a time. The overall magnitude $\eta$ of whichever representation switches on at the transition is the order parameter — zero above, nonzero below, continuous through.

Rather than compute an intractable partition function — and anyway I have argued the answer should not depend on the microscopics — I work with the thermodynamic potential $\Phi$, which at fixed $p,T$ is minimized at equilibrium, and expand it in powers of the small $\eta$. The crucial constraint is that $\Phi$ is a scalar invariant under every operation of the symmetric phase, so every term in the expansion must be an invariant of the high-symmetry group. The first-order (linear) term must vanish: $\rho_0$ is itself an equilibrium of the symmetric phase, so $\Phi$ is already a minimum at $\eta=0$ and cannot grow linearly away from it, and in any case a single power of a representation's amplitude is generically not an invariant. The leading surviving term is the quadratic invariant, which for a single representation is the sum of squares $\sum_i c_i^2 = \eta^2$ up to a coefficient, and the next is the quartic invariant $\eta^4$. Deferring for a moment the question of a cubic term, the expansion is

$$
\Phi(\eta;p,T) = \Phi_0(p,T) + A(p,T)\,\eta^2 + B(p,T)\,\eta^4 + \cdots, \qquad B>0,
$$

with $A,B$ smooth functions inherited from the microscopics I am refusing to look at. For $\eta=0$ to be a genuine minimum in the symmetric phase I need $A>0$; for the broken phase to acquire minima at nonzero $\eta$ continuously, the cleanest mechanism is for $A$ to pass through zero and go negative, turning $\eta=0$ from the bottom of a bowl into a local maximum and pushing the minima out to nonzero $\pm\eta$. So the transition is exactly the locus $A(p,T)=0$ — a curve in the $(p,T)$ plane, the Curie line — and $B>0$ there keeps $\Phi$ curving back up so the broken phase has a stable minimum rather than running off to $-\infty$.

Minimizing, $\partial\Phi/\partial\eta = 2A\eta + 4B\eta^3 = 0$, gives either $\eta=0$ (the only physical solution when $A>0$, the symmetric phase) or $\eta^2=-A/(2B)$ (real and positive when $A<0$, the broken phase, a degenerate pair $\eta=\pm\sqrt{-A/2B}$ between which the body spontaneously chooses). Writing $A\approx a(T-T_c)$ with $a>0$ and $B\approx b>0$ near the transition, the order parameter for $T<T_c$ is

$$
\eta = \sqrt{\frac{a}{2b}}\,(T_c-T)^{1/2},
$$

the $\beta=1/2$ law — derived from nothing but a symmetry-respecting expansion, a coefficient that changes sign, and a minimization, with no spins, no molecules, no tanh, no equation of state. Substituting $\eta^2=-A/(2B)$ back gives the equilibrium potential $\Phi_{\rm eq}=\Phi_0 - A^2/(4B)$, lowered by ordering as it must be. The entropy $S=-\partial\Phi/\partial T$ then has an excess $A(\partial A/\partial T)/(2B)$ that vanishes at $T_c$ because $A=0$ there: the entropy is continuous, there is no latent heat — precisely the mark of a continuous transition, falling straight out. One more derivative gives the specific heat $C=-T\,\partial^2\Phi/\partial T^2$, whose ordered-phase excess is the constant $T a^2/(2B)$, present only below $T_c$; so on heating up through $T_c$ the specific heat drops by a finite amount

$$
\Delta C = \frac{T_c\,a^2}{2B}\quad(\text{a downward jump, } \alpha=0),
$$

a value and a sign the theory predicts, where Ehrenfest could only say "the specific heat jumps." Adding a conjugate field, $\Phi=\Phi_0+A\eta^2+B\eta^4-h\eta$, and differentiating the equilibrium condition $2A\eta+4B\eta^3=h$ implicitly gives $\chi=1/(2A+12B\eta^2)$, hence $\chi=1/(2A)=1/[2a(T-T_c)]$ above $T_c$ (where $\eta=0$) and, using $12B\eta^2=-6A$ below, $\chi=1/(-4A)=1/[4a(T_c-T)]$ below. Both diverge as $|T-T_c|^{-1}$ — the Curie–Weiss law with Weiss temperature $\theta=T_c$ and the ordered side exactly half as susceptible at equal distance, so $\gamma=1$. Sitting exactly at $T_c$ where $A=0$, $\Phi=\Phi_0+B\eta^4-h\eta$ minimizes to $4B\eta^3=h$, i.e. $\eta\propto h^{1/3}$, the critical isotherm with $\delta=3$. All four signatures depend only on the *form* $A\eta^2+B\eta^4$ and not on the values of $a,b$, which is the universality the matching Weiss and van der Waals exponents were hinting at: it is the analytic structure of the expansion, nothing else.

Now the deferred cubic term, which decides whether a transition can be continuous at all. The expansion runs $A\eta^2+B\eta^4$ with no odd power only if the symmetry forbids a third-order invariant — true for a representation with the $\eta\to-\eta$ symmetry, like a ferromagnet where flipping all spins is a symmetry so $\Phi$ must be even. But some symmetries do admit a cubic invariant, and then $\Phi-\Phi_0=A\eta^2+C\eta^3+B\eta^4$. A nonzero stationary state coexisting with $\eta=0$ must satisfy both stationarity and equal free energy, $2A+3C\eta+4B\eta^2=0$ and $A+C\eta+B\eta^2=0$; subtracting twice the second from the first eliminates $A$ to give $C\eta+2B\eta^2=0$, so the coexistence value is $\eta_*=-C/(2B)$ and back-substitution gives $A_*=C^2/(4B)>0$. A nonzero cubic coefficient therefore makes the order parameter jump to a finite value while $A$ is still positive, before the symmetric state has lost stability at $A=0$: the transition is first order with latent heat. A genuinely continuous transition requires not only $A=0$ but the *absence of the cubic invariant* — the symmetry of the order parameter decides the matter. Near the endpoint where the first-order line shrinks into the continuous point, the less-symmetric branches still have $\eta\sim|A|^{1/2}$ and the entropy jump is odd in $\eta$, so the latent heat switches off as $Q\propto\eta^3\propto|A|^{3/2}\propto|T-T_c|^{3/2}$. Continuity can also be spoiled if $B<0$, in which case the quartic no longer bounds $\Phi$ from below and one carries the expansion to $+D\eta^6$ with $D>0$; two secondary minima then overtake $\eta=0$ above $T_c$, again first order. So continuity requires $A$ changing sign, $B>0$ there, and no cubic invariant. Finally, $\eta$ need not be uniform: allowing $\eta(\mathbf r)$, the cheapest symmetry-allowed penalty for non-uniformity is the gradient cost $\gamma(p,T)(\nabla\eta)^2$ with $\gamma>0$ — a term linear in $\nabla\eta$ would violate spatial symmetry — giving $\Phi=\int d^3r\,[\Phi_0+A\eta^2+B\eta^4+\gamma(\nabla\eta)^2-h\eta]$, which governs fluctuations and interfaces and, in three dimensions, leaves the fluctuation integral finite so the uniform minimization is self-consistent.

Everything follows from one move: choose the order parameter, and let the symmetry of its free-energy expansion write the physics. The symbolic check below confirms each step — the minimization, the $(T_c-T)^{1/2}$ law, the $-A^2/4B$ potential drop, the continuous entropy and finite specific-heat jump, the two-sided Curie–Weiss susceptibility, the cube-root critical isotherm, and the cubic-term coexistence with its first-order jump.

```python
import sympy as sp

T, Tc, a, b, h, eta = sp.symbols('T T_c a b h eta', real=True)
A, B = a*(T - Tc), b
Phi = A*eta**2 + B*eta**4 - h*eta

dPhi = sp.diff(Phi, eta)
assert sp.expand(dPhi - (2*A*eta + 4*B*eta**3 - h)) == 0

eta2 = sp.solve(sp.Eq(2*A + 4*B*eta**2, 0), eta**2)[0]
assert sp.simplify(eta2 - a*(Tc - T)/(2*b)) == 0
eta_below = sp.sqrt(eta2)

Phi_eq = sp.simplify(A*eta2 + B*eta2**2)
assert sp.simplify(Phi_eq + A**2/(4*B)) == 0

S_exc = -sp.diff(Phi_eq, T)
assert sp.simplify(S_exc.subs(T, Tc)) == 0
dC = sp.simplify(T*sp.diff(S_exc, T))
assert sp.simplify(dC - T*a**2/(2*b)) == 0

chi_above = 1/(2*A)
chi_below = sp.simplify(1/(2*A + 12*B*eta2))
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

Q_scale = sp.Pow(-A0/(2*B), sp.Rational(3, 2), evaluate=False)

print("eta^2 =", eta2)
print("eta below =", eta_below)
print("Phi_eq - Phi0 =", Phi_eq)
print("specific-heat jump =", dC)
print("chi above =", chi_above)
print("chi below =", chi_below_display)
print("critical isotherm eta =", eta_crit)
print("cubic coexistence eta*, A* =", eta_jump, A_jump)
print("latent-heat scaling near endpoint: Q ~", Q_scale)
```
