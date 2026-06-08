# Landau theory of continuous phase transitions — the order parameter

## Problem

Continuous (second-order) phase transitions — the ferromagnetic Curie point, order–disorder in alloys, structural and ferroelectric transitions, the liquid–gas critical point — proceed with no latent heat: the state changes continuously while a response function (specific heat, susceptibility) does something abrupt at the critical temperature. Before this theory each such transition had only a system-specific model (Weiss for magnets, van der Waals for fluids) or a mere taxonomy (Ehrenfest). A suspicious fact demanded explanation: unrelated systems show the *same* near-critical power laws, so the critical behavior cannot live in the microscopic detail.

## Key idea

A continuous transition is a continuous change of state at a point where the symmetry group of the body drops discontinuously to a subgroup (symmetry is discrete, so it must jump). Introduce a single **order parameter** η — the amplitude of the symmetry-lowering part of the density, decomposed over the irreducible representations of the high-symmetry group — that is **zero in the symmetric phase and nonzero in the broken-symmetry phase**. Expand the thermodynamic potential Φ in powers of η, keeping only terms invariant under the high-symmetry group. The whole physics then follows from where the quadratic coefficient changes sign.

## The free-energy expansion

$$
\Phi(\eta;p,T) \;=\; \Phi_0(p,T) \;+\; A(p,T)\,\eta^2 \;+\; B(p,T)\,\eta^4 \;+\;\cdots, \qquad B>0.
$$

- The **linear term vanishes** because the symmetric state ρ₀ is an equilibrium (a minimum at η = 0).
- **Odd terms vanish** when the order parameter has the symmetry η → −η (the standard case); a cubic invariant, when symmetry permits it, changes the conclusions (below).
- Near the transition $A(p,T)\approx a\,(T-T_c)$ with $a>0$; $B\approx b>0$.

The continuous-transition locus in the (p,T) plane is $A(p,T)=0$ with $B>0$ there.

## Equilibrium order parameter and mean-field exponents

Minimize, $\partial\Phi/\partial\eta = 2A\eta + 4B\eta^3 = 0$:

$$
\eta = 0 \;\;(A>0,\ \text{symmetric}), \qquad
\eta^2 = -\frac{A}{2B}\;\;(A<0,\ \text{broken}).
$$

**Order parameter (β = 1/2).**
$$
\eta = \sqrt{\frac{a}{2b}}\,(T_c-T)^{1/2}, \qquad T<T_c.
$$

**Free energy and specific-heat jump (α = 0).** Substituting $\eta^2=-A/2B$,
$$
\Phi_{\rm eq} = \Phi_0 - \frac{A^2}{4B}.
$$
Entropy $S=-\partial\Phi/\partial T$ has excess $A(\partial A/\partial T)/2B$, which **vanishes at $T_c$** (since $A=0$): no entropy jump, no latent heat — the mark of a continuous transition. The specific heat $C=-T\,\partial^2\Phi/\partial T^2$ gains, in the ordered phase only, a constant term, so it **jumps by a finite amount**
$$
\Delta C = \frac{T_c\,a^2}{2B}\quad(\text{downward on heating through }T_c).
$$

**Susceptibility / Curie–Weiss law (γ = 1).** With a conjugate field, $\Phi=\Phi_0+A\eta^2+B\eta^4-h\eta$, implicit differentiation gives $\chi=1/(2A+12B\eta^2)$:
$$
\chi = \frac{1}{2A}=\frac{1}{2a\,(T-T_c)}\;\;(T>T_c), \qquad
\chi = \frac{1}{-4A}=\frac{1}{4a\,(T_c-T)}\;\;(T<T_c).
$$
Both diverge as $|T-T_c|^{-1}$ — the Curie–Weiss law with $\theta=T_c$ — the ordered side being half as susceptible at equal distance.

**Critical isotherm (δ = 3).** At $T=T_c$ ($A=0$): $4B\eta^3=h \Rightarrow \eta\propto h^{1/3}$.

All four signatures (β = 1/2, γ = 1, δ = 3, α = 0 jump) depend only on the *form* $A\eta^2+B\eta^4$, not on the values of $a,b$ — hence **universality**: any system whose order parameter breaks the same symmetry shares them. This is why Weiss (magnets) and van der Waals (fluids) gave identical exponents.

## When the transition is first order

If the symmetry of the order parameter **admits a third-order invariant**, the expansion has a cubic term $C\eta^3$. For
$$
\Phi-\Phi_0=A\eta^2+C\eta^3+B\eta^4,\qquad B>0,
$$
coexistence of $\eta=0$ with a nonzero stationary state requires
$$
2A+3C\eta+4B\eta^2=0,\qquad A+C\eta+B\eta^2=0,
$$
so
$$
\eta_*=-\frac{C}{2B},\qquad A_*=\frac{C^2}{4B}>0.
$$
Thus a nonzero cubic coefficient makes the order parameter jump before $A$ reaches zero: a **first-order** transition with latent heat. A genuinely continuous transition requires not only $A=0$ but **no cubic invariant**. Near the endpoint where the first-order line shrinks into the continuous point, $\eta\sim |A|^{1/2}$ on the less-symmetric branches and the entropy jump is odd in η, so the latent heat switches off as
$$
Q\propto \eta^3\propto |A|^{3/2}\propto |T-T_c|^{3/2}.
$$
A continuous transition can also be spoiled if $B<0$ (then carry the expansion to $+D\eta^6$, $D>0$); two secondary minima then overtake $\eta=0$ above $T_c$, again first order.

## Spatial variation (gradient term)

For a non-uniform order parameter η(**r**), symmetry permits a leading gradient cost $\gamma(p,T)\,(\nabla\eta)^2$ with $\gamma>0$, giving
$$
\Phi=\int d^3r\;\Big[\Phi_0 + A\,\eta^2 + B\,\eta^4 + \gamma\,(\nabla\eta)^2 - h\,\eta\Big].
$$
This governs fluctuations and interfaces; in three dimensions the fluctuation integral is finite, so the uniform minimization is self-consistent there.

## Worked symbolic check

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

The order parameter, the (T_c − T)^{1/2} law, the finite specific-heat jump, the Curie–Weiss susceptibility, and the cubic-term/first-order dichotomy all drop out of one move: choose the order parameter and let the symmetry of its free-energy expansion dictate the rest.
