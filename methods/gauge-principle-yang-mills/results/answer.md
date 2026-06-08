# The non-abelian gauge principle (Yang–Mills)

## Problem

Isotopic spin (SU(2), the proton–neutron doublet) is an excellent *global* symmetry of the strong interaction: a single rotation ψ → Sψ, S ∈ SU(2), applied identically at every spacetime point. But the choice of internal orientation is physically meaningless, and a meaningless convention should not be forced to be the same everywhere. Demand instead that the symmetry hold **locally** — independently at each spacetime point, ψ(x) → S(x)ψ(x) — and ask what structure the world needs to support that demand.

## Key idea

A local internal symmetry breaks the bare derivative: ∂_μ(Sψ) = (∂_μS)ψ + S∂_μψ, and the inhomogeneous (∂_μS)ψ term spoils invariance. The cure — exactly as in Weyl's abelian electromagnetic gauge principle, where local U(1) phase invariance forces the photon — is a compensating **gauge field** A_μ and a **covariant derivative** that transforms covariantly. The new feature is that SU(2) is non-commuting: the field strength acquires a commutator self-interaction term [A_μ, A_ν] with no electromagnetic counterpart, so the gauge field is necessarily **self-interacting** and carries the very isotopic spin that sources it.

## The construction (final form)

Matter doublet, local transformation:

  ψ(x) → S(x) ψ(x),  S(x) ∈ SU(2),  generators T^a = σ^a/2,  [T^a, T^b] = i ε^{abc} T^c.

**Covariant derivative**, defined so that D_μψ → S(D_μψ):

  D_μ = ∂_μ − i g A_μ,   A_μ = A_μ^a T^a  (Hermitian, traceless, Lie-algebra-valued).

**Gauge-field transformation**, solved by imposing covariance of D_μψ:

  A_μ → S A_μ S⁻¹ − (i/g) (∂_μ S) S⁻¹.

The first term is homogeneous rotation; the second is the inhomogeneous compensating term. Abelian limit S = e^{iα}: A_μ → A_μ + (1/g) ∂_μ α — Weyl's electromagnetic gauge transformation.

**Field strength**, from the commutator of covariant derivatives [D_μ, D_ν] = −i g F_μν:

  F_μν = ∂_μ A_ν − ∂_ν A_μ − i g [A_μ, A_ν].

It transforms homogeneously, F_μν → S F_μν S⁻¹. The term −i g [A_μ, A_ν] is quadratic in A; it vanishes for an abelian (commuting) group, recovering Maxwell's F_μν = ∂_μA_ν − ∂_νA_μ, and is nonzero for SU(2) — the **self-interaction**.

**Locally-invariant Lagrangian:**

  L = −¼ F_μν^a F^{aμν} + ψ̄ (i γ^μ D_μ − m) ψ.

With Tr(T^aT^b) = ½δ^{ab}, the gauge kinetic term can be written as −½ Tr(F_μνF^{μν}) = −¼F_μν^aF^{aμν}. Expanding it through the commutator yields cubic (three-field) and quartic (four-field) self-couplings of the gauge field: the field is its own source. The matter current ψ̄γ_μT^aψ is not separately conserved; the conserved total isotopic spin includes the gauge field's own contribution.

**Quanta:** spin 1 (vector field), isotopic spin 1 (triplet A^a), three charge states ±e and 0. A mass term m²A_μ^aA^{aμ} is forbidden by the local invariance (it is not invariant under the inhomogeneous transformation), so the quanta come out massless — an open difficulty, since charged massless quanta coupling to nucleons are not observed.

## Derivation of the field strength (the self-interaction)

Acting on ψ, with D_μ = ∂_μ − igA_μ:

  [D_μ, D_ν]ψ = (∂_μ − igA_μ)(∂_ν − igA_ν)ψ − (μ ↔ ν).

The ∂_μ∂_ν terms cancel by symmetry; the terms with derivatives hitting ψ (−igA_ν∂_μψ − igA_μ∂_νψ and their μ↔ν images) cancel pairwise. What survives is purely multiplicative:

  [D_μ, D_ν]ψ = [ −ig(∂_μA_ν − ∂_νA_μ) − g²(A_μA_ν − A_νA_μ) ] ψ
            = −ig[ (∂_μA_ν − ∂_νA_μ) − ig[A_μ, A_ν] ] ψ.

The factor is fixed because (−ig)(−ig) = −g², so −ig multiplying −ig[A_μ,A_ν] reproduces the quadratic term with the correct sign. Hence F_μν = ∂_μA_ν − ∂_νA_μ − ig[A_μ, A_ν] and F_μν → SF_μνS⁻¹ for free (because [D_μ,D_ν] → S[D_μ,D_ν]S⁻¹). The commutator term is exactly what cancels the non-homogeneous residue thrown off by the naive curl ∂_μA_ν − ∂_νA_μ when S does not commute.

## Worked symbolic check

```python
# Verifies the sign/factor of the non-abelian field strength from [D_mu, D_nu],
# and that it reduces to Maxwell in the abelian (commuting) limit.
import sympy as sp

# SU(2) generators T^a = sigma^a / 2
s1 = sp.Matrix([[0, 1], [1, 0]])
s2 = sp.Matrix([[0, -sp.I], [sp.I, 0]])
s3 = sp.Matrix([[1, 0], [0, -1]])
T = [s/2 for s in (s1, s2, s3)]

# check the su(2) algebra [T^a, T^b] = i eps^{abc} T^c
def comm(A, B): return A*B - B*A
eps = {(0,1,2):1,(1,2,0):1,(2,0,1):1,(1,0,2):-1,(0,2,1):-1,(2,1,0):-1}
for a in range(3):
    for b in range(3):
        lhs = comm(T[a], T[b])
        rhs = sum((sp.I*eps.get((a,b,c),0)*T[c] for c in range(3)), sp.zeros(2,2))
        assert sp.simplify(lhs - rhs) == sp.zeros(2,2)
print("su(2) algebra [T^a,T^b] = i eps^abc T^c verified")

# Field strength F_mu_nu = d_mu A_nu - d_nu A_mu - i g [A_mu, A_nu], built from Lie-algebra fields.
g = sp.symbols('g', real=True)
def field_strength(dmuAnu, dnuAmu, Amu, Anu):
    return dmuAnu - dnuAmu - sp.I*g*comm(Amu, Anu)

# The factor in [D_mu,D_nu] is fixed by (-i g)(-i g) = -g^2.
assert sp.expand((-sp.I*g)*(-sp.I*g)) == -g**2
dmuAnu, dnuAmu = sp.symbols('dmuAnu dnuAmu')
Amu_sym, Anu_sym = sp.symbols('Amu Anu', commutative=False)
curl = dmuAnu - dnuAmu
expanded_commutator = -sp.I*g*curl - g**2*(Amu_sym*Anu_sym - Anu_sym*Amu_sym)
factored_commutator = -sp.I*g*(curl - sp.I*g*(Amu_sym*Anu_sym - Anu_sym*Amu_sym))
assert sp.expand(expanded_commutator - factored_commutator) == 0

# Abelian limit: a single commuting generator -> commutator drops, recover the curl (Maxwell).
A1, A2, dA12, dA21 = sp.symbols('A1 A2 dA12 dA21')
abelian = (dA12 - dA21) - sp.I*g*(A1*A2 - A2*A1)  # scalars commute => last term = 0
assert sp.simplify(abelian - (dA12 - dA21)) == 0
print("abelian limit: F_mu_nu = d_mu A_nu - d_nu A_mu (Maxwell) recovered")

# Non-abelian: the commutator term is generally nonzero -> self-interaction present.
Amu = sp.symbols('a1 a2 a3')
Anu = sp.symbols('b1 b2 b3')
Amat = sum((Amu[i]*T[i] for i in range(3)), sp.zeros(2,2))
Bmat = sum((Anu[i]*T[i] for i in range(3)), sp.zeros(2,2))
self_term = -sp.I*g*comm(Amat, Bmat)
assert sp.simplify(self_term) != sp.zeros(2, 2)
print("non-abelian: [A_mu, A_nu] != 0 -> gauge field self-interacts")
```

The causal chain: local internal symmetry ⇒ broken bare derivative ⇒ covariant derivative D_μ = ∂_μ − igA_μ with an inhomogeneously-transforming A_μ ⇒ the naive curl fails to transform cleanly for non-commuting S ⇒ a commutator term −ig[A_μ,A_ν] restores covariance ⇒ that term, nonzero only because the group is non-abelian, makes the field self-interacting (3- and 4-field vertices), with massless spin-1 isospin-1 quanta of charge ±e and 0.
