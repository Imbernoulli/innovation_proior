# Noether's theorem: continuous symmetry ⇒ conserved current

## The problem it solves

Conservation laws (energy, momentum, angular momentum, charge) and continuous symmetries of a physical
system are tied together, but before this result the connection was a patchwork of special cases. The
sharp open question came from general relativity: its action is invariant under the infinite group of
all coordinate transformations, and in such a theory the energy-conservation law degenerates — Hilbert
had asserted, without proof, that proper energy equations "do not exist at all" in general relativity.
Noether's theorem gives the exact, general relationship and explains both the ordinary case and the
relativistic anomaly with one piece of machinery.

## The key idea

Demand invariance of the **action** I = ∫ f(x, u, ∂u/∂x, …) dx — not of the equations of motion. Action
invariance is an *off-shell* identity (true for all field configurations), so it can *construct* a
current. Integration by parts in the calculus of variations makes the boundary term of any variation a
total divergence; the symmetry turns this boundary term into a conserved current, and imposing the
Euler–Lagrange equations afterward is a single substitution.

## Noether's first theorem (precise statement)

Let I = ∫ f(x, u, ∂u/∂x, …) dx be invariant under a finite continuous group 𝔊_ρ of ρ parameters. Then
ρ linearly independent combinations of the Lagrangian expressions ψᵢ (the Euler–Lagrange left-hand
sides) are divergences:

  Σᵢ ψᵢ δ̄uᵢ^{(λ)} = Div B^{(λ)},   λ = 1,…,ρ,

and on the variational problem ψᵢ = 0 these become ρ conservation laws Div B^{(λ)} = 0 — in one
independent variable, ρ first integrals B^{(λ)} = const. The converse holds: ρ such divergence relations
imply invariance under a ρ-parameter group.

(**Second theorem**, for completeness: if I is invariant under an infinite group 𝔊_{∞ρ} depending on ρ
arbitrary functions p^λ and their derivatives up to order σ, and
δ̄uᵢ = Σ_{λ,|α|≤σ} aᵢ^{λ,α}∂_αp^λ, then the formal adjoint gives ρ off-shell identities
Σ_{i,|α|≤σ}(−1)^{|α|}∂_α(aᵢ^{λ,α}ψᵢ)=0. Such groups' conservation laws are "improper" — their currents
are combinations of the field equations, up to identically divergenceless terms — which is exactly why
proper energy conservation fails under general covariance. The converse holds too.)

## Proof of the first theorem

**Notation.** Independent variables x = (x₁,…,x_n), dependent u = (u₁,…,u_μ). Lagrangian expressions
ψᵢ = ∂f/∂uᵢ − d/dx(∂f/∂uᵢ′) + … . Div A = Σ_k ∂A_k/∂x_k.

**Step 1 — off-shell variational identity.** Integration by parts moves all derivatives off the variation
δu and produces a boundary divergence:

  Σᵢ ψᵢ δuᵢ = δf + Div A,   A linear in δu and its derivatives.

(For a single integral, first derivatives: A = −Σᵢ (∂f/∂uᵢ′) δuᵢ.) This holds for arbitrary δu.

**Step 2 — write out the symmetry.** Normalize ε=0 to the identity: yᵢ = xᵢ + Δxᵢ, vᵢ(y) = uᵢ + Δuᵢ,
with Δx, Δu linear in ε. Invariance ΔI = 0 means

  0 = ∫ f(y, v(y), …) dy − ∫ f(x, u(x), …) dx.

**Step 3 — pull back the moved domain.** For infinitesimal Δx,

  ∫ f(y, v(y), …) dy = ∫ f(x, v(x), …) dx + ∫ Div(f · Δx) dx,

the Div(f·Δx) term being the Jacobian/boundary-displacement contribution.

**Step 4 — fixed-point variation.** Define the field variation at a fixed point,

  δ̄uᵢ = vᵢ(x) − uᵢ(x) = Δuᵢ − Σ_λ (∂uᵢ/∂x_λ) Δx_λ

(subtracting the convective part). Then ΔI = 0 becomes ∫ { δ̄f + Div(f·Δx) } dx = 0 over every interval,
so the integrand vanishes — Lie's invariance equation:

  δ̄f + Div(f · Δx) = 0.

**Step 5 — combine.** Substitute Step 1 (with δ̄) into Step 4: δ̄f = Σ ψᵢ δ̄uᵢ − Div A, so

  Σ ψᵢ δ̄uᵢ − Div A + Div(f·Δx) = 0  ⇒  **Σᵢ ψᵢ δ̄uᵢ = Div B**,   B = A − f · Δx.

This is an identity in all arguments (off-shell).

**Step 6 — split by parameter.** B and δ̄u are linear in ε: B = Σ_λ B^{(λ)} ε_λ, δ̄u = Σ_λ δ̄u^{(λ)} ε_λ.
Equating coefficients of each ε_λ:

  Σᵢ ψᵢ δ̄uᵢ^{(λ)} = Div B^{(λ)},   λ = 1,…,ρ.

**Step 7 — impose the field equations.** On solutions ψᵢ = 0:

  Div B^{(λ)} = 0,   λ = 1,…,ρ.

ρ conservation laws. In one independent variable Div = d/dx, so B^{(λ)} = const: ρ first integrals. ∎

## Energy and momentum corollaries

In field-theory notation with density ℒ(φ, ∂_μφ) and spacetime variables x^μ, the first-derivative
boundary vector has A^μ = −(∂ℒ/∂(∂_μφ))δ̄φ in the identity used above, so the current of Step 5 is

  B^μ = −(∂ℒ/∂(∂_μφ)) δ̄φ − ℒ Δx^μ.

**Translations** x^μ → x^μ + ε^μ: Δx^μ = ε^μ, Δφ = 0 ⇒ δ̄φ = −ε^ν ∂_ν φ. Factoring out ε^ν gives the
canonical **energy–momentum tensor**

  T^μ_ν = (∂ℒ/∂(∂_μφ)) ∂_ν φ − δ^μ_ν ℒ,

with split identity −Σᵢψᵢ∂_νφᵢ = ∂_μT^μ_ν, hence ∂_μT^μ_ν = 0 on solutions. The conserved charges:

- **Time translation (ν=0) → energy:**  E = ∫ T⁰₀ d³x = ∫ [ (∂ℒ/∂φ̇) φ̇ − ℒ ] d³x = ∫ ℋ d³x  (the Hamiltonian).
- **Space translation (ν=i) → momentum:**  Pⁱ = ∫ T⁰ⁱ d³x = ∫ (∂ℒ/∂φ̇)(∂ⁱφ) d³x.

**Rotations/Lorentz** Δx^μ = ω^μ_νx^ν (ω antisymmetric): δ̄φ = −ω^ν_ρx^ρ∂_νφ, giving the
**angular-momentum current** M^{μ,ρσ} = x^ρ T^{μσ} − x^σ T^{μρ}, ∂_μM^{μ,ρσ} = 0, with conserved
angular momentum J^{ρσ} = ∫ (x^ρ T^{0σ} − x^σ T^{0ρ}) d³x.

So: time-translation ↔ energy, space-translation ↔ momentum, rotation ↔ angular momentum — all instances
of the one theorem, and conversely each conservation law reflects a symmetry of the action.

## Optional symbolic check (1-D, free particle)

A minimal sanity check that the constructed current is conserved exactly on solutions; the field-theory
result above needs no code.

```python
import sympy as sp

x = sp.symbols('x')
u = sp.Function('u')(x)
up = u.diff(x)

f = up**2 / 2                                   # f = ½ u'^2  (free particle in x)
psi = sp.diff(f, u) - sp.diff(sp.diff(f, up), x)  # ψ = -u''  (Euler–Lagrange expression)

# Time/x-translation: Δx = 1, Δu = 0  ⇒  δ̄u = -u'.
dbar_u = -up
# Boundary term A = -(∂f/∂u') δ̄u ; current B = A - f·Δx (the energy/Hamiltonian).
A = -sp.diff(f, up) * dbar_u
B = A - f * 1
# Master identity: ψ·δ̄u should equal d/dx of B  (off-shell).
print(sp.simplify(psi))                            # -> -u''
print(sp.simplify(psi * dbar_u - sp.diff(B, x)))   # -> 0  : Div B = ψ·δ̄u, so B'=0 when ψ=0.
print(sp.simplify(B))                              # -> u'^2/2  : the conserved energy.
```
