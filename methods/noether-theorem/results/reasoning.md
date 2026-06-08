The thing that won't leave me alone is the energy law in the new gravitation theory. In ordinary
physics the conservation of energy is a clean fact: you write down the energy, you show its time
derivative vanishes on the equations of motion, done. But in Einstein's theory, built so that the action
is the same in every coordinate system, the energy law goes soft. You can write ∂_μ(T^{μν} + t^{μν}) = 0,
but the gravitational piece t^{μν} isn't a tensor — it depends on which coordinates you chose, and at any
single point you can transform it away. Hilbert has said outright that in this theory the energy
equations "in the ordinary sense do not exist at all," and that he thinks this isn't a defect but a
*characteristic feature* — and that it ought to be provable. Einstein took Hilbert's candidate energy
law, derived a consequence from it that drained it of content, and asked what's going on. So that's the
puzzle in front of me, and it's not a vague philosophical one: there is a precise mathematical fact about
a generally-covariant action that makes its energy law behave unlike every other energy law I know, and
nobody can say exactly what that fact is.

Let me not start from gravity. Let me start from the only thing that's solid: the variational principle
itself, which all of this rests on. I have an integral

  I = ∫ f(x, u, ∂u/∂x, ∂²u/∂x², …) dx,

x = (x₁,…,x_n) the independent variables, u = (u₁,…,u_μ) the dependent ones, and the physics is δI = 0.
Everything I trust — the field equations, the conservation laws of mechanics — is downstream of this one
object. So whatever the relationship between symmetry and conservation is, it has to be readable *off the
action*, not bolted on afterward.

Now, what does a continuous symmetry even mean here? It means a group of transformations, in Lie's
sense, that leaves I unchanged: I = ∫ f(x,u,…) dx = ∫ f(y,v,…) dy, where the transformation carries
(x,u) to (y,v). A finite group 𝔊_ρ depends on ρ parameters ε; the infinite group 𝔊_{∞ρ} of general
covariance depends on ρ arbitrary functions p(x) and their derivatives. Near the identity — and I can
always arrange the parameters so that ε=0 is the identity — the transformation is

  yᵢ = xᵢ + Δxᵢ + …,   vᵢ(y) = uᵢ + Δuᵢ + …,

with Δx, Δu the lowest-order terms, linear in ε (or in p and its derivatives).

My first instinct is to demand that the *equations of motion* be invariant under the symmetry. That's how
a physicist usually thinks — the dynamics is symmetric. But the equations of motion are only the
stationary configurations. If I
demand symmetry of the solutions, I'm working on a thin set — the on-shell configurations — and I have
no leverage to *construct* anything. The action, by contrast, is defined for *every* field, on-shell or
not. Invariance of the action is a statement about all configurations at once. That's an enormously
stronger handle. So I'll demand invariance of the action and see what it forces, and only put in the
equations of motion at the very end. Let me hold that thought — it may be the whole point — and grind
through the variational machinery first.

The one piece of calculus-of-variations structure I'm sure of: take I and form its first variation,
varying u but holding it fixed at the boundary. Integration by parts gives

  δI = ∫ Σᵢ ψᵢ δuᵢ dx,

where the ψᵢ are the Lagrangian expressions — the left-hand sides of the Euler–Lagrange equations,
ψᵢ = ∂f/∂uᵢ − d/dx(∂f/∂uᵢ′) + …. Fine. But I threw away the boundary terms by assuming δu vanishes on
the boundary. I shouldn't throw them away — they're going to be the whole story. So let me *not* impose
the boundary condition, and write the integration-by-parts identity with the boundary terms put back in:

  Σᵢ ψᵢ δuᵢ = δf + Div A,

where Div A = ∂A₁/∂x₁ + … + ∂A_n/∂x_n and A is linear in δu and its derivatives. For a single integral
with first derivatives, this is just the central Lagrange equation,

  Σ ψᵢ δuᵢ = δf − d/dx( Σ (∂f/∂uᵢ′) δuᵢ ),

so A = −Σ (∂f/∂uᵢ′) δuᵢ; the partial integration moves every derivative off δu and dumps the cost into
that total derivative. And this is an *identity* — it holds for arbitrary δu, no equations of motion
assumed. That matters. The boundary term of the variation is a divergence. Stare at that. The boundary
term is *already* a current; integration by parts has manufactured a divergence for free.

So now I want to feed the symmetry into this. The symmetry says ΔI = 0 under the infinitesimal
transformation. Let me write out what ΔI actually is, carefully, because the transformation moves the
integration region too:

  0 = ΔI = ∫ f(y, v(y), ∂v/∂y, …) dy − ∫ f(x, u(x), ∂u/∂x, …) dx,

the first integral over the y-interval that corresponds to the x-interval. The annoyance is that the two
integrals are over different domains. I need to pull the y-integral back to the x-domain. For
infinitesimal Δx the change of variables gives a Jacobian factor, and to first order

  ∫ f(y, v(y), …) dy = ∫ f(x, v(x), …) dx + ∫ Div(f · Δx) dx.

Good — the domain-change is bookkept entirely by that Div(f·Δx) term; it's the f times the displacement
of the boundary. Now the two integrals are over the same x-domain, and the integrand difference involves
f evaluated at v(x) versus at u(x).

But v(x) − u(x) is not the same as Δu. Δu = vᵢ(y) − uᵢ(x) compares the *transformed field at the
transformed point* to the original — it has a piece coming purely from the point having moved. What I
actually want, to plug into the calculus-of-variations identity (which compares f at the *same* x), is
the variation of the field at a fixed point. So define

  δ̄uᵢ = vᵢ(x) − uᵢ(x) = Δuᵢ − Σ_λ (∂uᵢ/∂x_λ) Δx_λ.

The minus term is exactly the convective part: how much uᵢ changes just because I dragged its argument by
Δx. This δ̄ is the genuine field variation, at a point — the one the boundary-term identity is built for.
(Let me sanity-check the sign on the simplest case: a rigid shift x → x+ε with the field carried along
unchanged, Δu = 0. Then δ̄u = −(∂u/∂x)·ε. Negative, as it must be — at a fixed point the shifted field
takes the value the old field had a step back. Good.)

With δ̄ in hand, ΔI = 0 becomes

  0 = ∫ { δ̄f + Div(f · Δx) } dx,

over an *arbitrary* interval. An integral that vanishes over every interval forces its integrand to
vanish. So

  δ̄f + Div(f · Δx) = 0.

This is Lie's differential equation for the invariance of I — the local, integrand-level statement of
the symmetry. Now I substitute the calculus-of-variations identity Σ ψᵢ δ̄uᵢ = δ̄f + Div A into it.
Replace δ̄f by Σ ψᵢ δ̄uᵢ − Div A:

  ( Σ ψᵢ δ̄uᵢ − Div A ) + Div(f · Δx) = 0,
  Σ ψᵢ δ̄uᵢ = Div A − Div(f · Δx) = Div( A − f · Δx ).

Set B = A − f · Δx. Then

  Σᵢ ψᵢ δ̄uᵢ = Div B.

And there it is — stop and look at what just happened. This is an *identity*, true for every field
configuration, not just solutions. On the left, the Lagrangian expressions contracted with the symmetry
variation. On the right, a pure divergence, a current B built out of the boundary term A of the variation
minus the convective f·Δx. The symmetry of the action has *manufactured* a current whose divergence
equals (equations of motion) · (symmetry direction). Nothing has been assumed solved yet.

Now I do the one thing I deferred: I put in the equations of motion. Stationary configurations are
exactly ψᵢ = 0. Substitute that into the identity and the left side dies:

  Div B = 0.

A conservation law. *That's* the connection, and now I can see precisely why invariance of the action,
not invariance of the equations, was the right object: the off-shell identity Σ ψᵢ δ̄uᵢ = Div B is what
carries all the content, and setting ψ = 0 is a single clean substitution that converts it into Div B = 0.
If I'd only had a symmetry of the solution set, I'd never have produced B; B comes from the boundary term
of the variation of the full action, and the variation only makes sense off-shell. The conserved current
*is* the boundary term of the action's variation.

Let me make the ρ-parameter case explicit, because I want one conservation law per generator. Δu and Δx
are linear in the parameters ε₁,…,ε_ρ, so by the definition of δ̄ so is δ̄u, and so is B. Write
B = Σ_λ B^(λ) ε_λ and δ̄u = Σ_λ δ̄u^(λ) ε_λ. Since the identity holds for every ε independently,

  Σᵢ ψᵢ δ̄uᵢ^(λ) = Div B^(λ),   λ = 1,…,ρ.

ρ linearly independent combinations of the Lagrangian expressions are divergences. On the variational
problem ψ = 0, that's ρ conservation laws Div B^(λ) = 0. In one independent variable Div is just d/dx, so
Div B^(λ) = 0 means dB^(λ)/dx = 0, i.e. B^(λ) = const — ρ *first integrals*. The classical conservation
theorems of mechanics are exactly these. The converse is the same calculation read backward when the
divergence relations have this boundary-term form: multiply the ρ relations by ε_λ, add them, recover
Σψᵢδ̄uᵢ = Div B, combine with Σψᵢδ̄uᵢ = δ̄f + Div A, and Lie's invariance equation follows for the
corresponding infinitesimal transformations. So it's symmetry ⇔ conservation, both directions.

Now let me actually crank the handle on the cases I care about, because a theorem I can't turn into
energy and momentum is just decoration. Take the displacement group — rigid translations:

  Δxᵢ = εᵢ,   Δuᵢ = 0,   so   δ̄uᵢ = − Σ_λ (∂uᵢ/∂x_λ) ε_λ.

Plug into Σ ψᵢ δ̄uᵢ = Div B, peel off the coefficient of each ε_λ:

  −Σᵢ ψᵢ (∂uᵢ/∂x_λ) = Div B^(λ).

These are the energy relationships. On ψ = 0: Div B^(λ) = 0, and the B^(λ) are the energy components.
Let me see what B^(λ) actually is in field-theory notation, where the integrand is a Lagrangian density
ℒ(φ, ∂_μφ) and there are several spacetime independent variables x^μ. I have to keep the sign of the
boundary vector consistent with the central Lagrange identity: for first derivatives,
A^μ = −(∂ℒ/∂(∂_μφ)) δ̄φ, because
δ̄ℒ = ψ δ̄φ + ∂_μ[(∂ℒ/∂(∂_μφ))δ̄φ] and I am using ψδ̄φ = δ̄ℒ + ∂_μA^μ. The convective subtraction is
ℒΔx^μ with Δx^ν = ε^ν. So the current carries an upper spacetime index from Div and another from which
translation I picked, ε^ν:

  B^μ = −(∂ℒ/∂(∂_μφ)) δ̄φ − ℒ Δx^μ.

With δ̄φ = −ε^ν ∂_ν φ and Δx^μ = ε^μ = ε^ν δ^μ_ν, factor out ε^ν:

  B^μ = ε^ν [ (∂ℒ/∂(∂_μφ)) ∂_ν φ − δ^μ_ν ℒ ].

So the coefficient of ε^ν is

  T^μ_ν = (∂ℒ/∂(∂_μφ)) ∂_ν φ − δ^μ_ν ℒ.

The identity itself reads −ψ∂_νφ = ∂_μT^μ_ν, so on the field equations ∂_μT^μ_ν = 0. The canonical
energy–momentum tensor dropped straight out of translation invariance with no assumption about the form
of ℒ at all. Now read off the conserved charges by integrating the μ=0
component over space. The ν = 0 piece — invariance under time translation:

  E = ∫ T⁰₀ d³x = ∫ [ (∂ℒ/∂φ̇) φ̇ − ℒ ] d³x.

That bracket is precisely the Legendre transform of ℒ — the Hamiltonian density. So time-translation
symmetry ↔ conservation of energy, and the conserved quantity is the Hamiltonian, exactly as it must be.
The spatial pieces ν = i — invariance under space translation:

  Pⁱ = ∫ T⁰ⁱ d³x = ∫ (∂ℒ/∂φ̇)(∂ⁱφ) d³x,

the field momentum, conserved. There's the second half: space-translation symmetry ↔ conservation of
momentum. One theorem, both laws, and the same machine will grind out angular momentum from rotations:
an infinitesimal Lorentz rotation has Δx^μ = ω^μ_ν x^ν with ω antisymmetric, so for a scalar field
δ̄φ = −ω^ν_ρ x^ρ ∂_νφ. The current is B^μ = Δx^νT^μ_ν; separating the antisymmetric coefficients gives
M^{μ,ρσ} = x^ρ T^{μσ} − x^σ T^{μρ} with ∂_μM^{μ,ρσ} = 0; integrate the time component and you get the
conserved angular momentum J^{ρσ} = ∫ (x^ρ T^{0σ} − x^σ T^{0ρ}) d³x. Rotation ↔ angular momentum. The
antisymmetry of ω is what makes the current carry the antisymmetric ρσ index pair — that's why it's an
angular momentum and not something else.

So I have the whole edifice for finite groups: every ρ-parameter continuous symmetry of the action gives
ρ conserved currents, and time/space/rotation hand me energy/momentum/angular momentum. Now back to the
thing that started this — gravity, and Hilbert's claim. The symmetry of general relativity isn't a
finite group; it's the infinite group of *all* coordinate transformations y = p(x), depending on
arbitrary functions p, not on finitely many parameters ε. What does my machine do when the parameters
become arbitrary functions?

Let me rerun the derivation with p(x) in place of ε. δ̄u, and hence B, is now linear in the p's and their
derivatives up to some order σ. I can write the fixed-point variation as

  δ̄uᵢ = Σ_{λ,|α|≤σ} aᵢ^{λ,α}(x,u,∂u,...) ∂_αp^λ,

with α a multi-index. Then

  Σᵢ ψᵢδ̄uᵢ = Σ_{i,λ,|α|≤σ} ψᵢ aᵢ^{λ,α} ∂_αp^λ.

The move that worked before — equating coefficients of independent parameters — doesn't directly apply,
because the p's and their derivatives aren't independent constants; they're tied together. But I can use
integration by parts again, on the p-derivatives this time. Each integration by parts moves one derivative
off p^λ and onto its coefficient and contributes a minus sign. Modulo a divergence Div Γ,

  ψᵢ aᵢ^{λ,α} ∂_αp^λ = (−1)^{|α|} ∂_α(ψᵢ aᵢ^{λ,α}) p^λ + Div Γ_{iλα},

so after all derivatives have been moved,

  Σᵢ ψᵢ δ̄uᵢ
  = Σ_λ [ Σ_{i,|α|≤σ} (−1)^{|α|} ∂_α(aᵢ^{λ,α}ψᵢ) ] p^λ + Div Γ.

Combine with Σ ψᵢ δ̄uᵢ = Div B:

  Σ_λ [ Σ_{i,|α|≤σ} (−1)^{|α|} ∂_α(aᵢ^{λ,α}ψᵢ) ] p^λ = Div(B − Γ).

Now integrate over a region and choose the p's, and enough of their derivatives, to vanish on the
boundary. The integral of Div(B−Γ) is then a boundary integral, which vanishes — so the integral of the
left side vanishes too, for *arbitrary* p inside. By the fundamental lemma of the calculus of variations
the integrand itself must vanish for every p:

  Σ_{i,|α|≤σ} (−1)^{|α|} ∂_α(aᵢ^{λ,α}ψᵢ) = 0,   λ = 1,…,ρ.

These are *identities among the Lagrangian expressions and their derivatives*. Not conservation laws —
identities. They hold whether or not ψ = 0. In ordinary one-dimensional notation the last term is the
familiar (−1)^σ d^σ(cψ)/dx^σ piece; in several variables it is exactly the formal adjoint of the
operator that sends p^λ to δ̄uᵢ. With an infinite symmetry group I don't get ρ conserved currents; I get ρ
differential relations that the field equations themselves are forced to satisfy identically. The
arbitrary functions are too much symmetry: every extra arbitrary function spends itself not on a new
conservation law but on a constraint linking the equations of motion.

And now I can see Hilbert's puzzle dissolve. General covariance is exactly this infinite-group case. So
in gravity the "conservation law" of energy isn't free-standing the way it is for a finite group.
Compare the two sides directly. Take a finite subgroup of the infinite group — say the displacements,
generated by specializing p^{(i)}(x) = ε_i. The infinite group gives identities; the finite subgroup,
applied on its own, gives divergence relations Σ ψᵢ δ̄uᵢ^{(λ)} = Div B^{(λ)}. But because the
displacement group sits *inside* the full covariance group, those divergence relations can't be
independent of the identities — the ψ's appear linearly in both, so the divergence relations must be
linear combinations of the identities. Which means the energy current B^{(λ)} is itself a combination of
the Lagrangian expressions and their derivatives, plus a piece whose divergence vanishes identically.
Call a divergence relation whose B is built out of the ψ's in this way, up to an identically divergenceless
piece, *improper*; call the ordinary kind *proper*. The energy law of general relativity is improper:
B^{(λ)} is equivalent to a combination of the field equations and their derivatives, so "Div B = 0" has no
independent on-shell content. Its divergence vanishes as an algebraic consequence of general covariance
itself, not as a new conservation equation added to the field equations.

That is exactly what Hilbert asserted and couldn't prove — that in general relativity proper energy
equations "in the ordinary sense do not exist at all," and that this is a characteristic feature of the
theory. It's a characteristic feature of *any* theory invariant under a group depending on arbitrary
functions. Where the symmetry is finite — classical mechanics, the older Lorentz-invariant relativity
where Σ dx² maps to itself — no arbitrary functions occur, the divergence relations are proper, and
energy is genuinely conserved. Where the symmetry is the infinite covariance group, the energy relation
turns improper. The whole anomaly was never about energy; it was about the size of the symmetry group,
and the boundary-term identity is what makes the distinction visible: a finite group leaves the boundary
term as a genuine conserved current; an infinite group folds the boundary term back into the field
equations.

I can now state the result in its sharpest form. Start from an action I = ∫ f dx
invariant under a continuous group. Integration by parts gives the off-shell identity
Σ ψᵢ δuᵢ = δf + Div A, so the boundary term of any variation is a divergence. Feeding in the symmetry
through Lie's invariance equation δ̄f + Div(f·Δx) = 0 collapses this to the master identity

  Σᵢ ψᵢ δ̄uᵢ = Div B,   B = A − f · Δx,

true for all configurations. For a finite ρ-parameter group this splits into ρ identities
Σ ψᵢ δ̄uᵢ^{(λ)} = Div B^{(λ)}; imposing the Euler–Lagrange equations ψ = 0 gives ρ conservation laws
Div B^{(λ)} = 0 — first integrals in one variable, continuity equations in several — and translations,
rotations deliver energy, momentum, angular momentum as the cases
T^μ_ν = (∂ℒ/∂(∂_μφ))∂_ν φ − δ^μ_ν ℒ with E = ∫T⁰₀, Pⁱ = ∫T⁰ⁱ, J^{ρσ} = ∫(x^ρT^{0σ}−x^σT^{0ρ}). The
converse holds: ρ conservation laws imply invariance under a ρ-parameter group. For an infinite group
depending on ρ arbitrary functions the same integration by parts, now applied to the derivatives of the
arbitrary functions, yields instead ρ *identities* among the Lagrangian expressions, and the apparent
conservation laws of such a theory are improper — their currents are equivalent to combinations of the
field equations and identically divergenceless pieces — which is precisely why proper energy conservation
fails in a generally-covariant theory. The causal chain is one line: invariance of the action is an
off-shell statement, integration by parts turns it into (equations of motion)·(symmetry) = divergence of a
current, and then either the equations of motion kill the left side and leave a genuine conservation law
(finite group), or the arbitrariness of the symmetry functions forces the adjoint identities among the
equations themselves (infinite group). Symmetry and conservation are two faces of the boundary term.
