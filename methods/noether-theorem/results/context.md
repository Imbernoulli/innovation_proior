# Context: invariance and the law of conservation of energy in generally-covariant theories

## Research question

A variational principle sits underneath almost all of physics. A mechanical system or a field is
governed by an action — an integral

I = ∫ f(x, u, ∂u/∂x, ∂²u/∂x², …) dx

over independent variables x = (x₁,…,x_n) and dependent variables u = (u₁,…,u_μ) — and the physical
configurations are the stationary points δI = 0, the solutions of the Euler–Lagrange equations. Quite
separately, physics carries a list of conservation laws: energy, linear momentum, angular momentum. By
1917 these two facts are known to be related in scattered special cases (Hamel, Herglotz, Lorentz and
his pupils, Klein), but the relationship is a patchwork of individual results rather than one principle.

The pressing, concrete form of the question comes from the new general theory of relativity. Einstein's
field equations (November 1915) and Hilbert's "Foundations of Physics" (1916) build gravity on an action
that is invariant under *every* smooth coordinate transformation y = p(x) — general covariance, a
symmetry that depends on arbitrary functions, not finitely many parameters. In such a theory the law of
conservation of energy refuses to behave like the energy law of ordinary (Lorentz-invariant) physics.
One can write ∂_μ(T^{μν} + t^{μν}) = 0, but the gravitational contribution t^{μν} is not a tensor: it
depends on the coordinates and can be transformed away at a point. Hilbert went so far as to assert,
without proof, that for general relativity "energy equations which in your sense correspond to the energy
equations of the orthogonally invariant theories do not exist at all," and that he regarded this as a
*characteristic feature* of the theory, for which a mathematical proof should be possible. Einstein, in
a letter, had derived a consequence of Hilbert's proposed energy law that "deprives the theorem of its
sense" and asked for clarification.

So the question is sharp and twofold. (1) What, in full generality, is the relationship between a
continuous symmetry of the action and a conservation law — enough to make energy ↔ time-translation,
momentum ↔ space-translation, angular momentum ↔ rotation fall out as instances of one theorem? (2) Why
does that relationship *change character* when the symmetry group is the infinite-dimensional group of
all coordinate transformations, so that proper energy conservation fails in general relativity? A
satisfying answer must explain both with the same machinery.

## Background

**The calculus of variations and Hamilton's principle.** The settled tool is the first variation. Given
I = ∫ f(x,u,u') dx and a variation δu of the dependent variables that vanishes on the boundary,

δI = ∫ Σᵢ ψᵢ δuᵢ dx,   ψᵢ = ∂f/∂uᵢ − d/dx(∂f/∂uᵢ′) + … ,

where the ψᵢ are the **Lagrangian expressions** (the left-hand sides of the Euler–Lagrange equations).
This formula is obtained by *integration by parts*: every derivative of δu is moved off δu and onto f,
and the price of each integration by parts is a boundary term. Collecting those boundary terms, the
integration by parts produces, *before* any boundary condition is imposed, an identity

Σᵢ ψᵢ δuᵢ = δf + Div A,

with A linear in δu and its derivatives, and Div A = ∂A₁/∂x₁ + … + ∂A_n/∂x_n. For a single integral with
first derivatives this is the "central Lagrange equation" Σ ψᵢ δuᵢ = δf − d/dx(Σ (∂f/∂uᵢ′) δuᵢ). The key
structural fact, knowable from the calculus of variations alone, is that **the boundary term of the
first variation is a total divergence**, and that this identity holds for arbitrary δu — it is an
off-shell identity, not a statement about solutions.

**Lie's theory of continuous groups.** Sophus Lie's theory of finite continuous groups 𝔊_ρ (depending
analytically on ρ essential parameters ε) and infinite continuous groups 𝔊_{∞ρ} (depending on ρ
arbitrary functions p(x) and their derivatives) supplies the language for "continuous symmetry." A group
can always be normalized so that the zero values of the parameters (or arbitrary functions) give the
identity transformation; near the identity the most general transformation is

yᵢ = xᵢ + Δxᵢ + …,   vᵢ(y) = uᵢ + Δuᵢ + …,

with Δx, Δu the lowest-order (linear) terms in ε or in p and its derivatives. Lie already studied which
differential equations admit a given group, treating differential equations in general; equations arising
from a variational problem carry more structure than arbitrary group-admitting equations.

**Conservation laws as divergence relations.** A "law of conservation" is a divergence equation
Div B = 0 for some current B. In one independent variable this collapses to dB/dx = 0, i.e. B = const,
a **first integral** of the equations of motion. The classical conservation theorems of mechanics are
exactly such first integrals: when f does not depend explicitly on time, energy is a first integral;
when f is invariant under a rigid spatial shift, total momentum is a first integral; under a rigid
rotation, angular momentum. Each of these is, separately, known.

**The diagnostic phenomenon from general relativity.** It is already observed,
in the Göttingen circle around Klein, Hilbert and Einstein (1915–1918), that the energy law of
general relativity is anomalous: the candidate conservation relation does not behave like an
independent conservation law the way it does in ordinary physics. This is the phenomenon Hilbert named a
characteristic feature of the theory. It is a fact about the generally-covariant action — knowable before
any general theorem — and it is the diagnostic anchor that any correct account of "symmetry ⇒
conservation" must reproduce.

## Baselines

These are the partial results a general theorem would have to subsume and surpass.

- **Case-by-case first integrals in mechanics (cyclic coordinates; Routh, Jacobi).** If a coordinate q_j
  does not appear in the Lagrangian (an "ignorable" or cyclic coordinate), its conjugate momentum
  ∂L/∂q̇_j is conserved. If the Lagrangian has no explicit time dependence, the Jacobi integral / energy
  function h = Σ q̇_j ∂L/∂q̇_j − L is conserved. *Gap*: these are read off individually from the form of
  L; there is no single statement that produces all of them from the symmetry, and nothing tells you
  what to do for a symmetry that mixes the coordinates non-trivially or that depends on arbitrary
  functions.

- **Special-group results: Hamel, Herglotz, Fokker, Lorentz, Klein, Kneser.** Several authors had, for
  particular finite groups or particular variational problems, derived the corresponding invariants or
  conservation laws (Hamel and Herglotz for special finite groups; Lorentz and Fokker for relativistic
  invariance; Klein and Weyl for special infinite groups; Kneser on setting up invariants by a similar
  method). *Gap*: each is tied to a specific group or a specific Lagrangian. None states the general
  correspondence, none separates the finite-group case from the infinite-group case, and none has a
  converse (symmetry ⇔ conservation law in both directions).

- **Hilbert's energy vector and the GR pseudotensor (Hilbert 1916; Einstein 1916; Klein 1918).** In
  general relativity one constructs an "energy vector"/pseudotensor t^{μν} so that
  ∂_μ(T^{μν} + t^{μν}) = 0. Klein, in his dialogue with Hilbert (1918), showed Hilbert's intermediate
  calculations could be shortened "by the use of the ordinary Lagrange variation theorem." *Gap*: the
  object is coordinate-dependent, can be gauged away at a point, and Hilbert could only *assert* — not
  prove — that proper energy equations "do not exist at all" in GR. There is a conjecture in want of a
  theorem, and no explanation of *why* general covariance forces this.

- **Lie's theory of group-admitting differential equations.** Lie classified continuous groups and the
  equations admitting them. *Gap*: it is a theory of differential equations in general, not of equations
  *arising from a variational problem*; it does not draw on the extra structure that the action provides,
  and so cannot reach the precise symmetry ↔ conservation statements.

## Evaluation settings

The natural yardsticks are not benchmark datasets but the standard variational systems whose
conservation laws are already known and against which any general theorem must be checked:

- **Point mechanics**, single integral I = ∫ L(t, q, q̇) dt: time-translation t → t+ε (energy), spatial
  translation q → q+ε (momentum), rotation (angular momentum), Galilean boosts. These supply the
  textbook first integrals the theorem must reproduce.
- **Classical field theory**, multiple integral I = ∫ ℒ(φ, ∂_μφ, x) d⁴x: spacetime translations
  x^μ → x^μ + ε^μ (energy–momentum tensor), Lorentz rotations/boosts (angular-momentum tensor), and
  internal phase symmetries φ → e^{iα}φ (charge). The natural test is whether translation invariance of
  ℒ yields ∂_μ T^{μν} = 0 with the canonical T^{μν}, and whether E = ∫T⁰₀ and Pⁱ = ∫T⁰ⁱ come out as the
  conserved energy and momentum.
- **General relativity**, the generally-covariant gravitational action invariant under the
  infinite-dimensional group of all coordinate transformations y = p(x). The decisive test of the
  *second*, infinite-group side of the theory is whether it reproduces — and explains — Hilbert's
  assertion that proper energy conservation fails, in the form of identities among the field equations.
- **Worked single-integral examples** with f = ½u′² and its boundary-modified partners, used to exhibit
  first integrals, non-linear dependencies among them, and the freedom of adding a divergence to f.


