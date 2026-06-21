# Context: modularity of semistable elliptic curves over Q

## Research question

Does the equation x^n + y^n = z^n have any solution in positive integers when n > 2? Fermat
asserted in a margin around 1637 that it does not, and left no proof. For three hundred years the
question resisted every general method. By the 1980s the partial results came from one tradition:
Kummer's nineteenth-century work on cyclotomic fields and ideal class numbers had settled the
exponent p for every "regular" prime, and refinements of that circle of ideas pushed verification
to large but finite bounds.

The reframing that makes progress conceivable is to attach geometry to a hypothetical solution. Given a
counterexample a^p + b^p = c^p (p an odd prime), form the elliptic curve

    E:  y^2 = x(x - a^p)(x + b^p).

This curve is highly unusual. Its minimal discriminant is Delta = 2^{-8}(abc)^{2p} — a perfect
p-th power up to the prime 2 — and its conductor is rad(abc), the squarefree radical, so E is
semistable. The mod-p Galois representation on its p-torsion, rho: Gal(Qbar/Q) -> GL_2(F_p), is
then unramified at every odd bad prime q ≠ p and finite flat at p: the p-th-power structure of
the discriminant makes the tame inertia disappear modulo p. A representation that flat and that
unramified is exactly the kind that "should not exist" once it is forced to be modular. So the
research question becomes: can one prove that every semistable elliptic curve over Q is modular —
arising from a modular form?

## Background

**Elliptic curves and modularity.** An elliptic curve E/Q is *modular* if it admits a finite
covering by a modular curve X_0(N), equivalently if its Hasse-Weil L-function L(E,s) coincides with
the L-function of a weight-2 cusp form for Gamma_0(N). A conjecture growing out of work of Taniyama
and Shimura, made precise and propagated by Weil, asserts that *every*
elliptic curve over Q is modular. Before the work described here only finitely many j-invariants
were known to be modular; the conjecture was numerically supported but had no general method behind
it.

**Galois representations attached to modular forms (the first tradition).** Eichler and Shimura
(weight 2) and Deligne (higher weight) attach to each eigenform f of weight k, level N, character X
a family of p-adic Galois representations rho_{f,lambda}: Gal(Qbar/Q) -> GL_2(O_{f,lambda}),
unramified outside Np, with trace(rho(Frob_q)) = a_q(f) and det(rho(Frob_q)) = X(q) q^{k-1} for
q ∤ Np. The reverse direction — recognizing which Galois representations *come from* modular forms —
was almost untouched outside the special weight-one case of Langlands and Tunnell. Serre's
conjectures of the mid-1980s framed this reverse problem: every odd irreducible mod-p representation
rho_0: Gal(Qbar/Q) -> GL_2(Fbar_p) should be modular, and there should be a recipe for the minimal
weight and level.

**The Frey-curve mechanism (the motivating arithmetic fact).** Frey observed that the curve
above, built from a Fermat counterexample, has a mod-p representation so lightly ramified that, if
it were modular, level-lowering would push it to an impossibly small level. Serre formalized this as
the epsilon conjecture (a level-lowering statement for mod-p modular representations), and Ribet
proved it in the summer of 1986. The consequence is sharp: if the Frey curve were modular, its mod-p
representation would arise from a weight-2 newform of level 2 — but S_2(Gamma_0(2)) = 0 because
X_0(2) has genus zero, so no such form exists. Therefore a Fermat counterexample would produce a
*non-modular* semistable elliptic curve. Modularity of all semistable elliptic curves over Q would
then immediately yield Fermat's Last Theorem. This reduction turns a hopeless Diophantine equation
into a question about modularity.

**Deformation theory of Galois representations.** Mazur, seeking to understand Hida's one-parameter
families of Galois representations, developed the deformation theory of a fixed residual
representation rho_0: Gal(Q_Sigma/Q) -> GL_2(k) (k a finite field of characteristic p, Q_Sigma the
maximal extension unramified outside a finite set Sigma). For a suitable deformation problem D
(fixing determinant and the local behavior at each prime), there is a *universal deformation ring*
R_D, a complete Noetherian local W(k)-algebra carrying a universal lift rho^univ: every lift of
rho_0 of type D is obtained from rho^univ by a unique ring map R_D -> A. Mazur further conjectured
that, in good cases, R_D should equal a Hecke algebra — a first hint that "all lifts are modular."

**Hecke algebras, the Gorenstein and congruence structure.** The Hecke algebra T acting on the
relevant space of cusp forms, localized at the maximal ideal m attached to rho_0, parametrizes the
modular lifts. Mazur showed (prime level) that these localized Hecke rings are Gorenstein; the
property was extended as needed. The congruences between a chosen eigenform f and other forms of
varying level — a theory begun by Hida and Ribet — are measured by a congruence ideal eta, which
records a special value. Kunz's commutative algebra and Tate's account of Grothendieck duality for
complete intersections are available tools for relating ring-theoretic invariants of such local rings.

**Iwasawa theory and Galois cohomology (the second tradition).** The analytic class number formula
of Dirichlet, revived through the conjecture of Birch and Swinnerton-Dyer and through Iwasawa
theory, interprets sizes of class groups / Selmer groups as special values of L-functions. Iwasawa's
method studies how class numbers vary up a cyclotomic Z_p-tower of fields. Mazur and Wiles proved
the main conjecture for Q; Wiles proved it for totally real fields, where he introduced a technique
of replacing the cyclotomic tower by a sequence of auxiliary primes q_i ≡ 1 mod p^{n_i} with
n_i -> infinity. The Poitou-Tate duality theorems in Galois cohomology give the fundamental relations
between a Selmer group and its dual, and between Selmer groups as the allowed ramification set varies.

**Euler systems.** Kolyvagin's method of Euler systems produces, from a compatible family of
cohomology classes (e.g. Heegner points, cyclotomic units), sharp upper bounds on Selmer groups and
Tate-Shafarevich groups. Flach constructed what looks like the first step of such a system in the
setting needed here.

## Baselines

- **Kummer's cyclotomic approach to FLT (regular primes).** Works in Z[zeta_p]; for primes p not
  dividing the class number of Q(zeta_p) (regular primes), unique factorization obstructions vanish
  and FLT for exponent p follows.
- **Langlands-Tunnell theorem.** For a continuous odd irreducible complex representation with finite
  solvable projective image (dihedral, A_4, or S_4), the representation is modular and arises from a
  weight-one form. Read mod p with p = 3: since PGL_2(F_3) ≅ S_4, every odd irreducible mod-3
  representation with image defined over F_3 has projective image in a solvable group, and the
  weight-one result gives the needed residual modularity after the standard mod-3 weight adjustment.
- **Ribet's theorem / Serre's epsilon conjecture.** Level-lowering: a mod-p modular representation
  unramified at q (and finite flat at p) that occurs at level qN already occurs at level N.
  Consequence for the Frey curve: modular => weight-2 level-2 => contradiction. This converts FLT
  into the modularity of semistable curves.
- **Mazur's R=T expectation and deformation rings.** Gives the language (universal deformation ring,
  Hecke ring) and the conjecture that they agree in special cases.
- **Euler systems (Kolyvagin, Flach).** Bound Selmer groups when a compatible system of cohomology
  classes exists.
- **Iwasawa-theoretic class-number formulas.** Give sharp bounds on class groups / Selmer groups in
  the cyclotomic CM setting via special values of L-functions.

## Evaluation settings

The natural yardsticks are arithmetic, not numerical. A proof must (i) be valid for *all* semistable
elliptic curves over Q (square-free conductor), since Ribet's reduction needs the whole class, not
finitely many j-invariants; (ii) be checkable in the known cases — the CM cases via Rubin-Kolyvagin,
the dihedral / A_4 / S_4 residual cases via Langlands-Tunnell, the modularity already verified for
small conductor; (iii) connect to the Langlands-Tunnell residual base case and still reach the
semistable curves it does not directly cover. The criteria of success are exactness (the relevant
arithmetic invariants must match exactly, not merely be close), uniformity across the semistable
family, and internal consistency with Poitou-Tate duality and with the known special values of
L-functions.

## Proof framework

    Goal: every semistable elliptic curve E/Q is modular  (=> FLT via Frey + Ribet).

    Available reductions from prior art:
        - Frey + Ribet: a Fermat counterexample => a semistable E whose rho_{E,p}
          cannot be modular, so it suffices to prove every semistable E/Q is modular.
        - Eichler-Shimura / Faltings isogeny: E modular <=> its p-adic representation
          rho_{E,p} is modular, for one prime p.
        - Langlands-Tunnell: for an irreducible rho_0 = rhobar_{E,3} the projective
          image lies in PGL_2(F_3) ~= S_4 (solvable), giving residual modularity at p = 3.

    Open:
        - bridge from a modular residual representation rho_0 to modularity of the
          actual semistable curve E;
        - account for the curves not reached by the p = 3 residual hypothesis.
