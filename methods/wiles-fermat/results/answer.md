# Modularity of semistable elliptic curves, and Fermat's Last Theorem

## The problem solved

Fermat's Last Theorem: for every integer n > 2, x^n + y^n = z^n has no solution in positive
integers. The remaining prime-exponent form proved here is: if u^p + v^p + w^p = 0 with
u, v, w ∈ Q and p > 3 prime, then uvw = 0.

## The key idea

Do not attack the equation. Prove instead that **every semistable elliptic curve over Q is modular**
(it arises from a weight-2 modular form). By the Frey–Serre–Ribet reduction, a Fermat counterexample
would yield a semistable elliptic curve that is *not* modular; so semistable modularity implies FLT.
Modularity of a curve is reformulated as modularity of its p-adic Galois representation. In the
irreducible mod-3 case, Langlands–Tunnell supplies a modular residual representation; the remaining
task is a **modularity lifting theorem**, proved by identifying a universal deformation ring with a
Hecke algebra (R = T) via a numerical complete-intersection criterion. The required Selmer-group
bound is delivered by Taylor–Wiles patching, and the reducible mod-3 case is handled by the 3–5
switch.

## The proof strategy, stated precisely

**Step 1 — Reduction to modularity (Frey, Serre, Ribet).**
From a hypothetical solution a^p + b^p = c^p (p odd prime, a,b,c coprime), form the Frey curve
E: y^2 = x(x − a^p)(x + b^p). Its minimal discriminant is Δ = 2^{−8}(abc)^{2p} and conductor
rad(abc), so E is semistable; rhobar_{E,p} on the p-torsion is unramified outside 2p and finite flat
at p. Serre's epsilon conjecture (level-lowering), proved by Ribet (1986), then shows that if E were
modular, rhobar_{E,p} would arise from a weight-2 newform of level 2; but S_2(Γ_0(2)) = 0 (X_0(2)
has genus 0). Hence a Fermat counterexample gives a non-modular semistable elliptic curve.
**It therefore suffices to prove every semistable elliptic curve over Q is modular.**

**Step 2 — From curves to Galois representations.**
E/Q is modular ⇔ its p-adic representation rho_{E,p}: Gal(Qbar/Q) → GL_2(Z_p) is modular (one prime
p suffices; Eichler–Shimura, Deligne, with Faltings's isogeny theorem for the X_0(N) covering).

**Step 3 — A modular residual representation (p = 3, Langlands–Tunnell).**
For an irreducible residual rho_0 = rhobar_{E,3}: Gal(Qbar/Q) → GL_2(F_3), the projective image lies
in PGL_2(F_3) ≅ S_4 (solvable). Langlands–Tunnell gives a weight-one modular form for the associated
finite complex representation, and the standard weight-raising step gives the needed weight-2 mod-3
residual modularity. The problem becomes: **show every suitable lift of a modular rho_0 is modular**
(a lifting theorem), proved through the compatible GL_2(Z/3^n) lifts and passage to the limit.

**Step 4 — Modularity lifting as R = T.**
Fix a deformation problem D (determinant fixed; prescribed local types — ordinary or flat at p,
controlled ramification at Σ). Mazur's universal deformation ring R_D carries the universal lift of
rho_0 of type D; the modular lifts are parametrized by a localized Hecke algebra T_D. The universal
property gives a canonical surjection φ: R_D ↠ T_D, and

    "every lift of rho_0 of type D is modular"   ⇔   φ is an isomorphism (R_D = T_D).

**Step 5 — The numerical (complete-intersection) criterion.**
Let O be the coefficient ring, p_T = ker(T_D ↠ O). Two facts reduce R = T to commutative algebra:

  (a) *Isomorphism criterion.* If φ induces an isomorphism on reduced cotangent
      spaces p_R/p_R^2 ≅ p_T/p_T^2 and T has the complete-intersection presentation with the matching
      number of relations (r relations in the torsion cotangent case, more generally r-u after the
      free rank u is split off), then φ is an isomorphism. Match generators and relations via
      Fitting-ideal determinants.

  (b) *Complete-intersection criterion.* For a Gorenstein O-algebra T, finite
      and free over O, with T/p_T = O and p_T/p_T^2 torsion,

          T is a complete intersection over O   ⇔   l(p_T/p_T^2) = l(O/η_T),

      where η_T = (β(β̂(1))) is the congruence ideal (β: T ↠ T/p_T = O, β̂ its adjoint under the
      Gorenstein duality T ≅ Hom_O(T,O)). One always has l(p_T/p_T^2) ≥ l(O/η_T); equality is the
      complete-intersection condition.

The deformation cotangent length l(p_R/p_R^2) is measured by a **Selmer group**; the Hecke cotangent
p_T/p_T^2 is its quotient; and the congruence length l(O/η_T) measures a **special value of an
L-function**. The required chain is

    l(O/η_T) ≤ l(p_T/p_T^2) ≤ l(p_R/p_R^2) = l(Selmer) ≤ l(O/η_T).

The first inequality is automatic, the middle inequality comes from R_D ↠ T_D, and the last is the
Selmer upper bound. Equality gives the cotangent isomorphism, the complete-intersection property, and
therefore R = T.

**Step 6 — The Selmer-group bound via Taylor–Wiles patching (the engine).**
The required upper bound l(Selmer) ≤ l(O/η_T) is a generalized class-number formula. The cyclotomic
Iwasawa method fails because it needs unknown base-change relations between Hecke rings up the tower;
a Kolyvagin–Flach Euler system fails because it does not establish the bound in the needed generality
(the 1993 gap). The fix: for each n choose auxiliary primes Q_n with q ≡ 1 mod p^n, allowing controlled
extra ramification that kills the dual Selmer obstruction. The deformation rings R_{Q_n}, Hecke rings
T_{Q_n}, and Hecke modules at these auxiliary levels are patched through compatible finite quotients
into an object over a power series ring S_infty = O[[z_1,…,z_r]]. Hida/de Shalit freeness and
Poitou–Tate variable counts force the patched ring to have the expected complete-intersection shape.
Specializing by killing the auxiliary variables gives the minimal Hecke ring T as a complete
intersection and yields the equality l(p_T/p_T^2) = l(O/η_T). This is the Taylor–Wiles patching
argument; it delivers the Selmer bound structurally, without an Euler system.

**Step 7 — Removing the irreducibility hypothesis (the 3–5 switch).**
If rhobar_{E,3} is reducible (Langlands–Tunnell does not apply), use the family of elliptic curves
with the same 5-torsion to find an auxiliary curve E′, after the needed local choices and possible
twist, with rhobar_{E′,3} irreducible and rhobar_{E′,5} ≅ rhobar_{E,5}. Then E′ is modular by
Steps 3–6 at p = 3, so rhobar_{E,5} is modular, and Steps 4–6 at p = 5 make E modular.

**Conclusion.** Every semistable elliptic curve over Q is modular (modularity lifting + 3–5 switch).
Combined with Frey–Ribet, no non-modular Frey curve exists, so no Fermat counterexample exists:

    u^p + v^p + w^p = 0,  u,v,w ∈ Q,  p > 3   ⟹   uvw = 0.   ∎

## Why each choice is forced

- **Work with Galois representations, not curves**: modularity of the curve is inaccessible directly;
  the representation is deformable and countable.
- **p = 3**: only here does a solvable projective image (PGL_2(F_3) = S_4) let Langlands–Tunnell seed
  an irreducible modular residual representation; p = 2 is too degenerate.
- **R = T via a complete-intersection criterion**: lifting modularity is exactly the statement that
  the universal lift ring equals the modular one; the numerical criterion pairs the Hecke cotangent
  length with the congruence number, while the Selmer bound controls the deformation cotangent above it.
- **Taylor–Wiles patching instead of an Euler system or a cyclotomic tower**: the auxiliary primes
  q ≡ 1 mod p^n build the patched power-series object the Iwasawa method needed and the Euler system
  could not supply — turning the broken Euler system's own special primes into the missing key.
- **3–5 switch**: the only device that covers semistable curves with reducible mod-3 representation,
  by transporting modularity through a shared mod-5 representation.
