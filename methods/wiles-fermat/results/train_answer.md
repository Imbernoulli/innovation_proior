For three hundred years the equation $x^n + y^n = z^n$ with $n > 2$ has been treated as what it appears to be: a single Diophantine statement to be attacked directly. That is exactly why it has resisted. Kummer's cyclotomic machinery — class numbers of $\mathbb{Q}(\zeta_p)$, regular primes, ideal-theoretic obstructions to unique factorization — goes after the equation head-on, one prime at a time, and never reaches infinity: the irregular primes are themselves infinite and are handled by no uniform principle. The honest diagnosis is that there is no leverage *inside* the equation. If it is ever to move, one must stop staring at "no solutions" and instead manufacture, from a hypothetical solution, a deeper arithmetic object whose general theory — once proved — annihilates the equation as a by-product. The pieces that point the way are real but each falls short on its own: Frey's observation builds such an object but assumes its modularity; Serre's epsilon conjecture and Ribet's theorem turn that modularity into a contradiction but prove modularity of nothing; Langlands–Tunnell delivers modularity only of a residual representation in the solvable case, and only in characteristic zero in disguise; Mazur's deformation theory supplies the language $R = T$ but no general technique; and the Iwasawa-theoretic and Euler-system methods that bound Selmer groups either demand unknown base-change relations or fail to be long enough in the generality required. What is missing is the bridge from a modular residual representation to the modularity of the actual curve, and a Selmer bound to carry it.

I propose to prove that every semistable elliptic curve over $\mathbb{Q}$ is modular, and to obtain Fermat's Last Theorem as a corollary; the load-bearing engine is a modularity lifting theorem proved through a numerical complete-intersection criterion and Taylor–Wiles patching, with the 3–5 switch closing the last case. The reduction comes first. From a putative solution $a^p + b^p = c^p$ ($p$ an odd prime, $a,b,c$ coprime) form the Frey curve $E: y^2 = x(x-a^p)(x+b^p)$. Its three roots $0, a^p, -b^p$ have pairwise differences $a^p, b^p, a^p+b^p = c^p$, all perfect $p$-th powers, so the discriminant — which for this Legendre-type model is the product of squared root-differences up to a power of $2$ — is $\Delta = 2^{-8}(abc)^{2p}$, a perfect $p$-th power away from the prime $2$, and the conductor is the squarefree radical $\mathrm{rad}(abc)$, making $E$ semistable. The decisive consequence is that the mod-$p$ representation $\bar\rho_{E,p}: \mathrm{Gal}(\bar{\mathbb{Q}}/\mathbb{Q}) \to \mathrm{GL}_2(\mathbb{F}_p)$ is *unnaturally* unramified: ramification at a bad prime $q$ is governed by how $p$ divides the valuation of $\Delta$ there, and since every such valuation is a multiple of $p$, modulo $p$ the ramification at every odd $q \neq p$ simply vanishes, leaving a representation unramified outside $2p$ and finite flat at $p$. Such a representation, *if modular*, is forced by Serre's epsilon conjecture — level-lowering, proved by Ribet in 1986 — down to a weight-2 newform of level $2$; but $S_2(\Gamma_0(2)) = 0$ since $X_0(2)$ has genus zero, so none exists. A Fermat counterexample therefore produces a *non-modular* semistable elliptic curve, and proving every semistable $E/\mathbb{Q}$ modular kills the equation.

To gain a handle on "modular," I trade the curve for its Galois representation: $E/\mathbb{Q}$ is modular if and only if its $p$-adic representation $\rho_{E,p}: \mathrm{Gal}(\bar{\mathbb{Q}}/\mathbb{Q}) \to \mathrm{GL}_2(\mathbb{Z}_p)$ agrees with the $\rho_f$ attached by Eichler–Shimura and Deligne to some eigenform (one prime $p$ suffices, with Faltings's isogeny theorem handling the $X_0(N)$ covering). The choice of prime is the first real design decision. The instinct is $p=2$, closest to the Frey curve's 2-torsion, but that case is too degenerate to seed anything. I take $p=3$ instead, because $\mathrm{PGL}_2(\mathbb{F}_3) \cong S_4$ is solvable: for an irreducible residual $\rho_0 = \bar\rho_{E,3}$ the projective image lands in a solvable group, so Langlands–Tunnell gives a weight-one form, and the standard mod-3 weight adjustment yields the weight-2 residual modularity matching the local type of $E$. The problem is now a *lifting* problem — given the modular residual $\rho_0$, show that the genuine $p$-adic lift $\rho_{E,3}$ reducing to it is itself modular, inductively over the $\mathrm{GL}_2(\mathbb{Z}/3^n)$ representations and in the limit.

The lifting problem is organized by Mazur's deformation theory. Fixing $\rho_0: \mathrm{Gal}(\mathbb{Q}_\Sigma/\mathbb{Q}) \to \mathrm{GL}_2(k)$ and a deformation problem $D$ (determinant fixed; prescribed local behavior — ordinary or flat at $p$, controlled ramification on $\Sigma$) yields a universal deformation ring $R_D$ carrying $\rho^{\mathrm{univ}}$, through which every type-$D$ lift factors uniquely. The modular lifts sit inside a localized Hecke algebra $T_D$, and since every modular lift is a lift of type $D$, the universal property gives a canonical surjection
$$\varphi: R_D \twoheadrightarrow T_D,$$
with the content that "every lift of $\rho_0$ of type $D$ is modular" is *exactly* the assertion that $\varphi$ is an isomorphism, $R_D = T_D$. So I need a criterion that forces a surjection of complete local $O$-algebras ($O = W(k)$, possibly extended) to be injective. Two facts do it. The first is an isomorphism criterion: if $\varphi$ is an isomorphism on reduced cotangent spaces $\mathfrak{p}_R/\mathfrak{p}_R^2 \cong \mathfrak{p}_T/\mathfrak{p}_T^2$ and $T$ admits the complete-intersection presentation with the matching number of relations, then $\varphi$ is an isomorphism. Concretely, present $R \twoheadrightarrow T$ by ideals $I_R \subset I_T$ in $S = O[[x_1,\dots,x_r]]$; with $T$ a complete intersection $I_T = (f_1,\dots,f_r)$, choose an $r$-tuple of generators $g_i$ of $I_R$ realizing the relevant Fitting minor, write linear terms $f_i \equiv \sum_j a_{ij}x_j$ and $g_i \equiv \sum_j b_{ij}x_j$, so $\mathrm{Fitt}(\mathfrak{p}_T/\mathfrak{p}_T^2) = (\det a)$ and the selected minor is $(\det b)$; the cotangent isomorphism forces these to have equal valuation, while $g_i = \sum_j r_{ij} f_j$ gives $b \equiv ra$ at the closed point, hence $\det b = \det r \cdot \det a$ up to a unit, hence $\det r$ a unit, hence $I_T \subset I_R$ and $R = T$ (split off the free rank $u$ first when the cotangent is not pure torsion).

The second fact converts the problem into two checkable numbers. The Hecke rings are Gorenstein — $T \cong \mathrm{Hom}_O(T,O)$ from the duality on the modular Jacobian — and the complete-intersection criterion upgrades Gorenstein to complete intersection numerically. Let $\beta: T \twoheadrightarrow T/\mathfrak{p}_T = O$ be the chosen modular point, $\hat\beta$ its adjoint under the Gorenstein pairing, and $\eta_T = (\beta(\hat\beta(1)))$ the congruence ideal measuring congruences between the chosen eigenform and the rest — a special $L$-value. Then for a Gorenstein $O$-algebra $T$, finite free over $O$, with $T/\mathfrak{p}_T = O$ and $\mathfrak{p}_T/\mathfrak{p}_T^2$ torsion,
$$T \text{ is a complete intersection over } O \iff \ell(\mathfrak{p}_T/\mathfrak{p}_T^2) = \ell(O/\eta_T).$$
One inequality is automatic: the Gorenstein pairing gives $\mathrm{Fitt}_O(\mathfrak{p}_T/\mathfrak{p}_T^2) \subset (\eta_T)$, and over a DVR a containment $(\pi^a) \subset (\pi^b)$ forces $a \ge b$, so $\ell(\mathfrak{p}_T/\mathfrak{p}_T^2) \ge \ell(O/\eta_T)$ always — the congruence number is a free *lower* bound for the Hecke cotangent length. Complete intersection is exactly the case of equality. This is what makes the whole architecture meet: the deformation cotangent $\mathfrak{p}_R/\mathfrak{p}_R^2$ is a Selmer group (counting first-order deformations), $\mathfrak{p}_T/\mathfrak{p}_T^2$ is its quotient under $\varphi$, and $\eta_T$ is an $L$-value, so
$$\ell(O/\eta_T) \le \ell(\mathfrak{p}_T/\mathfrak{p}_T^2) \le \ell(\mathfrak{p}_R/\mathfrak{p}_R^2) = \ell(\mathrm{Selmer}) \le \ell(O/\eta_T),$$
where only the last inequality — an *upper* bound on the Selmer group by the congruence number — is open. If it holds, all four lengths collapse to equality: the cotangent map is an isomorphism, $T$ is a complete intersection, and $R = T$.

The entire weight of the proof rests on that single Selmer upper bound, a generalized class-number formula. The cyclotomic Iwasawa method — controlling class groups up a $\mathbb{Z}_p$-tower — fails here because, translated into the deformation language, it requires unknown principles of base change: exact relations between the Hecke rings for different fields in the tower, not merely up to torsion, and the projective limit refuses to become a power series ring. A Kolyvagin–Flach Euler system, which would bound the Selmer group directly by annihilating cohomology with a compatible family of classes, also fails: the system extending Flach's first step is not long enough, and the claimed bound is simply not there in the needed generality. The resolution comes from reading the Euler system's failure carefully: extending Flach forced the introduction of *special* auxiliary primes $q \equiv 1 \bmod p^n$ (not the usual $q \equiv 1 \bmod p$), chosen so the residual representation acquires extra deformations exactly there — and those are precisely the primes the abandoned Iwasawa approach always needed. With de Shalit's generalization of Hida's theory, which gives a step toward a power series ring for such primes, together with Poitou–Tate duality, the Hecke rings at the auxiliary levels can be glued. For each $n$ choose a finite set $Q_n$ of primes $q \equiv 1 \bmod p^n$ of controlled size, allowing extra ramification in $D_{Q_n}$ tuned to kill the dual Selmer obstruction and count the new local deformation directions exactly; the rings $R_{Q_n}$, $T_{Q_n}$, and the Hecke modules (with their diamond-operator action from the $p$-power quotients at the $q$'s) are patched — not through honest transition maps, but through congruent finite quotients modulo $p^n$ and the auxiliary augmentation ideals — into a single object over a power series ring $S_\infty = O[[z_1,\dots,z_r]]$. Hida/de Shalit freeness of the patched Hecke module and the Poitou–Tate variable count force the patched ring into the expected complete-intersection shape; specializing by killing the auxiliary variables descends this to the minimal Hecke ring $T$, which is therefore a complete intersection with $\ell(\mathfrak{p}_T/\mathfrak{p}_T^2) = \ell(O/\eta_T)$ and $R = T$ at minimal level. The patching *is* the missing Selmer bound, obtained structurally rather than by an Euler system; level-raising and level-lowering congruences then carry $R = T$ from the minimal level to all allowed levels, giving the full modularity lifting theorem.

One case remains: $\rho_0 = \bar\rho_{E,3}$ reducible, where Langlands–Tunnell's irreducibility hypothesis fails. Here the 3–5 switch, built from Mazur's twisted modular curves, transports modularity through the 5-torsion. If $\bar\rho_{E,5}$ were also reducible, $E$ would give a rational point on $X_0(15)$, whose only non-cuspidal rational points correspond to a handful of non-semistable curves already modular for this purpose; so I may assume $\bar\rho_{E,5}$ irreducible. Twisting the full-level-5 curve $X(5)$ by $\bar\rho_{E,5}$, the curve $E$ gives a rational point on a genus-zero component, and any rational point there yields an elliptic curve $E'$ with $E'[5] \cong E[5]$ as Galois modules. Hilbert irreducibility lets me choose such a point so that $\bar\rho_{E',3}$ is irreducible, and $5$-adically close enough that $E'$ (or a quadratic twist) is semistable at $5$. Then $E'$ is modular by the $p=3$ argument, so $\bar\rho_{E',5} \cong \bar\rho_{E,5}$ is modular, and the lifting theorem at $p=5$ makes $E$ modular. This is the one place that deforms the elliptic curve itself rather than only its Galois representation.

Assembling everything: take a semistable $E/\mathbb{Q}$ and look at $\bar\rho_{E,3}$; if irreducible, Langlands–Tunnell plus $R=T$ at $p=3$ makes $E$ modular, and if reducible, the 3–5 switch plus $R=T$ at $p=5$ makes $E$ modular. Every semistable elliptic curve over $\mathbb{Q}$ is modular, and the Frey–Ribet reduction closes the loop. The final artifact, stated as the field would present it:

```
Theorem (Modularity of semistable elliptic curves ⇒ Fermat's Last Theorem).
    Every semistable elliptic curve E/Q is modular. Consequently, for p > 3 prime,
        u^p + v^p + w^p = 0,  u, v, w ∈ Q   ⟹   uvw = 0.

Step 1 — Reduction to modularity (Frey, Serre, Ribet).
    From a hypothetical solution a^p + b^p = c^p (p odd prime, a,b,c coprime), form the
    Frey curve  E : y^2 = x(x − a^p)(x + b^p).  Its minimal discriminant is
    Δ = 2^{−8}(abc)^{2p} and its conductor rad(abc), so E is semistable; rhobar_{E,p} is
    unramified outside 2p and finite flat at p. Serre's epsilon conjecture (level-lowering),
    proved by Ribet (1986), then shows that if E were modular, rhobar_{E,p} would arise from
    a weight-2 newform of level 2; but S_2(Γ_0(2)) = 0 since X_0(2) has genus 0. Hence a
    Fermat counterexample gives a non-modular semistable elliptic curve, and it suffices to
    prove every semistable elliptic curve over Q is modular.

Step 2 — From curves to Galois representations.
    E/Q is modular ⇔ its p-adic representation rho_{E,p} : Gal(Qbar/Q) → GL_2(Z_p) is modular
    (one prime p suffices; Eichler–Shimura, Deligne, with Faltings's isogeny theorem).

Step 3 — A modular residual representation (p = 3, Langlands–Tunnell).
    For an irreducible residual rho_0 = rhobar_{E,3} : Gal(Qbar/Q) → GL_2(F_3), the projective
    image lies in PGL_2(F_3) ≅ S_4 (solvable). Langlands–Tunnell gives a weight-one form, and
    the standard weight-raising step gives weight-2 mod-3 residual modularity. The problem
    becomes: show every suitable lift of a modular rho_0 is modular.

Step 4 — Modularity lifting as R = T.
    Fix a deformation problem D. Mazur's universal deformation ring R_D carries the universal
    lift of rho_0 of type D; the modular lifts are parametrized by a localized Hecke algebra
    T_D. The universal property gives a canonical surjection φ : R_D ↠ T_D, and
        "every lift of rho_0 of type D is modular"   ⇔   φ is an isomorphism (R_D = T_D).

Step 5 — The numerical (complete-intersection) criterion.
    Let O be the coefficient ring, p_T = ker(T_D ↠ O).
      (a) Isomorphism criterion. If φ induces an isomorphism on reduced cotangent spaces
          p_R/p_R^2 ≅ p_T/p_T^2 and T has the complete-intersection presentation with the
          matching number of relations, then φ is an isomorphism (Fitting-ideal determinants).
      (b) Complete-intersection criterion. For a Gorenstein O-algebra T, finite and free over
          O, with T/p_T = O and p_T/p_T^2 torsion,
              T is a complete intersection over O   ⇔   l(p_T/p_T^2) = l(O/η_T),
          where η_T = (β(β̂(1))) is the congruence ideal. One always has
          l(p_T/p_T^2) ≥ l(O/η_T); equality is the complete-intersection condition.
    With l(p_R/p_R^2) = l(Selmer) and l(O/η_T) a special L-value, the chain
        l(O/η_T) ≤ l(p_T/p_T^2) ≤ l(p_R/p_R^2) = l(Selmer) ≤ l(O/η_T)
    collapses to equality, giving the cotangent isomorphism, the complete intersection, R = T.

Step 6 — The Selmer-group bound via Taylor–Wiles patching.
    The required upper bound l(Selmer) ≤ l(O/η_T) is a generalized class-number formula. The
    cyclotomic Iwasawa method fails (unknown base-change relations); a Kolyvagin–Flach Euler
    system fails (bound not established in the needed generality). The fix: for each n choose
    auxiliary primes Q_n with q ≡ 1 mod p^n, allowing controlled extra ramification that kills
    the dual Selmer obstruction. The rings R_{Q_n}, T_{Q_n}, and Hecke modules are patched
    through compatible finite quotients into an object over a power series ring
    S_infty = O[[z_1,…,z_r]]; Hida/de Shalit freeness and Poitou–Tate variable counts force
    the patched ring into the complete-intersection shape. Specializing by killing the
    auxiliary variables gives the minimal Hecke ring T as a complete intersection and the
    equality l(p_T/p_T^2) = l(O/η_T).

Step 7 — Removing the irreducibility hypothesis (the 3–5 switch).
    If rhobar_{E,3} is reducible, use the family of elliptic curves with the same 5-torsion
    to find an auxiliary curve E′ (after local choices and possible twist) with rhobar_{E′,3}
    irreducible and rhobar_{E′,5} ≅ rhobar_{E,5}. Then E′ is modular by Steps 3–6 at p = 3,
    so rhobar_{E,5} is modular, and Steps 4–6 at p = 5 make E modular.

Conclusion. Every semistable elliptic curve over Q is modular. Combined with Frey–Ribet, no
    non-modular Frey curve exists, so no Fermat counterexample exists:
        u^p + v^p + w^p = 0,  u,v,w ∈ Q,  p > 3   ⟹   uvw = 0.   ∎
```
