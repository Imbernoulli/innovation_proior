## Research question

Take a proper simply connected domain `Omega` in the complex plane and a chosen interior point
`a`. The question is whether the shape of `Omega` is only superficial from the point of view of
one complex variable: can there be a holomorphic coordinate `f` on `Omega` that sends the whole
domain onto the unit disk, sends `a` to `0`, and fixes the remaining rotational freedom by requiring
`f'(a)>0`?

The point is stronger than finding many clever maps. It asks whether the topology of the domain and
the rigidity of holomorphicity force a canonical conformal coordinate. If the answer is yes, then
every proper simply connected planar domain is analytically the same as the disk once one basepoint
and one tangent direction are chosen. The arbitrary boundary geometry is absorbed into the coordinate.

## Background

Complex differentiability is not ordinary two-variable differentiability. The Cauchy-Riemann
equations force the differential of a noncritical holomorphic map to be multiplication by a complex
number, so infinitesimal figures are rotated and scaled rather than sheared. Riemann's own
formulation described this geometrically as a mapping in which corresponding smallest parts are
similar.

Simple connectivity supplies the global bookkeeping that local conformality does not. On a simply
connected domain, holomorphic functions with no zeros admit logarithms, and therefore square roots;
closed periods vanish; harmonic conjugates exist when the relevant differential has zero period.
Those facts are what let local holomorphic flexibility become a single-valued global coordinate.

The potential-theoretic intuition is the Green-function picture. If a normalized disk map `H` exists
and `H(a)=0`, then near `a` it looks like a nonzero constant times `z-a`, while on the boundary of
the disk `|H|` should approach `1`. Thus `log|H|` should be a harmonic function on
`Omega \ {a}` with a logarithmic singularity at `a` and boundary value `0`. Equivalently,
`log|H(z)| = log|z-a| + u(z)`, where `u` solves a Dirichlet problem with boundary data
`-log|z-a|`. Once this harmonic function has a conjugate, exponentiation should recover `H`.

The analytic compactness toolkit gives another route. Montel's theorem says a locally bounded
family of holomorphic functions is compact for convergence on compact sets. Hurwitz's theorem
keeps nonvanishing and one-to-one behavior from collapsing in a nonconstant limit. Schwarz's lemma
then identifies the only automorphisms of the disk fixing `0`. These facts turn the search for the
coordinate into an extremal problem: among all injective holomorphic maps into the disk normalized
by `f(a)=0`, maximize the positive derivative at `a`.

## Baselines

- **Explicit elementary conformal maps.** Linear fractional maps handle disks and half-planes, and
  special functions handle some polygonal or slit domains. Gap: arbitrary simply connected domains,
  including rough boundaries, have no explicit boundary formula to assemble.

- **Local conformal charts.** A holomorphic map with nonzero derivative is locally a conformal
  coordinate, and inverse-function reasoning gives local equivalences. Gap: local coordinates do
  not decide whether one coordinate covers the whole domain once and only once.

- **Dirichlet principle and Green functions.** The harmonic function with logarithmic singularity
  and zero boundary value is exactly the modulus data a disk coordinate would need. Gap: the
  original variational principle does not by itself justify existence for arbitrary boundaries, and
  even after the harmonic function is built one must still prove the resulting holomorphic map is
  one-to-one and onto.

- **Normal-family compactness alone.** A bounded family of holomorphic maps has convergent
  subsequences. Gap: compactness has to be tied to a normalization that prevents the limit from
  degenerating and selects the largest possible conformal scale at the basepoint.

- **Injective maps into the disk.** Simply connectivity and square roots can produce at least one
  bounded injective holomorphic test map. Gap: a test map may land in a proper subdomain of the
  disk, so the proof must show that missing any disk value would contradict extremality.

## Evaluation settings

The artifact is a theorem and proof. The input is a proper simply connected domain
`Omega subset C` and a basepoint `a in Omega`. The required output is a biholomorphic map
`f: Omega -> D` with `f(a)=0` and `f'(a)>0`, and a proof that this normalized map is unique.

Stress cases include unbounded domains, domains with rough or non-Jordan boundary, the unit disk
itself, and excluded cases such as `Omega=C` or punctured domains where a global logarithm or square
root can fail. Success means proving nonemptiness of the normalized test family, existence of an
extremal map, preservation of injectivity in the limit, surjectivity onto the disk, and uniqueness
under the basepoint/tangent normalization.

## Code framework

No executable implementation is needed. The field-appropriate scaffold is a proof sequence:

1. Use simple connectivity to create a bounded injective holomorphic test map into the disk.
2. Normalize all test maps by `f(a)=0` and `f'(a)>0`.
3. Maximize `f'(a)` over the normalized family.
4. Use Montel compactness to obtain a holomorphic extremal limit.
5. Use Hurwitz's theorem to preserve injectivity in the nonconstant limit.
6. If the extremal image omits a disk value, use a disk automorphism and a holomorphic square root
   to build another normalized injective disk map with larger derivative.
7. Use Schwarz's lemma to prove that the normalized disk coordinate is unique.
