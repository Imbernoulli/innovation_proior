## Research question

Classical algebraic geometry begins with equations and their solution sets. Over an algebraically closed field, one studies subsets of affine or projective space cut out by polynomial equations, then recovers functions on the set by quotienting the polynomial ring. This works beautifully for many geometric questions, but it builds geometry from visible points first and only later asks what functions remain on those points.

Grothendieck's question reverses the priority. Suppose the polynomial ring, or more generally a commutative ring `A`, is the real object. What is the space whose functions are `A`? Which points should it contain if `A` has nilpotents, if the base is `Z` rather than an algebraically closed field, or if the object is varying in a family over another object?

The scheme answer is: use all prime ideals of `A`, not just classical solutions or maximal ideals, and attach to them a sheaf of local rings. Then define general algebraic-geometric objects as locally ringed spaces locally isomorphic to these prime spectra, patched together by gluing.

## Background

For a commutative ring `A`, `Spec A` is the set of prime ideals of `A` with the Zariski topology. Basic opens have the form `D(f) = { p in Spec A : f notin p }`. The structure sheaf `O_Spec A` assigns functions locally represented by fractions; on `D(f)` it is governed by the localization `A_f`, and the stalk at a prime `p` is the local ring `A_p`.

The pair `(Spec A, O_Spec A)` is not just a topological space. It is a locally ringed space: every point has a local ring of functions, with a unique maximal ideal recording functions that vanish at that point. This local ring is the device that makes infinitesimal and algebraic information part of the geometry, rather than an afterthought.

A scheme is a locally ringed space covered by open subsets isomorphic to affine schemes `Spec A`. The phrase "patched from prime spectra" is literal: affine pieces are glued along open subschemes, and the sheaves of rings are glued with the spaces. The local model is algebra, but the final object can be global, singular, reducible, nonreduced, and defined over an arbitrary base.

## Baselines

- **Classical affine varieties.** Start from zeros of polynomials in `k^n`, usually over algebraically closed `k`, and use coordinate rings of regular functions. This misses the generic points naturally associated with irreducible subvarieties, and it usually treats nilpotent functions as invisible because they vanish at every ordinary point.

- **Maximal ideal spectra.** For finitely generated reduced `k`-algebras over algebraically closed `k`, closed points correspond to maximal ideals by the Nullstellensatz. This identifies ordinary points but not all geometric specializations: prime ideals record irreducible subvarieties as points, including generic points.

- **Manifold or analytic-space intuition.** A space is often imagined as a set of points with functions on it. Schemes keep that slogan but change what a point is and what a function is: points are prime ideals, and functions are sheaf-theoretic local ring elements.

- **Separate treatments of arithmetic and geometry.** Algebraic number theory studies rings such as `Z` and their prime ideals; algebraic geometry studies polynomial equations over fields. Schemes put both under the same formal object, so `Spec Z` is a geometric base with a generic point and closed prime fibers.

## Evaluation settings

The construction must explain why nilpotents matter. `Spec k[epsilon]/(epsilon^2)` and `Spec k` have the same underlying single topological point, but different structure sheaves. The former has a nonzero infinitesimal function `epsilon`, so the scheme remembers first-order thickening that the point set alone would erase.

It must explain arithmetic fibers. A scheme `X` over `Spec Z` has a generic fiber over `(0)`, usually a characteristic-zero object, and closed fibers over `(p)`, objects in characteristic `p`. Reduction modulo primes is not a side operation; it is taking fibers of one morphism of schemes.

It must explain families. A morphism `X -> S` is a family of schemes parameterized by `S`; for each point `s in S`, the fiber is `X_s = X x_S Spec k(s)`. Because points of `S` include nonclosed and generic points, the family includes generic behavior, special fibers, degeneration, and base change in one language.

It must also recover the classical case. When `A` is a finitely generated reduced algebra over an algebraically closed field, the closed points of `Spec A` recover the classical affine variety, while the full scheme adds generic points and the structure sheaf that makes gluing and morphisms intrinsic.

## Code framework

The neutral formal scaffold is:

1. Replace a system of equations by its coordinate ring `A`.
2. Replace the visible solution set by `Spec A`, the set of all prime ideals.
3. Put the Zariski topology on `Spec A`, where containment of primes encodes specialization.
4. Attach the structure sheaf `O_Spec A`, with local rings `A_p` as stalks.
5. Define general objects by gluing affine schemes as locally ringed spaces.
6. Define maps by morphisms of locally ringed spaces, equivalently ring homomorphisms in the affine opposite direction.

The key invariant is therefore not just a point set but the pair `(space, sheaf of local rings)`. This is why nilpotents, arithmetic fibers, and families are not separate extensions of the theory; they are ordinary features of the same definition.
