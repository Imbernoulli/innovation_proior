## Research question

Classical algebraic geometry begins with equations and their solution sets. Over an algebraically closed field, one studies subsets of affine or projective space cut out by polynomial equations, then recovers functions on the set by quotienting the polynomial ring. Geometry is built from visible points first, and the functions on those points come afterward.

The question is whether the priority can be reversed. Suppose the polynomial ring, or more generally a commutative ring `A`, is taken as the primary object. What is the geometric space whose functions are `A`? Which points should such a space contain when `A` has nilpotents, when the base is `Z` rather than an algebraically closed field, or when the object varies in a family over another object?

## Background

For a commutative ring `A`, the classical study uses the points of `k^n` satisfying a system of polynomials and the coordinate ring of regular functions on those points. The maximal ideals of a finitely generated reduced `k`-algebra over algebraically closed `k` correspond, by the Nullstellensatz, to the ordinary points of the variety. The prime ideals of `A` form a larger set: each prime corresponds to an irreducible subvariety, and containment of primes records specialization.

A locally ringed space is a topological space together with a sheaf of rings whose stalk at every point is a local ring, with a unique maximal ideal recording functions that vanish at that point. The local ring is the device that carries infinitesimal and algebraic information at a point, beyond the bare value of a function there.

In algebraic number theory one studies rings such as `Z` together with their prime ideals; in algebraic geometry one studies polynomial equations over fields. These are pursued as separate subjects with their own languages.

## Baselines

- **Classical affine varieties.** Start from zeros of polynomials in `k^n`, usually over algebraically closed `k`, and use coordinate rings of regular functions. Points are the ordinary solutions; functions are polynomial functions on those solutions modulo the ideal of the variety.

- **Maximal ideal spectra.** For finitely generated reduced `k`-algebras over algebraically closed `k`, closed points correspond to maximal ideals by the Nullstellensatz. This identifies the ordinary points of the variety with the maximal ideals of its coordinate ring.

- **Manifold or analytic-space intuition.** A space is imagined as a set of points with functions on it, with charts and transition maps describing local models patched together.

- **Separate treatments of arithmetic and geometry.** Algebraic number theory studies rings such as `Z` and their prime ideals; algebraic geometry studies polynomial equations over fields. Each has its own objects, points, and notion of function.

## Evaluation settings

Several phenomena are at stake.

Nilpotents. The rings `k[epsilon]/(epsilon^2)` and `k` each have a single ordinary point, yet the first carries a nonzero element `epsilon` with `epsilon^2 = 0` that vanishes at every ordinary point. A purely point-set description erases this first-order thickening; an adequate geometric object should retain it.

Arithmetic fibers. Reduction of an arithmetic object modulo a prime `p`, and its behavior in characteristic zero, are usually treated as separate operations. The primes of `Z` are `(0)` and the `(p)`, and one would like characteristic-zero and characteristic-`p` behavior to appear as parts of a single object over `Z`.

Families. A family of geometric objects parameterized by another object, with its generic members, special members, degenerations, and base changes, is usually described case by case. One would like the family and its fibers to be expressed in one language.

The classical case. When `A` is a finitely generated reduced algebra over an algebraically closed field, the ordinary points of the variety are already described by the maximal ideals of `A`; any new object should recover this classical picture.

## Code framework

The neutral formal scaffold is:

1. Replace a system of equations by its coordinate ring `A`.
2. Decide which set of objects associated with `A` plays the role of "points".
3. Put a topology on that set.
4. Attach a sheaf of rings recording the functions, with appropriate stalks at each point.
5. Define general objects by patching together such local models.
6. Define maps between objects, and relate them to ring homomorphisms.

The recurring choice is what data an object should carry: a point set alone, or a point set together with a sheaf of rings of functions.
