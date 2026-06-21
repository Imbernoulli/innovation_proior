## Research question

The problem is to decide whether a continuous measurement of a sphere into the same number of real coordinates can distinguish every antipodal pair. In symbols, given a continuous map `g : S^n -> R^n`, must there exist `x in S^n` with `g(x) = g(-x)`?

The point is not a coordinate accident. Locally one can choose charts on the sphere and write down `n` coordinate functions, and locally the condition `g(x)=g(-x)` looks like `n` scalar equations. That framing is misleading because the unknowns live in antipodal pairs spread over the whole sphere. The natural negation is global: if every antipodal pair is separated, then

`h(x) = g(x) - g(-x)`

is a continuous nonzero vector in `R^n` satisfying `h(-x) = -h(x)`. Normalizing would give an antipode-preserving, or odd, map `S^n -> S^{n-1}`. The real question is therefore whether an odd continuous map can collapse the `n`-sphere onto a lower-dimensional sphere. The theorem says no.

## Background

Borsuk's original 1933 paper formulates the sphere as the boundary of a ball in `R^{n+1}`, writes the antipodal point of `p` as `p*`, and calls a map antipode-faithful when it sends antipodes to antipodes. The paper states three results: every antipode-faithful self-map of `S_n` is essential; if `f` maps `S_n` into `R^n`, then some `p` satisfies `f(p)=f(p*)`; and if `S_n` is covered by `n+1` compact sets none of which contains an antipodal pair, then the cover cannot be all of `S_n`. The footnote to the second statement says that this statement was posed as a conjecture by St. Ulam.

The invariant in the background is degree. For a continuous map `u : S^m -> S^m`, the induced homomorphism on top homology `H_m(S^m) ~= Z` is multiplication by an integer, `deg(u)`. The identity has degree `1`; a non-surjective map has degree `0`; homotopic maps have the same degree; the antipodal map has degree `(-1)^{m+1}`. Degree is the global count that a coordinate chart cannot see.

The special fact needed here is sharper than ordinary degree bookkeeping: an odd map `u : S^m -> S^m`, satisfying `u(-x)=-u(x)`, has odd degree. In modern homology this can be proved with `Z/2` coefficients using the double covering `S^m -> RP^m` and the transfer exact sequence. Borsuk's own proof uses the language of essential maps and explicitly notes that Hopf supplied shorter arguments based on deep degree theory, while Borsuk wanted a more elementary route.

## Baselines

The one-dimensional intuition is reliable only as a warning. For a continuous `g : S^1 -> R`, the difference `h(x)=g(x)-g(-x)` changes sign under the antipodal move, so it must vanish by connectedness or the intermediate value principle. This explains the answer for `n=1`, but it does not scale by choosing more coordinates: there is no total order on `R^n`, and a nonzero vector field can rotate rather than change sign.

Projection examples show why the conclusion is plausible but not obvious. Orthogonal projection `S^2 -> R^2` has exactly the expected antipodal coincidences at the poles, but a general continuous map can fold and tangle the sphere without respecting any visible coordinate axis. The proof cannot depend on a chosen projection direction.

Brouwer-style fixed-point reasoning supplies the right topological mood: negate the desired coincidence and manufacture an impossible global object. For fixed points the impossible object is a retraction of a ball onto its boundary. Here the impossible object is an odd collapse from `S^n` to `S^{n-1}`. Both arguments work by replacing a local search for a point with a global obstruction.

Degree on its own also has a tempting false start. The normalized difference `S^n -> S^{n-1}` has no ordinary degree because the domain and target dimensions differ. The move is to restrict it to an equatorial `S^{n-1}`. On that equator the map is an odd self-map, so its degree must be odd. But the same equatorial map extends across a hemisphere of `S^n`, so its degree must be zero. The obstruction appears only after choosing the invariant and the right equator.

## Evaluation settings

The result is a theorem, so success means a complete proof under the exact hypotheses: `S^n` is the standard sphere, `g : S^n -> R^n` is continuous, and no differentiability, linearity, or metric regularity is assumed. The proof must cover the low-dimensional cases and the general case without relying on local coordinates.

The standard equivalent form is: there is no continuous odd map `S^n -> S^{n-1}`. A proof of this equivalent form immediately gives the coincidence theorem by applying it to the normalized difference `g(x)-g(-x)`, if that difference were never zero.

The proof should also explain why the target dimension is exact. Mapping `S^n` to `R^{n+1}` can separate antipodes by the inclusion `x |-> x`, so the obstruction is not "spheres always identify antipodes"; it is the mismatch between antipodal symmetry and one fewer Euclidean coordinate.

## Proof framework

The final artifact is a theorem and proof, not code. The useful scaffold is:

1. Reduce coincidence failure to a nonzero odd map `h : S^n -> R^n`.
2. Normalize to an odd map `F : S^n -> S^{n-1}`.
3. Restrict `F` to an equator `S^{n-1} subset S^n`; the restriction is an odd self-map of `S^{n-1}` and therefore has odd degree.
4. Observe that the same restriction extends over a hemisphere `D^n`, so its degree is zero because the top homology of the disk vanishes.
5. The same map cannot have odd degree and degree zero. Therefore the normalized odd collapse cannot exist, so the original antipodal coincidence must exist.

This is the distinctive move: the proof does not solve `g(x)=g(-x)` in coordinates. It turns the absence of a solution into a symmetry-respecting global collapse and then measures that collapse with degree mod 2.
