## Research question

The problem is to decide whether a continuous measurement of a sphere into the same number of real coordinates can distinguish every antipodal pair. In symbols, given a continuous map `g : S^n -> R^n`, must there exist `x in S^n` with `g(x) = g(-x)`?

Locally one can choose charts on the sphere and write down `n` coordinate functions, and locally the condition `g(x)=g(-x)` looks like `n` scalar equations. But the unknowns live in antipodal pairs spread over the whole sphere. The natural negation is global: if every antipodal pair is separated, then

`h(x) = g(x) - g(-x)`

is a continuous nonzero vector in `R^n` satisfying `h(-x) = -h(x)`. Normalizing gives an antipode-preserving, or odd, map `S^n -> S^{n-1}`. So the question can be recast as whether an odd continuous map can collapse the `n`-sphere onto a lower-dimensional sphere.

## Background

Borsuk's original 1933 paper formulates the sphere as the boundary of a ball in `R^{n+1}`, writes the antipodal point of `p` as `p*`, and calls a map antipode-faithful when it sends antipodes to antipodes. The paper states three results: every antipode-faithful self-map of `S_n` is essential; if `f` maps `S_n` into `R^n`, then some `p` satisfies `f(p)=f(p*)`; and if `S_n` is covered by `n+1` compact sets none of which contains an antipodal pair, then the cover cannot be all of `S_n`. The footnote to the second statement says that this statement was posed as a conjecture by St. Ulam.

The invariant in the background is degree. For a continuous map `u : S^m -> S^m`, the induced homomorphism on top homology `H_m(S^m) ~= Z` is multiplication by an integer, `deg(u)`. The identity has degree `1`; a non-surjective map has degree `0`; homotopic maps have the same degree; the antipodal map has degree `(-1)^{m+1}`. Degree is the global count that a coordinate chart cannot see.

A sharper fact than ordinary degree bookkeeping is available: an odd map `u : S^m -> S^m`, satisfying `u(-x)=-u(x)`, has odd degree. In modern homology this can be proved with `Z/2` coefficients using the double covering `S^m -> RP^m` and the transfer exact sequence. Borsuk's own proof uses the language of essential maps; he notes that Hopf supplied shorter arguments based on degree theory, while Borsuk wanted a more elementary route.

## Baselines

For a continuous `g : S^1 -> R`, the difference `h(x)=g(x)-g(-x)` changes sign under the antipodal move, so it vanishes by connectedness or the intermediate value principle. This settles the answer for `n=1`. There is no total order on `R^n` for `n >= 2`, and a nonzero vector field can rotate rather than change sign.

Projection examples are instructive. Orthogonal projection `S^2 -> R^2` has antipodal coincidences at the poles, but a general continuous map can fold and tangle the sphere without respecting any visible coordinate axis.

Brouwer-style fixed-point reasoning sets the topological mood: negate the desired coincidence and manufacture an impossible global object. For fixed points the impossible object is a retraction of a ball onto its boundary. Here the negation manufactures an odd collapse from `S^n` to `S^{n-1}`. Both arguments replace a local search for a point with a global obstruction.

The normalized difference `S^n -> S^{n-1}` has no ordinary degree because the domain and target dimensions differ.

## Evaluation settings

The result is a theorem, so success means a complete proof under the exact hypotheses: `S^n` is the standard sphere, `g : S^n -> R^n` is continuous, and no differentiability, linearity, or metric regularity is assumed. The proof must cover the low-dimensional cases and the general case without relying on local coordinates.

The standard equivalent form is: there is no continuous odd map `S^n -> S^{n-1}`. A proof of this equivalent form gives the coincidence theorem by applying it to the normalized difference `g(x)-g(-x)`, supposing that difference were never zero.

The proof should also explain why the target dimension is exact. Mapping `S^n` to `R^{n+1}` can separate antipodes by the inclusion `x |-> x`; the relevant feature is the mismatch between antipodal symmetry and one fewer Euclidean coordinate.

## Proof framework

The final artifact is a theorem and proof, not code. The deliverable is a complete proof of the coincidence statement for continuous `g : S^n -> R^n`, equivalently the nonexistence of a continuous odd map `S^n -> S^{n-1}`, established from the standard homological invariants above and covering both the low-dimensional and general cases.
