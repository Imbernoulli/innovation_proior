## Research question

A smooth manifold carries local coordinates, tangent spaces, and smooth functions, but its global topology is harder to see directly. A useful probe would turn the manifold into a filtered object: for a smooth real-valued function `f:M->R`, define

`M^a = {x in M : f(x) <= a}`.

The question is whether the topology of `M` can be read by increasing `a` and watching the sublevel sets. A satisfactory answer has to distinguish ordinary levels, where the function has no stationary behavior, from exceptional levels where the differential vanishes. It also has to say what local differential data at an exceptional point determines the global topological change.

The target setting is a smooth `n`-manifold and a smooth function whose critical points are isolated and locally stable under perturbation. Compactness enters through bands `f^{-1}([a,b])`, so that gradient-flow arguments do not run off to infinity while comparing nearby sublevel sets.

## Background

At a point `p`, criticality means `df_p=0`. At such a point the Hessian is a well-defined quadratic form on `T_pM`, independent of the local coordinate chart up to the usual change of basis. If this quadratic form has no kernel, the critical point is nondegenerate. Its index is the number of negative directions of the Hessian, counted with multiplicity.

The ordinary second-derivative test already suggests why the Hessian matters: near a nondegenerate critical point, negative directions are locally descending directions and positive directions are locally ascending directions. The Morse lemma sharpens this from a second-order approximation into an exact local normal form. In suitable coordinates centered at a nondegenerate critical point `p`,

`f = f(p) - x_1^2 - ... - x_lambda^2 + x_{lambda+1}^2 + ... + x_n^2`,

where `lambda` is the Hessian index.

Away from critical points, the gradient does not vanish. After choosing a Riemannian metric, a vector field proportional to `grad f` can be arranged to satisfy `df(X)=1` on a compact regular band. Its flow moves points from one regular level to another. This turns a regular band into a product-like region and makes it plausible that sublevel topology cannot change inside such a band.

Genericity is important because degenerate critical points do not have a stable local model. A one-variable example such as `f(x)=x^3` has a critical point with zero Hessian, but crossing the critical value does not produce the kind of controlled quadratic event seen in the nondegenerate case. A small perturbation can remove the critical point or split it into nondegenerate ones. The useful theory therefore asks for smooth functions with only nondegenerate critical points; on compact finite-dimensional manifolds such functions form a dense open class in the appropriate `C^k` topology.

## Baselines

- **Triangulate or build a CW complex directly.** A topological decomposition can describe the homotopy type of a manifold, but it does not explain how the smooth structure itself produces the cells or why their dimensions should be controlled by differential data.

- **Use ordinary extrema only.** Minima and maxima explain where sublevel sets first appear and finally close off, but manifolds such as a torus require intermediate events. A method that sees only extrema misses saddles, where connected components merge or tunnels open.

- **Track all level sets equally.** Level sets give a contour picture, but most levels are redundant. If the gradient is nonzero throughout a band, flowing along the gradient identifies the levels inside that band. A contour-by-contour description spends effort where no topological event is available.

- **Use the Hessian approximation without a normal form.** Taylor expansion suggests the signs of the quadratic part, but a proof needs more than a local approximation with higher-order terms. The higher-order terms must be removed by a smooth coordinate change, or the local topological model remains unclear.

- **Allow degenerate critical points.** Degenerate points can have flat directions and unstable behavior. Their local topology is not determined by a single integer index, so they do not provide a clean event type for a filtration of sublevel sets.

## Evaluation settings

The proof artifact should work for a smooth `n`-manifold `M`, a smooth function `f:M->R`, and compact bands `f^{-1}([a,b])`. The main comparison is between `M^a` and `M^b` when the band has no critical points, and then when the band has exactly one nondegenerate critical point.

Natural examples include the height function on a sphere, where a minimum starts the sublevel set and a maximum caps it; the height function on a torus, where saddles account for one-dimensional attachments; a compact regular band with no critical points; a degenerate one-variable critical point such as `x^3`; and a function with two critical points at the same level, which motivates separating critical values by a small perturbation or by treating one critical level at a time.

Success means a theorem that identifies ordinary levels as topologically inert and nondegenerate critical levels as controlled handle attachments. The local model should use the Hessian index, and the global conclusion should imply a handle decomposition, hence a CW model, with one cell of dimension `lambda` for each index-`lambda` critical point when critical values are separated.

## Proof artifact

The final artifact should state and prove two linked results.

If `f^{-1}([a,b])` is compact and contains no critical points, then `M^a` and `M^b` are diffeomorphic, and `M^a` is a deformation retract of `M^b`.

If `f^{-1}([c-epsilon,c+epsilon])` is compact and contains exactly one critical point `p`, with `f(p)=c` and Hessian index `lambda`, then crossing from `c-epsilon` to `c+epsilon` attaches a `lambda`-handle `D^lambda x D^{n-lambda}` along `S^{lambda-1} x D^{n-lambda}`. In particular, the upper sublevel set has the homotopy type of the lower sublevel set with one `lambda`-cell attached.

The proof should use normalized gradient flow for the regular-band statement and the Morse lemma normal form for the critical-level statement. The reason topology changes only at critical levels should be explicit: regular levels are traversed by a nonsingular flow, while a nondegenerate critical point is locally exactly a quadratic saddle whose negative eigenspace determines the new handle dimension.
