## Research question

A smooth manifold carries local coordinates, tangent spaces, and smooth functions, but its global topology is harder to see directly. A useful probe would turn the manifold into a filtered object: for a smooth real-valued function `f:M->R`, define

`M^a = {x in M : f(x) <= a}`.

The question is whether the topology of `M` can be read by increasing `a` and watching the sublevel sets, and what role the critical points of `f` play in any topological changes that occur.

The target setting is a smooth `n`-manifold and a smooth function whose critical points are isolated and locally stable under perturbation. Compactness enters through bands `f^{-1}([a,b])`, so that gradient-flow arguments do not run off to infinity while comparing nearby sublevel sets.

## Background

At a point `p`, criticality means `df_p=0`. At such a point the Hessian is a well-defined quadratic form on `T_pM`, independent of the local coordinate chart up to the usual change of basis. If this quadratic form has no kernel, the critical point is nondegenerate. Its index is the number of negative directions of the Hessian, counted with multiplicity.

The ordinary second-derivative test already suggests why the Hessian matters: near a nondegenerate critical point, negative directions are locally descending directions and positive directions are locally ascending directions. The Morse lemma sharpens this from a second-order approximation into an exact local normal form. In suitable coordinates centered at a nondegenerate critical point `p`,

`f = f(p) - x_1^2 - ... - x_lambda^2 + x_{lambda+1}^2 + ... + x_n^2`,

where `lambda` is the Hessian index.

Away from critical points, the gradient does not vanish. After choosing a Riemannian metric, a vector field proportional to `grad f` can be arranged to satisfy `df(X)=1` on a compact regular band. Its flow moves points from one regular level to another.

Genericity is important because degenerate critical points do not have a stable local model. A one-variable example such as `f(x)=x^3` has a critical point with zero Hessian; a small perturbation can remove the critical point or split it into nondegenerate ones. The useful theory therefore asks for smooth functions with only nondegenerate critical points; on compact finite-dimensional manifolds such functions form a dense open class in the appropriate `C^k` topology.

## Baselines

- **Triangulate or build a CW complex directly.** A topological decomposition can describe the homotopy type of a manifold, but it does not explain how the smooth structure itself produces the cells or why their dimensions should be controlled by differential data.

- **Use ordinary extrema only.** Minima and maxima explain where sublevel sets first appear and finally close off. A torus requires intermediate critical events such as saddles, where connected components merge or tunnels open.

- **Track all level sets equally.** Level sets give a contour picture. When the gradient is nonzero throughout a band, flowing along the gradient identifies the levels inside that band.

- **Use the Hessian approximation without a normal form.** Taylor expansion gives the signs of the quadratic part. A proof needs a smooth coordinate change that removes higher-order terms and produces an exact local model.

- **Allow degenerate critical points.** Degenerate points can have flat directions and unstable behavior, with local topology not determined by a single integer index.

## Evaluation settings

The proof artifact should work for a smooth `n`-manifold `M`, a smooth function `f:M->R`, and compact bands `f^{-1}([a,b])`. The main comparison is between `M^a` and `M^b` when the band has no critical points, and then when the band has exactly one nondegenerate critical point.

Natural examples include the height function on a sphere, where a minimum starts the sublevel set and a maximum caps it; the height function on a torus, where saddles account for one-dimensional changes; a compact regular band with no critical points; a degenerate one-variable critical point such as `x^3`; and a function with two critical points at the same level, which motivates separating critical values by a small perturbation or by treating one critical level at a time.
