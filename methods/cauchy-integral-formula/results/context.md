## Research question

For a holomorphic function on a plane domain, how much information is local? In real analysis, differentiability at each point does not determine a function from its boundary values: smooth bump functions can vanish on the boundary and still change the interior. Complex differentiability is much more restrictive. The question is whether holomorphicity, together with contour integration, gives a reconstruction principle: can the value of `f` at an interior point `a`, and its derivatives at `a`, be recovered from values of `f` on a surrounding contour?

The setting is a domain `Omega subset C`, a point `a in Omega`, and a positively oriented closed contour `gamma` whose inside stays in `Omega`. The question is how a contour integral relates a single interior value to the boundary values, how the choice of surrounding contour enters, and how complex differentiability once relates to differentiability of higher order.

## Background

Complex line integration treats a parametrized curve `gamma:[alpha,beta]->C` by integrating `f(gamma(t)) gamma'(t) dt`. This makes integrals sensitive to orientation and to singularities enclosed by the path, rather than only to endpoint values.

For holomorphic functions, the decisive preliminary principle is Cauchy's theorem: under the usual hypotheses that the region swept by a closed contour stays inside the holomorphic domain, the integral of a holomorphic function around that closed contour is zero. One consequence is contour deformation: if two closed contours can be deformed into each other without crossing a singularity, the integrals of a holomorphic integrand over them agree. Another is path independence in simply connected regions, equivalently the existence of primitives for holomorphic functions there.

The function `1/(z-a)` is the basic example of a nonzero contour integral. It is holomorphic away from `a`, but a positively oriented simple loop around `a` has integral `2*pi*i` for this kernel. Thus a contour can detect the presence of a point singularity inside it. Multiplying this kernel by a holomorphic function `f(z)` creates a singularity at `a`.

The contrast with real differentiability is structural. A real derivative constrains one direction at a time, and smooth functions can be locally adjusted without disturbing distant boundary data. Complex differentiability imposes compatibility in two real directions at once through the Cauchy-Riemann equations, so local first-order behavior is tied to global integral identities.

## Baselines

- **Real-variable local calculus.** Taylor's theorem recovers a smooth function near a point from its derivatives there when the function is sufficiently analytic.

- **Endpoint path independence for gradients.** In real vector calculus, exact differentials integrate to endpoint differences, and closed integrals of exact differentials vanish.

- **Cauchy's theorem.** Closed contour integrals of holomorphic functions vanish on suitable regions.

- **Shrinking of contours.** If a contour is a small circle around `a`, continuity makes `f(z)` close to `f(a)` on that circle.

- **Pole detection.** The kernel `1/(z-a)` has loop integral `2*pi*i` around `a`, so a contour integral registers the location of the pole.

## Evaluation settings

The artifact is a theorem and proof. The main case is a function holomorphic on an open set containing a closed disk or the interior of a positively oriented simple closed contour. The point `a` lies inside the contour and not on it. The contour is piecewise continuously differentiable.

Stress cases include constant functions, polynomials, functions with removable behavior at the selected point after subtracting `f(a)`, annular regions where contour deformation must avoid the singularity at `a`, and real smooth functions as a contrast class where the same reconstruction principle fails.

Success means proving the interior value formula, showing why the result is unchanged when the contour is deformed without crossing `a`, deriving the higher derivative formulas, and relating holomorphicity once to differentiability of all orders.

## Code framework

No executable implementation is needed. The field-appropriate artifact is a theorem with a proof, using the holomorphic-integrand tools above: closed-contour cancellation from Cauchy's theorem, contour deformation between homologous loops, the loop integral of `1/(z-a)`, and differentiation of an integral representation with respect to the interior point.
