## Research question

For a holomorphic function on a plane domain, how much information is really local? In real analysis, differentiability at each point does not determine a function from its boundary values: smooth bump functions can vanish on the boundary and still change the interior. Complex differentiability is much more restrictive. The question is whether holomorphicity, together with contour integration, forces a reconstruction principle: can the value of `f` at an interior point `a`, and then all derivatives at `a`, be recovered from values of `f` on a surrounding contour?

The target setting is a domain `Omega subset C`, a point `a in Omega`, and a positively oriented closed contour `gamma` whose inside stays in `Omega`. A satisfactory result should not merely say that closed contour integrals of holomorphic functions vanish. It should explain how a contour integral can isolate a single interior value, why the choice of surrounding contour does not matter, and why differentiability once in the complex sense produces derivatives of every order.

## Background

Complex line integration treats a parametrized curve `gamma:[alpha,beta]->C` by integrating `f(gamma(t)) gamma'(t) dt`. This makes integrals sensitive to orientation and to singularities enclosed by the path, rather than only to endpoint values.

For holomorphic functions, the decisive preliminary principle is Cauchy's theorem: under the usual hypotheses that the region swept by a closed contour stays inside the holomorphic domain, the integral of a holomorphic function around that closed contour is zero. One consequence is contour deformation: if two closed contours can be deformed into each other without crossing a singularity, the integrals of a holomorphic integrand over them agree. Another is path independence in simply connected regions, equivalently the existence of primitives for holomorphic functions there.

The function `1/(z-a)` is the basic obstruction to zero contour integral. It is holomorphic away from `a`, but a positively oriented simple loop around `a` has integral `2*pi*i` for this kernel. Thus a contour can detect the presence of a point singularity inside it. Multiplying this kernel by a holomorphic function `f(z)` creates a controlled singularity at `a`: the non-holomorphic part should carry exactly the value `f(a)`, while the remaining part should be holomorphic and invisible to closed-contour integration.

The contrast with real differentiability is structural. A real derivative constrains one direction at a time, and smooth functions can be locally adjusted without disturbing distant boundary data. Complex differentiability imposes compatibility in two real directions at once through the Cauchy-Riemann equations, so local first-order behavior is tied to global integral identities. This overdetermination is the source of rigidity.

## Baselines

- **Real-variable local calculus.** Taylor's theorem recovers a smooth function near a point only when the needed derivatives are known and the function is sufficiently analytic. Gap: ordinary real differentiability alone leaves enormous freedom; boundary values do not determine interior values.

- **Endpoint path independence for gradients.** In real vector calculus, exact differentials integrate to endpoint differences. Gap: closed integrals being zero gives cancellation, but by itself it does not explain how to extract a value at a chosen interior point.

- **Cauchy's theorem alone.** Closed contour integrals of holomorphic functions vanish on suitable regions. Gap: the theorem seems to erase information rather than recover it, until a controlled singularity is introduced.

- **Direct shrinking of contours.** If a contour is a small circle around `a`, continuity suggests that `f(z)` is close to `f(a)` on that circle. Gap: this gives only a local circle computation unless one also proves that the integral is independent of the enclosing contour.

- **Residue-style pole detection.** The kernel `1/(z-a)` has a nonzero loop integral around `a`. Gap: the pole detects the location, but the reconstruction of a holomorphic function requires separating the singular constant part from the holomorphic remainder.

## Evaluation settings

The artifact is a theorem and proof. The main case is a function holomorphic on an open set containing a closed disk or the interior of a positively oriented simple closed contour. The point `a` lies inside the contour and not on it. The contour is piecewise continuously differentiable.

Stress cases include constant functions, polynomials, functions with removable behavior at the selected point after subtracting `f(a)`, annular regions where contour deformation must avoid the singularity at `a`, and real smooth functions as a contrast class where the same reconstruction principle fails.

Success means proving the interior value formula, showing why the result is unchanged when the contour is deformed without crossing `a`, deriving the higher derivative formulas, and making clear that holomorphicity once implies holomorphicity and differentiability of all orders.

## Code framework

No executable implementation is needed. The field-appropriate scaffold is a proof sequence:

1. Establish the closed-contour cancellation available for holomorphic integrands.
2. Isolate the only non-holomorphic part of `f(z)/(z-a)` by writing `f(z)=f(a)+(f(z)-f(a))`.
3. Show that `(f(z)-f(a))/(z-a)` extends holomorphically at `a`.
4. Compute the contour integral of `1/(z-a)` around a positively oriented loop enclosing `a`.
5. Use deformation or subtraction of contours to see that the chosen contour can be replaced by any homologous surrounding contour.
6. Differentiate the resulting integral representation with respect to `a` to obtain all derivative formulas.
