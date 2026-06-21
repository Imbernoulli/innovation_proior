I start from the tension that makes complex analysis feel unlike real analysis. In one real variable, knowing that a function is differentiable is weak. I can make a smooth bump inside an interval that vanishes at the endpoints, and the boundary sees nothing. Even knowing many local slopes does not force the interior from the boundary unless I assume real analyticity and carry an entire power series. So if holomorphic functions behave differently, the reason cannot be "differentiable functions are nice." It has to be that complex differentiability is a much narrower condition: the derivative has to be the same no matter which complex direction I approach from, so two real directions are tied together at every point.

The contour integral is the place where that rigidity becomes global. If `f` is holomorphic on a region with no holes relevant to the curve, Cauchy's theorem says `int_gamma f(z) dz = 0` for a closed contour. At first that sounds like loss of information. A closed loop around a holomorphic function gives me zero, so how can a loop reconstruct anything? If I only integrate `f`, it cannot. I need an integrand that is holomorphic everywhere except at the point whose value I want, because Cauchy's theorem then says every part of the contour integral is blind except the part forced by that one obstruction.

The obstruction I understand completely is `1/(z-a)`. On a small positively oriented circle `z=a+re^{it}`, `dz=i r e^{it} dt`, and
`dz/(z-a)=i dt`, so `int dz/(z-a)=2*pi*i`. This is not an accident of the circle. If I move the contour without crossing `a`, the difference between the old and new contours bounds a region where `1/(z-a)` is holomorphic, so Cauchy's theorem makes the two integrals equal. The contour is measuring winding around the point. A closed curve that winds once around `a` gives `2*pi*i`; one that misses `a` gives zero.

Now I try to put `f` into that detector. The first naive object is `f(z)/(z-a)`. It has the same singularity as `1/(z-a)`, but its numerator varies along the contour. I want only `f(a)` to survive. Split the numerator at the point:
`f(z)/(z-a) = f(a)/(z-a) + (f(z)-f(a))/(z-a)`.
The first term is perfect: its integral is `2*pi*i f(a)` on a once-winding contour. The second term is the test of holomorphic rigidity. In real variables this quotient would merely be a difference quotient with a possible directional problem. Here, because `f` is complex differentiable at `a`, `(f(z)-f(a))/(z-a)` has a limit as `z -> a`, namely `f'(a)`. If I define its value at `a` to be `f'(a)`, it is continuous at the point and holomorphic away from it. To integrate it, I remove a tiny circle around `a`; Cauchy's theorem erases the annular part, and the tiny circle contributes at most its length times a bounded value, so that contribution goes to zero as the radius goes to zero. Thus the whole second term has closed-contour integral zero:
`int_gamma (f(z)-f(a))/(z-a) dz = 0`.

That leaves exactly
`int_gamma f(z)/(z-a) dz = f(a) int_gamma dz/(z-a) = 2*pi*i f(a)`.
So the interior value is not an independent datum. It is the coefficient of the singular kernel detected by the boundary contour:
`f(a) = (1/(2*pi*i)) int_gamma f(z)/(z-a) dz`.

This is already stronger than a clever integral identity. It says the contour can be far from `a`. If I know `f` only along a surrounding boundary and know it is holomorphic in the enclosed region, I know `f(a)`. The local differentiability condition has become a global reconstruction rule. The reason the boundary can move is also clear now: for two contours enclosing `a` once, the integrand `f(z)/(z-a)` is holomorphic in the annular region between them, because the only singularity is at `a` and neither contour crosses it. The integral over one contour minus the integral over the other is a closed integral over a holomorphic region, hence zero. I can shrink a large contour to a tiny circle around `a`, read off the value locally, and then expand it back out to the boundary without changing the integral.

The shrinking picture is worth checking because it tells me why the constant is right without hiding behind notation. On `|z-a|=r`,
`int f(z)/(z-a) dz = int [f(a) + (f(z)-f(a))]/(z-a) dz`.
The `f(a)` part is `2*pi*i f(a)`. The error has size at most `(max_{|z-a|=r} |f(z)-f(a)|/r) * 2*pi*r`, which is `2*pi max |f(z)-f(a)|`, and that goes to zero by continuity as `r -> 0`. The singular kernel turns a shrinking circumference into a finite mass. Without the pole, the integral of a bounded function over a shrinking circle would go to zero. The pole is exactly calibrated to leave one value behind.

Once the value formula is true, I look at derivatives. I could try to repeat the removable-singularity argument for derivatives, but there is a cleaner route: the formula already writes `f(a)` as an integral whose dependence on `a` is explicit. The contour variable `z` stays on `gamma`, while `a` moves inside at positive distance from `gamma`. For nearby `a`, the denominator never vanishes on the contour, so differentiating under the integral should be legitimate. The first derivative ought to be
`f'(a) = (1/(2*pi*i)) int_gamma f(z)/(z-a)^2 dz`,
because differentiating `(z-a)^{-1}` with respect to `a` gives `(z-a)^{-2}`.

Let me verify without assuming what I am trying to prove. For `b` near `a`, both points lie inside the same contour, so
`f(b)-f(a) = (1/(2*pi*i)) int_gamma f(z)(1/(z-b)-1/(z-a)) dz`.
The bracket equals `(b-a)/((z-b)(z-a))`. After division by `b-a`,
`(f(b)-f(a))/(b-a) = (1/(2*pi*i)) int_gamma f(z)/((z-b)(z-a)) dz`.
As `b -> a`, the integrand converges uniformly on the contour to `f(z)/(z-a)^2`, because the contour has a positive minimum distance from `a`. The limit passes through the integral. The derivative exists and has the displayed integral formula. So one complex derivative plus Cauchy's value formula already forces another derivative.

The same argument repeats. If
`f^{(n)}(a) = (n!/(2*pi*i)) int_gamma f(z)/(z-a)^{n+1} dz`,
then differentiating with respect to `a` multiplies the kernel by `n+1`, since `d/da (z-a)^{-(n+1)} = (n+1)(z-a)^{-(n+2)}`. Therefore
`f^{(n+1)}(a) = ((n+1)!/(2*pi*i)) int_gamma f(z)/(z-a)^{n+2} dz`.
The induction begins at `n=0`, the value formula. Holomorphic once means holomorphic infinitely many times. That is the rigidity in its sharpest elementary form: the function is not merely differentiable; all derivatives are boundary moments against singular kernels.

This also explains why path independence is not a side lemma. Suppose I know `f` on a large boundary and I want the value at `a`. I can compute the integral on that boundary, but I can also contract the contour until it nearly sits on `a`. Every deformation step is allowed exactly because the difference contour sees a holomorphic integrand away from the isolated pole. The nonzero answer is not contradicting Cauchy's theorem; it is using Cauchy's theorem to prove that the only thing the contour can remember is the pole it cannot cross. The singular kernel carries the address of the point, and holomorphicity prevents any other interior freedom from changing the answer.

The real-variable contrast now becomes precise. A real smooth function can have compact support inside a disk while vanishing on the boundary, so no boundary integral depending only on boundary values can recover every interior value in that class. In the holomorphic class, such a bump cannot exist unless it is identically zero, because the value formula would give zero at every interior point from the zero boundary data. The boundary determines the inside not by a numerical accident but because local complex differentiability gives closed-contour cancellation, closed-contour cancellation gives contour deformation, and contour deformation lets the singular kernel shrink to the point and read off the coefficient `f(a)`. The final artifact is therefore the integral reconstruction formula and its derivative family: for every point inside the contour, the boundary values determine `f(a)` and all `f^{(n)}(a)` through `1/(z-a)^{n+1}`.
