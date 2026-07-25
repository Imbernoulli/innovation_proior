The research question asks how much local information is really carried by a holomorphic function on a plane domain. In real analysis, smoothness at each point is weak: a bump function can vanish on a boundary yet be nonzero inside, so boundary values do not determine interior values. Complex differentiability is far more rigid because the derivative must be the same in every complex direction, tying together the two real partial derivatives through the Cauchy-Riemann equations. The challenge is to turn that rigidity into a concrete reconstruction principle: given a closed contour enclosing a point, can the boundary values of a holomorphic function recover the value and all derivatives at that interior point?

Existing ideas fall short in different ways. Real-variable calculus only reconstructs a function near a point if we already know its derivatives and assume analyticity. Cauchy's theorem alone says closed contour integrals of holomorphic functions vanish, which seems to erase information rather than recover it. The pole of 1/(z-a) detects the presence of a point singularity, but by itself it does not explain how the value f(a) is isolated. Direct shrinking of a small circle around a only gives a local computation unless we also prove independence of the enclosing contour. What is missing is a controlled singularity that lets the contour isolate exactly the coefficient f(a) while Cauchy's theorem removes everything else.

The method that resolves this is the Cauchy integral formula. It states that if f is holomorphic on a domain containing a positively oriented piecewise smooth simple closed contour gamma and its interior, then for every point a inside gamma,

f(a) = (1 / (2πi)) ∫_gamma f(z)/(z-a) dz.

Moreover, f has derivatives of every order at a, and for every integer n ≥ 0,

f^(n)(a) = (n! / (2πi)) ∫_gamma f(z)/(z-a)^(n+1) dz.

The idea is to split the integrand so that the singular part carries the value f(a) and the remaining part becomes holomorphic. Write f(z)/(z-a) as f(a)/(z-a) plus (f(z)-f(a))/(z-a). The quotient (f(z)-f(a))/(z-a) extends to a holomorphic function at a because complex differentiability gives it the limit f'(a) there. Cauchy's theorem then forces the integral of that remainder over any closed contour to be zero. The only surviving contribution comes from f(a)/(z-a), whose contour integral is 2πi f(a) because a positively oriented loop around a winds once around the pole. This proves that the boundary values of f determine f(a).

Contour independence follows immediately. If two contours both enclose a once and can be deformed into each other without crossing a, then f(z)/(z-a) is holomorphic on the region between them. Cauchy's theorem says the integrals over the two contours agree, so a large boundary contour may be shrunk to a tiny circle around a without changing the answer. For derivatives, the value formula writes f(a) as an integral whose dependence on a is explicit. Differentiating under the integral sign is justified because z stays on gamma while a stays inside at positive distance from gamma, so the denominator never vanishes on the contour. The first derivative comes from differentiating (z-a)^(-1) to obtain (z-a)^(-2), and repeating the argument by induction gives the formula for every higher derivative. Thus complex differentiability once at a implies differentiability of all orders.

The final artifact is this reconstruction principle, stated with its hypotheses made explicit. Let $\Omega \subset \mathbb{C}$ be a domain, let $\gamma$ be a positively oriented, piecewise $C^1$ simple closed contour whose interior and image both lie in $\Omega$, and let $f$ be holomorphic on $\Omega$. Then for every point $a$ inside $\gamma$,

$$f(a) = \frac{1}{2\pi i}\int_\gamma \frac{f(z)}{z-a}\,dz,$$

and $f$ has derivatives of every order at $a$, given for every integer $n \ge 0$ by

$$f^{(n)}(a) = \frac{n!}{2\pi i}\int_\gamma \frac{f(z)}{(z-a)^{n+1}}\,dz.$$

Equivalently: the boundary values of a holomorphic function on any contour enclosing $a$ determine the value and every derivative at $a$, through the same singular kernel $1/(z-a)$ raised to successive powers. This is the deliverable in full, not merely an existence claim — the constant $1/(2\pi i)$, the factorial $n!$, and the contour $\gamma$ are exactly the objects the removable-singularity argument produces, and none of them is a free parameter left to fit. A single complex derivative at $a$ is therefore not a local fact at all; it is the first instance of a family of boundary moments of $f$ against $1/(z-a)^{n+1}$, and the whole family is fixed the moment $f$ is known to be holomorphic on a neighborhood of $\overline{\mathrm{int}(\gamma)}$.
