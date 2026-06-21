The research question asks how much local information is really carried by a holomorphic function on a plane domain. In real analysis, smoothness at each point is weak: a bump function can vanish on a boundary yet be nonzero inside, so boundary values do not determine interior values. Complex differentiability is far more rigid because the derivative must be the same in every complex direction, tying together the two real partial derivatives through the Cauchy-Riemann equations. The challenge is to turn that rigidity into a concrete reconstruction principle: given a closed contour enclosing a point, can the boundary values of a holomorphic function recover the value and all derivatives at that interior point?

Existing ideas fall short in different ways. Real-variable calculus only reconstructs a function near a point if we already know its derivatives and assume analyticity. Cauchy's theorem alone says closed contour integrals of holomorphic functions vanish, which seems to erase information rather than recover it. The pole of 1/(z-a) detects the presence of a point singularity, but by itself it does not explain how the value f(a) is isolated. Direct shrinking of a small circle around a only gives a local computation unless we also prove independence of the enclosing contour. What is missing is a controlled singularity that lets the contour isolate exactly the coefficient f(a) while Cauchy's theorem removes everything else.

The method that resolves this is the Cauchy integral formula. It states that if f is holomorphic on a domain containing a positively oriented piecewise smooth simple closed contour gamma and its interior, then for every point a inside gamma,

f(a) = (1 / (2πi)) ∫_gamma f(z)/(z-a) dz.

Moreover, f has derivatives of every order at a, and for every integer n ≥ 0,

f^(n)(a) = (n! / (2πi)) ∫_gamma f(z)/(z-a)^(n+1) dz.

The idea is to split the integrand so that the singular part carries the value f(a) and the remaining part becomes holomorphic. Write f(z)/(z-a) as f(a)/(z-a) plus (f(z)-f(a))/(z-a). The quotient (f(z)-f(a))/(z-a) extends to a holomorphic function at a because complex differentiability gives it the limit f'(a) there. Cauchy's theorem then forces the integral of that remainder over any closed contour to be zero. The only surviving contribution comes from f(a)/(z-a), whose contour integral is 2πi f(a) because a positively oriented loop around a winds once around the pole. This proves that the boundary values of f determine f(a).

Contour independence follows immediately. If two contours both enclose a once and can be deformed into each other without crossing a, then f(z)/(z-a) is holomorphic on the region between them. Cauchy's theorem says the integrals over the two contours agree, so a large boundary contour may be shrunk to a tiny circle around a without changing the answer. For derivatives, the value formula writes f(a) as an integral whose dependence on a is explicit. Differentiating under the integral sign is justified because z stays on gamma while a stays inside at positive distance from gamma, so the denominator never vanishes on the contour. The first derivative comes from differentiating (z-a)^(-1) to obtain (z-a)^(-2), and repeating the argument by induction gives the formula for every higher derivative. Thus complex differentiability once at a implies differentiability of all orders.

The numerical verification below implements the Cauchy integral formula for a holomorphic function on a circular contour. It parametrizes the contour, computes the contour integral with the trapezoidal rule, and compares the recovered values and derivatives against the exact analytic expressions.

```python
import math
import numpy as np

def cauchy_integral(f, center, radius, n=10_000):
    """Evaluate f at the center of a positively oriented circle via Cauchy."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    dz = 1j * radius * np.exp(1j * t) * (2 * np.pi / n)
    z = center + radius * np.exp(1j * t)
    return np.sum(f(z) / (z - center) * dz) / (2j * np.pi)

def cauchy_derivative(f, n, center, radius, n_pts=10_000):
    """Recover the n-th derivative f^(n)(center) from a circular contour."""
    t = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    dz = 1j * radius * np.exp(1j * t) * (2 * np.pi / n_pts)
    z = center + radius * np.exp(1j * t)
    return math.factorial(n) * np.sum(f(z) / (z - center)**(n + 1) * dz) / (2j * np.pi)

# Example: f(z) = exp(z) at a = 0.5 + 0.3i using a surrounding circle.
a = 0.5 + 0.3j
R = 1.0
f = lambda z: np.exp(z)

recovered_value = cauchy_integral(f, a, R)
exact_value = np.exp(a)
print(f"f(a) recovered: {recovered_value:.12f}")
print(f"f(a) exact:     {exact_value:.12f}")

for n in range(1, 4):
    rec = cauchy_derivative(f, n, a, R)
    ext = np.exp(a)
    print(f"f^({n})(a) recovered: {rec:.12f}, exact: {ext:.12f}")

# Example: f(z) = z^3 + 2z + 1 at a = 1.0.
a2 = 1.0 + 0.0j
R2 = 0.5
f2 = lambda z: z**3 + 2*z + 1
rec2 = cauchy_integral(f2, a2, R2)
print(f"\nPolynomial f(a) recovered: {rec2:.12f}, exact: {a2**3 + 2*a2 + 1:.12f}")
```
