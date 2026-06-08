# The Riemann zeta function, its functional equation, and the explicit formula for primes

## The problem

Count the primes: find an exact analytic expression for $\pi(x)$, the number of primes $\le x$.
Empirically $\pi(x)\approx \mathrm{Li}(x)=\int_0^x dt/\log t$ (Gauss, Legendre), with a small,
unexplained oscillating discrepancy. The goal is a law that captures both the smooth trend and the
fluctuations.

## The key idea

Move the problem into complex analysis through a single function of a complex variable $s$ that
encodes every prime, then continue it to the whole plane, exploit its symmetry, and read the prime
count off its **zeros**.

## The construction

**1. The zeta function (Euler product).** For $\mathrm{Re}(s)>1$,
$$\zeta(s)=\sum_{n=1}^\infty \frac{1}{n^s}=\prod_{p\ \text{prime}}\frac{1}{1-p^{-s}}.$$

**2. Analytic continuation to all of $\mathbb{C}$.** Via the Gamma integral,
$\Pi(s-1)\zeta(s)=\int_0^\infty \frac{x^{s-1}}{e^x-1}\,dx$. Converting to the keyhole contour integral
$$2\sin(\pi s)\,\Pi(s-1)\,\zeta(s)=i\int \frac{(-x)^{s-1}}{e^x-1}\,dx$$
(taken on the Hankel contour from $+\infty$ around the origin and back to $+\infty$) defines
$\zeta(s)$ for **every** $s\ne 1$, single-valued, with a simple pole only at $s=1$ and **trivial
zeros** at $s=-2,-4,-6,\dots$ (here $\Pi(s-1)=\Gamma(s)$).

**3. Functional equation.** Equivalently, using $\psi(x)=\sum_{n\ge1}e^{-n^2\pi x}$ and Jacobi's
theta self-duality $2\psi(x)+1=x^{-1/2}(2\psi(1/x)+1)$,
$$\Pi\!\Bigl(\tfrac{s}{2}-1\Bigr)\pi^{-s/2}\zeta(s)=\frac{1}{s(s-1)}+\int_1^\infty\psi(x)\bigl(x^{s/2-1}+x^{(1-s)/2-1}\bigr)\,dx,$$
whose right-hand side is invariant under $s\mapsto 1-s$. Hence the **symmetric functional equation**
$$\Pi\!\Bigl(\tfrac{s}{2}-1\Bigr)\pi^{-s/2}\zeta(s)=\Pi\!\Bigl(\tfrac{1-s}{2}-1\Bigr)\pi^{-(1-s)/2}\zeta(1-s).$$

**4. The entire function $\xi$.** Multiplying by the pole-killer $\tfrac12 s(s-1)$,
$$\boxed{\;\xi(s)=\tfrac12\,s(s-1)\,\pi^{-s/2}\,\Pi\!\Bigl(\tfrac{s}{2}-1\Bigr)\,\zeta(s),\qquad \xi(s)=\xi(1-s)\;}$$
is **entire**, real on the real axis, and its zeros are exactly the **nontrivial** zeros of $\zeta$,
all lying in the critical strip $0<\mathrm{Re}(s)<1$ and symmetric about $\mathrm{Re}(s)=\tfrac12$.

**5. Product over zeros and the zero count.** Since $\xi$ is entire, it has the symmetric product
$$\xi(s)=\xi(0)\prod_{\rho}\Bigl(1-\frac{s}{\rho}\Bigr),$$
with $\xi(0)=\tfrac12$ and the product over the nontrivial zeros $\rho$ taken in reflected pairs. The
number of zeros with imaginary part in $(0,T)$ is approximately
$\dfrac{T}{2\pi}\log\dfrac{T}{2\pi}-\dfrac{T}{2\pi}$.

**6. Explicit formula linking the prime count to the zeros.** Let $F(x)$ be the prime-power counting
function that jumps by $1/m$ at each $p^m$ and takes midpoint values at jumps. Then
$\dfrac{\log\zeta(s)}{s}=\int_1^\infty F(x)x^{-s-1}\,dx$; inverting by Fourier's theorem and inserting
the product over zeros gives
$$F(x)=\mathrm{Li}(x)-\sum_{\mathrm{Im}\,\rho>0}\Bigl(\mathrm{Li}(x^\rho)+\mathrm{Li}(x^{1-\rho})\Bigr)+\int_x^\infty\frac{dt}{t(t^2-1)\log t}+\log\xi(0),$$
where $\log\xi(0)=-\log2$ and the zero sum is taken in Riemann's symmetric limiting order. Under RH,
the paired terms become $\mathrm{Li}(x^{1/2+i\gamma})+\mathrm{Li}(x^{1/2-i\gamma})$. With the same
midpoint convention at jumps, Möbius inversion of $F(x)=\sum_n\tfrac1n\pi(x^{1/n})$ yields
$$\pi(x)=\sum_{n=1}^\infty\frac{\mu(n)}{n}\,F(x^{1/n})=F(x)-\tfrac12 F(x^{1/2})-\tfrac13 F(x^{1/3})-\tfrac15 F(x^{1/5})+\tfrac16 F(x^{1/6})+\cdots.$$

**Why the zeros govern the primes.** A conjugate pair $\rho=\beta+i\gamma$, $\bar\rho$ gives an
oscillatory contribution whose leading size is proportional to
$x^\beta\cos(\gamma\log x+\phi)/\log x$. The real part controls the power of the amplitude and the
imaginary part controls the frequency in $\log x$. The fluctuations of $\pi(x)$ about
$\mathrm{Li}(x)$ are the superposition of these zero-indexed oscillations.

## The Riemann Hypothesis

**All nontrivial zeros $\rho$ of $\zeta$ satisfy $\mathrm{Re}(\rho)=\tfrac12$**. Equivalently, in the
$s=\tfrac12+it$ variable, all roots of the corresponding $\Xi(t)$ are real. This is a conjecture,
not a proof; if true, every zero-indexed oscillation has square-root-size power, up to logarithmic
factors and the prime-power correction.
