# Context: Counting the prime numbers below a given magnitude

## Research question

Let $\pi(x)$ denote the number of primes not exceeding $x$. The primes begin
$2,3,5,7,11,13,\dots$ and thin out as one climbs: there is no formula that produces the next prime
from the last, and inspection of the tables shows, in Euler's own words, "no order and no rule."
Yet the *count* $\pi(x)$ is far from lawless. From the prime tables compiled up to a few million,
Gauss and Legendre had observed that $\pi(x)$ tracks a smooth function astonishingly closely. The
question is to find an analytic expression for $\pi(x)$, or for the closely related density of primes
near $x$, that accounts both for the smooth trend and for the local fluctuations the tables exhibit.

Why it matters: the primes are the multiplicative atoms of the integers (every whole number factors
uniquely into primes), so the way they are distributed is the most basic quantitative question in
number theory. Locally the primes look random, but globally $\pi(x)$ follows a smooth curve to
within small, oscillating corrections. The task is to connect the discrete, irregular primes to the
smooth count they obey by an expression one could analyze.

## Background

**The empirical law of Gauss and Legendre.** From extensive hand counts of primes in successive
intervals, Gauss conjectured (already as a youth, working from prime tables) that the *density* of
primes near $x$ is about $1/\log x$, so that
$$\pi(x) \;\approx\; \mathrm{Li}(x) \;=\; \int_0^x \frac{dt}{\log t},$$
the logarithmic integral. Legendre published the closely related fit $\pi(x)\approx x/(\log x - A)$
with an empirical constant $A\approx 1.08$. Comparison of $\mathrm{Li}(x)$ against the counted
$\pi(x)$ up to about three million (the counts of Gauss and of Goldschmidt) shows agreement that is
strikingly good and improving with $x$, with the counted value running a little *below*
$\mathrm{Li}(x)$, the difference growing slowly and with many fluctuations. These fluctuations — an
observed irregular oscillation of the prime density about its smooth trend — were noticed but
obeyed no stated law. This is the central empirical fact on the table: a smooth approximation
$\mathrm{Li}(x)$ that is almost right, plus an oscillating discrepancy.

**Euler's product.** In 1737 Euler observed, as a consequence of unique factorization together with
the geometric series $\frac1{1-p^{-s}} = 1 + p^{-s} + p^{-2s} + \cdots$, that
$$\prod_{p\ \text{prime}} \frac{1}{1-p^{-s}} \;=\; \sum_{n=1}^{\infty} \frac{1}{n^s},$$
the product over all primes equalling the sum over all whole numbers, valid for real $s>1$. This is
the one identity that packages *every* prime into a single analytic expression. Euler used it to
reprove that there are infinitely many primes and, more, that $\sum_p 1/p$ diverges — a quantitative
strengthening of Euclid. Euler treated $s$ as a real variable greater than $1$; the product and the
series both converge there.

**The Gamma (factorial) function.** Euler's interpolation of the factorial,
$\Gamma(s) = \int_0^\infty e^{-x}x^{s-1}\,dx$ (here written $\Pi(s-1)$, with $\Pi(s-1)=\Gamma(s)$ and
$\Pi(n)=n!$), supplies the basic integral identity
$$\int_0^\infty e^{-nx} x^{s-1}\,dx \;=\; \frac{\Pi(s-1)}{n^s},$$
which turns a single term $n^{-s}$ into an integral. It is the standard identity relating the
factorial integral to a power $n^{-s}$.

**Jacobi's theta transformation.** For the series $\psi(x) = \sum_{n=1}^{\infty} e^{-n^2\pi x}$ and
its symmetrized form $\Theta(x)=2\psi(x)+1=\sum_{n=-\infty}^{\infty} e^{-n^2\pi x}$, Jacobi recorded
(Fundamenta Nova, S.184) the transformation
$$2\psi(x)+1 \;=\; x^{-1/2}\bigl(2\psi(1/x)+1\bigr), \qquad\text{i.e.}\qquad \Theta(x)=x^{-1/2}\Theta(1/x),$$
a self-duality under $x \leftrightarrow 1/x$ that comes from the self-reciprocity of the Gaussian
under the Fourier transform (Poisson summation). It is a known identity about a known function,
arising in the theory of elliptic and theta functions.

**Complex function theory and contour integration.** Cauchy's theory of functions of a complex
variable was in place: a function analytic except at isolated poles can be integrated along a
contour, and the integral is determined by the residues enclosed; a contour may be deformed freely
across regions where the integrand is analytic. The notion of an *analytic continuation* — that a
function given by one convergent expression on part of the plane is, if it extends analytically at
all, determined uniquely on the rest — belongs to this same circle of ideas.

**Fourier's inversion theorem.** Fourier's theorem permits recovering a function from an integral
transform of it: if $g$ is built from $h$ by an exponential/multiplicative integral, $h$ can be
expressed back in terms of $g$ by an inverse integral. It is the general tool for recovering a
function from an integral transform of it.

## Baselines

**Direct counting / sieving.** Compute $\pi(x)$ by listing or sieving the primes up to $x$
(Eratosthenes). It yields the exact value at one $x$ by enumeration.

**Gauss's logarithmic-integral approximation $\pi(x)\approx \mathrm{Li}(x)$.** The smooth
description available: a single clean function matching the counts to within a slowly growing,
oscillating error. It is conjectured from tables and describes the trend.

**Legendre's fit $\pi(x)\approx x/(\log x - A)$.** A two-parameter empirical formula with a fitted
constant $A\approx 1.08$.

**Euler's product as an analytic identity.** The product $\prod_p(1-p^{-s})^{-1}=\sum_n n^{-s}$ is
the one baseline object that contains all primes analytically, and Euler extracted qualitative facts
from it (infinitude of primes, divergence of $\sum 1/p$). As written it holds on the real half-line
$s>1$, where both sides converge.

## Evaluation settings

The natural yardstick is the existing prime data and the existing smooth approximations against
which any analytic law for $\pi(x)$ would be checked:
- The tabulated prime counts $\pi(x)$ up to roughly $x = 3\times 10^6$ (Gauss; Goldschmidt), with
  per-decade or per-interval prime densities derived from them.
- The smooth comparison curves $\mathrm{Li}(x) = \int_0^x dt/\log t$ and Legendre's
  $x/(\log x - A)$, and the observed signed discrepancy $\pi(x) - \mathrm{Li}(x)$ as a function of
  $x$, including its sign and its fluctuation.
- The functions of analytic number theory available to phrase the count: $\pi(x)$ itself; the
  logarithmic integral $\mathrm{Li}$; and the Möbius coefficients $\mu(n)\in\{-1,0,1\}$ of
  elementary number theory.
- The domain of validity to be respected: the Euler product and the Dirichlet series both require
  $\mathrm{Re}(s)>1$, so any analytic claim must be checked first where these converge and only then
  extended.
