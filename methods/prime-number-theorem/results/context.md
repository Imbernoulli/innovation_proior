## Research question

Let \(\pi(x)\) be the number of primes not exceeding \(x\). Tables of primes suggest that
\(\pi(x)\) grows like a smooth function even though individual primes have no visible local
rule. Gauss and Legendre proposed, from computation, that the density near \(x\) is governed
by \(1/\log x\), equivalently that the main scale should be \(x/\log x\) or the logarithmic
integral. The problem is to turn that empirical law into a theorem.

The obstruction is that direct enumeration sees only the primes already listed. It gives
\(\pi(x)\) exactly for a fixed \(x\), but it does not explain why the cumulative count should
have a stable asymptotic. A proof has to replace direct counting by an object whose global
behavior can be controlled.

## Background

Euler's product is the first object that contains all primes at once:
\[
\zeta(s)=\sum_{n\ge 1} n^{-s}=\prod_p (1-p^{-s})^{-1},\qquad \operatorname{Re}s>1.
\]
It follows from unique factorization and geometric series. Riemann's 1859 translation starts
from this identity, treats \(s\) as complex, and notes that both the series and product converge
only for \(\operatorname{Re}s>1\), while an integral representation gives a continuation of the
same function beyond that half-plane, with a simple pole at \(s=1\) and trivial zeros at the
negative even integers.

Chebyshev introduced weighted prime-counting functions that are better adapted to analysis:
\[
\theta(x)=\sum_{p\le x}\log p,\qquad
\psi(x)=\sum_{p^a\le x}\log p=\sum_{n\le x}\Lambda(n),
\]
where \(\Lambda(n)=\log p\) if \(n=p^a\) and \(0\) otherwise. Avigad's account records that
Chebyshev showed the prime number theorem is equivalent to \(\theta(x)/x\to 1\) and also to
\(\psi(x)/x\to 1\). These functions replace a bare count of primes by logarithmic weights that
match the logarithmic derivative of Euler's product.

Riemann's explicit formula shows why zeros matter. In the notation of his translated memoir,
the logarithm of \(\zeta\) is inverted by a Fourier/Mellin integral to express a weighted prime
count \(f(x)\); after inserting the product formula for the completed \(\xi\)-function, terms
from zeros of \(\zeta\) appear as oscillatory corrections to the logarithmic integral. The same
source states that the observed discrepancy between \(\operatorname{Li}(x)\) and the counted
primes up to three million has fluctuations, and that the periodic terms explain why such
fluctuations should be expected.

Hadamard's 1896 article begins from the logarithmic form of \(\zeta\) for
\(\operatorname{Re}s>1\), states that \(\zeta\) has no zeros there, and identifies the missing
boundary assertion as the absence of zeros on the line \(\operatorname{Re}s=1\). The article's
opening explicitly sets out to prove that conclusion, then uses it for arithmetic consequences.

## Baselines

Direct sieving gives exact finite counts. It is indispensable for data but remains an
enumeration procedure. It cannot convert the irregular local sequence of primes into an
asymptotic law.

The logarithmic-integral approximation fits the observed trend. It captures the empirical
scale but does not by itself explain why the scale is forced, nor why the errors fluctuate.

Chebyshev's real-variable estimates move from \(\pi(x)\) to \(\theta(x)\) and \(\psi(x)\), and
they prove that \(\pi(x)\log x/x\) stays between explicit positive constants for large \(x\).
This shows the right order of magnitude, but not the limiting constant.

Riemann's explicit formula gives a structural account of the fluctuations through zeros of
\(\zeta\). Its difficulty is that the formula is only as useful as the information available
about those zeros, especially near the boundary line \(\operatorname{Re}s=1\).

## Evaluation settings

The main target is the asymptotic statement
\[
\pi(x)\sim \frac{x}{\log x}.
\]
Equivalent targets are \(\theta(x)\sim x\) and \(\psi(x)\sim x\). A proof may naturally pass
through \(\psi\), because \(\psi\) is tied to the Dirichlet series for \(-\zeta'(s)/\zeta(s)\).

The analytic checks are: convergence in \(\operatorname{Re}s>1\), meromorphic continuation of
\(\zeta\), the simple pole at \(s=1\), nonvanishing on the line \(\operatorname{Re}s=1\), and a
valid bridge from singular behavior of a Dirichlet series to asymptotics of its coefficient
sums.

## Code framework

For this pure theorem there is no computational implementation to inherit. The natural scaffold
is a proof pipeline:

- Define \(\pi(x)\), \(\theta(x)\), \(\psi(x)\), and \(\Lambda(n)\).
- Establish the equivalences among \(\pi(x)\sim x/\log x\), \(\theta(x)\sim x\), and
  \(\psi(x)\sim x\).
- Express the Dirichlet series of \(\Lambda\) using the logarithmic derivative of Euler's
  product.
- Insert the analytic facts about \(\zeta\) at and near \(\operatorname{Re}s=1\).
- Use a contour/Perron or Tauberian bridge to pass from the singularity at \(s=1\) to the
  asymptotic for \(\psi(x)\), then return to \(\pi(x)\).
