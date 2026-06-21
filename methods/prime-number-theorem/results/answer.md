Prime number theorem via the zeta function.

**Theorem.** As \(x\to\infty\),
\[
\pi(x)\sim \frac{x}{\log x}.
\]

It is enough to prove the equivalent weighted statement
\[
\psi(x)=\sum_{n\le x}\Lambda(n)\sim x,
\]
where \(\Lambda(n)=\log p\) if \(n=p^k\) for a prime \(p\) and integer \(k\ge1\), and
\(\Lambda(n)=0\) otherwise.

**Proof.** For \(\operatorname{Re}s>1\), Euler's product gives
\[
\zeta(s)=\prod_p(1-p^{-s})^{-1}.
\]
Taking a logarithmic derivative,
\[
-\frac{\zeta'(s)}{\zeta(s)}
=\sum_p\sum_{k\ge1}(\log p)p^{-ks}
=\sum_{n\ge1}\frac{\Lambda(n)}{n^s}.
\]

The meromorphic continuation of \(\zeta\) has a simple pole at \(s=1\). Hence
\[
-\frac{\zeta'(s)}{\zeta(s)}-\frac1{s-1}
\]
is holomorphic near \(s=1\). The Euler product excludes zeros in \(\operatorname{Re}s>1\), and the
zero-free-line theorem excludes zeros on \(\operatorname{Re}s=1\). Therefore the same difference has
no boundary singularity on \(\operatorname{Re}s=1\).

Apply the Tauberian form appropriate to Dirichlet series with nonnegative coefficients: if
\[
A(s)=\sum_{n\ge1}a_n n^{-s}
\]
converges for \(\operatorname{Re}s>1\), the \(a_n\) are nonnegative, and
\[
A(s)-\frac{c}{s-1}
\]
extends without singularity to the boundary line \(\operatorname{Re}s=1\), then
\(\sum_{n\le x}a_n\sim cx\). With \(a_n=\Lambda(n)\) and \(c=1\), this gives
\[
\psi(x)\sim x.
\]

Now remove prime powers. Since
\[
\theta(x)=\sum_{p\le x}\log p,
\]
we have
\[
0\le \psi(x)-\theta(x)
=\sum_{\substack{p^k\le x\\k\ge2}}\log p
\le \sum_{2\le k\le \log_2 x}\theta(x^{1/k})
=O(x^{1/2}\log^2 x)=o(x).
\]
Thus \(\theta(x)\sim x\).

Partial summation converts \(\theta\) to \(\pi\):
\[
\pi(x)=\frac{\theta(x)}{\log x}+\int_2^x\frac{\theta(t)}{t\log^2 t}\,dt.
\]
Since \(\theta(t)\sim t\), the first term is asymptotic to \(x/\log x\), and the integral is
\(O(x/\log^2 x)\) at the main scale. Therefore
\[
\pi(x)\sim \frac{x}{\log x}.
\]

The method is to encode primes in the analytic function \(\zeta\), use the logarithmic
derivative to expose the Chebyshev/von Mangoldt counting function, and extract the real
counting asymptotic from the pole at \(s=1\) together with the absence of competing boundary
zeros.
