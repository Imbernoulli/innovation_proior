I start with the thing that feels least analytic: I want to count primes. If I write \(\pi(x)=\sum_{p\le x}1\), I have only renamed the list. A sieve can compute this number, but every step still asks whether each integer has a prime divisor. That is exact and local; it gives me no reason that the cumulative count should settle near \(x/\log x\). The tables suggest a smooth density, and direct counting gives me a jagged step function. I need to stop staring at the primes one at a time.

Euler's product is the first place where the primes stop being a list and become a function. For \(\operatorname{Re}s>1\),
\[
\zeta(s)=\sum_{n\ge1}\frac1{n^s}=\prod_p(1-p^{-s})^{-1}.
\]
The right side is still built from primes, but the left side has forgotten the primes completely and sums over all integers. Unique factorization is doing the compression. If I can learn something analytic about this function, I may learn something about the primes without checking primality one integer at a time.

But the product itself is not yet a counting theorem. In \(\operatorname{Re}s>1\), the product converges and has no zeros, but the prime number theorem asks what happens at the boundary of that convergence. The pole at \(s=1\) is exactly where the harmonic series appears, and the prime density scale \(1/\log x\) smells like a boundary phenomenon. If I remain only in the safe half-plane, I never touch the place where the main term should be born.

Riemann's move changes the problem. The same \(\zeta(s)\) that Euler defines by a product in \(\operatorname{Re}s>1\) can be continued as a meromorphic function, with a simple pole at \(s=1\). This matters because a singularity is a global analytic event. It is not a prime-by-prime datum. If the primes are encoded in \(\zeta\), then perhaps the main term in their count is encoded by the singularity at \(1\), and the error terms are encoded by the other singularities and zeros.

Taking logarithms is tempting because
\[
\log\zeta(s)=-\sum_p\log(1-p^{-s})
       =\sum_p\sum_{k\ge1}\frac{p^{-ks}}{k}.
\]
This already says I should not count only primes. The logarithm naturally counts prime powers, with weight \(1/k\). Riemann's weighted function \(f(x)\) appears from precisely that expansion: primes, half prime squares, a third prime cubes, and so on. It is smoother than \(\pi(x)\), and Möbius inversion can later recover the prime count. That is useful, but the logarithm has branch issues and the inverse integral for \(\log\zeta(s)\) is delicate.

The derivative removes some of that ambiguity. Differentiating the logarithm of Euler's product gives
\[
-\frac{\zeta'(s)}{\zeta(s)}
=\sum_p\sum_{k\ge1}(\log p)p^{-ks}
=\sum_{n\ge1}\frac{\Lambda(n)}{n^s},
\qquad \operatorname{Re}s>1,
\]
where \(\Lambda(n)=\log p\) when \(n=p^k\), and \(0\) otherwise. This is cleaner. The coefficient sum of this Dirichlet series is
\[
\psi(x)=\sum_{n\le x}\Lambda(n)=\sum_{p^k\le x}\log p.
\]
So the object I should try to estimate is not \(\pi(x)\) first. It is \(\psi(x)\). This is not cosmetic. The logarithmic derivative is where primes enter linearly, and the pole and zeros of \(\zeta\) become poles of \(-\zeta'/\zeta\). The counting problem has become a question about the singularities of a Dirichlet series.

Now I ask what singularity I should expect. Near \(s=1\), \(\zeta(s)\) has a simple pole, so locally \(\zeta(s)=a/(s-1)+\) holomorphic terms with \(a\neq0\). Then
\[
\frac{\zeta'(s)}{\zeta(s)}=-\frac1{s-1}+\text{holomorphic terms},
\]
and therefore
\[
-\frac{\zeta'(s)}{\zeta(s)}=\frac1{s-1}+\text{holomorphic terms}.
\]
That residue \(1\) is exactly the candidate main term for \(\psi(x)\sim x\). If a Dirichlet series with nonnegative coefficients has a single simple pole of residue \(1\) at the boundary \(s=1\), and no other boundary singularity interferes, then its partial sums should grow like \(x\). This is the Tauberian shape of the argument.

There is a trap here. A pole at \(s=1\) alone is not enough if other singularities sit on the same boundary line. A zero of \(\zeta\) at \(1+it\), \(t\neq0\), would become a pole of \(-\zeta'/\zeta\). In the inverse Mellin picture, such a pole contributes an oscillating term of size \(x^{1+it}/(1+it)\), whose magnitude is on the same order as \(x\). That would disturb the limit \(\psi(x)/x\). The boundary line is not a technical nuisance; it is the place where possible main-size oscillations live.

So the analytic statement I need is very specific: \(\zeta(s)\) has no zeros on \(\operatorname{Re}s=1\). In \(\operatorname{Re}s>1\), the Euler product already rules out zeros. Hadamard's article isolates the missing boundary conclusion and proves it. Once that line is clear, \(-\zeta'/\zeta(s)-1/(s-1)\) has no boundary pole at any point \(1+it\). The pole at \(1\) is the only boundary singularity left.

The proof of that nonvanishing has the right flavor. It does not try to count primes directly. It uses the positivity hidden in the logarithm of the Euler product. For real \(\sigma>1\),
\(\log\zeta(\sigma)\) is a sum with positive contributions. If I compare values at \(\sigma\), \(\sigma+it\), and \(\sigma+2it\), the trigonometric inequality
\[
3+4\cos u+\cos 2u=2(1+\cos u)^2\ge0
\]
keeps the prime-power sum from producing the wrong sign. In modern terms this makes the product
\[
|\zeta(\sigma)^3\zeta(\sigma+it)^4\zeta(\sigma+2it)|
\]
large enough, as \(\sigma\downarrow1\), to contradict the possibility that \(\zeta(1+it)=0\) while \(\zeta\) has only its simple pole at \(1\). The exact proof can be organized in different ways, but the force of it is this positivity: the Euler product still controls the boundary when its logarithm is combined so that every prime-power coefficient is nonnegative.

With the zero-free boundary in place, the counting theorem is no longer a direct enumeration problem. I have the Dirichlet series
\[
A(s)=\sum_{n\ge1}\Lambda(n)n^{-s}=-\frac{\zeta'(s)}{\zeta(s)}
\]
for \(\operatorname{Re}s>1\), and I know that
\[
A(s)-\frac1{s-1}
\]
extends without singularity along the boundary line \(\operatorname{Re}s=1\). The coefficients \(\Lambda(n)\) are nonnegative. A Tauberian theorem is exactly the bridge from this analytic boundary behavior to the real-variable asymptotic:
\[
\sum_{n\le x}\Lambda(n)\sim x.
\]
This is \(\psi(x)\sim x\).

Now I have to return from \(\psi\) to actual primes. The extra terms in \(\psi\) come from prime powers \(p^k\) with \(k\ge2\). They are lower order:
\[
0\le \psi(x)-\theta(x)
=\sum_{\substack{p^k\le x\\k\ge2}}\log p
\le \sum_{2\le k\le \log_2 x}\theta(x^{1/k})
\le \sum_{2\le k\le \log_2 x} x^{1/k}\log x
=O(x^{1/2}\log^2 x).
\]
This is \(o(x)\), so \(\theta(x)\sim x\) as well.

Finally \(\theta\) and \(\pi\) differ only by the logarithmic weight. Partial summation gives
\[
\pi(x)=\frac{\theta(x)}{\log x}+\int_2^x\frac{\theta(t)}{t\log^2 t}\,dt.
\]
If \(\theta(t)\sim t\), the first term is \(x/\log x\), and the integral is smaller, of order \(x/\log^2 x\) at the main scale. Therefore
\[
\pi(x)\sim \frac{x}{\log x}.
\]
