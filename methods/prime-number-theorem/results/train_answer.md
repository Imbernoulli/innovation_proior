I will explain the canonical method for the prime number theorem, which I take to be the proof through the analytic properties of the Riemann zeta function. The goal is to understand why the number of primes up to $x$, denoted $\pi(x)$, is asymptotic to $x/\log x$ as $x$ grows. This is not a fact that becomes obvious by listing primes, because the primes themselves are irregular; the regularity appears only in the cumulative count, and the right tool for capturing that regularity is a global analytic function rather than a local sieve.

The starting point is Euler's product, which gathers all primes into a single identity. For $\operatorname{Re}s>1$,
\[
\zeta(s)=\sum_{n\ge 1}\frac{1}{n^s}=\prod_p\frac{1}{1-p^{-s}}.
\]
The sum is over all positive integers, and the product is over all primes; unique factorization is what makes the two sides equal. This product is the first place where the primes stop being a sequence of isolated numbers and become part of a continuous object. It is already a compression of the multiplicative structure of the integers, but it is still restricted to a half-plane where it converges absolutely.

The prime number theorem asks about a boundary phenomenon, because the empirical density $1/\log x$ corresponds to the pole of the zeta function at $s=1$. Riemann's contribution was to continue $\zeta(s)$ meromorphically beyond $\operatorname{Re}s>1$, leaving only a simple pole at $s=1$. That continuation lets us bring complex analysis to bear on a counting problem. The pole is a global singularity; its residue will turn out to be the coefficient in the main term of the prime-counting asymptotic.

To turn the product into a counting statement, I take the logarithmic derivative. Differentiating $\log\zeta(s)$ using the Euler product gives
\[
-\frac{\zeta'(s)}{\zeta(s)}=\sum_p\sum_{k\ge 1}(\log p)\,p^{-ks}=\sum_{n\ge 1}\frac{\Lambda(n)}{n^s},
\]
where $\Lambda(n)=\log p$ if $n=p^k$ for some prime $p$ and integer $k\ge 1$, and $\Lambda(n)=0$ otherwise. This is the Dirichlet series whose partial sums are Chebyshev's function
\[
\psi(x)=\sum_{n\le x}\Lambda(n)=\sum_{p^k\le x}\log p.
\]
So instead of attacking $\pi(x)$ directly, I study $\psi(x)$. The logarithmic derivative makes the primes appear linearly, and every zero or pole of $\zeta$ becomes a singularity of $-\zeta'/\zeta$.

Near $s=1$, the simple pole of $\zeta$ produces
\[
-\frac{\zeta'(s)}{\zeta(s)}=\frac{1}{s-1}+\text{holomorphic terms}.
\]
The residue $1$ is exactly the candidate main term in $\psi(x)\sim x$. If this were the only boundary singularity, a standard Tauberian theorem for Dirichlet series with nonnegative coefficients would immediately give $\psi(x)\sim x$. The difficulty is that a zero of $\zeta$ on the line $\operatorname{Re}s=1$ would create another pole of $-\zeta'/\zeta$ of comparable size, producing oscillatory terms that could destroy the limit.

Therefore the crucial analytic input is the zero-free line: $\zeta(s)$ has no zeros on $\operatorname{Re}s=1$. The Euler product already rules out zeros in $\operatorname{Re}s>1$, so the boundary is the only remaining danger. The classical proof of the zero-free line relies on positivity hidden inside the logarithm of the Euler product. For real $\sigma>1$ and real $t\neq 0$, the trigonometric inequality $3+4\cos u+\cos 2u=2(1+\cos u)^2\ge 0$ can be used to show that
\[
|\zeta(\sigma)^3\zeta(\sigma+it)^4\zeta(\sigma+2it)|
\]
remains large as $\sigma\downarrow 1$. This contradicts the possibility that $\zeta(1+it)=0$, because a simple pole at $1$ cannot balance a fourth-order zero at $1+it$ against the third-order behavior at $1$.

Once the boundary line is clear, the difference
\[
-\frac{\zeta'(s)}{\zeta(s)}-\frac{1}{s-1}
\]
extends without singularity to $\operatorname{Re}s\ge 1$. Since the coefficients $\Lambda(n)$ are nonnegative, the appropriate Tauberian theorem applies: a Dirichlet series with nonnegative coefficients whose only boundary singularity is a simple pole of residue $c$ at $s=1$ has partial sums asymptotic to $cx$. With $c=1$ this gives
\[
\psi(x)\sim x.
\]

It remains to return from $\psi$ to the ordinary prime count $\pi$. The difference between $\psi$ and Chebyshev's theta function $\theta(x)=\sum_{p\le x}\log p$ comes from prime powers $p^k$ with $k\ge 2$. Their total contribution is at most
\[
\sum_{2\le k\le \log_2 x}\theta(x^{1/k})=O(x^{1/2}\log^2 x)=o(x),
\]
so $\theta(x)\sim x$ as well. Finally, partial summation converts $\theta$ to $\pi$:
\[
\pi(x)=\frac{\theta(x)}{\log x}+\int_2^x\frac{\theta(t)}{t\log^2 t}\,dt.
\]
The first term is asymptotic to $x/\log x$, while the integral is $O(x/\log^2 x)$, so the integral is lower order and we obtain
\[
\pi(x)\sim\frac{x}{\log x}.
\]

The canonical method name I propose is "prime number theorem via the Riemann zeta function." The essence is to stop counting primes one by one, encode them in Euler's product, differentiate the logarithm to expose the von Mangoldt weights, use the analytic continuation and the zero-free line to isolate the pole at $s=1$, and let a Tauberian bridge carry that analytic information back to the real asymptotic. The code below gives a small numerical illustration: it sieves primes, reports $\pi(x)$, and compares the ratio $\pi(x)\log x/x$ to $1$ over a range of $x$.

```python
def sieve(limit):
    is_prime = bytearray(b"\x01") * (limit + 1)
    is_prime[0:2] = b"\x00\x00"
    for p in range(2, int(limit**0.5) + 1):
        if is_prime[p]:
            step = p
            start = p * p
            is_prime[start:limit + 1:step] = b"\x00" * ((limit - start) // step + 1)
    return is_prime

def pi_table(limit):
    primes = sieve(limit)
    counts = [0] * (limit + 1)
    c = 0
    for n in range(limit + 1):
        if primes[n]:
            c += 1
        counts[n] = c
    return counts

LIMIT = 2_000_000
counts = pi_table(LIMIT)
print(f"{'x':>12} {'pi(x)':>10} {'pi(x)*log(x)/x':>18}")
for x in [100, 1_000, 10_000, 100_000, 1_000_000, 2_000_000]:
    import math
    ratio = counts[x] * math.log(x) / x
    print(f"{x:>12} {counts[x]:>10} {ratio:>18.6f}")
```
