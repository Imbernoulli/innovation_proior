The problem is to decide whether a large odd integer $n$ — possibly a thousand bits — is prime, in time polynomial in $\log n$, without factoring $n$, and with a guarantee I can actually prove today. Public-key cryptography needs a steady supply of large primes, so candidates are sampled and must be tested before anything else can proceed. Trial division by every integer up to $\sqrt{n}$ is correct but costs $\Theta(\sqrt{n})$, exponential in the bit-length, so it is useful only as a small-factor pre-filter. The Fermat test is fast — pick a base $a$, compute $a^{n-1} \bmod n$ by square-and-multiply, and declare $n$ composite if the result is not $1$ — but it has a fatal structured failure: the Carmichael numbers, composite $n$ such as $561 = 3\cdot 11\cdot 17$ for which $a^{n-1} \equiv 1 \pmod n$ for *every* coprime base $a$. On those, random coprime sampling reduces the error not at all. The Solovay–Strassen test repairs this by way of the Euler criterion and the Jacobi symbol, but it needs that extra machinery and still only buys a $1/2$ bound. And there is already a fast *deterministic* test, Miller's: it checks a structured congruence over a fixed short list of small bases. The trouble is its correctness rests on the unproven extended Riemann hypothesis, which is the only thing guaranteeing that such a short fixed list must contain an exposer of every composite. I do not want a primality verdict that is hostage to an open problem in analytic number theory; I want a guarantee that stands on proven ground.

I propose the Miller–Rabin primality test. The escape from the Riemann hypothesis is to stop demanding that a *fixed short list* of bases provably works and instead prove, unconditionally, that a *large fraction* of all bases expose any composite — then I sample bases at random and let independent trials drive the error to zero geometrically, trading certainty for a controllable, vanishingly small probability of error. So I need two things: a compositeness condition on a base $a$ that is cheap to check, and a proof that its exposers are dense for every composite $n$. The starting point is Fermat's little theorem, $a^{n-1} \equiv 1 \pmod n$ for prime $n$ and $\gcd(a,n)=1$, a fast necessary condition. Its weakness is that it inspects only the single end-value $a^{n-1}$ and throws away the path that led there — which is exactly why Carmichael numbers slip through. The fix is to keep the path. Since $n$ is odd, write $n-1 = 2^s d$ with $d$ odd and $s \ge 1$, so that $a^{n-1}$ is reached by squaring $a^d$ exactly $s$ times, and watch the whole chain
$$a^d,\ a^{2d},\ a^{4d},\ \ldots,\ a^{2^{s-1}d},\ a^{2^s d} = a^{n-1},$$
each term the square of the one before. The load-bearing fact about primes is that modulo a prime the only square roots of $1$ are $\pm 1$: from $x^2 - 1 = (x-1)(x+1) \equiv 0$ a prime must divide one factor, so $x \equiv \pm 1$. Walking the chain backward from $a^{n-1} \equiv 1$, every predecessor squares to $1$ and so must be $\pm 1$; either the chain is $1$ all the way down, or the first non-$1$ value it meets is forced to be $-1$. Equivalently, factoring $x^{2^s d} - 1 = (x^d-1)(x^d+1)(x^{2d}+1)\cdots(x^{2^{s-1}d}+1)$ and using that a prime divides one factor, for prime $n$ every base satisfies
$$a^d \equiv 1 \pmod n \quad\text{or}\quad a^{2^i d} \equiv -1 \pmod n \text{ for some } i \in \{0,\ldots,s-1\}. \qquad (\star)$$
I call $a$ a nonwitness when $(\star)$ holds and a witness otherwise; a witness proves $n$ composite. A witness fails $(\star)$ in one of two ways: either the top value $a^{n-1}$ is not $1$ (the old Fermat certificate), or the chain reaches $1$ from a predecessor that is neither $1$ nor $-1$ — a nontrivial square root of $1$, which a prime cannot possess. This is what defeats $561$: with $s=4,\ d=35$, base $a=2$ gives the chain $(263,166,67,1)$, which reaches $1$ from $67$, a nontrivial square root of $1$, exposing $561$ even though $2^{560}\equiv 1$ left Fermat silent. The whole check costs only the squarings I was doing anyway, so $O(\log n)$ modular multiplications per base. I sample from $\{2,\ldots,n-2\}$, skipping $1$ and $n-1$, which are always nonwitnesses ($1^d=1$, and $(n-1)^d \equiv (-1)^d \equiv -1$ since $d$ is odd).

What makes the method reliable is the density guarantee: for every odd composite $n$, more than $3/4$ of the bases in $\{2,\ldots,n-2\}$ are witnesses, with no Carmichael exception. The first instinct — mirror Fermat and show the nonwitnesses form a proper subgroup — fails, because the nonwitnesses are not even a subgroup: modulo $65$ the nonwitnesses $8$ and $18$ both hit $-1$ at the same position, and $8\cdot 18 \equiv 14$ is a witness, the matching $-1$s cancelling to $+1$ too early. The repair is to build a genuine subgroup that *contains* all nonwitnesses by collapsing every chain to one fixed exponent. Let $i_0$ be the largest index in $\{0,\ldots,s-1\}$ for which some unit satisfies $a_0^{2^{i_0} d} \equiv -1 \pmod n$ — it exists because $a_0=-1$ works at $i=0$ — and set
$$G_n = \{\, a \in (\mathbb{Z}/n\mathbb{Z})^* : a^{2^{i_0} d} \equiv \pm 1 \pmod n \,\}.$$
Since $a \mapsto a^{2^{i_0}d}$ is a homomorphism and $\{\pm 1\}$ is closed, $G_n$ is a subgroup. It contains every nonwitness: if $a^d\equiv 1$ then $a^{2^{i_0}d}\equiv 1$; if $a^{2^i d}\equiv -1$ then by maximality $i \le i_0$, and squaring $i_0-i$ times sends $-1$ to $+1$, so $a^{2^{i_0}d}\equiv \pm 1$. To force $G_n$ proper when $n=p^\alpha n'$ is not a prime power, CRT picks $a \equiv a_0 \pmod{p^\alpha}$, $a\equiv 1 \pmod{n'}$, so $a^{2^{i_0}d}$ is $-1$ mod $p^\alpha$ but $+1$ mod $n'$, hence neither $\pm 1$ mod $n$: a unit outside $G_n$. To sharpen the index from $2$ to at least $4$, note first that every $a\in G_n$ satisfies $a^{n-1}\equiv 1$ (one more squaring of $\pm 1$ gives $1$, since $2^{i_0+1}d \mid n-1$), so $G_n$ lies inside the Fermat group $F_n = \{a : a^{n-1}\equiv 1\}$. When $n$ is not Carmichael, $F_n$ is itself a proper subgroup of the units, and the CRT element above lies in $F_n$ but not $G_n$, giving the strict chain $\text{units} \supsetneq F_n \supsetneq G_n$ — two doublings, so $\phi(n)/|G_n| \ge 4$. When $n$ is Carmichael, $F_n$ is everything and that chain stalls at $1/2$, but a Carmichael number is squarefree with $r\ge 3$ distinct prime factors $n=p_1\cdots p_r$, which is the resource: define $H_n = \{a : a^{2^{i_0}d}\equiv \pm 1 \pmod{p_l}\ \text{for each } l\}$, allowing each factor its own independent sign, and let $f(a) = (a^{2^{i_0}d}\bmod p_l)_l$ map $H_n \to \prod_l \{\pm 1\}$. Using $a_0$ and CRT to realize each one-coordinate sign flip, $f$ is surjective onto a target of order $2^r$, while $G_n$ — which demands the *same* sign at every factor — maps only to the two diagonal patterns. Hence $|H_n|/|G_n| = 2^{r-1} \ge 4$, and again $\phi(n)/|G_n| \ge 4$. For prime powers $n=p^\alpha$, $\alpha\ge 2$, a direct count finishes it: the nonwitnesses are exactly the $p-1$ solutions of $a^{p-1}\equiv 1 \pmod{p^\alpha}$ in the cyclic unit group, so their proportion is $(p-1)/(p^\alpha-1) = 1/(1+p+\cdots+p^{\alpha-1}) \le 1/4$. In every case the nonwitnesses number at most $\phi(n)/4 < (n-1)/4$, and discarding the always-nonwitness bases $1$ and $n-1$ only improves the sampled range. So each random round catches a composite with probability $> 3/4$; running $t$ independent rounds, the test never rejects a prime and reports a false "probably prime" with probability $< 4^{-t}$ — an unconditional bound, no Riemann hypothesis anywhere — at a cost of $O(t\log n)$ modular multiplications.

```python
import random

def modpow(base, exp, mod):
    result = 1
    base %= mod
    while exp > 0:
        if exp & 1:
            result = (result * base) % mod
        base = (base * base) % mod
        exp >>= 1
    return result

def small_factor_prefilter(n):
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n == p:
            return True
        if n % p == 0:
            return False
    return None

def prepare_candidate(n):
    # n - 1 = 2^s * d, d odd
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    return s, d

def single_base_test(n, a, state):
    # False means a proves n composite; True means this base passes.
    s, d = state
    x = modpow(a, d, n)
    if x == 1 or x == n - 1:
        return True
    for _ in range(s - 1):
        x = (x * x) % n
        if x == n - 1:
            return True
        if x == 1:               # nontrivial square root of 1 -> composite
            return False
    return False

def is_probably_prime(n, trials=40):
    # randomized over 2..n-2: for trials > 0, composite error < 4^(-trials)
    if n < 2:
        return False
    pre = small_factor_prefilter(n)
    if pre is not None:
        return pre
    state = prepare_candidate(n)
    for _ in range(trials):
        a = random.randrange(2, n - 1)
        if not single_base_test(n, a, state):
            return False
    return True
```
