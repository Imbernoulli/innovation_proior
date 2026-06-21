What bothers me is not factoring as such — it is that I have in front of me a machine that obeys quantum mechanics and so, on paper, can hold $2^n$ complex amplitudes at once, and nobody has yet extracted from it a single useful thing that a classical computer with a coin could not already do. Deutsch–Jozsa, Bernstein–Vazirani: real separations, but either exact and fragile or built on a problem so contrived it feels like cheating. Meanwhile factoring an $l$-bit integer $N$ stays superpolynomial for every classical method — the number field sieve costs $\exp\!\big(c\,(\log N)^{1/3}(\log\log N)^{2/3}\big)$, subexponential but never polynomial — and RSA's security rests entirely on that hardness. The goal is to factor $N$ in time polynomial in $l=\log N$, and the deeper prize is a candidate counterexample to the strong Church's thesis on a problem people actually care about. So the question I really care about is sharper: what does a quantum computer naturally do well, and is there a valuable problem whose structure matches it?

The one piece of machinery that genuinely impressed me is Simon's. Given a black box $f$ on $n$-bit strings with the promise that $f(x)=f(y)$ exactly when $x\oplus y\in\{0,s\}$, classically you must find a collision, which is $\Theta(2^{n/2})$ queries; his quantum routine needs $O(n)$. The mechanism is the whole lesson: put the input register in uniform superposition, compute $f$ into a second register, *measure the second register* so the first collapses onto a single coset $\{x_0, x_0\oplus s\}$, then apply a Fourier transform over $(\mathbb{Z}_2)^n$ — a Hadamard on every wire — so the random offset $x_0$ contributes only a global sign and washes out of the probabilities, while the hidden period $s$ survives as the constraint that every observable $y$ satisfies $y\cdot s=0$. Gather $n-1$ such linear constraints over $\mathrm{GF}(2)$, solve, done. Strip away the $(\mathbb{Z}_2)^n$ and what remains is general: a quantum computer is good at taking a function with a *hidden periodicity*, superposing its whole domain, and using a Fourier transform — the basis in which a shift becomes a phase, so the unknown shift cancels and the period concentrates the amplitude — to make that period measurable. Simon's period happens to live in a binary vector space and his transform reaches no further than $\mathbb{Z}_2$. The natural move is to ask what becomes possible when the periodicity lives in a *cyclic* group instead, where the right transform is the honest discrete Fourier transform $|a\rangle\mapsto q^{-1/2}\sum_c e^{2\pi i ac/q}|c\rangle$.

I propose the method that follows from carrying Simon's strategy into $\mathbb{Z}_q$; it is Shor's algorithm, and its engine is quantum order-finding. The bridge from factoring to periodicity already exists in Miller's reduction, and I want to re-derive it so I trust every case. For odd $N$ that is not a prime power, choose a random $x$ with $1<x<N$; if $\gcd(x,N)\neq1$ that gcd is already a factor, otherwise let $r$ be the order of $x$, the least exponent with $x^r\equiv1\pmod N$. If $r$ is even, then $(x^{r/2}-1)(x^{r/2}+1)=x^r-1\equiv0\pmod N$, and since $r$ is the *least* exponent giving $1$ we have $x^{r/2}\not\equiv1$, so $N$ does not divide the left factor; if additionally $x^{r/2}\not\equiv-1\pmod N$, then $N$ divides the right factor either, so $N$'s primes are split between the two factors and $\gcd(x^{r/2}-1,N)$ is a nontrivial divisor. The reduction fails only when $r$ is odd or $x^{r/2}\equiv-1$. Writing $N=\prod_{i=1}^k p_i^{a_i}$ with $k$ distinct odd primes, the Chinese remainder theorem makes the choice of $x$ equivalent to independent choices mod each $p_i^{a_i}$, and with $r_i=\mathrm{ord}_{p_i^{a_i}}(x)$ and $r=\mathrm{lcm}(r_i)$ the failure happens exactly when the $2$-adic valuations $e_i$ of the $r_i$ all agree — all zero forces $r$ odd, a common positive value forces $x^{r/2}\equiv-1$ modulo every prime power and hence mod $N$. Each $(\mathbb{Z}/p_i^{a_i})^*$ is cyclic, so each $e_i$ takes any fixed value with probability at most $1/2$, and by CRT independence all $k$ agree with probability at most $1/2^{k-1}$. So a random $x$ yields a factor with probability $\ge 1-1/2^{k-1}\ge 1/2$. The whole problem is now to find the order $r$.

Order-finding is where the quantum machine earns its keep. Fix a transform length $q=2^m$ with $N^2\le q<2N^2$ — I will justify that size in a moment — and use two registers. First put register one into the uniform superposition $q^{-1/2}\sum_{a=0}^{q-1}|a\rangle|0\rangle$, which is just a Hadamard on each of the $m$ qubits. Second, compute $x^a\bmod N$ into register two reversibly, giving $q^{-1/2}\sum_a|a\rangle|x^a\bmod N\rangle$; keeping $a$ makes the map $(a,0)\mapsto(a,x^a)$ reversible, and since $x$ is fixed I precompute the constants $x^{2^i}\bmod N$ classically and apply controlled multiplications, building each multiplication mod $N$ from reversible controlled additions and erasing scratch by the Bennett uncomputation dance — $O((\log N)^3)$ time, $O(\log N)$ space. The crucial structure is now visible: $x^a\bmod N$ depends only on $a\bmod r$, so register two is periodic in $a$ with period $r$, exactly Simon's hidden-period situation but in the cyclic index group. Third, apply the quantum Fourier transform $|a\rangle\mapsto q^{-1/2}\sum_c e^{2\pi i ac/q}|c\rangle$ to register one, producing $\tfrac1q\sum_{a,c}e^{2\pi i ac/q}|c\rangle|x^a\bmod N\rangle$. Fourth, measure. To see the structure, imagine measuring both registers and obtaining $|c, x^k\bmod N\rangle$ for some $0\le k<r$; the amplitude is a sum over every $a$ with $x^a\equiv x^k$, i.e. $a\equiv k\pmod r$, so $a=br+k$, and pulling out the common modulus-one factor $e^{2\pi i kc/q}$ leaves
$$P_k(c)=\Big|\tfrac1q\sum_{b=0}^{\lfloor(q-k-1)/r\rfloor}e^{2\pi i\,b\,\{rc\}_q/q}\Big|^2,$$
where $\{rc\}_q\in(-q/2,q/2]$ is $rc\bmod q$. This is a geometric sum of unit phasors stepping by $2\pi rc/q$ per term: when $rc/q$ is near an integer all phasors align and the sum is large, otherwise they fan around the circle and cancel, so the probability concentrates on $c$ near multiples of $q/r$ — the period read out as the location of the Fourier peaks.

To make "near an integer" quantitative — because $r$ does not divide $q$, which is precisely the smearing I was worried about — suppose $|\{rc\}_q|\le r/2$. Comparing the geometric sum with the integral $\tfrac1q\int_0^{\lfloor(q-k-1)/r\rfloor}e^{2\pi i b\{rc\}_q/q}\,db$ costs only $O(1/q)$ in this regime, and substituting $u=rb/q$ turns it into $\tfrac1r\int_0^1 e^{2\pi i(\{rc\}_q/r)u}\,du$ (the upper limit is within $O(1/q)$ of $1$ since $k<r$). This sinc envelope has modulus minimized at $\{rc\}_q/r=\pm1/2$, where it equals $\tfrac1r\cdot\tfrac2\pi=\tfrac{2}{\pi r}$, so the amplitude has modulus at least $2/(\pi r)$ up to an $O(1/q)$ error that is negligible because $q\ge N^2$ and $r<N$. Hence $P_k(c)\ge 4/(\pi^2 r^2)$, and since $4/\pi^2>1/3$ the usable bound is $P_k(c)\ge 1/(3r^2)$ for large $N$.

The measurement is useful only if I can convert the peak location back into $r$. The good $c$ are exactly those with $|\{rc\}_q|\le r/2$, i.e. there is an integer $d$ with $|c/q-d/r|\le 1/(2q)$. I know $c$ and $q$ and want $r$, so I need $d/r$ to be the *unique* low-denominator fraction that close to $c/q$: two distinct fractions with denominators below $N$ differ by more than $1/N^2$, so if $1/q\le 1/N^2$, i.e. $q\ge N^2$, then $d/r$ is unique — and *that* is exactly where the transform length $N^2\le q<2N^2$ comes from. The continued-fraction expansion of $c/q$, whose convergents $p_n/q_n=a_n p_{n-1}+p_{n-2}$ over $a_n q_{n-1}+q_{n-2}$ are the best rational approximations, recovers $d/r$ in lowest terms in polynomial time; the denominator is $r$ exactly when $\gcd(d,r)=1$. To count successes, note there are $\varphi(r)$ values of $d$ coprime to $r$ (each giving one good $c$) and $r$ possible values of $x^k$, so $r\,\varphi(r)$ favorable measurement outcomes, each of probability at least $1/(3r^2)$, giving
$$P(\text{recover }r)\;\ge\;\frac{r\,\varphi(r)}{3r^2}\;=\;\frac{\varphi(r)}{3r}.$$
Because $\varphi(r)/r>\delta/\log\log r$ (Hardy–Wright, Thm 328), each run succeeds with probability $\gtrsim 1/\log\log r$, so $O(\log\log r)$ repetitions suffice.

The one thing I owe is that the cyclic QFT on $m$ qubits is buildable cheaply and really computes the claimed phase. Use only two gates: a Hadamard $R_j$ on bit $j$, and a controlled phase $S_{j,k}$ for $j<k$ applying $e^{i\theta}$ with $\theta_{k-j}=\pi/2^{k-j}$ on $|11\rangle$. Apply them as $R_{m-1}S_{m-2,m-1}R_{m-2}\cdots R_0$, the Hadamards in descending bit order with all controlled phases $S_{j,k}$ ($k>j$) between consecutive Hadamards — $m$ Hadamards plus $m(m-1)/2$ phase gates, so $O(m^2)$. Only the Hadamards flip bits, so the amplitude on a path $|a\rangle\to|b\rangle$ is fixed: $m$ factors of $1/\sqrt2$ give $1/\sqrt q$, with accumulated phase
$$\sum_{0\le j<m}\pi a_j b_j+\sum_{0\le j<k<m}\frac{\pi}{2^{k-j}}a_j b_k=\sum_{0\le j\le k<m}\frac{\pi}{2^{k-j}}a_j b_k.$$
The circuit emits the output in bit-reversed order, $b_k=c_{m-1-k}$; substituting and reindexing $k\leftarrow m-1-k$ turns the constraint $j\le k$ into $j+k<m$ and gives $\sum_{j+k<m}2\pi(2^j2^k/2^m)a_j c_k$. Terms with $j+k\ge m$ add integer multiples of $2\pi$, so the sum extends freely to all $j,k<m$, and by distributivity $\sum_{j,k}2\pi(2^j2^k/2^m)a_jc_k=(2\pi/2^m)(\sum_j 2^j a_j)(\sum_k 2^k c_k)=2\pi ac/q$, exactly $e^{2\pi i ac/q}$ with the $1/\sqrt q$ out front. The $S_{j,k}$ with large $k-j$ apply exponentially tiny phases that barely matter, so Coppersmith's approximate QFT drops them and still factors. The cost overall is dominated by modular exponentiation, $O(l^3)$ time (or $O(l^2\log l\log\log l)$ with Schönhage–Strassen) for $l=\log N$, with the QFT at $O(l^2)$, times $O(\log\log r)$ repetitions and polynomial classical post-processing — polynomial in $l$, and RSA's hardness assumption with it. (Worked instance: $N=91=7\cdot13$, $x=3$ has order $r=6$, $3^3=27\not\equiv-1\pmod{91}$, and $\gcd(27-1,91)=\gcd(26,91)=13$.)

```python
import math, random
from collections import defaultdict
from fractions import Fraction

def gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)

def modexp(base, e, mod):
    r = 1; base %= mod
    while e:
        if e & 1: r = (r * base) % mod
        base = (base * base) % mod; e >>= 1
    return r

def _integer_nth_root(n, k):
    lo, hi = 1, 1 << ((n.bit_length() + k - 1) // k)
    while lo <= hi:
        mid = (lo + hi) // 2
        p = mid ** k
        if p == n:
            return mid
        if p < n:
            lo = mid + 1
        else:
            hi = mid - 1
    return hi

def _perfect_power_factor(N):
    for k in range(2, N.bit_length() + 1):
        root = _integer_nth_root(N, k)
        if root > 1 and root ** k == N:
            return root
    return None

def find_period(x, N):
    q_bits = (N * N - 1).bit_length()     # q = 2^q_bits, N^2 <= q < 2 N^2
    q = 1 << q_bits
    # exact simulation of: superposition -> modexp -> QFT -> measure
    groups = defaultdict(list)
    value = 1
    for a in range(q):
        groups[value].append(a)
        value = (value * x) % N
    probs = [0.0] * q
    for alist in groups.values():
        for c in range(q):
            s = sum(complex(math.cos(2*math.pi*a*c/q),
                            math.sin(2*math.pi*a*c/q)) for a in alist)
            probs[c] += abs(s / q) ** 2
    t = random.random(); acc = 0.0
    c = q - 1
    for cc in range(q):
        acc += probs[cc]
        if t <= acc: c = cc; break
    frac = Fraction(c, q).limit_denominator(N - 1)
    candidate = frac.denominator
    if candidate > 0 and modexp(x, candidate, N) == 1:
        return candidate
    return None

def factor(N):
    if N < 2:
        raise ValueError("N must be at least 2")
    if N % 2 == 0: return 2
    pp = _perfect_power_factor(N)
    if pp is not None:
        return pp
    while True:
        x = random.randrange(2, N)
        g = gcd(x, N)
        if g != 1:
            return g                       # lucky common factor
        r = find_period(x, N)
        if r is None or r % 2 != 0:
            continue                       # need even order
        y = modexp(x, r // 2, N)
        if y == 1 or y == N - 1:
            continue                       # retry the failed reduction cases
        f = gcd(y - 1, N)
        if 1 < f < N:
            return f                       # gcd(x^{r/2}-1, N)
```
