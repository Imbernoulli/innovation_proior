Two strangers on a wire that everyone can tap want one of them to send the other something only the receiver can read, without their ever having met to agree on a secret — and, separately, the sender wants to attach to a message a tag that proves he sent it, that anyone can check but no one can forge. Every cipher of the day collapses both goals into one prerequisite: the two parties must already share a secret key, delivered ahead of time over a secure channel. That is the key distribution problem, and it is fatal for a large electronic network, since the number of pre-distributed courier keys grows quadratically in the number of users. Diffie and Hellman (1976) named the missing piece cleanly: each user should publish an enciphering procedure $E$ in a directory and keep a deciphering procedure $D$ private, where $D$ inverts $E$, both are cheap for the owner to run, and — the load-bearing requirement — publishing $E$ reveals no feasible way to compute $D$. Easy forward, infeasible backward, except trivial backward for whoever holds a secret baked into $E$ at construction time. That is a trap-door one-way function, and they proved that a single example would yield private mail, signatures, and key exchange all at once. What they lacked was the example. Their best public-key candidate — encipher by multiplying by an invertible $n \times n$ matrix, decrypt by inverting it — they discarded themselves, because inversion costs about $n^3$ against $n^2$ for the legitimate map, so the attacker is slower only by a factor of $n$, when the goal is a cryptanalytic-to-legitimate cost ratio of $10^6$ or more. Their one genuine one-way function, discrete exponentiation $Y = \alpha^X \bmod q$ over a prime field, has the right asymmetry — forward by repeated squaring is trivial, backward is the discrete logarithm needing on the order of $\sqrt{q}$ operations — but it builds only a key *exchange* between two interacting parties, with no published $E$ a stranger can use unilaterally, and no trap door at all: discrete log mod a prime is hard for everyone, the designer included. A one-way function with no back door is a lock with no key.

I propose RSA, a concrete trap-door one-way permutation built from modular exponentiation over a *composite* modulus. The form of an exponentiation cipher is right even if the prime-field use of it is wrong: encrypt by $C \equiv M^e \pmod n$, and hope the inverse is *another* exponentiation $M \equiv C^d \pmod n$, so that encryption and decryption are the same operation run by the same squaring routine with different exponents. Raising to $e$ then to $d$ is raising to $ed$, so the whole scheme reduces to finding a modulus $n$ and a pair $(e,d)$ for which $M^{ed} \equiv M \pmod n$ — the product $ed$ must act like the identity exponent. The question of when a power acts like the identity is exactly Fermat–Euler territory, and it is also where the prime-modulus version dies and the composite version is born. Modulo a prime $p$, Fermat gives $M^{p-1} \equiv 1$, so $ed \equiv 1 \pmod{p-1}$ would make decryption work; but $p$ is the public modulus, so everyone reads off $p-1$ and solves for $d$ from $e$ by the same extended Euclid the owner would use. There is no secret — this is the Pohlig–Hellman symmetric setting, never public-key. The fix is to name the obstruction precisely: the exponent arithmetic is governed by the count of units, and modulo a prime that count $p-1$ is handed to everyone by the modulus. I need a modulus whose governing constant cannot be read off it. Euler answers Fermat's question for composite moduli: the magic exponent is $\varphi(n)$, and $M^{\varphi(n)} \equiv 1 \pmod n$ for $M$ coprime to $n$. The crack is that $\varphi(n)$ is *not* a function of $n$ you can evaluate by looking at $n$ — it depends on how $n$ factors. If $n = pq$ is a product of two primes, multiplicativity gives

$$\varphi(n) = (p-1)(q-1),$$

which whoever built $n$ from $p$ and $q$ knows instantly, but whoever sees only $n$ must factor it first — the three-hundred-year-old problem nobody can do at scale. So the modulus must be composite precisely so that the exponent-arithmetic constant $\varphi(n)$ is hidden behind the difficulty of factoring $n$. The factorization is the secret tunnel: publish $n$, keep $p$ and $q$.

Building it carefully: pick two large random primes $p, q$, form $n = pq$, and compute $\varphi(n) = (p-1)(q-1)$ from the known factors. I want exponents with $ed \equiv 1 \pmod{\varphi(n)}$, so that $ed = k\varphi(n) + 1$ for some integer $k$ and $M^{ed} = M \cdot (M^{\varphi(n)})^k$. The pair exists exactly when one exponent is a unit modulo $\varphi(n)$: choosing $e$ coprime to $\varphi(n)$, that is $\gcd(e, \varphi(n)) = 1$, guarantees an inverse $d = e^{-1} \bmod \varphi(n)$, obtained by running extended Euclid on $\varphi(n)$ and $e$ in under $2\log_2 n$ steps. The public key is $(e, n)$; the private key is $(d, n)$; and $p, q, \varphi(n)$ are kept or destroyed. (One may instead pick $d$ first and derive $e$; the algebra is symmetric, but I fix $e$ first so the public operation stays fast, restarting only if a chosen $e$ fails the coprimality test.)

The correctness must hold for *every* message in $[0, n)$, not just for the typical case, or the map is not a permutation and signatures break. The bare Euler argument $M^{\varphi(n)} \equiv 1 \pmod n$ has a hole: it needs $M$ coprime to $n$, yet the message space includes multiples of $p$ and of $q$. The clean route is to prove $M^{ed} \equiv M$ modulo each prime separately and glue by the Chinese Remainder Theorem, since $p$ and $q$ are distinct hence coprime. Writing $ed = k(p-1)(q-1) + 1$ and working modulo $p$: if $p \nmid M$, Fermat gives $M^{p-1} \equiv 1$, so $M^{ed} = M \cdot (M^{p-1})^{k(q-1)} \equiv M \pmod p$; if $p \mid M$, then $M \equiv 0$ and $M^{ed} \equiv 0 \equiv M \pmod p$ trivially. The identical argument modulo $q$ uses $M^{q-1} \equiv 1$ when $q \nmid M$ and the trivial vanishing otherwise. The very case that defeats Euler — $M$ sharing a factor with $n$ — is exactly the case where both sides vanish modulo that prime, so it causes no trouble once handled prime by prime. Agreement modulo both coprime primes forces agreement modulo $pq = n$ by CRT, so $M^{ed} \equiv M \pmod n$ for all $M$ with no coprimality assumption left over. Hence $E : M \mapsto M^e$ and $D : C \mapsto C^d$ are mutually inverse *permutations* of $\{0, \dots, n-1\}$: every ciphertext is itself a valid message, which is what makes it well-defined to apply $D$ to an unenciphered message.

That permutation property is what hands me signatures essentially for free, and it is the reason I worked to make the construction a permutation and not merely a function. Because $E$ and $D$ are inverse permutations they commute, $E(D(M)) = D(E(M)) = M$. To sign $M$, Bob applies his private exponent first, $S \equiv M^{d} \pmod n$, and anyone verifies by applying his public exponent, $S^{e} \equiv M^{ed} \equiv M \pmod n$. Producing such an $S$ for a specified $M$ without the private exponent is exactly the public inversion problem — finding an $e$-th root of $M$ modulo $n$ — so the same hardness that protects decryption gives the signature its force, and the signed value is bound to the whole message because verification recomputes $M$ from $S$. One construction, run with the public exponent first for secrecy and the private exponent first for signatures, meets both of Diffie and Hellman's goals.

What makes it trustworthy is that the obvious shortcuts all collapse back to factoring. Factoring $n$ yields $p, q$, hence $\varphi(n)$, hence $d$ — so factoring suffices to break it, which is the trap door working in reverse for whoever can open it. Computing $\varphi(n)$ directly gains nothing, because $\varphi(n) = n - (p+q) + 1$ recovers $s = p+q$, and then $(p-q)^2 = s^2 - 4n$ gives $r = |p-q|$ by an integer square root, so $p, q = (s \pm r)/2$; computing $\varphi(n)$ is therefore no easier than factoring. Recovering any decrypting exponent $d$ also collapses to factoring: $ed - 1$ is a multiple of $\varphi(n)$, and Miller (1975) showed $n$ can be factored from any multiple of $\varphi(n)$; any equivalent exponent $d'$ satisfies $ed' \equiv 1$ modulo $\mathrm{lcm}(p-1, q-1)$, exposing a multiple of the same hidden structure and likewise factoring $n$. The one door I cannot bolt shut is computing $e$-th roots modulo $n$ by some route that never touches the factorization; that is not a classically named hard problem, so its intractability is conjectured rather than proved, and security rests on the difficulty of factoring together with the absence of such a root-extraction shortcut. The margin is the one the matrix scheme never had: with $\sim 100$-digit primes giving a $\sim 200$-digit $n$, the best known factoring is out past $10^{23}$ operations. Every piece is computable at that scale — exponentiation costs at most $2\log_2 e$ modular multiplications, growing as the cube of the digit length of $n$; inverting $e$ takes under $2\log_2 n$ Euclid steps; and finding the primes leans on the asymmetry that *testing* primality is far cheaper than factoring, using the Solovay–Strassen test (pick random $a$, check $\gcd(a, b) = 1$ and that the Jacobi symbol $J(a,b) \equiv a^{(b-1)/2} \pmod b$), where a composite fails at least half the rounds so $100$ rounds leave error below $2^{-100}$, and by the prime number theorem a prime is found after testing on the order of a hundred candidates. For safety against special-purpose factoring one makes $p, q$ differ in length by a few digits, ensures $p-1$ and $q-1$ each carry a large prime factor, and keeps $\gcd(p-1, q-1)$ small.

```python
import secrets

def modexp(base, exp, mod):
    # m^e mod n by repeated squaring & multiply
    result, base = 1, base % mod
    while exp > 0:
        if exp & 1:
            result = (result * base) % mod
        base = (base * base) % mod
        exp >>= 1
    return result

def egcd(a, b):                      # a*x + b*y = gcd(a, b)
    if b == 0:
        return (a, 1, 0)
    g, x1, y1 = egcd(b, a % b)
    return (g, y1, x1 - (a // b) * y1)

def modinv(a, m):                    # inverse of a mod m; needs gcd(a, m) == 1
    g, x, _ = egcd(a, m)
    if g != 1:
        raise ValueError("not invertible")
    return x % m

def jacobi(a, n):
    if n <= 0 or n % 2 == 0:
        raise ValueError("n must be a positive odd integer")
    a %= n
    result = 1
    while a:
        while a % 2 == 0:
            a //= 2
            if n % 8 in (3, 5):
                result = -result
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3:
            result = -result
        a %= n
    return result if n == 1 else 0

def is_probable_prime(b, rounds=100):
    if b in (2, 3):
        return True
    if b < 2 or b % 2 == 0:
        return False
    for _ in range(rounds):
        a = 2 + secrets.randbelow(b - 3)
        if egcd(a, b)[0] != 1:
            return False
        if modexp(a, (b - 1) // 2, b) != jacobi(a, b) % b:
            return False
    return True

def random_prime(num_bits, rounds=100):
    if num_bits < 2:
        raise ValueError("num_bits must be at least 2")
    while True:
        candidate = secrets.randbits(num_bits)
        candidate |= (1 << (num_bits - 1)) | 1
        if is_probable_prime(candidate, rounds):
            return candidate

def generate_keys(prime_bits, public_exponent=65537, rounds=100):
    if public_exponent <= 1:
        raise ValueError("public exponent must be greater than 1")
    while True:
        p = random_prime(prime_bits, rounds)
        q = random_prime(prime_bits, rounds)
        if p == q:
            continue
        n = p * q
        phi = (p - 1) * (q - 1)      # phi(n); needs the factors p, q
        e = public_exponent
        if e < phi and egcd(e, phi)[0] == 1:
            d = modinv(e, phi)       # e*d = k*phi + 1
            return (e, n), (d, n)    # public (e, n), private (d, n)

def require_residue(x, n, name):
    if not (0 <= x < n):
        raise ValueError(f"{name} must satisfy 0 <= {name} < n")

def encrypt(m, pub):
    e, n = pub
    require_residue(m, n, "message")
    return modexp(m, e, n)           # C = m^e mod n

def decrypt(c, priv):
    d, n = priv
    require_residue(c, n, "ciphertext")
    return modexp(c, d, n)           # m = C^d mod n

def sign(m, priv):
    d, n = priv
    require_residue(m, n, "message")
    return modexp(m, d, n)           # S = m^d mod n

def verify(m, s, pub):
    e, n = pub
    require_residue(m, n, "message")
    require_residue(s, n, "signature")
    return modexp(s, e, n) == m      # S^e = m^{ed} = m mod n
```
