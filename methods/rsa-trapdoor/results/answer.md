# RSA: a concrete trap-door one-way permutation from modular exponentiation

## The problem it solves

Diffie and Hellman (1976) defined the object public-key cryptography needs — a **trap-door one-way
function**: easy to compute forward, infeasible to invert from public data, yet trivial to invert
for whoever holds a secret "trap door" — but exhibited no example. RSA is a concrete instance: a
function easy for anyone to apply, whose obvious inversion routes all run through factoring a large
number unless you hold that factorization.

## The key idea

Encrypt by raising the message to a public power modulo a **composite** n = pq:
m ↦ m^e mod n. The inverse is *another* exponentiation, c ↦ c^d mod n, where the two exponents
satisfy e·d ≡ 1 (mod φ(n)). The decryption exponent d can be computed only from
φ(n) = (p−1)(q−1), which requires the factors p, q. So **the factorization of n is the trap door**:
the modulus n is public, but the exponent-arithmetic constant φ(n) it controls is hidden behind the
difficulty of factoring n. Over a *prime* modulus the corresponding constant p−1 is public and there
is no secret — which is exactly why n must be composite.

## Key generation

1. Choose two large random primes p, q (found via probabilistic primality testing); set n = pq.
2. Compute φ(n) = (p−1)(q−1).
3. Choose an encryption exponent e coprime to φ(n) (so e is invertible modulo φ(n)).
4. Compute the decryption exponent d = e^{-1} mod φ(n) by the extended Euclidean algorithm, so
   e·d ≡ 1 (mod φ(n)), i.e. ed = kφ(n) + 1 for some integer k.
5. Publish the public key (e, n); keep the private key (d, n); keep or destroy p, q, φ(n).

## Encryption / decryption / signatures

- Encrypt: C ≡ M^e (mod n).  Decrypt: M ≡ C^d (mod n).
- Sign: S ≡ M^d (mod n) (private exponent first).  Verify: check S^e ≡ M (mod n).
- Both directions are the same operation (modular exponentiation), computed by repeated squaring in
  at most 2 log₂(exponent) modular multiplications; cost grows as the cube of the digit-length of n.

## Correctness: M^{ed} ≡ M (mod n) for *every* M in [0, n)

Write ed = kφ(n) + 1 = k(p−1)(q−1) + 1. Prove the congruence modulo p and modulo q separately, then
glue by the Chinese Remainder Theorem (p, q distinct primes ⇒ coprime).

Modulo p, for any M:
- If p ∤ M, Fermat's little theorem gives M^{p−1} ≡ 1 (mod p), so
  M^{ed} = M · (M^{p−1})^{k(q−1)} ≡ M · 1 ≡ M (mod p).
- If p | M, then M ≡ 0 (mod p), so M^{ed} ≡ 0 ≡ M (mod p) trivially.

Modulo q, for any M:
- If q ∤ M, Fermat gives M^{q−1} ≡ 1 (mod q), so
  M^{ed} = M · (M^{q−1})^{k(p−1)} ≡ M · 1 ≡ M (mod q).
- If q | M, then M ≡ 0 (mod q), so M^{ed} ≡ 0 ≡ M (mod q) trivially.

Since M^{ed} and M agree modulo both coprime primes, by CRT M^{ed} ≡ M (mod pq = n) for **all** M in
[0, n) — no coprimality assumption needed. The case that defeats a bare Euler-theorem argument (M
sharing a factor with n) is exactly the case where both sides vanish modulo that prime. Therefore
E: M ↦ M^e and D: C ↦ C^d are mutually inverse **permutations** of {0, …, n−1}; every ciphertext is a
valid message, which is what makes signatures (applying D to an unenciphered message) well-defined.

## Why the main attacks reduce to factoring

- **Factor n ⇒ break.** Factoring yields p, q ⇒ φ(n) ⇒ d. So factoring suffices to break it.
- **Compute φ(n) ⇔ factor n.** From n and φ(n) = n − (p+q) + 1 recover s = p+q; then
  (p−q)² = s² − 4n, so r = |p−q| is the integer square root; finally p, q are
  (s+r)/2 and (s−r)/2 up to order. So computing φ(n) is no easier than factoring.
- **Recover d ⇒ factor n.** ed − 1 is a multiple of φ(n), and n can be factored given any multiple of
  φ(n) (Miller 1975). Any equivalent decrypting exponent d′ satisfies ed′ ≡ 1 modulo lcm(p−1, q−1);
  finding one exposes a multiple of that hidden exponent structure and likewise factors n. So recovering
  a decrypting exponent is no easier than factoring.
- **The residual gap.** Computing e-th roots mod n by some route that never recovers the factorization
  is not a classically named hard problem; its intractability is conjectured, not proved. Security
  therefore rests on the difficulty of factoring (a problem resisted for three centuries) together with
  the absence of such a root-extraction shortcut.

## Parameter guidance (period-appropriate)

Use ~100-digit primes so n has ~200 digits; published factoring estimates put a 200-digit n out of
reach (≈10²³ operations by the fastest method then known). For extra safety, make p, q differ in
length by a few digits, ensure p−1 and q−1 each have a large prime factor, and keep gcd(p−1, q−1)
small. In the choose-d-first variant, restart if the computed e comes out smaller than log₂ n; in the
standard e-first implementation, choose e in range and require gcd(e, φ(n)) = 1.

## Python implementation

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

A worked check at toy size: p = 47, q = 59 ⇒ n = 2773, φ(n) = 46·58 = 2668; pick d = 157 ⇒ e = 17
(since 17·157 = 2669 = φ(n) + 1). Then M = 920 encrypts to 920¹⁷ ≡ 948 (mod 2773), and
948¹⁵⁷ ≡ 920 (mod 2773) recovers it.
