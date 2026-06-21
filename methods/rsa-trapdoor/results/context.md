# Context: a concrete trap-door one-way function for public-key cryptography

## Research question

Two people who have never met want to communicate privately over a channel that everyone can
listen to — and, separately, one of them wants to "sign" a message so the other can prove to a
third party who sent it. With every cipher known at the time, both goals share one prerequisite:
the two parties must *already* share a secret key, delivered ahead of time over some secure channel
(a courier, registered mail, a trusted face-to-face meeting). The question is whether there exists
a concrete mathematical function that is easy to evaluate in the forward direction, yet practically
infeasible to invert for anyone who does not hold a secret piece of information used during its
construction — a *trap-door one-way function* — so that public-key encryption and message signing
become possible without pre-shared secrets.

## Background

The field state: classical cryptography is *symmetric*. A transmitter applies an invertible
transformation S_K (keyed by a secret K) to plaintext P to get ciphertext C = S_K(P); the legitimate
receiver, knowing the same K, applies S_K^{-1} to recover P. Security rests on the computational
difficulty, for a cryptanalyst lacking K, of recovering P from C. The one-time pad (1920s) is
provably unbreakable (Shannon's information-theoretic argument, a quarter-century later) but requires
key material as long as the message, so it is impractical at scale. The U.S. National Bureau of
Standards has just adopted the Data Encryption Standard (DES, 1975–77), developed at IBM; it is a
strong symmetric block cipher.

The reframing that sets up everything: Diffie & Hellman (1976) argued that the security of most
ciphers already rests not on information theory but on **computational difficulty** to the
cryptanalyst — which puts cryptography inside the theory of computational complexity, where some
problems are believed to require exponential effort. They named the object we need a **one-way
function**: f easy to compute, f^{-1} infeasible. They gave a genuine example — discrete
exponentiation over a finite field. Fix a prime q and a primitive element α; then Y = α^X mod q is
cheap to compute (≤ 2 log₂ q multiplications by repeated squaring), but recovering X = log_α Y mod q
(the *discrete logarithm*) appears to need on the order of q^{1/2} operations by the best known
method — exponential in the bit-length of q. From this asymmetry they built a key *exchange*: users i
and j publish Y_i = α^{X_i} and Y_j = α^{X_j}, each forms the shared value α^{X_i X_j} = Y_j^{X_i} =
Y_i^{X_j}, while an eavesdropper holding only Y_i, Y_j must compute a discrete log to get it. This
gives a *shared secret over a public channel* — but it is not a public-key cryptosystem: there is no
published E that a stranger can use, unilaterally, to send a private message.

Diffie & Hellman then isolated the harder object: a **trap-door one-way function**. Their analogy is
a combination lock — anyone who knows the combination opens it in seconds; a skilled locksmith
without it needs hours; and crucially the designer who *built* the lock with a chosen combination has
an advantage no amount of cleverness substitutes for. A trap-door one-way function is one that
"strongly resists" inversion by anyone not holding the trap-door information used in its construction,
yet is easy to invert for whoever does. They observed that a public-key cryptosystem is *exactly* a
family of trap-door one-way functions: the public E is a complete algorithm for the forward map, the
private trap-door bit-string is what makes the inverse D easy. They could prove the architecture —
that such functions would yield public-key encryption, signatures, and key exchange — but they had
**no example**. Their one candidate public-key construction (encipher a binary message m by an
invertible n×n matrix E; inverting requires matrix inversion) they called "suggestive, although
unfortunately useless," because matrix inversion costs only about n³ versus n² for the legitimate
operation — a cryptanalytic advantage ratio of at most n, when ratios of 10^6 or more are needed.
Their own summary of the situation: "There is currently little evidence for the existence of
trap-door ciphers."

The mathematical material that exists and is load-bearing for any number-theoretic attempt:

- **Fermat's little theorem.** For a prime p and any a not divisible by p, a^{p−1} ≡ 1 (mod p). One
  clean proof: for a coprime to p, the map x ↦ ax permutes the nonzero residues {1,…,p−1}, so
  ∏(a·k) ≡ ∏ k (mod p), i.e. a^{p−1}(p−1)! ≡ (p−1)! (mod p); cancel (p−1)! to get a^{p−1} ≡ 1.
- **Euler's theorem (the generalization to composite moduli).** Let φ(n) be Euler's totient — the
  count of integers in [1, n] coprime to n. Then for any m coprime to n, m^{φ(n)} ≡ 1 (mod n). φ is
  multiplicative on coprime arguments, and φ(prime p) = p − 1.
- **Chinese Remainder Theorem.** For coprime p, q: if x ≡ a (mod p) and x ≡ a (mod q) then x ≡ a (mod
  pq). (Proof: the difference x − a is a common multiple of p and q, hence a multiple of pq.) This lets
  a statement about residues modulo a product be assembled from the statement modulo each prime factor.
- **The difficulty of factoring.** Multiplying two large primes p, q to form n = pq is cheap; the
  inverse — recovering p, q from n — has been worked on for three centuries (Fermat, Legendre) with no
  known method that factors a 200-digit number in feasible time. By contrast, *testing* whether a
  number is prime is much easier than factoring it: the probabilistic Solovay–Strassen test (1977)
  picks random a and checks gcd(a,b)=1 and J(a,b) ≡ a^{(b−1)/2} (mod b) where J is the Jacobi symbol; a
  composite fails with probability ≥ ½, so 100 independent trials drive the error below 2^{-100}. By
  the prime number theorem a random 100-digit odd number is prime with probability ~2/ln(10^100), so a
  prime is found after testing on the order of 115 candidates.
- **Modular exponentiation is cheap.** Computing m^e mod n costs at most 2 log₂ e modular
  multiplications by repeated squaring and multiplication; the cost grows only as the cube of the digit
  length of n.
- **A multiple of φ(n) is as good as the factorization.** Miller (1975) showed that given any
  multiple of φ(n), n can be factored.
- A related but distinct line: Pohlig & Hellman study exponentiation ciphers done modulo a **prime**.
  There the order of the multiplicative group, p − 1, is public.

## Baselines

- **Symmetric ciphers in general (and DES in particular).** C = S_K(P), decrypt with the same K.
  The encrypting and decrypting capability are the same secret, so the key must be shared in advance
  over a secure channel.

- **Diffie–Hellman exponential key exchange (1976).** Over GF(q) with primitive α, users publish
  Y = α^X mod q and combine to the shared α^{X_i X_j}; security rests on the hardness of discrete
  logarithm mod q. It produces a shared key between two parties who interact.

- **The Diffie–Hellman matrix public-key sketch (1976).** Encipher binary message m by an invertible
  n×n matrix E (m ↦ Em); the legitimate decryptor knows D = E^{-1}; an attacker must invert E.
  Inversion costs ~n³ versus ~n² for the forward map; Diffie and Hellman call it "useless" as a
  practical construction. It establishes the *shape* of a public-key construction: publish a forward
  map, hide an easy inverse.

- **Merkle's puzzles (≈1976).** A sender prepares a large number of "puzzles," each modestly hard;
  the receiver solves one at random and they use its contents as a shared key, forcing an eavesdropper
  to solve about as many puzzles as were prepared.

- **Pohlig–Hellman exponentiation cipher (modulo a prime).** Encryption by m ↦ m^e mod p, decryption
  by c ↦ c^d mod p over a *prime* modulus. Over a prime the group order p − 1 is public information.

## Evaluation settings

The natural yardsticks are not benchmark datasets but the defining properties of a trap-door
one-way function and the computational asymmetry they demand:

- **Correctness.** Does deciphering invert enciphering for *every* admissible message M, with no
  message silently corrupted? (For a permutation, every ciphertext must be a valid message and vice
  versa.) For signatures, additionally enciphering-after-deciphering must return M.
- **Forward cost.** Encryption/decryption must be feasible at the chosen size: a 200-digit message
  encryptable in seconds on a general-purpose computer; cost growing no faster than the cube of the
  digit-length of n.
- **Inversion cost without the trap door.** The number of operations an attacker needs to recover the
  plaintext (or the secret key) from public data — measured against the best known algorithms for
  whatever hard problem the construction ends up resting on. As a rough sense of the scale such
  estimates can reach, the best discrete-logarithm methods need on the order of q^{1/2} operations for
  a modulus of bit-length log₂ q. A meaningful target is a cryptanalytic-to-legitimate cost *ratio* of
  10^6 or more — a ratio Diffie and Hellman identified as necessary for practical security.
- **Key generation cost.** Producing the keys — whatever they turn out to be — must itself be cheap,
  using only the available machinery (probabilistic primality testing, extended Euclid), so that
  setting up a user is fast even at ~100-digit sizes.
- **Reductions among attacks.** The strongest evidence available, absent a proof of security, is to
  show that the obvious routes for breaking a candidate scheme each reduce to some problem already
  studied, and resisted attack, over a long history — so that defeating the scheme is no easier than
  solving a long-standing hard problem.

## Code framework

The available machinery is big-integer arithmetic, fast modular exponentiation by repeated
squaring, the extended Euclidean algorithm for modular inverses, and a probabilistic primality test.
The open construction slot is the choice of keys and maps that make a public forward procedure easy,
its inverse easy with a private secret, and inversion from public data infeasible.

```python
# ---- primitives that already exist ----

def modexp(base, exp, mod):
    # m^e mod n by repeated squaring & multiply; <= 2*log2(exp) modular mults
    result = 1
    base %= mod
    while exp > 0:
        if exp & 1:
            result = (result * base) % mod
        base = (base * base) % mod
        exp >>= 1
    return result

def egcd(a, b):
    # extended Euclid: returns (g, x, y) with a*x + b*y = g = gcd(a, b)
    if b == 0:
        return (a, 1, 0)
    g, x1, y1 = egcd(b, a % b)
    return (g, y1, x1 - (a // b) * y1)

def modinv(a, m):
    # multiplicative inverse of a modulo m (requires gcd(a, m) == 1)
    g, x, _ = egcd(a, m)
    if g != 1:
        raise ValueError("not invertible")
    return x % m

def is_probable_prime(b, rounds=100):
    # probabilistic primality test (Jacobi-symbol / Solovay-Strassen style);
    # a composite is rejected with probability >= 1/2 per round
    ...  # TODO

def random_prime(num_bits, rounds=100):
    # generate random odd candidates, return the first that passes the test
    ...  # TODO


# ---- open construction slot ----

def generate_keys(prime_bits, public_exponent, rounds=100):
    """Produce a public key and a private key once the construction slot is filled.
    The secret, the permutation condition, and the key relation are still unspecified."""
    # TODO: the trap-door construction
    pass

def encrypt(message, public_key):
    # TODO: the easy forward direction
    pass

def decrypt(ciphertext, private_key):
    # TODO: the inverse, easy ONLY with the trap door
    pass

def sign(message, private_key):
    # TODO: the private-key-first operation, if the maps are permutations
    pass

def verify(message, signature, public_key):
    # TODO: the public check corresponding to sign
    pass
```
