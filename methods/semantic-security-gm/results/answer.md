# Semantic security, indistinguishability, and the Goldwasser–Micali cryptosystem

## The problem

Public-key encryption existed (Diffie–Hellman, RSA, Rabin), but "secure" meant only that the adversary
cannot *invert* — recover the whole plaintext or the key. That is too weak: in applications such as
mental poker, learning even one bit of partial information (the color of a hidden card) breaks the
system though the plaintext is never recovered. The goal is to define security so the ciphertext leaks
*nothing efficiently computable* about the plaintext, for every message distribution, and to build a
scheme provably meeting it.

## Key idea

1. **Semantic security** is the right definition: whatever a polynomially bounded adversary can compute
   about the plaintext given the ciphertext, it can compute without the ciphertext.
2. It is **equivalent to ciphertext-indistinguishability** (no efficient adversary distinguishes
   encryptions of two chosen messages) — the clean, single-game definition one actually proves against.
3. These force encryption to be **randomized**: a deterministic public-key scheme leaks the predicate
   "ciphertext is even," leaks message *equality*, and is broken by encrypt-the-guess-and-compare, so it
   can never be indistinguishable.
4. A semantically secure scheme is built by **encrypting one bit at a time** with an *unapproximable
   trapdoor predicate*; the **quadratic residuosity** predicate is one, under the QR assumption.

## Definitions (security parameter k; adversaries = polynomial-size circuits)

**Indistinguishability (polynomial security).** A PKC Π is *indistinguishable* if for every message
generator no polynomial-size message-finder F can output a pair m₀, m₁ whose encryptions a poly-size
line-tapper T tells apart: with p_b^T = Pr[T(E, a)=1 : a ∈ E(m_b)], no T achieves |p₀^T − p₁^T| > 1/poly(k)
for an F-findable pair (with non-negligible probability). Deterministic schemes are not indistinguishable.

**Semantic security.** For any function f on the message space M (need not be recursive), let
p_f = max_v Pr[f(m)=v] be the best a-priori guessing probability. Π is *semantically secure* if for every
M, every f, and every polynomial Q, no poly-size circuit, given E and a ∈ E(m), computes f(m) with
probability exceeding p_f + 1/Q(k) (on a 1/poly fraction of public keys E). Equivalently: the ciphertext
gives no computational advantage in computing any function of the plaintext — the polynomially bounded
analogue of Shannon perfect secrecy (a-posteriori = a-priori probabilities).

## Equivalence theorem

**Theorem. Π is semantically secure ⇔ Π is indistinguishable.**

*Semantic ⇒ indistinguishable (contrapositive).* If a finder/tapper pair distinguishes m₀, m₁ with
advantage > 1/poly, let the semantic adversary choose f(m) = [m = m₁] and a distribution supported on
{m₀, m₁}, and output the tapper's bit as its guess of f(m). The tapper fires noticeably more often on
encryptions of m₁ than of m₀, so this beats the a-priori p_f by 1/poly — semantic security fails.

*Indistinguishable ⇒ semantic (contrapositive).* Suppose a poly-size circuit C computes f(m) from a∈E(m)
with probability > p_E + ε, ε = 1/Q(k). For a message m and value v let r_v^m = Pr[C(E, E(m)) = v]. Fix a
random reference μ; call ζ *good* if some v has |r_v^ζ − r_v^μ| > ε/10.
- **Mass lemma.** Σ_{ζ good} Pr[ζ] > ε/10. (Messages whose C-output distribution stays within ε/10 of μ's
  on the relevant value can contribute at most ≈ p_E + 13ε/10 to C's overall success; since C succeeds
  above p_E + ε, the remaining good-message mass must exceed ε/10.)
- **Sampling lemma.** For a good ζ, drawing s = O(k^c/ε²) fresh encryptions of ζ and of μ, running C, and
  comparing empirical output frequencies finds a witnessing v with |r_v^ζ − r_v^μ| > ε/20 w.h.p. (weak law
  of large numbers; E is public so encryptions are samplable).
Finder: pick random μ, ζ; with prob ≥ ε/10, ζ is good; find v; output (m₀,m₁)=(ζ,μ). Tapper: output
[C(E, x)=v]. It fires at rate r_v^ζ on ζ and r_v^μ on μ, gap > ε/20 — indistinguishability fails. ∎

## Why encryption must be randomized

If E is deterministic: (i) B(m) = "E(m) is even" is a nonconstant predicate on the plaintext computable
from the ciphertext — partial information leaks; (ii) equal plaintexts give equal ciphertexts, so message
equality leaks; (iii) with the public key the adversary encrypts any guess and compares to the challenge,
distinguishing trivially. Hence a deterministic scheme is never indistinguishable, hence never
semantically secure. Randomizing (many ciphertexts per plaintext) removes all three.

## Unapproximable trapdoor predicate (the building block)

A family {B_i : Ω_i → {0,1}} is an **unapproximable trapdoor predicate** if:
- *(both bit-values samplable)* one can efficiently sample a uniform x with B_i(x)=0 or with B_i(x)=1;
- *(trapdoor)* there is a secret σ(i) with which, given any z, B_i(z) is poly-time computable;
- *(unapproximable)* no poly-size circuit computes B_i correctly on more than a 1/2 + 1/poly fraction;
- *(constructible)* the pair (i, σ(i)) is samplable in poly time, essentially uniformly.

**Bit-by-bit PKC.** Public key i. Encrypt bit b: send a random x with B_i(x) = b. Encrypt a message:
encrypt each bit independently into a tuple. Decrypt: apply the trapdoor to each component.

**Theorem (hybrid). The bit-by-bit PKC is indistinguishable** (hence semantically secure). *Proof.* If a
tapper distinguishes l-bit messages m₀, m₁ with advantage > ε, view messages as vertices of {0,1}^l
labeled by T's 1-frequency; a one-bit-flip path from m₀ to m₁ (length ≤ l) must contain adjacent words
s, t (differing only in position d) with frequency gap > ε/l. Sampling the other l−1 positions from their
agreed-bit encryption distributions and splicing the challenge into position d turns T into a circuit
approximating B (advantage > ε/(5l)), contradicting unapproximability. Frequencies are estimated by
sampling (weak law). ∎ Per-bit security lifts to the whole message at a 1/l cost.

## Quadratic residuosity gives such a predicate

n = p·q, distinct k-bit primes; Z_n^*(+1) = Jacobi-+1 elements (half QR, half QNR). Predicate
**Q_n(x) = [x is a quadratic residue mod n]** on Z_n^*(+1).
- *Trapdoor:* with (p,q), test residuosity mod p and mod q by Euler's criterion + CRT, O(k³). Without
  factorization the Jacobi symbol is poly-time but Jacobi-+1 leaves QR-vs-QNR undecided (the QR Problem).
- *Both bit-values samplable:* a random square x² is a uniform QR; publishing one Jacobi-+1 nonresidue y
  (a *pseudosquare*), y·x² is a uniform Jacobi-+1 QNR.
- *Unapproximable under the QR assumption*, via **random self-reducibility**: an oracle right on a 1/2+ε
  fraction yields a near-perfect decider, so QR is everywhere-hard-or-everywhere-easy. To test a challenge
  z, form z·x_i for random squares x_i — if z is a QR these are random QRs, if a QNR they are random QNRs
  (multiplication by a square is a residue-class-preserving bijection on Z_n^*(+1)); calibrate the
  oracle's "QR" rate on genuine squares and compare. Hence no poly-size circuit approximates Q_n.
- *Publishing y is safe:* revealing a random Jacobi-+1 nonresidue leaves QR exactly as hard.
- *Constructible:* 4k coin flips give candidate primes p,q (Prime Number Theorem + primality test) and a
  pseudosquare y, uniformly, in random poly time.

## The Goldwasser–Micali cryptosystem

- **KeyGen:** k-bit primes p, q; n = pq; y a Jacobi-+1 nonresidue (pseudosquare). Public (n, y); secret (p, q).
- **Encrypt bit b:** random x ∈ Z_n^*; send e = y^b · x² mod n (b=0 → QR; b=1 → Jacobi-+1 QNR). Whole
  message → tuple of such e's. One plaintext bit expands to k ciphertext bits.
- **Decrypt e:** with (p, q), b = 0 if e is a QR mod n (QR mod p and mod q), else b = 1.

Semantically secure under the QR assumption; any break decides quadratic residuosity. (Ciphertexts are
re-randomizable by multiplying each component by a fresh random square — the same self-reducibility.)

```python
from math import gcd
import random

def jacobi(a, n):
    a %= n; result = 1
    while a:
        while a % 2 == 0:
            a //= 2
            if n % 8 in (3, 5): result = -result
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3: result = -result
        a %= n
    return result if n == 1 else 0

def is_qr_mod_prime(a, p):                       # Euler's criterion
    return pow(a % p, (p - 1) // 2, p) == 1

def keygen(p, q):                                # public (n, y), secret (p, q)
    n = p * q
    while True:                                  # y: nonresidue mod BOTH primes => Jacobi(y/n) = +1
        y = random.randrange(2, n)
        if gcd(y, n) == 1 and not is_qr_mod_prime(y, p) and not is_qr_mod_prime(y, q):
            return (n, y), (p, q)

def encrypt_bit(b, n, y):                        # b=0 -> x^2 (QR); b=1 -> y*x^2 (Jacobi-+1 QNR)
    x = random.randrange(1, n)
    while gcd(x, n) != 1: x = random.randrange(1, n)
    return (pow(y, b, n) * pow(x, 2, n)) % n

def decrypt_bit(e, p, q):                        # QR mod n iff QR mod p and mod q
    return 0 if (is_qr_mod_prime(e, p) and is_qr_mod_prime(e, q)) else 1

def encrypt(bits, pub): n, y = pub;        return [encrypt_bit(b, n, y) for b in bits]
def decrypt(ct, sec):   p, q = sec;        return [decrypt_bit(e, p, q) for e in ct]
```
