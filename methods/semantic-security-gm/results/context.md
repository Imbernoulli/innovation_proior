# Context: defining secure public-key encryption

## Research question

Public-key encryption exists as a concept and as concrete schemes, but no one has said precisely what
it means for an encryption scheme to be *secure*. The working notion inherited from the founders is:
an encryption is secure if a computationally bounded adversary, seeing the ciphertext, cannot recover
the whole plaintext (or the secret key). The goal here is to expose how weak that is and to replace it
with a definition that captures the real-world demand — that the ciphertext leak *nothing useful* — and
then to exhibit a scheme provably meeting it under a clean number-theoretic assumption.

The demand is sharpened by a concrete application: playing card games (mental poker) over a telephone
line, where the cards are exchanged as ciphertexts. If even a single bit of partial information about a
hidden card — its color, its suit, whether it equals a card already seen — can be computed from the
ciphertext, the game is compromised even though no one ever recovers the full card. So the requirement a
solution must meet is strong: hide *every* efficiently computable function of the plaintext, for *every*
message distribution, not merely the identity of the message, and not only on random-looking messages.

## Background

**The public-key idea and its security notion (Diffie–Hellman, 1976).** Diffie and Hellman introduced
the public-key cryptosystem: each user publishes an enciphering key E and keeps a deciphering key D such
that computing D from E is computationally infeasible. Anyone can send by computing E(m); only the
holder of D recovers m. Their analysis frames security as a *work factor*: a system is "computationally
secure" if breaking it (recovering plaintext/key) costs a finite but impossibly large amount of
computation. They classify attacks (ciphertext-only, known-plaintext, chosen-plaintext) and note that
the one-time pad is the only unconditionally (information-theoretically) secure system in common use,
but its key must be as long as the message — Shannon's perfect secrecy, where the a-posteriori message
probabilities equal the a-priori ones, is unachievable with short keys. The entire framing is about the
infeasibility of *inverting*; it says nothing about partial information.

**Trapdoor functions and one-way functions.** The implementation primitive is a *trapdoor one-way
function*: f easy to compute, hard to invert, but invertible given secret trapdoor information. A
one-way function is easy forward, infeasible backward (e.g. modular exponentiation g^x mod p, whose
inverse is the discrete logarithm). These are the objects on which public-key encryption is built.

**Number theory of quadratic residues.** For an odd prime p, half of Z_p^* are quadratic residues
(squares) and half are not; residuosity mod a prime is decided in polynomial time by Euler's criterion:
for a in Z_p^*, a^((p−1)/2) is +1 mod p exactly on residues and −1 mod p on nonresidues. For a composite
n = p·q, the *Jacobi symbol* (a/n) = (a/p)(a/q) is computable in polynomial time *without* knowing the
factorization. If (a/n) = −1 then a is certainly a nonresidue; but if (a/n) = +1, then either a is a
residue modulo both primes, hence a residue mod n, or a is a nonresidue modulo both primes, hence a
Jacobi-+1 nonresidue. *Deciding which* — the Quadratic Residuosity Problem — has no known efficient
algorithm when the factorization of n is unknown.
It is one of the four main algorithmic problems Gauss singled out in the *Disquisitiones Arithmeticae*
(1801). Exactly half of the Jacobi-+1 elements are residues and half nonresidues. With the factorization,
residuosity mod n is decided in O(|n|^3) by testing mod p and mod q and combining via the Chinese
Remainder Theorem.

**Known leakage of single bits from one-way functions.** It is documented that hardness of a one-way
function on its whole input says nothing about a *designated bit* of that input. For g^x mod p the last
bit of x leaks for free: x is even iff g^x is a quadratic residue mod p, and residuosity mod a prime is
poly-time testable. So embedding a secret bit inside the input of a one-way function does not protect it.

## Baselines

**RSA (Rivest–Shamir–Adleman, 1978).** A user publishes n = p·q (two large secret primes) and an
exponent s coprime to φ(n); encryption is E(m) = m^s mod n, decryption m = c^d mod n with s·d ≡ 1 mod
φ(n). E is a trapdoor one-way *permutation* of Z_n^*. The scheme is deterministic and bijective. Gaps it
leaves open: (1) being hard to invert on random inputs does not preclude easy inversion on *structured*
inputs — a function hard on generic x can be easy on ASCII English. (2) It does not preclude leaking
*partial* information: for integer-valued deterministic encryption, any visible output bit that varies
with m becomes a plaintext predicate read directly from the ciphertext, and Lipton showed that in
RSA-based mental poker a bit that must stay hidden is easily computed. (3) Being deterministic,
identical plaintexts yield identical
ciphertexts, so *equality* of messages always leaks, and with the public key an adversary can encrypt
any candidate and compare. No proof exists that decoding is hard without assumptions on the message
space.

**Rabin (1979).** Choose s = 2: E(x) = x^2 mod n. Decryption takes square roots mod n (possible with the
factorization, via CRT). Rabin proved a tight link to factoring: if one could extract a square root for
even a 1/log n fraction of quadratic residues, one could factor n in random polynomial time (because two
square roots x, y of the same residue with x ≠ ±y give gcd(n, x±y) as a factor). The gap: this
"break ⇒ factor" guarantee holds only when the message set is *dense* in Z_n^*. If messages are sparse,
a decoder for messages never yields a second square root inside the message set, so it does not factor —
the equivalence to factoring evaporates, and again partial information is unaddressed. Rabin's E is also
deterministic (4-to-1), so the parity-leak and equality-leak objections apply.

**Attempts to send one bit securely with a trapdoor function.** Two natural constructions fail. (i)
Embed the secret bit as the i-th bit of an otherwise-random r and send E(r): a one-way E can leak a
specific bit (the discrete-log last-bit example above). (ii) Place the bit at a random position i in a
100-bit x, with the first 7 bits encoding i and the rest random: if E leaks those 7 bits and any one of
the remaining bits, the adversary recovers the message bit with probability 1/2 + 1/2·(1/93). The lesson:
infeasibility of recovering all of x grants no protection to one designated bit of x.

## Evaluation settings

The yardstick is the adversary model and the message regime against which any proposed definition and
scheme must hold. The adversary is a passive line-tapper who knows the message space and its probability
distribution, knows the (public) encryption algorithm, intercepts the ciphertext, and computes. The
computational model for adversary and parties is polynomially bounded — polynomial-time Turing machines
or, to make "after the key is fixed" attacks well-defined, families of polynomial-size Boolean circuits
indexed by a security parameter k. Security must hold for *every* message space with *any* probability
distribution (no density or randomness assumption on messages), and for *every* function of the plaintext
the adversary might try to compute (the identity, a predicate, a hash — recursive or not). The standing
hardness benchmarks are: factoring n = p·q for large primes (the assumed-hard problem behind RSA/Rabin),
and deciding quadratic residuosity mod a composite of unknown factorization on Jacobi-+1 inputs.
Shannon's perfect-secrecy criterion (a-posteriori message probabilities equal to the a-priori ones)
is the information-theoretic reference point, but it is unachievable with short keys against the
unbounded adversary it presumes.

## Code framework

A public-key scheme over the integers mod n can already rely on standard number-theoretic primitives
(Jacobi symbol, Euler's criterion, CRT square roots) and the usual public-key harness (key generation,
encrypt, decrypt). The encoding rule and the corresponding key material are left open.

```python
from math import gcd

# --- primitives that already exist (number theory) ---
def jacobi(a, n): ...          # poly-time, no factorization needed; +1/-1/0
def is_qr_mod_prime(a, p): ... # Euler's criterion: a^((p-1)/2) mod p == 1

# --- public-key harness ---
def keygen(k):
    # pick two k-bit primes p, q; n = p*q.
    # TODO: fill in the key material.
    public_key = None   # TODO
    secret_key = None   # TODO
    return public_key, secret_key

def encrypt(message, public_key):
    # TODO: choose the encoding rule.
    pass

def decrypt(ciphertext, secret_key):
    # invert encrypt using the trapdoor in secret_key
    pass

# --- the missing piece: what does "secure" even mean? ---
def is_secure(scheme):
    # TODO: state the security goal precisely enough to prove a scheme meets it.
    #       "adversary cannot recover the whole plaintext" is the inherited notion;
    #       it is the thing we suspect is too weak.
    pass
```
