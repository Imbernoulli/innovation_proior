The central difficulty in practical cryptography has always been key distribution. Conventional symmetric ciphers assume that the sender and receiver already share a secret key, which means every pair of users must arrange a separate, secure channel to exchange that key before they can communicate. In a network of n users this creates roughly n² pairwise keys, and for strangers who want to transact immediately there is simply no time for a courier or registered mail. The same obstacle appears for digital signatures: anything encrypted under a shared secret can be forged by anyone who holds that secret, so shared-key authentication cannot settle a dispute between sender and receiver. Existing partial fixes each guard only one flank. Challenge-response schemes defeat replay by an eavesdropper but collapse if the secret is captured, while one-way password hashes survive capture of the password file but do nothing against an intercepted login message. What is missing is a way for two parties to agree on a secret value using only messages sent in the clear, with security resting not on a pre-shared key but on the gap between easy and hard computation.

The solution is Diffie-Hellman key exchange. It works in a finite prime field GF(q), where q is a large prime and α is a primitive element whose powers cycle through every nonzero field element. Each user draws a secret exponent X uniformly from {1, ..., q−1}, keeps it private, and publishes the public value Y = α^X mod q. The forward direction, computing Y from X, is cheap: repeated squaring needs at most about 2·log₂ q multiplications. The backward direction, recovering X from Y, is the discrete logarithm problem, and the best known algorithm needs roughly q^(1/2) operations. For a b-bit prime this is an exponential gap, around 2^(b/2) work for the attacker against only O(b) work for the legitimate user. No unconditional security is possible here, because the public data determines the secret among a finite set and an infinitely powerful adversary could always search it; the guarantee is instead computational, and it is vastly stronger than the merely quadratic advantage offered by earlier schemes such as Merkle's puzzles.

The algebraic structure that makes agreement possible is the commutativity of exponentiation. When users i and j want to share a key, i takes j's public value Y_j and raises it to i's own secret X_i, obtaining Y_j^{X_i} = (α^{X_j})^{X_i} = α^{X_i X_j} mod q. Symmetrically, j computes Y_i^{X_j} = (α^{X_i})^{X_j} = α^{X_i X_j} mod q. Both parties arrive at the same field element K_ij = α^{X_i X_j} mod q even though neither ever transmits their secret exponent. This shared value can then be used directly as the key for an ordinary fast symmetric cipher. An eavesdropper who sees q, α, Y_i, and Y_j is left wanting K_ij; as far as anyone knows, the only way to reach it is to take a discrete logarithm of one of the public values first, and that is the problem we have chosen to make infeasible. The security depends on choosing q large enough, typically hundreds to about a thousand bits, so that 2^(b/2) lies outside any realistic budget. It is also essential that fresh randomness comes from both sides: if only one party contributed a secret, the eavesdropper would see exactly the same transmission as the intended receiver and could not be distinguished from the legitimate party.

The same conceptual framework also explains how public-key encryption and digital signatures are possible, even though the exponential exchange itself only solves key agreement. A trap-door one-way function is a function that is easy to compute in one direction and infeasible to invert unless one possesses special secret information, the trap-door. Publishing the forward function lets anyone encrypt a message to the trap-door holder, while only the holder can invert and read it; running the trap-door function backward on a message produces a signature that anyone can verify with the public forward function but that no one else could have created. Diffie-Hellman key exchange is the cleanest concrete realization of the weaker but already sufficient goal: two strangers, broadcasting only public values, end up with a shared secret over an entirely public channel.

```python
import secrets

def random_int(low, high):
    if low > high:
        raise ValueError("empty range")
    return low + secrets.randbelow(high - low + 1)

def modexp(base, exp, mod):
    # Repeated-squaring exponentiation: O(log exp) multiplications mod `mod`.
    result = 1
    base %= mod
    while exp > 0:
        if exp & 1:
            result = (result * base) % mod
        base = (base * base) % mod
        exp >>= 1
    return result

class PublicParameters:
    def __init__(self, q, alpha):
        self.q = q          # large prime; attack cost ~ 2^(b/2) for a b-bit q
        self.alpha = alpha  # primitive element of GF(q)

def generate_keypair(params):
    X = random_int(1, params.q - 1)          # secret exponent, stays on the terminal
    Y = modexp(params.alpha, X, params.q)    # public value, goes in the directory
    return X, Y

def derive_shared_secret(my_secret_X, their_public_Y, params):
    # (alpha^X_other)^X_mine = alpha^(X_mine * X_other) mod q -- same for both parties.
    return modexp(their_public_Y, my_secret_X, params.q)

# Eavesdropper has q, alpha, Y_i, Y_j but not X_i or X_j; reaching alpha^(X_i X_j)
# requires (as far as known) first taking a discrete logarithm.
```