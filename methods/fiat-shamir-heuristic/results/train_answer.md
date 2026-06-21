The problem is how to turn an interactive identification protocol into something that can be checked later by anyone, without revealing the secret held by the card or relying on a live verifier. Interactive zero-knowledge proofs solve the leakage issue by having the prover commit first, then answer a random challenge from the verifier. The trouble is that this leaves no durable evidence: a verifier who took part in the conversation can later write a fake-looking transcript by choosing its own questions and answers, so third parties cannot trust the transcript as proof. A static certificate avoids interaction but usually gives away the witness, and simply hashing a completed transcript only labels it; it does not recreate the moment when the prover is bound to a commitment before seeing the challenge.

The right move is to preserve only the irreplaceable job of the verifier, which is to supply unpredictable randomness after the commitment is fixed. Everything else the verifier does can be moved into a public recomputation. So the challenge should be generated from the fixed commitment and the public statement by a function that looks random and is deterministic for later verification. In the signature setting the message is also included so the proof is bound to one specific message and cannot be reused elsewhere.

The method is the Fiat-Shamir heuristic. Given a public-coin three-message proof with transcript (commitment, challenge, response), replace the verifier's sampled challenge with the output of a public hash or random function applied to the domain separator, the public statement, and the commitment. The proof becomes just the commitment and response. To verify, anyone recomputes the challenge from the same inputs and runs the original verifier predicate. For signatures derived from identification schemes, the public key and the message are also hashed together with the commitment, so the signature is tied to that exact message.

In the original Fiat-Shamir scheme based on modular square roots, the public identity values are derived as u_j from the identity string and index, and the card stores secrets s_j satisfying s_j^2 * u_j = 1 modulo the public composite n. The signer samples r_i, computes commitments z_i = r_i^2 mod n, derives the challenge matrix from the first k*t bits of a public function applied to the message and the commitments, and then computes responses y_i = r_i * product of s_j over the selected entries. A verifier reconstructs the commitments from the responses and challenge matrix and checks that the same public function reproduces the challenge matrix. The security intuition is unchanged from the interactive protocol: because the commitment is fixed before the challenge is known, being able to answer two different challenges for the same commitment would reveal the hidden square roots.

```python
import hashlib
import os
import secrets

def hash_challenge(domain: bytes, statement: bytes, commitment: bytes, message: bytes = b"") -> int:
    """Deterministically derive a challenge from public inputs."""
    h = hashlib.sha256(domain + statement + commitment + message)
    return int.from_bytes(h.digest(), "big")

# --- Example with a small proof-of-knowledge of discrete logarithm ---
# Public parameters: prime p, generator g, public key y = g^x mod p.
# Prover knows x.

p = 0xFFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7EDEE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F83655D23DCA3AD961C62F356208552BB9ED529077096966D670C354E4ABC9804F1746C08CA18217C32905E462E36CE3BE39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF6955817183995497CEA956AE515D2261898FA051015728E5A8AACAA68FFFFFFFFFFFFFFFF
q = (p - 1) // 2
g = 2

# Prover's secret x and public key y.
x = secrets.randbelow(q)
y = pow(g, x, p)

def prove(message: bytes) -> tuple[int, int, int]:
    # Commitment.
    r = secrets.randbelow(q)
    a = pow(g, r, p)

    # Challenge derived from public inputs and message.
    domain = b"fiat-shamir-dlog-example"
    commitment_bytes = a.to_bytes((a.bit_length() + 7) // 8, "big")
    statement_bytes = y.to_bytes((y.bit_length() + 7) // 8, "big")
    c = hash_challenge(domain, statement_bytes, commitment_bytes, message) % q

    # Response: z = r + c*x mod q.
    z = (r + c * x) % q
    return a, c, z

def verify(message: bytes, a: int, c: int, z: int) -> bool:
    # Recompute challenge from public information and commitment.
    domain = b"fiat-shamir-dlog-example"
    commitment_bytes = a.to_bytes((a.bit_length() + 7) // 8, "big")
    statement_bytes = y.to_bytes((y.bit_length() + 7) // 8, "big")
    c_expected = hash_challenge(domain, statement_bytes, commitment_bytes, message) % q
    if c != c_expected:
        return False

    # Check g^z == a * y^c (mod p).
    left = pow(g, z, p)
    right = (a * pow(y, c, p)) % p
    return left == right

# Demo.
msg = b"hello, world"
a, c, z = prove(msg)
print("Verification:", verify(msg, a, c, z))
print("Bad message rejected:", verify(b"other message", a, c, z))
```
