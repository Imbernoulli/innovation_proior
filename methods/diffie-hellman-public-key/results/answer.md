# Public-key cryptography and Diffie-Hellman key exchange

## The problem

Conventional (symmetric) cryptography requires the two communicating parties to share a secret key in advance, delivered over a *separate secure channel* (courier, registered mail). In a network of `n` users this means up to `n(n-1)/2` pre-shared keys, and it forces two strangers to wait for a physical key delivery before they can transact — an impossible barrier for cheap, instantaneous digital networks. A second, apparently separate problem: a perfectly copyable digital message cannot obviously carry an unforgeable *signature* that binds its sender against even the recipient.

## The key idea

Both problems dissolve into one object once you stop demanding *unconditional* (information-theoretic) security and accept *computational* security — security resting on the gap between feasible and infeasible computation. The unifying object is a **one-way function** (easy to compute, infeasible to invert) and its enrichment, a **trap-door one-way function** (a one-way function that *does* have an easily computed inverse, but the inverse is infeasible to find from the forward algorithm alone — recoverable only with secret "trap-door" information).

This separates the capacity to **encrypt** from the capacity to **decrypt**:

- **Public-key cryptosystem.** Publish the forward (enciphering) function `E`; keep the inverse (deciphering) function `D` secret. Anyone can encrypt to you; only you can decrypt. **Signatures** follow by running the trap-door function backward on the message (`D_A(M)`, which only the holder can produce) and letting anyone verify with the public `E_A`.
- **Public-key *distribution*.** The weaker, more reachable goal: two parties agree on a *shared secret* over a fully public channel, with no prior shared key. This is realized concretely by Diffie-Hellman exponential key exchange.

### Public-key cryptosystem — definition

A pair of families `{E_K}`, `{D_K}` of invertible transformations on a finite message space `{M}`, with:

1. for every `K`, `E_K` is the inverse of `D_K`: `D_K(E_K(M)) = M`;
2. for every `K`, both `E_K` and `D_K` are easy to compute;
3. for almost every `K`, it is computationally infeasible to derive `D_K` (or any equivalent algorithm) from `E_K`;
4. for every `K`, it is feasible to compute the inverse pair `(E_K, D_K)` from `K`.

Property 3 lets `E_K` be published; property 4 lets every user generate his own pair locally from fresh randomness. Such a system is exactly a set of trap-door one-way functions.

### One-way function — definition

`f` is one-way if for any `x` in its domain `f(x)` is easy to compute, yet for almost all `y` in its range it is computationally infeasible to solve `y = f(x)` for **any** argument `x'` with `f(x') = y`. The hardness is *not* the ordinary non-uniqueness of an inverse — non-uniqueness would only *help* an attacker, since any pre-image suffices; the function must not be too degenerate.

## Diffie-Hellman exponential key exchange — the protocol

Public, agreed in advance, not secret:
- a prime `q`,
- a fixed **primitive element** `α` of `GF(q)` (its powers `α^1, α^2, ...` run through all nonzero elements `1, ..., q-1`).

The construction uses the forward/backward asymmetry of modular exponentiation:

| direction | operation | cost |
|---|---|---|
| forward (easy) | `Y = α^X mod q` | `≤ 2·log₂ q` multiplications (repeated squaring) |
| backward (hard) | `X = log_α Y mod q` (discrete log) | `~ q^(1/2)` operations, best known algorithm |

For a `b`-bit prime, legitimate work is `~2b` multiplications while the attack is `~2^(b/2)` — an **exponential** gap in `b`.

**Protocol.**
1. Each user `i` draws a secret `X_i` uniformly from `{1, ..., q-1}` and publishes `Y_i = α^{X_i} mod q` (in a directory, beside name/address). `X_i` never leaves the terminal.
2. To share a key, users `i` and `j` each compute
   `K_ij = α^{X_i X_j} mod q`:
   - user `i` computes `K_ij = Y_j^{X_i} mod q = (α^{X_j})^{X_i} = α^{X_i X_j} mod q`;
   - user `j` computes `K_ij = Y_i^{X_j} mod q = (α^{X_i})^{X_j} = α^{X_i X_j} mod q`.
   They land on the **same** value because the exponents commute: `(α^a)^b = α^{ab} = (α^b)^a`. This composition law — beyond mere one-wayness — is what lets two strangers, broadcasting only `α^{X_i}`, meet on a common secret.
3. `K_ij` is used as the key in an ordinary fast symmetric cipher.

**Security.** An eavesdropper holds `q, α, Y_i, Y_j` and wants `K_ij = α^{X_i X_j}`. If discrete logs mod `q` are easy, he recovers `X_i` from `Y_i` and the system breaks. The converse — that computing `K_ij` from `Y_i, Y_j` *requires* a discrete log — is not proved; no method is known that avoids first extracting an exponent. Security therefore rests on the (believed, not proven) hardness of the discrete logarithm; `q` must be large (hundreds to ~1000 bits) so that `2^(b/2)` is out of reach. The exchange is necessarily two-way: fresh randomness from *both* ends is what distinguishes the legitimate receiver from the eavesdropper. Because public data uniquely determines the secret among a finite set, no such system can be unconditionally secure — brute force always works given unlimited computation; the entire guarantee is computational.

This is qualitatively stronger than the prior public-channel scheme (Merkle's puzzles), whose legitimate-vs-attacker advantage is only *quadratic* (`n` vs `n²`); Diffie-Hellman's is exponential in the bit length.

## Worked example (`q = 23`, `α = 5`)

`5` is a primitive root mod `23`.

- `A` picks `X_A = 6`: `5^2 ≡ 2`, `5^4 ≡ 4`, so `Y_A = 5^6 ≡ 4·2 = 8`.
- `B` picks `X_B = 15`: `5^8 ≡ 16`, so `Y_B = 5^15 ≡ 16·4·2·5 = 640 ≡ 19`.
- `A` computes `Y_B^{X_A} = 19^6`: with `19 ≡ -4`, `19^2 ≡ 16`, `19^4 ≡ 3`, so `19^6 ≡ 3·16 = 48 ≡ 2`.
- `B` computes `Y_A^{X_B} = 8^15`: since `8 = 2^3` and `2^11 ≡ 1`, `8^15 = 2^45 ≡ 2`.
- Check: `α^{X_A X_B} = 5^90`; since `90 = 4·22 + 2` and `5^22 ≡ 1`, `5^90 ≡ 5^2 ≡ 2`.

Shared key `= 2`; the only public data was `23, 5, 8, 19`.

## Minimal implementation

```python
import secrets

def random_int(low, high):
    if low > high:
        raise ValueError("empty range")
    return low + secrets.randbelow(high - low + 1)

def modexp(base, exp, mod):
    # repeated-squaring exponentiation: O(log exp) multiplications mod `mod`
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
    # (alpha^X_other)^X_mine = alpha^(X_mine * X_other) mod q  -- same for both parties
    return modexp(their_public_Y, my_secret_X, params.q)

# Eavesdropper has q, alpha, Y_i, Y_j but not X_i, X_j; reaching alpha^(X_i X_j)
# requires (as far as known) first taking a discrete logarithm.
```
