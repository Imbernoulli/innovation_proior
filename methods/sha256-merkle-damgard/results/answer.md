# Merkle–Damgård collision-resistant hashing, instantiated as SHA-256

## Problem

Build one public, keyless function H mapping a message of *any* length to a
fixed n-bit digest such that finding any pair M ≠ M′ with H(M) = H(M′) is
infeasible. The adversary knows H exactly. Two difficulties: collisions exist by
pigeonhole (so the goal is "hard to find", proved against a clean assumption,
not "none"); and we can only design and analyze functions on a *fixed* number of
input bits, never on unbounded input directly. The generic ceiling is the
birthday bound — ≈ 2^{n/2} work for a collision (vs. ≈ 2^n for (second-)preimage)
— so a construction is judged by whether its best attack stays at that ceiling,
and 128-bit collision security forces a 256-bit digest.

## Key idea

Two pieces that compose.

1. **Length-encoded chaining (Merkle–Damgård).** Posit a fixed-shape
   compression function f : {0,1}^{n+b} → {0,1}^n. Chain it from a fixed IV,
   folding b message bits per call: H_0 = IV, H_i = f(H_{i-1} ‖ M_i), output H_L.
   Claim: if f is collision-resistant, so is H. Proof (contrapositive): given an
   H-collision x ≠ x′, run both chains and walk *backward* from the equal
   outputs. At each step either the two f-inputs differ — an f-collision, done —
   or they match and we recurse. The walk cannot match all the way to the shared
   IV unless x = x′, *provided distinct messages give distinct padded block
   sequences*. Plain zero-fill violates this (a message ending in d zeros and one
   padded with d zeros become identical), so the padding must encode length: Pad
   is injective with (i) M a prefix of Pad(M), (ii) equal length ⇒ equal padded
   length, (iii) different length ⇒ different final block. Condition (iii) — the
   length field, "MD-strengthening" — is the hinge the reduction turns on.

2. **A fast collision-resistant f from a block cipher (Davies–Meyer).** The bare
   cipher map f(H, M) = E_M(H) is invertible in H — pick any M, set
   H = E_M^{-1}(c) to hit target c — hence trivially collidable. Re-inject the
   input after encryption: f(H, M) = E_M(H) ⊕ H. Now H is tangled into the
   output and cannot be decrypted away. In the ideal-cipher model (each key names
   an independent uniform random permutation) each of q queries reveals a hash
   value near-uniform over ≈ 2^l values (l = the cipher's block width = the digest
   length n), so Pr[collision] ≤ q²/2^l: a collision needs q ≈ 2^{l/2} queries, the
   birthday ceiling. Chaining this f gives a full hash collision-resistant to ≈ 2^{n/2}.

SHA-256 instantiates this with n = 256, b = 512: E is a 64-round permutation on
the 256-bit state keyed by the 512-bit block, and the ⊕H feed-forward is realized
as word-wise addition mod 2^32.

## Algorithm

State = eight 32-bit words; all arithmetic mod 2^32 (the carry chain is the only
cheap cross-bit *nonlinearity* — XOR and rotation are GF(2)-linear).

- **Padding.** Append a 1 bit (0x80 byte), then zeros until the length ≡ 448 mod
  512, then the 64-bit big-endian message bit-length L. Different L ⇒ different
  final block (condition iii). Defined for |M| < 2^64 bits.
- **IV** H^(0): first 32 bits of the fractional parts of √p for the first 8
  primes (nothing-up-my-sleeve).
- **Round constants** K_0…K_63: first 32 bits of the fractional parts of the
  cube roots of the first 64 primes (a distinct public recipe from the IV's square
  roots; distinct per round to defeat slide/fixed-point attacks).
- **Message schedule.** Sixteen block words W_0…W_15, then for t = 16…63
  W_t = σ1(W_{t-2}) + W_{t-7} + σ0(W_{t-15}) + W_{t-16}, with
  σ0(x) = ROTR^7 x ⊕ ROTR^18 x ⊕ SHR^3 x, σ1(x) = ROTR^17 x ⊕ ROTR^19 x ⊕ SHR^10 x.
  The *acyclic shift* (drops bits instead of wrapping) breaks rotational
  symmetry so no rotate-the-message differential survives — σ stays bijective but
  is no longer rotation-commuting; two recent/old taps avalanche each block word
  across many rounds.
- **Round.** Working register a…h from the chaining value; per round
  T1 = h + Σ1(e) + Ch(e,f,g) + K_t + W_t, T2 = Σ0(a) + Maj(a,b,c), then slide
  the register injecting d+T1 into the e-track and T1+T2 into the a-track:
  Σ0(x) = ROTR^2 x ⊕ ROTR^13 x ⊕ ROTR^22 x, Σ1(x) = ROTR^6 x ⊕ ROTR^11 x ⊕ ROTR^25 x
  (three rotations, no shift — here a bijective full-word stir), with
  Ch(e,f,g) = (e∧f) ⊕ (¬e∧g) and Maj(a,b,c) = (a∧b) ⊕ (a∧c) ⊕ (b∧c) the bitwise
  nonlinear mixers.
- **Davies–Meyer feed-forward.** Add the *incoming* state back word-wise into the
  64-round output. This is the ⊕H in the word ring — it tangles input into output
  so f is non-invertible. Without it the round map is a permutation and f
  collapses back to the Rabin failure.
- **Iterate** f from the IV across all padded blocks; emit the eight final words
  concatenated = 256-bit digest.

The digest *is* the entire final chaining value, so revealing H(M) and |M| lets
an adversary resume the state and compute H(pad(M) ‖ Y) without M — the
length-extension property. Not a collision break (the theorem stands); it is
remedied outside f by nesting/HMAC or truncation.

## Code

```python
# IV H^(0): first 32 bits of frac(sqrt(p)) for the first 8 primes (nothing-up-my-sleeve).
H0 = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
      0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]

# Round constants K_t: first 32 bits of frac(cbrt(p)) for the first 64 primes.
K = [0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
     0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
     0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
     0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
     0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
     0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
     0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
     0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2]

MASK = 0xffffffff
def rotr(x, n): return ((x >> n) | (x << (32 - n))) & MASK
def shr(x, n):  return x >> n

# Sigma: three rotations, no shift -- bijective intra-word stir (the round).
def Sigma0(x): return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22)
def Sigma1(x): return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25)
# sigma: two rotations + one shift -- the shift breaks rotational symmetry (schedule).
def sigma0(x): return rotr(x, 7)  ^ rotr(x, 18) ^ shr(x, 3)
def sigma1(x): return rotr(x, 17) ^ rotr(x, 19) ^ shr(x, 10)

# The only bitwise nonlinear mixers.
def Ch(x, y, z):  return (x & y) ^ (~x & z)
def Maj(x, y, z): return (x & y) ^ (x & z) ^ (y & z)

def pad(message: bytes) -> bytes:
    # MD-strengthening: 1-bit separator, zeros, 64-bit big-endian bitlen.
    bit_len = (len(message) * 8) & ((1 << 64) - 1)
    p = message + b"\x80"
    p += b"\x00" * ((56 - len(p) % 64) % 64)   # fill so len == 448 mod 512
    p += bit_len.to_bytes(8, "big")            # different lengths => different last block
    return p

def compress(state, block: bytes):
    # message expansion: 16 block words -> 64 schedule words, avalanching each word
    w = [int.from_bytes(block[4*i:4*i+4], "big") for i in range(16)]
    for t in range(16, 64):
        w.append((sigma1(w[t-2]) + w[t-7] + sigma0(w[t-15]) + w[t-16]) & MASK)
    a, b, c, d, e, f, g, h = state             # working register = incoming chaining value
    for t in range(64):                        # 64 rounds of the keyed permutation E_M(.)
        t1 = (h + Sigma1(e) + Ch(e, f, g) + K[t] + w[t]) & MASK   # e-track injection
        t2 = (Sigma0(a) + Maj(a, b, c)) & MASK                    # a-track injection
        h, g, f, e, d, c, b, a = g, f, e, (d + t1) & MASK, c, b, a, (t1 + t2) & MASK
    # Davies-Meyer feed-forward: add the input state back in => non-invertible f.
    return [(x + y) & MASK for x, y in zip(state, (a, b, c, d, e, f, g, h))]

def sha256(message: bytes) -> str:
    state = list(H0)                           # chaining starts at the fixed IV
    data = pad(message)
    for off in range(0, len(data), 64):        # iterate f block by block (Merkle-Damgard)
        state = compress(state, data[off:off+64])
    return "".join(f"{x:08x}" for x in state)   # output = final chaining value (256 bits)


if __name__ == "__main__":
    assert sha256(b"") == \
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert sha256(b"abc") == \
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    print(sha256(b"abc"))
```
