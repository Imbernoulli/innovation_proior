# Keccak and the sponge construction (SHA-3)

## Problem

Build a variable-input, variable-output cryptographic hash whose only deviation from a random oracle is quantifiable and can be pushed arbitrarily small with one parameter — in particular, one that is free of length-extension and does not depend on engineering a collision-resistant keyed compression function (the layer that broke in MD5 and SHA-1).

## Key idea: the sponge

Take a single fixed, public permutation `f` on `b = r + c` bits. Split the state into an outer **rate** of `r` bits and an inner **capacity** of `c` bits. The message only ever enters the outer `r` bits; the inner `c` bits are never set by input and never output.

- **Absorb.** `S = 0^b`; pad the message and split into `r`-bit blocks `P_i`; for each block `S ← f(S ⊕ (P_i ‖ 0^c))`.
- **Squeeze.** Output the outer `r` bits of `S`; for more output, `S ← f(S)` and read the next `r` bits; truncate to the requested length `d`.

Because the digest exposes only outer bits (truncated to `d ≤ r`) and never the `c` inner bits, an attacker cannot recover the full state and so cannot resume `f` — **length-extension is structurally absent**, not patched.

### Why a permutation, and the security knob

A permutation needs no key schedule (unlike a Davies–Meyer block-cipher compression function) yet keeps all block-cipher design intuition. Modeling `f` as a random permutation, the sponge is indistinguishable/indifferentiable from a random oracle except via an **inner collision** (two distinct absorbed paths reaching the same inner `c`-bit value). After `N` calls to `f` and `f^{-1}`,

  P(inner collision) ≈ 1 − ∏_{i=1}^{N}(1 − i/2^c) ≈ N(N+1) / 2^{c+1}.

The advantage depends on `c` only — never on `r`. So security ~`2^{c/2}`, throughput tuned by `r`, with `r + c = b` fixed. Collision/(2nd)preimage resistance are obtained from an easily invertible `f` because the secret is the hidden inner state, not `f`. Choosing **`c = 2d`** matches a `d`-bit truncated random oracle (collisions `2^{d/2}`, preimages `2^d`).

### Padding and domain separation

- **pad10*1** (multi-rate): append `1`, then `j = (−m−2) mod r` zeros, then `1`. The closing `1` makes padding injective and lets one `f` serve different rates safely; at least one `f`-call always occurs.
- A short **suffix** before pad10*1 separates domains: hashes append `01`, the extendable-output functions append `1111`, so a hash and an XOF can never collide on the same message.

## Keccak-f[1600]

State `A[x][y][z]`, `0 ≤ x,y < 5`, `0 ≤ z < w=64`: 5×5 lanes of 64 bits, `b = 1600`. With `l = log2 w = 6`, the round count is `n_r = 12 + 2l = 24`. Each round is `R = ι ∘ χ ∘ π ∘ ρ ∘ θ`:

- **θ (diffusion).** `C[x,z] = ⊕_y A[x,y,z]`; `D[x,z] = C[x−1,z] ⊕ C[x+1,z−1]`; `A'[x,y,z] = A[x,y,z] ⊕ D[x,z]`. XOR each bit with the parities of two neighbouring columns (one shifted in `z`). Cheap; branch number only 4 (the column-parity kernel slips through), addressed by π rather than by making θ heavier.
- **ρ (inter-slice dispersion).** Rotate each lane `(x,y)` along `z` by the triangular offset `(t+1)(t+2)/2 mod 64` marched around the lanes; lane `(0,0)` offset 0.
- **π (dispersion / long-term diffusion).** `A'[x,y] = A[(x+3y) mod 5, x]`. Shuffles lane positions so column-kernel trails cannot persist round to round.
- **χ (nonlinearity).** Per row of 5: `A'[x] = A[x] ⊕ ((¬A[x+1]) ∧ A[x+2])`. The only nonlinear step; algebraic degree 2; invertible.
- **ι (symmetry breaking).** XOR an LFSR-generated round constant into lane `(0,0)`, defeating the `z`-translation symmetry and slide attacks; one round of θ/χ diffuses it to all lanes.

θ comes first so the round mixes the inner (capacity) and outer (rate) parts early in the sponge; the order of the rest is essentially free.

## Code

Faithful to the public domain implementation (Gilles Van Assche, CC0). Known-answer outputs: `SHA3-256("") = a7ffc6f8…434a` and `SHA3-256("abc") = 3a985da7…1532`.

```python
def rol64(a, n):
    return ((a >> (64 - (n % 64))) | (a << (n % 64))) & ((1 << 64) - 1)

def keccak_f1600_on_lanes(lanes):
    R = 1
    for _ in range(24):                                   # n_r = 12 + 2l = 24
        # θ
        C = [lanes[x][0] ^ lanes[x][1] ^ lanes[x][2] ^ lanes[x][3] ^ lanes[x][4]
             for x in range(5)]
        D = [C[(x + 4) % 5] ^ rol64(C[(x + 1) % 5], 1) for x in range(5)]
        lanes = [[lanes[x][y] ^ D[x] for y in range(5)] for x in range(5)]
        # ρ and π (fused)
        (x, y) = (1, 0)
        current = lanes[x][y]
        for t in range(24):
            (x, y) = (y, (2 * x + 3 * y) % 5)
            (current, lanes[x][y]) = (lanes[x][y], rol64(current, (t + 1) * (t + 2) // 2))
        # χ
        for y in range(5):
            T = [lanes[x][y] for x in range(5)]
            for x in range(5):
                lanes[x][y] = T[x] ^ ((~T[(x + 1) % 5]) & T[(x + 2) % 5])
        # ι
        for j in range(7):
            R = ((R << 1) ^ ((R >> 7) * 0x71)) % 256
            if (R & 2):
                lanes[0][0] ^= (1 << ((1 << j) - 1))
    return lanes

def load64(b):  return sum(b[i] << (8 * i) for i in range(8))
def store64(a): return list((a >> (8 * i)) % 256 for i in range(8))

def keccak_f1600(state):
    lanes = [[load64(state[8 * (x + 5 * y):8 * (x + 5 * y) + 8]) for y in range(5)]
             for x in range(5)]
    lanes = keccak_f1600_on_lanes(lanes)
    state = bytearray(200)                                # b = 1600 bits = 200 bytes
    for x in range(5):
        for y in range(5):
            state[8 * (x + 5 * y):8 * (x + 5 * y) + 8] = store64(lanes[x][y])
    return state

def sponge(rate, capacity, input_bytes, delimited_suffix, output_len):
    assert rate + capacity == 1600 and rate % 8 == 0
    rate_bytes = rate // 8
    state = bytearray(200)
    off = block = 0
    # absorb
    while off < len(input_bytes):
        block = min(len(input_bytes) - off, rate_bytes)
        for i in range(block):
            state[i] ^= input_bytes[i + off]
        off += block
        if block == rate_bytes:
            state = keccak_f1600(state); block = 0
    # pad10*1 with domain suffix folded into the first padding byte
    state[block] ^= delimited_suffix
    if (delimited_suffix & 0x80) and block == rate_bytes - 1:
        state = keccak_f1600(state)
    state[rate_bytes - 1] ^= 0x80
    state = keccak_f1600(state)
    # squeeze
    out = bytearray()
    while output_len > 0:
        block = min(output_len, rate_bytes)
        out += state[0:block]; output_len -= block
        if output_len > 0:
            state = keccak_f1600(state)
    return out

def sha3_224(m):    return sponge(1152, 448,  m, 0x06, 28)
def sha3_256(m):    return sponge(1088, 512,  m, 0x06, 32)
def sha3_384(m):    return sponge(832,  768,  m, 0x06, 48)
def sha3_512(m):    return sponge(576,  1024, m, 0x06, 64)
def shake128(m, n): return sponge(1344, 256,  m, 0x1F, n)
def shake256(m, n): return sponge(1088, 512,  m, 0x1F, n)
```
