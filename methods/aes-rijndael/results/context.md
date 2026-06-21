## Research question

The incumbent symmetric block cipher of the 1990s operates on 64-bit blocks under a key with 56 effective key bits. A 56-bit key space can be searched exhaustively: special-purpose hardware can run through all 2^56 keys and recover a key by brute force in tens of hours. A successor standard is to accept keys of at least 128 bits, putting exhaustive search beyond any foreseeable brute-force budget.

A second consideration concerns how the cipher's security relates to its public structure. The incumbent's substitution boxes were designed against criteria that were kept secret. When differential cryptanalysis became public in 1990, those S-boxes turned out to resist it well, indicating the designers had used a technique the public did not have at the time.

So the question is how to design a block cipher that accepts keys of 128 bits or more; runs fast in both software (including on 8-bit smartcards and 32-bit CPUs) and hardware; and whose resistance to the two most powerful general attacks known — differential and linear cryptanalysis — can be related to its public structure.

## Background

**Iterated block ciphers and substitution–permutation networks.** A block cipher is a keyed permutation on fixed-length blocks. The standard way to build one is to iterate a simple keyed *round* transformation many times. Shannon's two principles guide the round: *confusion* (make the relation between key and ciphertext complex) via a nonlinear substitution, and *diffusion* (spread the influence of each input bit over many output bits) via a linear mixing/permutation. A round that alternates a layer of small nonlinear substitution boxes (S-boxes) with a linear permutation is a substitution–permutation network (SPN). A round key, derived from the cipher key by a key schedule, is mixed in each round.

**Differential cryptanalysis (Biham and Shamir, 1990).** Feed the cipher pairs of plaintexts with a fixed input difference a' = a ⊕ a*; observe the output difference b'. Through any function h, a' propagates to b' with a *difference propagation probability* P_h(a',b') — the fraction of inputs for which h(a⊕a') ⊕ h(a) = b'. Across a multi-round cipher, a *differential trail* is a sequence of differences q^(0), q^(1), …, q^(r), one per round; its probability is the product of the per-round step probabilities. A linear step (the mixing layer, the key XOR) propagates a difference deterministically; only the S-boxes are probabilistic. An S-box whose input difference is zero passes a zero difference with probability 1 — it is *passive*. S-boxes with a nonzero input difference, the *active* ones, contribute a factor below 1 (each at most the S-box's maximum difference propagation probability). If some (a',b') holds over almost the whole cipher with probability far above 2^(1-n), a chosen-plaintext attack recovers key material. The incumbent was broken with roughly 2^47 chosen plaintexts.

**Linear cryptanalysis (Matsui, 1993).** Find a linear relation between a parity of plaintext bits, a parity of ciphertext bits, and a parity of key bits that holds with probability noticeably different from 1/2. The signed bias is captured by the *correlation* C = 2·Prob(·) − 1. A *linear trail* is a sequence of bit-selection patterns u^(0), …, u^(r); its correlation contribution is the product of the per-step correlations. Again only the S-boxes matter: a passive S-box (zero selection pattern in and out) contributes correlation ±1, while an active S-box contributes a factor of magnitude below 1, at most the S-box's maximum correlation. A high-correlation approximation over most of the cipher yields a known-plaintext attack; the incumbent was broken with roughly 2^43 known plaintexts.

In both analyses a trail's strength is the product over its *active* S-boxes of a per-S-box factor strictly below 1, while passive S-boxes and the linear layers contribute a factor of magnitude 1. Both attacks succeed when the best trail stays above the feasibility threshold over the rounds attacked.

**Finite field GF(2^8).** A byte {b7…b0} is read as a polynomial b7·x^7 + … + b1·x + b0 over GF(2). Addition of two bytes is coefficient-wise XOR. Multiplication is polynomial multiplication followed by reduction modulo a fixed irreducible degree-8 polynomial m(x) = x^8 + x^4 + x^3 + x + 1 (hex 0x11B), giving the field GF(2^8) of 256 elements. The map "multiply by x", i.e. by the byte {02}, is a one-bit left shift, conditionally XOR-ed with the low byte of m(x), 0x1B, when the top bit overflows. Every nonzero byte b has a unique multiplicative inverse b^(-1) with b · b^(-1) = {01}; it equals b^254, or can be found by the extended Euclidean algorithm on b(x) and m(x). These operations are cheap (shifts and XORs, or small tables) and carry algebraic structure that supports proving properties of a construction rather than only testing them.

**S-box quality measures.** For an S-box S on m-bit bundles, the *differential uniformity* is the largest number of solutions x to S(x⊕a) ⊕ S(x) = b over all nonzero a and all b; dividing by 2^m gives the maximum difference propagation probability. The *maximum correlation* (linear bias) is the largest magnitude of correlation between any nonzero input parity and any nonzero output parity. A good S-box minimizes both. Differential uniformity and maximum correlation are invariant under composing the S-box with invertible affine maps on input and output.

## Baselines

**The incumbent 56-bit-key Feistel cipher (DES, 1977).** A 16-round Feistel network on 64-bit blocks: each round splits the block in half, runs one half through a key-dependent function (expansion, key XOR, eight 6→4-bit lookup S-boxes, a bit permutation) and XORs the result into the other half. It is compact, well-studied, and its S-boxes resist differential cryptanalysis; the S-box design criteria are not public. The key is 56 bits and the block is 64 bits.

**Triple application of the incumbent (3-DES).** Run the 64-bit cipher three times with two or three keys to stretch the effective key to ~112 bits. It uses the same 64-bit block and runs roughly three times slower than a single application.

**Early "provable-resistance" SPNs.** A line of work builds an SPN from S-boxes with bounded differential uniformity together with a mixing layer chosen so that a number of S-boxes are forced active across rounds, and from this bounds trail probabilities. These demonstrate structure-derived bounds: given the S-box differential uniformity and a count of active S-boxes per fixed number of rounds, one obtains an upper bound on any trail's probability or correlation.

## Evaluation settings

The natural yardstick is resistance to public cryptanalytic attacks, expressed structurally: the guaranteed minimum number of active S-boxes over a fixed number of rounds, and the resulting upper bound on any differential trail probability and any linear trail correlation, compared against the single-trail feasibility thresholds (2^(1-n) for differential trails over an n-bit block; the data complexity 1/C^2 for a linear approximation of correlation C). Secondary yardsticks are implementation cost across platforms — code size and speed on 8-bit and 32-bit processors, and gate count/throughput in hardware — and the inspectability of the design. Key sizes to support are 128, 192, and 256 bits; the block is 128 bits.

## Code framework

The available primitives are byte arithmetic in GF(2^8), XOR round-key injection, and a generic SPN harness that iterates a keyed round. The open design slot is the round transformation; the key schedule and stream wrapper are placeholders showing how that round will be used.

```python
# GF(2^8) with modulus m(x) = x^8 + x^4 + x^3 + x + 1  (0x11B)
def xtime(a):
    """Multiply a byte by {02} in GF(2^8)."""
    a <<= 1
    return (a ^ 0x1B) & 0xFF if a & 0x100 else a & 0xFF

def gf_mul(a, b):
    """Multiply two bytes in GF(2^8)."""
    p = 0
    for _ in range(8):
        if b & 1:
            p ^= a
        carry = a & 0x80
        a = (a << 1) & 0xFF
        if carry:
            a ^= 0x1B
        b >>= 1
    return p

# --- the round transformation ---
def round_transform(state, round_index, n_rounds):
    pass  # TODO: fill in the round transformation

def add_round_key(state, round_key):
    for i in range(len(state)):
        state[i] ^= round_key[i]

# --- key schedule (TODO: expand the cipher key into per-round keys) ---
def expand_key(key, n_rounds):
    pass  # TODO: derive n_rounds+1 round keys from the cipher key

# --- the SPN harness ---
def cipher(block, key, n_rounds):
    state = list(block)
    round_keys = expand_key(key, n_rounds)
    add_round_key(state, round_keys[0])          # initial whitening
    for r in range(1, n_rounds + 1):
        round_transform(state, r, n_rounds)
        add_round_key(state, round_keys[r])
    return bytes(state)

# --- turning the block permutation into a stream over arbitrary-length data ---
def encrypt_stream(plaintext, key, nonce, n_rounds):
    pass  # TODO: a mode of operation built on top of `cipher`
```
