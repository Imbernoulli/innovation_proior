We want a single public, keyless function $H$ that maps a message of *any* length to a short fixed-size digest of $n$ bits and behaves like a fingerprint: given $H$ in full — there is no secret to hide behind, the adversary evaluates it as freely as we do — it must be infeasible to find two distinct messages $M \neq M'$ with $H(M) = H(M')$. This collision resistance is the property digital signatures rest on, and it is the demanding one. Two facts bound the problem from the start. Collisions exist densely by pigeonhole, so the goal can only be "hard to *find*", and we want that infeasibility reduced to a clean assumption rather than asserted. And the generic ceiling is the birthday bound: throw $t$ random messages at $H$, the number of pairs is $\approx t^2/2$, each collides with probability $2^{-n}$, so the expected count $\approx t^2/2^{n+1}$ crosses one near $t \approx 2^{n/2}$. No construction can beat that, and 128-bit collision security therefore forces a 256-bit digest. Preimage and second-preimage searches cost $\approx 2^n$, so collision resistance is the binding constraint.

The deeper obstacle is that we have no way to design, let alone analyze, a function on unbounded input; every primitive we can scrutinize takes a *fixed* number of input bits. The prior families each broke on one horn. The number-theoretic provable hashes — claw-free permutation pairs, modular squaring — could be *proved* collision-resistant under a hard problem, but cost roughly an RSA operation per block, unusable for bulk data. Rabin's cipher-based iteration $h_{i+1} = E_{x_{i+1}}(h_i)$ was fast but had no proof and was concretely weak: a 64-bit chaining value falls to a $2^{32}$ birthday search, and the per-block map is trivially collidable. So the shape — chain a fixed compression from a fixed start — is right and fast, but nobody had a clean way to *prove* the composite inherits collision resistance, and the proofs were observed to get harder exactly as messages got longer.

I propose SHA-256, an instance of length-encoded Merkle–Damgård chaining over a Davies–Meyer compression function, built from two pieces that compose into a provable lift. The first piece posits a fixed-shape compression $f : \{0,1\}^{n+b} \to \{0,1\}^n$ — read its input as $n$ bits of running state plus $b$ fresh message bits — and chains it from a fixed initialization vector:
$$H_0 = \mathrm{IV}, \qquad H_i = f(H_{i-1} \,\|\, M_i), \qquad \text{output } H_L.$$
The claim is that if $f$ is collision-resistant, so is $H$, and the proof is by contrapositive. Given an $H$-collision $x \neq x'$, run both chains and walk *backward* from the equal outputs $H_L = H'_{L'}$. At the last step, either the two inputs to $f$ differ — and then I have two distinct inputs to $f$ with the same output, an $f$-collision, done — or they are equal, which forces $H_{L-1} = H'_{L'-1}$ and $M_L = M'_{L'}$, and I step back and repeat the case split. The walk either coughs up an $f$-collision somewhere or marches all the way to the shared IV with every block matching, which would mean $x = x'$, a contradiction — *provided distinct messages always give distinct padded block sequences*. Plain zero-fill violates that: a message genuinely ending in $d$ zeros and a shorter one padded with $d$ zeros become the same block sequence, and the reduction has nothing to grab. This is precisely the wall the long-message proofs kept hitting. The fix lives in the padding, which must be injective in a way the chain can see across different lengths: $\mathrm{Pad}$ must satisfy (i) $M$ is a prefix of $\mathrm{Pad}(M)$, (ii) equal-length messages get equal-length padding, and (iii) different-length messages get a *different final block*. Condition (iii) — the length field, MD-strengthening — is the hinge the whole reduction turns on: with different lengths the final blocks already differ, so the very last step feeds $f$ two distinct inputs with equal output immediately.

The second piece supplies a fast $f$. The natural fast engine is a block cipher, but the bare map $f(H, M) = E_M(H)$ is invertible in its data argument: to hit any target $c$, pick any key $M$ and set $H = E_M^{-1}(c)$, so colliding it is free — exactly Rabin's failure. The cure is to re-inject the input *after* encryption, the Davies–Meyer shape:
$$f(H, M) = E_M(H) \oplus H.$$
Now $H$ appears tangled on both sides and cannot be decrypted away. Modeling $E$ as an *ideal cipher* — each key names an independent uniformly random permutation, accessible only by oracle queries to $E$ and $E^{-1}$, a deliberately strong idealization because a mere strong-PRP assumption is not known to give collision resistance here — I count an adversary's chances. Writing $l$ for the cipher's block width (which equals the digest length $n$), each of $q$ queries, forward or backward, reveals one hash value $E_M(H) \oplus H$ that is near-uniform over $\geq 2^l - (i-1)$ values not yet pinned down under that key. With $q < 2^{l/2}$ each of the $\binom{q}{2}$ pairs collides with probability $\lesssim 2/2^l$, and the union bound gives
$$\Pr[\text{collision in } q \text{ queries}] \leq q^2/2^l,$$
so a collision needs $q \approx 2^{l/2}$ — the birthday ceiling, nothing better. The two pieces fit: a provably collision-resistant fixed-size $f$ from a cipher, lifted to arbitrary length by length-encoded chaining, with no number-theoretic operation per block.

SHA-256 instantiates this with $n = 256$ and $b = 512$. The state is eight 32-bit words and all arithmetic is mod $2^{32}$, because the carry chain of modular addition is the one cheap source of *cross-bit* nonlinearity — the bitwise mixers act position-by-position, while XOR and rotation move bits sideways but are GF(2)-linear, so without carries the only cross-bit coupling would be linear and the hash trivially invertible. Padding appends a single 1 bit (the byte 0x80) as an unambiguous separator, then zeros until the length is $\equiv 448 \pmod{512}$, then the 64-bit big-endian message bit-length $L$, which makes condition (iii) concrete and limits the hash to $|M| < 2^{64}$ bits. The IV is eight fixed words taken as the first 32 bits of the fractional parts of $\sqrt{p}$ over the first 8 primes, a nothing-up-my-sleeve recipe forced by a public rule so no trapdoor can be alleged. The block of sixteen words $W_0 \ldots W_{15}$ is expanded to 64 schedule words so that 64 rounds are not driven by a mere cyclic reuse that a one-word difference could track differentially:
$$W_t = \sigma_1(W_{t-2}) + W_{t-7} + \sigma_0(W_{t-15}) + W_{t-16}, \qquad t = 16 \ldots 63,$$
with $\sigma_0(x) = \mathrm{ROTR}^{7}x \oplus \mathrm{ROTR}^{18}x \oplus \mathrm{SHR}^{3}x$ and $\sigma_1(x) = \mathrm{ROTR}^{17}x \oplus \mathrm{ROTR}^{19}x \oplus \mathrm{SHR}^{10}x$. The *shift* rather than a third rotation is deliberate: a shift is acyclic — it drops bits off one end and feeds zeros in — so $\sigma$ stays bijective but is no longer rotation-commuting, which is exactly what kills a rotate-the-whole-message differential; the recent and old taps avalanche each block word across many future rounds.

Inside the cipher, the working register $a,b,c,d,e,f,g,h$ starts from the incoming chaining value, and most words simply slide down each round while only two receive fresh nonlinear injection, keeping each round cheap while 64 rounds still reach full diffusion. The nonlinearity comes from $\mathrm{Ch}(e,f,g) = (e \wedge f) \oplus (\neg e \wedge g)$, where $e$ chooses bit by bit between $f$ and $g$, and $\mathrm{Maj}(a,b,c) = (a \wedge b) \oplus (a \wedge c) \oplus (b \wedge c)$, the per-bit majority, together with the add-carries. For intra-word stirring the round uses $\Sigma_0(a) = \mathrm{ROTR}^{2}a \oplus \mathrm{ROTR}^{13}a \oplus \mathrm{ROTR}^{22}a$ and $\Sigma_1(e) = \mathrm{ROTR}^{6}e \oplus \mathrm{ROTR}^{11}e \oplus \mathrm{ROTR}^{25}e$ — three rotations and no shift this time, the opposite choice from the schedule, because here every input bit should survive into the output rather than fall off the end. Per-round constants $K_0 \ldots K_{63}$, the first 32 bits of the fractional parts of the *cube* roots of the first 64 primes (a distinct public recipe from the IV's square roots), make the rounds non-identical to defeat slide and fixed-point attacks. One round computes the two injections
$$T_1 = h + \Sigma_1(e) + \mathrm{Ch}(e,f,g) + K_t + W_t, \qquad T_2 = \Sigma_0(a) + \mathrm{Maj}(a,b,c),$$
all mod $2^{32}$, then slides the register injecting $d + T_1$ into the $e$-track and $T_1 + T_2$ into the $a$-track. After 64 rounds the register holds $E_M(H)$, and the Davies–Meyer feed-forward adds the *incoming* state back in word-wise, $H_i^{\text{new}} = H_i^{\text{old}} + (\text{working var})_i$ — the $\oplus H$ realized in the word ring with the same carry machinery. That add is what makes $f$ non-invertible; without it the round map is a permutation and the whole compression collapses back to the Rabin failure. Iterating $f$ from the IV across all padded blocks and concatenating the eight final words gives the 256-bit digest, collision-resistant down to the $2^{128}$ ceiling. One structural caveat does not break the theorem but bites in use: the digest *is* the entire final chaining value, so revealing $H(M)$ and $|M|$ lets an adversary resume the internal state and compute $H(\mathrm{pad}(M) \| Y)$ without knowing $M$ — the length-extension property — which is remedied outside $f$ by nesting (HMAC) or truncating the output, not by changing the compression.

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
