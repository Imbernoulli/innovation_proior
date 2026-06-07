OK, let me think this through from scratch. I want one public function H that takes a message of arbitrary length and spits out a short fixed-size fingerprint — n bits — such that nobody can find two different messages with the same fingerprint. No secret key anywhere; the adversary knows H exactly and runs it as freely as I do. That last point kills any "security through a hidden key" idea before it starts. The whole game is structural: the function itself has to be hard to collide.

Let me be honest about what "hard to collide" can even mean. The input is unbounded and the output is n bits, so by pigeonhole there are infinitely many collisions — they exist, densely. So I can't ask for "no collisions"; I can only ask "infeasible to *find* one." And I want more than a vibe; I want to reduce that infeasibility to one clean, scrutinizable assumption.

How hard *should* finding a collision be, ideally? Suppose H behaved like a random function into n bits. I throw t random messages at it and look for a repeat. The number of pairs is about t²/2, each pair collides with probability 2^{-n}, so the expected number of collisions is roughly t²/2^{n+1}. That hits 1 around t ≈ 2^{n/2}. So *generically*, with no knowledge of the internals at all, a collision costs only about 2^{n/2} work — the birthday bound. Not 2^n. That's a hard ceiling: I can never beat it, and the best I can hope for is that my construction's best attack is no better than this generic one. Already a design consequence falls out: if I want 128-bit collision security, I need a 256-bit digest. (Preimage and second-preimage are easier on the defender — those cost about 2^n, since I'm aiming at one fixed target rather than any-pair — so collision resistance is the binding constraint and the one signatures depend on.)

Now the real obstacle. I have no idea how to design, let alone *analyze*, a function on unbounded input. Every primitive I can actually build and pick apart takes a *fixed* number of input bits. So let me posit the thing I *can* hope to build: a fixed-shape compression function f : {0,1}^{n+b} → {0,1}^n. It eats n+b bits, emits n, shrinks by b. Collision resistance of a fixed-shape f is a self-contained combinatorial property — I can stare at it, model it, maybe reduce it to a hard problem. The unbounded-input monster I cannot.

So the question sharpens: can I *lift* a fixed-size collision-resistant f to an arbitrary-length collision-resistant H — and ideally prove the lift preserves collision resistance, without smuggling in a new assumption that only bites for long messages? Because that's exactly where it's been getting stuck: people noticed the proofs get harder as the messages get longer. That's the thing to crack.

The obvious skeleton: chain. f takes n+b bits — read that as "n bits of running state, plus b fresh message bits." Start from a fixed value H_0 = IV, eat the message b bits at a time, fold each block into the state:

  H_i = f(H_{i-1} ‖ M_i),  output H_L.

Rabin already did essentially this with a block cipher (h_{i+1} = E_{x_{i+1}}(h_i)), so the *shape* isn't new and it's fast. But it had no proof, and worse, the cipher version was concretely breakable. So the shape is right; the question is whether *this* shape, with the right f, comes with a theorem.

Let me try to prove it and watch where it breaks. Claim I want: if f is collision-resistant, so is H. Prove the contrapositive — suppose someone hands me a collision in H, two messages x ≠ x' with H(x) = H(x'); I'll squeeze an f-collision out of it.

Run both chains. Say x gives blocks M_1…M_L and states H_0,…,H_L; x' gives M'_1…M'_{L'} and states H'_0,…,H'_{L'}. The outputs are equal: H_L = H'_{L'}. Now walk *backward* from the end. At the last step, H_L = f(H_{L-1} ‖ M_L) and H'_{L'} = f(H'_{L'-1} ‖ M'_{L'}). Two cases. If the two inputs to f here differ — i.e. (H_{L-1}, M_L) ≠ (H'_{L'-1}, M'_{L'}) — then I have two distinct inputs to f with the same f-output. That's an f-collision; done. If instead the inputs are *equal*, then in particular H_{L-1} = H'_{L'-1} and M_L = M'_{L'}, and I step back one position and repeat the same case split on the previous step.

So the backward walk either coughs up an f-collision somewhere, or it marches all the way back with every pair of inputs matching. When does it march all the way back? Only if L = L' and M_i = M'_i for all i and the states agree down to H_0 = H'_0 = IV. But matching block-for-block with the same length means x and x' have the same padded message, hence x = x' — contradicting x ≠ x'. So the walk *must* hit a differing step, and that step is my f-collision.

Wait. I waved my hand over "L = L'." Look again. If the two messages have *different* lengths, the backward walk doesn't even line up cleanly — the chains have different numbers of steps. Concretely: what stops a short message and a long message from colliding through pure chaining without ever colliding f? Here's the failure I'm dreading. Suppose padding is just "append zeros to fill the last block." Take a message M, and take M followed by a block that happens to map the state back to itself — or simpler, M versus M with some trailing zero bits that the padding absorbs identically. The two genuinely-different messages can produce the *same sequence of blocks* after padding, so H never sees a difference and there's no f-collision to extract, yet H(x) = H(x'). My reduction has nothing to grab. That's the wall the long-message proofs kept hitting.

Stare at why. The reduction needs: distinct messages ⟹ distinct padded block sequences (so somewhere the chains' inputs must differ). Plain zero-fill violates this — "ends in d zeros" and "padded with d zeros" become indistinguishable. The fix has to be in the *padding*: make the padding injective in a way the chain can see, especially across different lengths.

Damgård's device, stated plainly: after splitting into blocks, append one *extra* block that holds d, the number of zeros I used to fill the last partial block. Now "a message that really ends in d zeros" and "a message padded with d zeros" disagree in that final length-bearing block, so their padded sequences differ, and the backward walk is guaranteed a differing step. And for the awkward edge case where f shrinks by only one bit per call (b minimal), the same idea works via prefix-free encoding of the message, or by seeding from a random first value and leaning on f's one-wayness — but the principle is the one device: encode enough about the length that no two distinct messages share a padded block sequence.

Generalize the requirement, because the *exact* encoding doesn't matter, only the property. The padding map Pad must be such that: (i) M is a prefix of Pad(M); (ii) equal-length messages get equal-length padding; (iii) messages of *different* length get a *different final block*. With those three, two cases of the backward walk close cleanly. Same length L = L': differ in some block, walk back, hit an f-collision before reaching the shared IV. Different length: the final blocks already differ (condition iii), so the very last step feeds f two distinct inputs with equal output — an f-collision immediately. Either way, an H-collision yields an f-collision. Contrapositive: f collision-resistant ⟹ H collision-resistant. The length encoding wasn't decoration; it's the hinge the whole reduction turns on. This is the design principle — "be able to cut one bit off the length in a collision-free way and you can hash arbitrary lengths" — and it tells me exactly what padding has to accomplish.

Good. The lift is proved, *conditional on having a collision-resistant fixed-size f*. Now I actually need an f, and a fast one — fast is the whole reason to bother, since the provable number-theoretic hashes (claw-free pairs, modular squaring) cost an RSA op per block and are unusable for bulk data.

Fast and trusted as a mixing engine means: a block cipher. Rabin's per-block map was f(a,b) = E_a(b), key a = message bits, plaintext b = state. Is that f collision-resistant? Let me try to break it, which is more honest than hoping. Pick any target output c. Solve f(a,b) = c, i.e. E_a(b) = c, for (a,b): choose literally any key a and set b = E_a^{-1}(c). Done — I produced a preimage for free, and two different keys give two colliding inputs trivially. The map is a permutation in its data argument, hence invertible, hence I can steer its output anywhere. So E_a(b) is hopeless as a compression function. (And empirically this is exactly why the DES-based iterated hash fell over: that, plus a 64-bit chaining value that a 2^{32} birthday search eats alive.)

The disease is invertibility: I can run the cipher backward to control the output. Cure: re-inject the input *after* encryption so the output can't be cleanly inverted. The ISO-era patch I'd reach for:

  f(a,b) = E_a(b) ⊕ b.

Now to hit a target c I'd need E_a(b) ⊕ b = c, i.e. E_a(b) ⊕ b = c with b appearing on both sides of the cipher — I can't just decrypt, because b is tangled into the output. The XOR feed-forward is doing the load-bearing work: it converts an invertible permutation into a map that's one-way-ish in the state. Let me re-cast it for the chaining picture, key = message block M, plaintext = chaining value H:

  f(H, M) = E_M(H) ⊕ H.

This is the Davies–Meyer shape. Does the feed-forward actually buy collision resistance, or am I just hoping? Model E as an *ideal cipher*: each key M names an independent uniformly-random permutation E(M,·), and the only access is querying an oracle for E and E^{-1}. (I need the idealization to be this strong — independent across keys, no related-key structure, no weak keys, random even when the key is known — because a mere strong-PRP assumption isn't known to give collision resistance here; the proof leans on E(M,·) being a *fresh random permutation* per key.)

Count an adversary's chances. It makes q queries. Each query, forward or backward, reveals one hash value f(M,H) = E_M(H) ⊕ H. On a forward query (M_i, H_i), E_{M_i}(H_i) is uniform over the l output values not yet pinned down by earlier queries under that same key — so the revealed hash is uniform over ≥ 2^l − (i−1) values; a backward query through E^{-1} is symmetric. So the probability the i-th query's hash equals a *fixed* earlier hash is at most 1/(2^l − (i−1)) for each of the forward/backward cases, ≤ 2/(2^l − (i−1)). With q < 2^{l/2} queries, 2^l − (i−1) ≈ 2^l, so each pair collides with probability ≲ 2/2^l, and there are C(q,2) ≤ q²/2 pairs. Union bound:

  Pr[collision in q queries] ≤ q²/2^l.

So a collision needs q ≈ 2^{l/2} queries — exactly the birthday ceiling, nothing better. Davies–Meyer is collision-resistant in the ideal-cipher model, and feeding it into the chaining theorem above gives a full hash that's collision-resistant down to the same birthday bound. The two pieces fit: a provably-CR fixed-size f from a cipher, lifted to arbitrary length by length-encoded chaining.

Now I have to actually *instantiate* this in fast word operations, n = 256 bits of state, and pick the cipher E. I'll build E internally as a round-based permutation on the 256-bit state keyed by the 512-bit message block — so b = 512, the block size — and realize the Davies–Meyer feed-forward at the end. 256-bit output means a 128-bit collision ceiling, which is the point of going to 256 rather than 128.

State as eight 32-bit words; arithmetic on 32-bit words because that's what hardware and software both do cheaply: bitwise AND/OR/NOT/XOR, rotate-right ROTR^n, shift-right SHR^n, and add mod 2^32. The mod-2^32 addition matters more than it looks: its carry chain is the one operation that mixes information *across* bit positions nonlinearly and cheaply — XOR and rotation are linear over GF(2), so without the carries everything would be a linear map and trivially invertible/collidable. Carries are my nonlinearity budget.

First, padding — I derived exactly what it must do, now make it concrete and length-encoding. Append a single 1 bit (so the boundary between message and padding is unambiguous — that's a separator, giving condition (i)), then zeros, then the message bit-length L as a fixed 64-bit big-endian field, sized so the whole thing is a multiple of the 512-bit block. Concretely: after the 1 bit and zeros, reserve the final 64 bits of the last block for L, i.e. zero-fill until the length ≡ 448 mod 512, then append the 64-bit L. The 64-bit length field is the (iii) condition made concrete — different-length messages put a different number in that field, so they can't share a final block. (Note the design corollary: with a 64-bit length field the hash is only defined for messages under 2^64 bits, which is fine.) In bytes: append 0x80, then zero bytes, then the 8-byte big-endian bit length.

The IV — eight fixed 32-bit words. I want constants that visibly hide no trapdoor ("nothing up my sleeve"), so derive them from something I couldn't have rigged: take the fractional parts of the square roots of the first 8 primes and read off the first 32 bits. That gives 6a09e667, bb67ae85, 3c6ef372, a54ff53a, 510e527f, 9b05688c, 1f83d9ab, 5be0cd19. Nobody can claim I chose these to plant a weakness — they're forced by a public recipe.

Now the cipher E_M(H), the engine. I'll run R rounds of a register that mixes the state, injecting message-derived material each round. Two design pressures: (a) every message bit must influence every output bit (avalanche), and (b) the round must be nonlinear enough to resist linear and differential attacks.

Message expansion first. The block is 512 bits = sixteen 32-bit words M_0…M_15, but I want enough rounds that the state is thoroughly stirred — 64 rounds. If I just reused the 16 words cyclically, a difference in one message word would recur on a fixed schedule and be easy to track differentially. So expand: build 64 schedule words W_0…W_63, the first 16 equal to the message words, and each later one a mix of earlier ones. I want the mix to diffuse and to *not* be a pure rotation (or the whole schedule inherits a rotational symmetry an attacker can exploit). So combine two earlier words through small mixing functions and add (mod 2^32) two more:

  W_t = σ1(W_{t-2}) + W_{t-7} + σ0(W_{t-15}) + W_{t-16},  t = 16…63,

where σ0, σ1 each fold a word with two rotations and — crucially — one *shift*:

  σ0(x) = ROTR^7(x) ⊕ ROTR^18(x) ⊕ SHR^3(x)
  σ1(x) = ROTR^17(x) ⊕ ROTR^19(x) ⊕ SHR^10(x).

Why a shift and not a third rotation? A shift loses the top/bottom bits — it's not a bijection — which breaks the clean rotational structure of the schedule so it isn't rotation-invariant; that asymmetry is exactly what stops a rotate-the-whole-message differential. The two source taps W_{t-2} (recent) and W_{t-15} (older) plus the raw adds of W_{t-7}, W_{t-16} spread each original message word across many future rounds: a one-bit change in M avalanches through the schedule.

Now the round itself. Working variables a,b,c,d,e,f,g,h initialized from the incoming chaining value H_0…H_7. I'll make most of the eight words a shift register — each round they just slide down (b←a, c←b, …) — and only *two* of them receive fresh nonlinear injection. That keeps each round cheap while 64 rounds still achieve full diffusion. Two injection points, call them the "a-track" and the "e-track."

The nonlinearity comes from two bitwise functions and the add-carries. Choose:

  Ch(e,f,g) = (e ∧ f) ⊕ (¬e ∧ g)       — e chooses, bit by bit, between f and g
  Maj(a,b,c) = (a ∧ b) ⊕ (a ∧ c) ⊕ (b ∧ c) — the per-bit majority of a,b,c.

Ch and Maj are the nonlinear bitwise mixers; one feeds the e-track, one the a-track. And for intra-word diffusion I want each word's bits stirred among themselves before they propagate — use functions of *three rotations* (no shift this time, because here I want a bijective stir of the full word, all bits preserved):

  Σ0(a) = ROTR^2(a) ⊕ ROTR^13(a) ⊕ ROTR^22(a)
  Σ1(e) = ROTR^6(e) ⊕ ROTR^11(e) ⊕ ROTR^25(e).

(Contrast with the schedule's σ: there I *wanted* the symmetry-breaking shift; here in the round I want full-word bijective mixing, so three rotations, no shift. Same letter family, opposite choice, for opposite reasons.)

I also need per-round constants so the 64 rounds aren't identical (identical rounds invite slide/fixed-point attacks). Same nothing-up-my-sleeve trick, different irrational source so they're independent of the IV: first 32 bits of the fractional parts of the *cube* roots of the first 64 primes — K_0 = 428a2f98, … K_63 = c67178f2.

Assemble one round. Compute the two injections:

  T1 = h + Σ1(e) + Ch(e,f,g) + K_t + W_t
  T2 = Σ0(a) + Maj(a,b,c)

— all additions mod 2^32, so the carries are doing their nonlinear cross-bit work. T1 gathers the e-track's nonlinear mix (Σ1, Ch), the tail word h, the round constant, and this round's schedule word; T2 gathers the a-track's (Σ0, Maj). Then slide the register and inject:

  h←g, g←f, f←e, e←d+T1, d←c, c←b, b←a, a←T1+T2.

So d+T1 enters the e-track and T1+T2 enters the a-track; the other six words just shift. Over 64 rounds, with each W_t carrying avalanched message material and Ch/Maj/carries supplying nonlinearity, every message bit reaches every state bit and the map is a thorough pseudorandom-looking permutation of the 256-bit state keyed by the block — that's my E_M(H).

Last, the Davies–Meyer feed-forward, the piece that makes the *compression* one-way rather than the cipher invertible. After the 64 rounds produce (a,…,h) = E_M(H), I add the *incoming* chaining words back in, word-wise mod 2^32:

  H_i^{new} = H_i^{old} + (working var)_i,  i = 0…7.

That word-wise add is the ⊕H of Davies–Meyer realized in the word ring (add mod 2^32 instead of XOR — either re-injects the input; the add reuses the same carry machinery). It's the reason I can't invert the compression: to run it backward I'd need the pre-round state, but it's been added into the post-round state, tangling input and output exactly as in f(H,M)=E_M(H)⊕H. Without this step the round function would be an invertible permutation and the whole f collapses back to the Rabin failure I diagnosed.

Step back and look at the whole pipeline. pad(M) makes a length-encoded block sequence (so the chaining theorem's reduction has a differing block to grab). For each 512-bit block: expand the 16 words to 64 schedule words; run 64 register rounds keyed by the schedule, with Σ/σ for diffusion, Ch/Maj plus mod-2^32 carries for nonlinearity; then Davies–Meyer-add the input state back in to get a non-invertible compression f. Iterate f from the nothing-up-my-sleeve IV across all blocks. Output the eight final words concatenated = 256-bit digest. The chaining theorem says any collision in this hash hands me a collision in f; Davies–Meyer says colliding f in the ideal-cipher model costs ≈ 2^{128} queries — the birthday ceiling for a 256-bit output. Provable lift, fast provable-ish compression, no number-theoretic op per block.

One structural fact I should flag to myself, because it bites in usage even though it doesn't break the collision theorem. The hash output *is* the entire final chaining value — the whole internal state after the last block. So if I publish H(M) and reveal |M|, an adversary can set its internal state to exactly H(M) and keep going: append more padded blocks Y and compute H(pad(M) ‖ Y) without ever knowing M. That's a length-extension property, and it means the naive keyed construction H(key ‖ msg) is forgeable — an attacker extends it to a valid digest of (key ‖ msg ‖ Y) without the key. It's not a collision break (the theorem stands); it's that "output = internal state" leaks the resumption point. The remedy lives outside the compression — nest the hash (hash twice / HMAC), or truncate the output so the full state isn't exposed — but that's a usage wrapper, not a change to f or the chaining.

Now the code, mirroring the pipeline exactly.

```python
# Initial chaining value H^(0): first 32 bits of the fractional parts of the
# square roots of the first 8 primes (nothing-up-my-sleeve).
H0 = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
      0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]

# Per-round constants K_t: first 32 bits of the fractional parts of the cube
# roots of the first 64 primes (independent of the IV).
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

# Σ: three rotations, no shift — bijective intra-word stir (round).
def Sigma0(x): return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22)
def Sigma1(x): return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25)
# σ: two rotations + one shift — the shift breaks rotational symmetry (schedule).
def sigma0(x): return rotr(x, 7)  ^ rotr(x, 18) ^ shr(x, 3)
def sigma1(x): return rotr(x, 17) ^ rotr(x, 19) ^ shr(x, 10)

# The only bitwise nonlinear mixers.
def Ch(x, y, z):  return (x & y) ^ (~x & z)
def Maj(x, y, z): return (x & y) ^ (x & z) ^ (y & z)

def pad(message: bytes) -> bytes:
    # length encoding (MD-strengthening): 1-bit separator, zeros, 64-bit bitlen.
    bit_len = (len(message) * 8) & ((1 << 64) - 1)
    p = message + b"\x80"
    p += b"\x00" * ((56 - len(p) % 64) % 64)   # fill so len ≡ 448 mod 512
    p += bit_len.to_bytes(8, "big")            # different lengths ⇒ different last block
    return p

def compress(state, block: bytes):
    # message expansion: 16 words → 64 schedule words, avalanching each block word
    w = [int.from_bytes(block[4*i:4*i+4], "big") for i in range(16)]
    for t in range(16, 64):
        w.append((sigma1(w[t-2]) + w[t-7] + sigma0(w[t-15]) + w[t-16]) & MASK)
    a, b, c, d, e, f, g, h = state            # working register = incoming chaining value
    for t in range(64):                       # 64 rounds of the keyed permutation E_M(·)
        t1 = (h + Sigma1(e) + Ch(e, f, g) + K[t] + w[t]) & MASK   # e-track injection
        t2 = (Sigma0(a) + Maj(a, b, c)) & MASK                    # a-track injection
        h, g, f, e, d, c, b, a = g, f, e, (d + t1) & MASK, c, b, a, (t1 + t2) & MASK
    # Davies–Meyer feed-forward: add the input state back in ⇒ non-invertible f.
    return [(x + y) & MASK for x, y in zip(state, (a, b, c, d, e, f, g, h))]

def sha256(message: bytes) -> str:
    state = list(H0)                          # chaining starts at the fixed IV
    data = pad(message)
    for off in range(0, len(data), 64):       # iterate f block by block (Merkle–Damgård)
        state = compress(state, data[off:off+64])
    return "".join(f"{x:08x}" for x in state)  # output = final chaining value (256 bits)
```

The causal chain, end to end: I needed an arbitrary-length collision-resistant fingerprint but could only analyze fixed-size compressions, so I chained one from a fixed IV; the chaining proof demanded that distinct messages give distinct block sequences, which forced a length-encoding padding; I then needed a concrete fast collision-resistant compression, found the bare cipher map invertible-and-collidable, and cured it with the Davies–Meyer feed-forward (provably birthday-bound collision-resistant in the ideal-cipher model); instantiated that cipher as a 64-round 32-bit-word register with Σ/σ for diffusion and Ch/Maj-plus-carries for nonlinearity, keyed by an avalanching message schedule, with nothing-up-my-sleeve IV and round constants — and the feed-forward add at the end is what keeps the whole thing one-way; the price, output = internal state, is the length-extension property to wrap around in use.
