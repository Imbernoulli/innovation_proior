Let me start from what's actually going wrong, not from a clean redesign. Every cryptographic hash anyone deploys — MD5, SHA-0, SHA-1, SHA-2, RIPEMD — is the same shape. There's a compression function f that eats a chaining value and a message block and spits out a new chaining value; you fix an initialization vector H_0, pad the message, split it into blocks M_1…M_t, and iterate H_i = f(H_{i-1}, M_i), and the digest is H_t. Merkle and Damgård gave the reason this design caught on: if f is collision-resistant and you pad with the message length encoded in the last block (strengthening), then the whole hash is collision-resistant — any collision in the hash forces a collision in f. That's a real theorem and it's why everyone builds this way.

But two things are breaking at once, and I want to be careful to separate them because they have different cures. One: the concrete compression functions are falling. Wang showed practical MD5 collisions; SHA-1 collisions are within theoretical reach below the 2^80 birthday bound. That's a failure of the *primitive*. Two — and this is the one I want to chew on — even if I had a perfect f, the *construction wrapped around it* leaks properties that the ideal object doesn't have.

What's the ideal object? A random oracle: it maps each distinct input to its own independent uniformly-random output. Truncate it to n bits and it costs 2^{n/2} to collide, 2^n to (second-)preimage, and it has no other exploitable structure. The appeal of "behaves like a random oracle" as a spec is that it's *one* criterion that implies all the others — collision resistance, preimage resistance, correlation-freeness, length-extension resistance, all of it. People keep discovering new criteria a hash ought to satisfy, and the random oracle satisfies all of them, including ones not yet named. So if I could just say "my function is indistinguishable from a random oracle up to some bound," I'd be done — that's the compact security claim I want.

So can an iterated hash *be* a random oracle? Stare at the iteration. The chaining value lives in a finite set. Finite means collisions: there exist two distinct messages M1, M2 with H(M1)=H(M2) as *chaining states*, not just as final digests. Now take any suffix N. The continuation H_i = f(H_{i-1}, ·) is deterministic and starts from that shared state, so M1‖N and M2‖N produce identical outputs. Forever. For every N. A random oracle does not do this — even if RO(M1) happened to equal RO(M2) on the truncated digest, RO(M1‖N) and RO(M2‖N) are independent of each other and of the M1/M2 outputs. So the iterated hash has an externally visible behavior — a "state collision propagates through suffixes" behavior — that the random oracle simply lacks. The finite state guarantees these collisions exist. Therefore *no* iterated hash with a finite chaining state can equal a random oracle. The goal "be a random oracle" is unreachable for anything streamable.

Okay. That kills the naive aspiration but it sharpens the real question. The deviation from a random oracle comes specifically from collisions in the internal state. If I can't eliminate them, can I (a) make the probability of an adversary *finding* one negligible, controlled by a parameter I choose, and (b) make a *reachable* reference claim — "indistinguishable from a random oracle except via these internal collisions, with advantage below some explicit function of my parameter"? That reframing feels right. Don't chase the unreachable ideal; quantify the one unavoidable gap and drive it down.

Before I design anything, let me name the most embarrassing concrete symptom of the gap, because fixing it will probably tell me what to change. Length-extension. In Merkle–Damgård the emitted digest *is* the final chaining state H_t (maybe after a finalization, but classically it's just H_t). So if I know H(M) and len(M) — not M itself — I know the exact internal state after absorbing M's padded blocks. I can resume the iteration: keep feeding f with my own blocks Y and compute H(pad(M)‖Y) without ever knowing M. The textbook casualty is the MAC H(K‖M): I see a tag, I extend it to a valid tag on K‖M‖pad‖Y, and I never learned K. The world's fix is HMAC — nest two keyed hashes — and MGF1 wraps similar scaffolding. The fact that the safe way to use these hashes is to *wrap them in more structure* is the tell. The base construction hands you its internal state as the output. That's the leak.

So here's the thing I keep circling back to: **the digest exposes the entire state that determines the future.** Length-extension is just the statement "output = resumable state." What if the output exposed only *part* of the state, and held the rest back?

Let me also reconsider the primitive itself, because the competition wants efficiency and a clean argument, and the Davies–Meyer route — build f from a block cipher as h(H,M)=E_M(H)⊕H, encrypt-the-chaining-value-under-the-message-as-key plus feed-forward — drags in a *keyed* object. A block cipher has a key schedule, with its cost and its related-key headaches, and you only use it to get a fixed-length compression function that you then have to wrap (prefix-free encoding, chop-MD, NMAC) to make it indifferentiable from a random oracle in Coron's sense. That's a lot of layers: keyed cipher → Davies–Meyer compression → strengthened iteration → indifferentiability repair. Each layer is there to patch the layer below.

Let me try to collapse the stack. What is the simplest fixed-length primitive I could iterate? Not a keyed cipher — just one fixed permutation f on b bits. No key, no key schedule, nothing secret inside it; f is public and fixed. A good block cipher is, after all, a *family* of random-looking permutations indexed by the key; if I don't need the family — if I don't need a key — then I just want *one* random-looking permutation, and I get to keep all the block-cipher design intuition (diffusion, nonlinearity, trails) while throwing away the key schedule. That's strictly less machinery. The PANAMA/RadioGatún/Grindahl line already hashes by iterating a big permutation, so this isn't crazy; what those lacked was a *provable* reduction to the permutation's strength. If I can pair a permutation with a construction that has a clean indifferentiability proof, I get the best of both.

Now combine the two ideas — "expose only part of the state" and "iterate one fixed permutation." Let the state be b bits, and split it into an outer part of r bits and an inner part of c bits, with r + c = b. Call r the rate and c the capacity. The message only ever touches the outer r bits; the inner c bits are never directly set by input and never directly output. Concretely:

Absorbing. Start S = 0^b. Pad the message and cut it into r-bit blocks P_0, P_1, …. For each block, XOR it into the outer r bits and apply the permutation: S ← f(S ⊕ (P_i ‖ 0^c)). The 0^c says: the message never gets XORed into the inner part; the inner part only ever changes because f mixes the outer part into it.

Squeezing. To produce output, read off the outer r bits of S. Need more than r bits? Apply f again and read the next r bits: S ← f(S), emit the outer r, repeat, and truncate to the requested length d. Like a sponge — soak the message in through the rate, then wring output back out of the rate.

Does this kill length-extension? The digest is (a prefix of) the *outer* r bits — and for a hash we'll truncate to d ≤ r bits, so we reveal at most the rate, never the c inner bits. To resume the permutation and extend, an attacker needs the *whole* b-bit state, including the inner c bits. Those were never output and are not determined by anything the attacker saw. So they can't resume. Length-extension is gone — not patched with an outer wrapper, but structurally absent, because the construction *withholds* the part of the state that would let you continue. And it costs nothing extra: I get a single tunable knob, c, that I can make as large as I want by spending state on it.

Now I have to earn the security claim, because "feels safe" isn't a bound. Replace f by a random transformation (or random permutation) and ask: in a black-box setting, what can an adversary learn that distinguishes this thing from a random oracle? Walk the absorbing/squeezing as a walk on a graph whose nodes are state values and whose edges are S → f(S). Group nodes by their inner-part value — call each group of all states sharing an inner value a *supernode* (there are 2^c of them, one per inner value; each holds 2^r nodes indexed by the outer value). Absorbing a message is a path from the root supernode (inner value 0) hopping supernode to supernode; the outer value you land on, truncated, is the output. Here's the key: as long as the adversary's queries never produce two distinct paths reaching the same *inner* value — an **inner collision** — the output characters she gets back are uniformly and independently distributed, exactly like a random oracle. Why: each fresh path lands in a supernode whose outer coordinate hasn't been pinned down yet, so its output bits are still free and uniform. The *only* way the sponge betrays itself is an inner collision; absent one, it's a perfect random oracle on what she's seen.

So bound the probability of an inner collision after N calls. Think of it as a birthday process on the inner state, which has C = 2^c values. Before the i-th call the adversary has at most i previously-reached inner states; in the random-transformation case the i-th new edge collides into one of them with probability about i/C, optimizing her strategy so each new edge starts from a reachable node. Multiply the survival probabilities:

P(no inner collision) = ∏_{i=1}^{N} (1 − i/C).

For N ≪ C use log(1+ε) ≈ ε: the log-probability is about −∑ i/C = −N(N+1)/(2C), so

P(inner collision) ≈ 1 − exp(−N(N+1)/(2C)) ≈ N(N+1)/(2C) = N(N+1)/2^{c+1}.

There it is. The distinguishing advantage in the black-box setting is about N(N+1)/2^{c+1}. It depends on c and *not at all* on r. The rate is free to be as large as I like for throughput; the security lives entirely in the capacity. (For a random *permutation* the count is even slightly more favorable — edges can be added both forward and backward and the available inner targets shrink as the permutation fills up — so the bound only improves.)

But wait — black-box means the adversary can only query the sponge, not f. In a real hash, f is public; the adversary can compute f and f^{-1} herself. The bound I just derived is about the sponge interface alone, and that's of limited use once f is out in the open. I need the stronger statement: indifferentiability in Coron's sense, where the distinguisher queries *both* the construction *and* the underlying primitive (and, since f is a permutation, also f^{-1}), and I must exhibit a simulator that fakes f using only random-oracle access so the two worlds are indistinguishable. This is harder than anything Coron et al. did, on two counts: their proofs were for an ideal *compression function* or ideal *cipher*, never for a bare random *permutation* the adversary can invert at will; and their constructions produce fixed-length output, whereas the sponge produces arbitrarily long output. The worry with a permutation is exactly the inverse queries — an adversary running f^{-1} could try to walk *backwards* into the inner state from a chosen output. The simulator has to maintain a consistent graph of forward and backward f-edges and answer new queries with fresh randomness unless they're forced by a path the distinguisher already pinned down through the construction; the place it can fail is precisely when a query forces an inner collision — the same event as before. Carrying that through, the differentiating advantage comes out at about N(N+1)/2^{c+1} for a random transformation, and a smaller bound for a random permutation, with N the total number of f and f^{-1} calls.

That's the whole payoff. Up to ~2^{c/2} queries, the sponge calling a random permutation is indifferentiable from a random oracle — so it inherits *every* random-oracle property, not just collision resistance: collisions cost ~2^{c/2}, (second) preimages more, and there's no length-extension because a random oracle has none. And note the lovely paradox falling out: I'm getting collision resistance and preimage resistance from a permutation that is *trivially invertible*. There's no contradiction — the thing the adversary can't reach is the hidden inner state, not f. Invertibility of f buys her nothing because she still can't see c bits of every state.

Now I get to *choose* parameters against this bound. b = r + c is the state width, fixed by the permutation I'll build. Generic security is ~2^{c/2} against collisions and richer for preimages, but a digest of d bits can't beat a truncated random oracle: collisions cost min(2^{c/2}, 2^{d/2}). To not waste capacity I want 2^{c/2} ≥ 2^{d/2}, i.e. c ≥ d for collisions; and to also match preimage resistance ~2^d of a d-bit RO I want 2^{c/2} ≥ 2^d, i.e. **c = 2d**. So for a 256-bit digest, c = 512 and the rate is r = b − 512; for 512-bit, c = 1024. Larger digest ⇒ larger capacity ⇒ smaller rate ⇒ slower, exactly the throughput-versus-security trade-off, all on the one dial. Settle the width at b = 1600 (big enough that even c=1024 leaves a healthy rate r=576), so SHA3-256 runs at r=1088, c=512; SHA3-512 at r=576, c=1024.

Padding. Absorbing needs the input cut into exact r-bit blocks, so I must pad to a multiple of r, and the padding must be injective or two messages collide for free. The minimal honest rule: append a 1, then enough 0s, then a 1 — pad10*1 — landing on a multiple of r. Why the *final* 1 and not just 1 followed by zeros? Two reasons. It guarantees the last block is nonzero in the rate even when the message already ended on a block boundary, and — the subtle one — appending the closing 1 makes the padding *multi-rate*: the same f can be safely reused across different rates r without a message at one rate ever colliding with a message at another, because the padded forms differ. Concretely the number of zeros is j = (−m − 2) mod r for a message of m bits, giving P = 1 0^j 1; and I append at least one bit so f is always called at least once (the empty input still gets a well-defined, f-dependent output rather than the trivial all-zero outer state).

One more wrinkle: I want this one permutation to drive several functions — fixed-length hashes and arbitrary-length output functions — without their outputs ever colliding. So before pad10*1 I append a short domain-separation suffix: hashes get one suffix, the extendable-output functions another. In bits the hashes append 01 and the XOFs append 1111 to the message; then pad10*1 closes it. Two callers in different domains can never produce the same absorbed string, so a clash between a hash and an XOF on the same message is impossible by construction.

Now the part I deferred: I assumed a random permutation f on 1600 bits. I have to actually *build* one — a fixed, public, fast permutation that behaves like a random permutation under differential and linear cryptanalysis — and it must be invertible (it's a permutation) but, as we saw, invertibility costs nothing. Lay the 1600 bits out as a 3-D array: 5×5 lanes of w = 64 bits each, A[x][y][z] with 0 ≤ x,y < 5 and 0 ≤ z < 64. Columns run along y (5 bits), rows along x (5 bits), lanes along z (64 bits). A round will be a sequence of simple step mappings, repeated; each step must earn its place by removing a specific weakness, and together they have to give fast diffusion and real nonlinearity.

Nonlinearity first, because without it the whole permutation is linear and trivially broken — solve a linear system and you invert it with no work. I want a *cheap*, low-degree, invertible nonlinear map. Operate within a row of 5 bits and set, for each bit, A'[x] = A[x] ⊕ ((¬A[x+1]) ∧ A[x+2]) (indices mod 5 within the row, same y,z). Call it χ. It's algebraic degree 2 — the minimum that's nonlinear — so it's cheap and analyzable, it's invertible (a known closed form recovers the row), and it's the *only* nonlinear step; everything else can be linear. Degree 2 means differentials through χ behave like an affine variety, which is exactly what lets me bound differential and linear trail weights later. Good.

But χ only mixes within a row of 5 bits; it touches nothing across rows, columns, or lanes. I need diffusion to spread a single-bit change across the whole 1600-bit state fast. Diffusion can be linear. Start with the columns. θ: compute each column's parity, and XOR into every bit the parities of two neighboring columns — specifically C[x,z] = ⊕_y A[x,y,z], then add to A[x,y,z] the value D[x,z] = C[x−1,z] ⊕ C[x+1,z−1]. The z−1 twist on the second column couples the lanes slightly; without it θ would act slice-by-slice. θ is linear and cheap (a handful of XORs per bit). But it has a weakness I should name now so I can fix it: if *every* column already has even parity, all the C's are zero, D is zero, and θ does nothing — it's the identity on that whole set of states (the "column-parity kernel"). That set contains states with as few as two active bits (two bits in one column). So θ's worst-case diffusion (branch number) is only 4; low-weight differences sitting in the kernel slide through θ untouched. I shouldn't try to fix this by beefing up θ — that would blow up its cost. Instead I'll fix it with the *next* steps, by making sure such low-weight kernel differences don't survive round after round.

θ diffuses *within* a slice (the columns live in a slice) but barely *between* slices — between different z. So add ρ: rotate each lane (x,y) along z by a fixed offset that depends on (x,y). Pick the offsets as the triangular numbers (t+1)(t+2)/2 mod 64 marched around the lanes, leaving lane (0,0) at offset 0. Now a bit that was at z moves to z+offset, and since the offsets differ across lanes, bits that were aligned in z get sheared apart — inter-slice dispersion. ρ is pure wiring in hardware (free) and translation-invariant in z. It's linear and invertible (rotate back).

But ρ and θ still leave the *lanes* sitting where they were in the (x,y) plane; column structure persists across rounds, which is exactly what lets the column-parity-kernel trails live. So permute the lane positions: π sends the lane at (x,y) to a new (x,y) by A'[x,y] = A[(x+3y) mod 5, x]. This shuffles the 25 lanes around the (x,y) grid (a single 24-cycle plus the fixed origin). Its job is long-term diffusion: it tears apart the alignment θ relies on, so a difference that was a cheap kernel pattern in one round gets scattered into different rows and columns the next round and can no longer reproduce itself cheaply. Together θ, ρ, π are the linear layer λ = π∘ρ∘θ that separates one χ from the next, and analyzing differential/linear *trails* through λ between χ steps is how the design margin gets argued — minimum trail weights climb fast enough over a few rounds that no low-weight trail spans the full round count. (I won't grind out the full trail tables here; the point is that π's lane shuffle is what makes the kernel trails die instead of persisting, which is why I can leave θ's branch number at 4 and pay nothing more.)

There's still a symmetry problem. θ, ρ, π, χ are all translation-invariant along z (they treat every z-slice the same), and identical every round. So the whole round function is z-translation-invariant and all rounds are equal, which invites slide attacks and symmetric fixed points (e.g. all-equal-lanes states mapping to all-equal-lanes states). Break it: ι XORs a round-dependent constant into a single lane, lane (0,0). The constants must (a) differ from round to round and (b) be irregular along z. Generate them with a maximum-length LFSR: R starts at 0x01, and each step R ← (R≪1) ⊕ ((R≫7)·0x71) mod 256, reading off bits to place at lane positions 2^j − 1 for j = 0…l. Adding it into just one lane is enough — θ and χ diffuse the asymmetry to all 25 lanes within a single round. ι is one XOR (plus the tiny LFSR). It doesn't affect whether differential/linear trails *exist* (it's a constant), only the symmetry, which is precisely its job.

Order within a round? The classic order is θ, ρ, π, χ, ι — that is, R = ι∘χ∘π∘ρ∘θ, applied left of the state. θ goes *first* for a sponge-specific reason: at the start of the round it mixes the inner (capacity) part of the state with the outer (rate) part, so the part of the state the adversary doesn't control gets stirred into the part she does early. The order of the remaining four is essentially free — what matters is that the linear layer λ = π∘ρ∘θ sits between consecutive χ's so each nonlinear step is well separated. How many rounds? Take w = 64 ⇒ l = log2(64) = 6, and use n_r = 12 + 2l = 24 rounds — a margin chosen so the climbing trail weights leave no usable differential or linear trail across the full permutation.

Let me land it in code. The state is 25 lanes of 64 bits; I'll keep it as a 5×5 array of Python ints and pack/unpack little-endian. ρ and π fold together neatly into one walk: start at (x,y)=(1,0), and at step t move (x,y) ← (y, (2x+3y) mod 5) while rotating the carried lane by the triangular offset (t+1)(t+2)/2 — that single loop realizes both the rotations and the lane permutation. The sponge wrapper absorbs by XORing each rate-sized block into the leading bytes of the state and applying f on every full block; pads by XORing the domain-suffix-plus-first-1 byte at the current offset and the closing 1 (0x80) at the last rate byte, then one more f; and squeezes the leading rate bytes, applying f between blocks, until d bytes are out.

```python
def rol64(a, n):
    # rotate a 64-bit lane left by n; ρ is built from these
    return ((a >> (64 - (n % 64))) | (a << (n % 64))) & ((1 << 64) - 1)

def keccak_f1600_on_lanes(lanes):
    R = 1
    for _ in range(24):                       # n_r = 12 + 2l = 24 rounds
        # θ: column parities, then XOR two neighbouring columns (one shifted in z)
        C = [lanes[x][0] ^ lanes[x][1] ^ lanes[x][2] ^ lanes[x][3] ^ lanes[x][4]
             for x in range(5)]
        D = [C[(x + 4) % 5] ^ rol64(C[(x + 1) % 5], 1) for x in range(5)]
        lanes = [[lanes[x][y] ^ D[x] for y in range(5)] for x in range(5)]
        # ρ and π fused: walk the 24-cycle, rotating each lane by a triangular offset
        (x, y) = (1, 0)
        current = lanes[x][y]
        for t in range(24):
            (x, y) = (y, (2 * x + 3 * y) % 5)
            (current, lanes[x][y]) = (lanes[x][y], rol64(current, (t + 1) * (t + 2) // 2))
        # χ: the only nonlinear step, within each row of 5 lanes
        for y in range(5):
            T = [lanes[x][y] for x in range(5)]
            for x in range(5):
                lanes[x][y] = T[x] ^ ((~T[(x + 1) % 5]) & T[(x + 2) % 5])
        # ι: XOR an LFSR-generated round constant into lane (0,0) to break symmetry
        for j in range(7):
            R = ((R << 1) ^ ((R >> 7) * 0x71)) % 256
            if (R & 2):
                lanes[0][0] ^= (1 << ((1 << j) - 1))
    return lanes

def load64(b):  return sum(b[i] << (8 * i) for i in range(8))   # little-endian lane pack
def store64(a): return list((a >> (8 * i)) % 256 for i in range(8))

def keccak_f1600(state):
    lanes = [[load64(state[8 * (x + 5 * y):8 * (x + 5 * y) + 8]) for y in range(5)]
             for x in range(5)]
    lanes = keccak_f1600_on_lanes(lanes)
    state = bytearray(200)                    # b = 1600 bits = 200 bytes
    for x in range(5):
        for y in range(5):
            state[8 * (x + 5 * y):8 * (x + 5 * y) + 8] = store64(lanes[x][y])
    return state

def sponge(rate, capacity, input_bytes, delimited_suffix, output_len):
    assert rate + capacity == 1600 and rate % 8 == 0
    rate_bytes = rate // 8
    state = bytearray(200)
    off = 0
    block = 0
    # absorb: XOR each rate-sized block into the OUTER bytes, f on each full block
    while off < len(input_bytes):
        block = min(len(input_bytes) - off, rate_bytes)
        for i in range(block):
            state[i] ^= input_bytes[i + off]
        off += block
        if block == rate_bytes:
            state = keccak_f1600(state)
            block = 0
    # pad10*1 with the domain suffix folded into the first padding byte
    state[block] ^= delimited_suffix          # domain bits ‖ first pad-1 bit
    if (delimited_suffix & 0x80) and block == rate_bytes - 1:
        state = keccak_f1600(state)
    state[rate_bytes - 1] ^= 0x80             # closing pad-1 bit
    state = keccak_f1600(state)
    # squeeze: emit OUTER rate bytes, f between blocks, truncate to output_len
    out = bytearray()
    while output_len > 0:
        block = min(output_len, rate_bytes)
        out += state[0:block]
        output_len -= block
        if output_len > 0:
            state = keccak_f1600(state)
    return out

# c = 2d picks the capacities; the suffix separates hashes (0x06) from XOFs (0x1F)
def sha3_256(m):       return sponge(1088, 512, m, 0x06, 32)
def sha3_512(m):       return sponge(576, 1024, m, 0x06, 64)
def shake128(m, n):    return sponge(1344, 256, m, 0x1F, n)
def shake256(m, n):    return sponge(1088, 512, m, 0x1F, n)
```

Tracing the whole chain back: I started from the fact that any finite-state iterated hash *cannot* be a random oracle — internal state collisions give it a behavior the oracle lacks — and that the most visible symptom, length-extension, is just "the output is the resumable state." So I stopped exposing the whole state: split b = r + c, let the message in through r, hold c back, never output c. That one move kills length-extension for free and turns the unavoidable gap into a single event — an inner collision — whose probability after N queries is about N(N+1)/2^{c+1}, controlled entirely by the capacity, independent of the rate; the indifferentiability argument lifts that from "can't query f" to "can query f and f^{-1}," so the construction inherits all random-oracle properties up to ~2^{c/2}. Choosing c = 2d matches a d-bit truncated oracle, with the rate left as the throughput dial. Then, needing a public fixed permutation in place of a keyed compression function, I built one out of five cheap steps on a 5×5×64 state: χ for nonlinearity, θ for column diffusion (with its kernel weakness deliberately left to π), ρ for inter-slice dispersion, π for the lane shuffle that kills the kernel trails, ι to break the z-symmetry — 24 rounds for margin — and dropped the sponge directly onto it.
