We want a cryptographic hash function — a map from arbitrary-length messages to fixed-length (or even arbitrary-length) digests — that resists collisions, preimages, and second preimages, and, more demandingly, that exhibits *no* structural weakness a random oracle would lack. The appeal of "behaves like a random oracle" as a specification is that it is a *single* criterion implying all the others: collision resistance, preimage resistance, correlation-freeness, length-extension resistance, and the new criteria people keep inventing. If I could just claim "my function is indistinguishable from a random oracle up to some explicit bound," I would be done. The trouble is that every deployed hash — MD5, SHA-0/1, SHA-2, RIPEMD — is built the same way, and that shared construction is failing on two independent fronts at once. The concrete compression functions are falling to collision search (Wang's practical MD5 collisions; SHA-1 collisions below the $2^{80}$ birthday bound), which is a failure of the *primitive*. And separately, even granting a perfect compression function, the iterated *construction wrapped around it* leaks behavior the ideal object does not have.

I want to be precise about that second failure, because it is the one that shapes the design. Merkle–Damgård fixes an initialization vector $H_0$, strengthens the padding with the encoded message length, splits into blocks $M_1\dots M_t$, iterates $H_i = f(H_{i-1}, M_i)$, and emits $H_t$. Its theorem is real: if $f$ is collision-resistant and the padding is strengthened, the hash is collision-resistant, because a collision in the hash forces a collision in $f$. But collision resistance is the *only* property that theorem transfers. Worse, no iterated hash with a finite chaining state can ever *be* a random oracle. The chaining value lives in a finite set, so state collisions necessarily exist: two distinct messages $M_1, M_2$ reaching the same chaining value. For any suffix $N$, the continuation $f(H_{i-1}, \cdot)$ is deterministic from that shared state, so $M_1\|N$ and $M_2\|N$ produce identical outputs forever — a behavior a random oracle simply lacks, since $\mathrm{RO}(M_1\|N)$ and $\mathrm{RO}(M_2\|N)$ are independent. So the right question is not "is it a random oracle" but "how far is it, and can that gap be driven down by a parameter I choose." The most visible symptom of the gap is length-extension: because the emitted digest *is* the final chaining state, knowing $H(M)$ and $\mathrm{len}(M)$ — not $M$ — lets an attacker resume the iteration and compute $H(\mathrm{pad}(M)\|Y)$, forging the naive MAC $H(K\|M)$. The world patches this with HMAC and MGF1, and the fact that the safe way to use these hashes is to wrap them in *more* structure is the tell: the base construction hands you its resumable state as the output. The Davies–Meyer route to $f$ — $h(H,M) = E_M(H)\oplus H$ — only deepens the stack, dragging in a *keyed* block cipher with its key schedule and related-key concerns, just to obtain a fixed-length compression function that must then be wrapped (prefix-free encoding, chop-MD, NMAC) to be indifferentiable in Coron's sense. Every layer is there to patch the one below it.

I propose the sponge construction, instantiated as the permutation Keccak — the design behind SHA-3. The single move that collapses the whole stack is to stop exposing the entire state. Take one fixed, *public* permutation $f$ on $b = r + c$ bits — no key, no key schedule, nothing secret inside; a block cipher is a key-indexed *family* of random-looking permutations, and if I do not need the family I just want *one* such permutation, keeping all the diffusion/nonlinearity/trail intuition while throwing away the key schedule. Split the $b$-bit state into an outer **rate** of $r$ bits and an inner **capacity** of $c$ bits. The message only ever enters the outer $r$ bits; the inner $c$ bits are never directly set by input and never directly output. Absorbing starts at $S = 0^b$, pads the message and cuts it into $r$-bit blocks $P_i$, and for each block XORs it into the rate and applies the permutation,
$$S \leftarrow f\bigl(S \oplus (P_i \| 0^c)\bigr),$$
where the $0^c$ enforces that input never touches the inner part — the inner bits change only because $f$ mixes the outer part into them. Squeezing reads off the outer $r$ bits of $S$, and to produce more than $r$ bits applies $f$ again, $S \leftarrow f(S)$, and reads the next $r$, truncating to the requested length $d$. Soak the message in through the rate; wring the output back out of the rate.

This kills length-extension structurally, not by an outer wrapper. The digest is a prefix of the outer bits, truncated to $d \le r$, so we reveal at most the rate and never the $c$ inner bits. To resume $f$ and extend, an attacker needs the *whole* $b$-bit state including the inner $c$ bits, which were never output and are not determined by anything she saw. The construction withholds exactly the part of the state that would let her continue, and it costs nothing — it hands me one tunable knob $c$. To earn the security claim I model $f$ as ideal and walk absorbing/squeezing as a walk on the graph of states under $S\mapsto f(S)$, grouping states by their inner value into $2^c$ supernodes. As long as the adversary's queries never produce two distinct absorbed paths reaching the same inner value — an **inner collision** — each fresh path lands in a supernode whose outer coordinate is still free, so the output bits she gets are uniform and independent, exactly like a random oracle. The *only* way the sponge betrays itself is an inner collision. So I bound that as a birthday process on the $C = 2^c$ inner values: before the $i$-th call she has at most $i$ reachable inner states, the $i$-th new edge collides with probability about $i/C$, and
$$P(\text{no inner collision}) = \prod_{i=1}^{N}\Bigl(1 - \frac{i}{C}\Bigr),\qquad P(\text{inner collision}) \approx 1 - \exp\!\Bigl(-\frac{N(N+1)}{2C}\Bigr) \approx \frac{N(N+1)}{2^{c+1}}.$$
The advantage depends on $c$ and *not at all* on $r$: the rate is free to be large for throughput, the security lives entirely in the capacity. Because $f$ is public the adversary can also compute $f^{-1}$, so I need the stronger statement — indifferentiability, with a simulator that fakes $f$ from random-oracle access and stays consistent across forward and backward edges; it can fail only when a query forces an inner collision, the same event, and the differentiating advantage comes out at about $N(N+1)/2^{c+1}$ for a random transformation (smaller for a random permutation). Up to $\sim 2^{c/2}$ queries the sponge is indifferentiable from a random oracle, inheriting *every* random-oracle property, and the surprise is that this collision and preimage resistance comes from an $f$ that is *trivially invertible* — invertibility buys the adversary nothing because she still cannot see $c$ bits of every state. A $d$-bit truncated oracle gives collisions at $2^{d/2}$ and preimages at $2^d$, so to keep the sponge term from being the bottleneck I want $2^{c/2}\ge 2^d$, i.e. **$c = 2d$**: SHA3-256 runs at $r=1088, c=512$; SHA3-512 at $r=576, c=1024$, with the width fixed at $b = 1600$.

Absorbing needs the input cut into exact $r$-bit blocks, so the padding must be injective. The minimal honest rule is **pad10*1**: append a $1$, then $j = (-m-2)\bmod r$ zeros, then a closing $1$. The final $1$ matters for two reasons — it forces the last rate block nonzero even on a boundary, and it makes the rule *multi-rate*, so one $f$ serves different rates $r$ without messages at different rates ever colliding; appending at least one bit also guarantees $f$ is called at least once, giving the empty input a well-defined $f$-dependent digest. Before pad10*1 a short domain-separation suffix is appended — hashes append $01$, the extendable-output functions append $1111$ — so a hash and an XOF can never collide on the same message.

That leaves building the permutation. Lay the $1600$ bits out as $A[x][y][z]$, $0\le x,y<5$, $0\le z<w=64$: a $5\times5$ array of $64$-bit lanes. A round is $R = \iota\circ\chi\circ\pi\circ\rho\circ\theta$, and each step earns its place. **$\chi$** supplies nonlinearity — without it the whole map is linear and invertible by solving a system. Within each row of $5$ bits, $A'[x] = A[x] \oplus ((\neg A[x+1]) \wedge A[x+2])$ (indices mod $5$). It is the *only* nonlinear step, has algebraic degree $2$ (the minimum that is nonlinear, hence cheap and analyzable, and the property that lets differential/linear trail weights be bounded), and is invertible. But $\chi$ mixes only within a row, so I need linear diffusion. **$\theta$** diffuses columns: $C[x,z] = \bigoplus_y A[x,y,z]$, $D[x,z] = C[x-1,z]\oplus C[x+1,z-1]$, and $A'[x,y,z] = A[x,y,z]\oplus D[x,z]$; the $z-1$ twist couples lanes so $\theta$ does not act slice-by-slice. $\theta$ is cheap but has branch number only $4$ — if every column already has even parity it is the identity (the column-parity kernel), and low-weight kernel differences slide through untouched. I deliberately do *not* fix this by making $\theta$ heavier; I fix it downstream. **$\rho$** rotates each lane along $z$ by the triangular offset $(t+1)(t+2)/2 \bmod 64$ marched around the lanes (lane $(0,0)$ at offset $0$), shearing apart bits that were aligned in $z$ — inter-slice dispersion, pure wiring, invertible. **$\pi$** permutes lane positions, $A'[x,y] = A[(x+3y)\bmod 5,\,x]$, a single $24$-cycle that tears apart the alignment $\theta$ relies on so a cheap kernel pattern in one round is scattered into different rows and columns the next and cannot reproduce itself cheaply; this is why $\pi$ must run through all axes in one cycle. The chosen $\pi,\rho$ offsets rule out four-point kernel rectangles for $w>16$, push the best three-round vortex to length $6$ (weight $36$, not the tempting $24$), and tighten the admissible four-round patterns from $\sim2^{20}$ down through $2^{16}$, $2^{12}$. Finally **$\iota$** breaks symmetry: $\theta,\rho,\pi,\chi$ are all $z$-translation-invariant and identical every round, inviting slide attacks and symmetric fixed points, so $\iota$ XORs an LFSR-generated round constant into lane $(0,0)$ — $R$ starts at $\texttt{0x01}$, each step $R \leftarrow (R\ll1)\oplus((R\gg7)\cdot\texttt{0x71})\bmod 256$, bits placed at positions $2^j-1$ — which $\theta$ and $\chi$ diffuse to all lanes within one round. $\theta$ goes first so the round mixes the inner capacity into the outer rate early in the sponge. With $w=64$, $l=\log_2 w = 6$, the round count is $n_r = 12 + 2l = 24$, a margin chosen so the climbing trail weights leave no usable differential or linear trail across the full permutation. In code, $\rho$ and $\pi$ fold into one walk over the $24$-cycle, rotating the carried lane by the triangular offset as it goes.

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
