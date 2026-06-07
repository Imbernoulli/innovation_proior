# Synthesis — Keccak / sponge construction (SHA-3)

## Pain point (pre-method, knowable ~2007-2008)
- All significant hashes (MD4, MD5, SHA-0/1/2, RIPEMD) are iterated Merkle–Damgård (MD): H_i = f(H_{i-1}, M_i), IV fixed, output = H_last (possibly + finalization). MD strengthening = pad with a 1, zeros, then encode message length in the last block.
- MD collision theorem (Merkle/Damgård 1989): if f is collision-resistant and MD-strengthened padding used, the hash is collision-resistant. Attractive but ONLY transfers collision-resistance; not all RO-properties.
- Length-extension: output H(M) IS the chaining state after absorbing M. Given H(M) and len(M) (not M itself), an attacker resumes the iteration: compute H(pad(M)||Y) = f-chain starting from H(M). Breaks the naive MAC = H(K||M): forge H(K||M||pad||Y) without knowing K. Patched in practice by HMAC, MGF1 — extra structure to compensate for a structurally leaky construction.
- Empirical failures driving SHA-3 competition: Wang et al. 2004 practical MD5 collisions; theoretical SHA-1 collision attacks (Wang 2005, < 2^80). NIST call Nov 2 2007 (Federal Register), submissions closed Oct 31 2008, 51 → 14 (Jul 2009) → 5 finalists (Dec 2010: BLAKE, Grøstl, JH, Keccak, Skein) → Keccak winner Oct 2 2012, standardized FIPS 202 (Aug 2015).
- Ideal target: random oracle (RO), Bellare–Rogaway 1993. RO satisfies every known criterion. But an iterated hash with finite chaining state HAS inner/state collisions a RO lacks: M1,M2 colliding state ⇒ M1||N, M2||N collide for all N. So an iterated hash can NEVER be a RO. Need a reference model that is reachable and quantifies the gap.
- Coron et al. 2005 (MD revisited): cast hashing in Maurer's indifferentiability framework; gave MD variants (prefix-free, chop-MD, NMAC/HMAC) provably indifferentiable from RO if f is an ideal compression fn / ideal cipher (Davies–Meyer). So: "design f that behaves like a FIL-RO / ideal block cipher" rather than "design a collision-resistant compression fn."

## Davies–Meyer / block-cipher hashing (ancestor)
- Compression from block cipher E: h(H,M) = E_M(H) ⊕ H (Davies–Meyer). Underlies MD5/SHA-1/SHA-2. Feed-forward XOR makes it one-way/collision-resistant under ideal-cipher model. But: needs a keyed cipher with a key schedule (overhead, related-key worries), and the MD wrapper still leaks state.

## The move: sponge
- Replace the (compress-then-iterate) paradigm with a single FIXED permutation f on b = r + c bits (b=1600 for SHA-3). No key schedule. State split: outer r bits ("rate") + inner c bits ("capacity").
- ABSORB: S=0^b; pad input to multiple of r; for each r-bit block P_i: S ← f(S ⊕ (P_i || 0^c)). Message bits XOR only into the outer r bits; inner c bits are never directly touched by input.
- SQUEEZE: output = first r bits of S; if more needed, S ← f(S), take r more, repeat. Truncate to d bits.
- Capacity c is NEVER output and NEVER directly set by input. This is the structural fix:
  - Length-extension defeated: the digest exposes only r outer bits (in SHA-3, truncated to d ≤ r); the c inner bits that also determine future output stay hidden. You cannot resume the permutation from the digest because you don't know the inner state. (For SHA3-256, d=256 < r=1088, so even the full rate isn't revealed.)
- Padding: pad10*1 (multi-rate). Append 1, then minimal 0s, then 1, to reach a multiple of r. The final 1 makes the last block's rate contribution nonzero and makes padding injective across different rates (multi-rate). FIPS j = (-m-2) mod x; P = 1 0^j 1.
- Domain separation suffix (FIPS 202): SHA3 hashes append "01" before pad10*1 (so N = M||01); SHAKE XOFs append "1111". In the byte/bit-reflected convention of the reference code these merge with the first pad-1 bit: delimitedSuffix = 0x06 (= bits 0,1,1) for SHA3, 0x1F for SHAKE; the trailing pad-1 is XORed as 0x80 at the last rate byte.

## Security (sponge)
- Indistinguishability (Sponge functions 2007): a random sponge differs from a RO only via inner collisions; P(inner collision after N calls) ≈ N(N+1)/2^{c+1}. So bounded by capacity, not rate.
- Indifferentiability (EUROCRYPT 2008): sponge calling a random permutation/transformation is indifferentiable from a RO up to advantage ≈ N(N+1)/2^{c+1}, N = #calls to f and f^{-1}. First indifferentiability proof for a construction calling a random PERMUTATION (adversary may query f AND f^{-1}) and for arbitrary-length output. ⇒ generic security ~2^{c/2}. Collision resistance min(c/2, d/2); (2nd) preimage ~min(c/2, d). SHA-3 sets c = 2d so collision = d/2, preimage = d (matches RO-at-output-length).
- Throughput/security trade-off: cost per absorbed bit ~ one f-call per r bits. Larger r = faster but smaller c = weaker. r + c = b fixed. SHA-3 picks c = 2d (e.g. SHA3-256: c=512, r=1088).
- Why permutation not random transformation: P-sponge bound is slightly better than T-sponge; a permutation is what good block-cipher design already produces (minus the key schedule); invertibility of f does NOT hurt — collision/preimage resistance coexist with an easily invertible f because the secret is the hidden inner state, not f.

## Keccak-f[1600] round: b=1600 = 25 lanes × w=64 bits. State A[x][y][z], 0≤x,y<5, 0≤z<w. l=log2(w)=6, n_r = 12+2l = 24 rounds. Round = ι∘χ∘π∘ρ∘θ.
- θ (theta) — DIFFUSION (linear). Column parity: C[x,z] = ⊕_y A[x,y,z]; D[x,z] = C[(x-1)mod5, z] ⊕ C[(x+1)mod5, (z-1)mod w]; A'[x,y,z] = A[x,y,z] ⊕ D[x,z]. Each bit XORed with parity of two neighboring columns (one rotated 1 in z). Provides inter-column diffusion. Branch number only 4 (CP-kernel = states with all-even column parity, where θ=identity) — addressed by π.
- ρ (rho) — INTER-SLICE DISPERSION (linear). Rotate each lane (x,y) in z by triangular offset. Generated: start (x,y)=(1,0), for t=0..23: offset = (t+1)(t+2)/2 mod w, then (x,y)=(y,(2x+3y)mod5). Lane (0,0) offset 0. Without ρ, no diffusion BETWEEN slices.
- π (pi) — DISPERSION / lane transposition (linear). A'[x,y] = A[(x+3y)mod5, x]. Rearranges lanes; breaks up the column alignment so the CP-kernel low-weight trails can't persist round to round (long-term diffusion).
- χ (chi) — NONLINEARITY (the only nonlinear step). Per row: A'[x,y,z] = A[x,y,z] ⊕ ((¬A[(x+1)%5,y,z]) ∧ A[(x+2)%5,y,z]). Algebraic degree 2. Invertible. This is what makes f cryptographically strong (without it f is linear ⇒ trivially broken).
- ι (iota) — SYMMETRY BREAKING. XOR a round-dependent constant into lane (0,0). Constants from a max-length LFSR (rc(t)): R=0x01; iterate R = (R<<1) ⊕ ((R>>7)*0x71) mod 256; bit at positions 2^j−1 of the lane for j=0..l. Without ι all rounds identical and round fn is z-translation-invariant ⇒ slide attacks / symmetric fixed points. ι breaks both.
- Order: θ first because in the sponge it mixes inner (capacity) and outer (rate) parts early; the order of the rest is essentially arbitrary (χ must be separated from itself by the linear layer λ=π∘ρ∘θ).

## Canonical code: CompactFIPS202.py (Gilles Van Assche, CC0, keccak.team). Validated locally:
- SHA3-256("") = a7ffc6...434a ✓; SHA3-256("abc") = 3a985d...1532 ✓.
- Lane-oriented: ROL64 rotate; θ via C/D; ρ+π fused via the (x,y)←(y,(2x+3y)%5) walk with triangular rotation (t+1)(t+2)/2; χ per plane; ι via LFSR R, XOR at bit (1<<j)-1 when (R&2). load64/store64 little-endian lane packing. Keccak(rate,cap,input,delimitedSuffix,outLen): absorb loop XORs into state[i], f every full block; pad = XOR delimitedSuffix at offset, XOR 0x80 at last rate byte, f; squeeze loop emits state[0:rate] blocks, f between.

## Design-decision → why (no holes)
- Permutation (not compression fn): no key schedule, reuse block-cipher design intuition, invertibility harmless (hidden inner state). ✓
- r+c split, c never output: defeats length-extension + gives a single tunable security knob (c). ✓
- pad10*1 final 1: injective multi-rate padding, ensures ≥1 f-call and last block nonzero. ✓
- domain suffix 01/1111: separates hash vs XOF (and future) domains. ✓
- c = 2d: collision resistance d/2 = output-truncated-RO level; preimage d. ✓
- 24 rounds = 12+2l: design margin from differential/linear trail weights. ✓
- five steps each: θ diffusion, ρ inter-slice, π disperse/long-term, χ nonlinearity, ι symmetry. Each removes a specific weakness. ✓
