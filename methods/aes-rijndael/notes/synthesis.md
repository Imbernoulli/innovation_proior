# Synthesis — Rijndael / AES

## Pain point (research question)
- DES (1977) is the incumbent block cipher. Two fatal/serious problems:
  1. **Key too short.** 56 effective key bits → brute force is feasible (EFF "Deep Crack" 1998 ~ 56 hours; key space exhaustible). A modern standard needs ≥128-bit keys.
  2. **Unjustifiable security.** DES S-boxes were designed by IBM/NSA with the design criteria kept SECRET. Later (1990) Biham–Shamir's differential cryptanalysis showed the DES S-boxes were suspiciously well-chosen to resist it (IBM knew the technique in 1974, asked to keep it secret) — but the public could not *argue* DES security from its structure. The secrecy bred backdoor suspicion and prevented public confidence.
- Goal: a cipher whose resistance to the *known* powerful attacks (differential, linear) is **arguable from the structure itself**, with a transparent design, fast in SW and HW, key ≥128.

## Attacks the cipher must resist (background, pre-method)
- **Differential cryptanalysis (Biham–Shamir 1990).** Track how an input XOR-difference a' propagates to output difference b' through the cipher; if some (a',b') holds with probability >> 2^(1-n), a chosen-plaintext attack recovers key. DES broken with 2^47 chosen plaintexts. Difference propagation probability over a *trail* = product of per-S-box difference propagation probs; non-active S-boxes (zero input difference) contribute prob 1.
- **Linear cryptanalysis (Matsui 1993).** Find a linear approximation (parity of plaintext bits ⊕ parity of ciphertext bits = parity of key bits) holding with prob ≠ 1/2; bias = correlation. DES broken with 2^43 known plaintexts. Correlation of a linear *trail* = product of per-S-box correlations; non-active S-boxes contribute correlation ±1.
- Both attacks' power is governed by the **number of ACTIVE S-boxes** in the best trail: each active S-box multiplies in a factor < 1 (its max diff prob, or its max correlation). Many active S-boxes ⇒ trail probability/correlation driven below the threshold ⇒ attack infeasible. So: BOUND THE MINIMUM NUMBER OF ACTIVE S-BOXES PER TRAIL.
- **GF(2^8)** finite field: bytes as polynomials of degree <8 over GF(2), arithmetic mod m(x)=x^8+x^4+x^3+x+1 (0x11B). Add = XOR. Multiply = poly mult then reduce mod m(x). xtime(b) = b·{02}: left shift, XOR 0x1B if MSB was 1.
- **SPN (substitution–permutation network):** alternate a nonlinear substitution layer (confusion, Shannon) with a linear permutation/mixing layer (diffusion). Iterate, XOR a round key each round.

## The method (derived)
**Key-alternating SPN over a 4×4 byte state (128 bits).** Round = SubBytes ∘ ShiftRows ∘ MixColumns, then AddRoundKey; initial whitening AddRoundKey before round 1; last round drops MixColumns. Nr = 10/12/14 for 128/192/256-bit keys.

### γ = SubBytes — the nonlinear layer
- Need an S-box with **low differential uniformity** (small max # of solutions to S(x⊕a)⊕S(x)=b over x) and **low maximum linear bias** (max correlation between input/output parities), to make each active S-box maximally damp a trail.
- The map x ↦ x^(-1) in GF(2^8) (with 0↦0) is the **inverse function**. Nyberg: over GF(2^n), n even, the inverse function has **differential uniformity 4** (any nonzero (a,b): ≤4 solutions ⇒ max diff prob 4/256 = 2^-6) and **max linear correlation 2^-3** (max bias) — provably near-optimal among 8-bit S-boxes. This is the *reason* to use the algebraic inverse: it's optimal and you can *prove* it, vs DES's table you can only test.
- But the pure inverse is algebraically simple (a single GF equation; has fixed points x with x^(-1)=x: x=0,1; could enable algebraic/interpolation attacks and looks "too clean"). Compose with an **invertible GF(2)-affine map** over the bits: b'_i = b̃_i ⊕ b̃_{(i+4)%8} ⊕ b̃_{(i+5)%8} ⊕ b̃_{(i+6)%8} ⊕ b̃_{(i+7)%8} ⊕ c_i, c = 0x63. The affine map is GF(2)-linear+constant so it does NOT change the differential uniformity or the linear bias (those are invariant under affine equivalence) — it only complicates the algebraic description, removes fixed points (no x with S(x)=x) and opposite fixed points (no x with S(x)=x⊕0xff), and frustrates interpolation attacks. So: best of both — provable DC/LC resistance from the inverse, no exploitable algebra from the affine.
- The affine matrix is a circulant (each row = previous rotated): chosen so its branch number / simplicity is good and it's easily implemented as a byte rotate-and-XOR. Constant 0x63 chosen so S has no fixed/opposite-fixed points.

### λ = ShiftRows + MixColumns — the linear diffusion layer, wide-trail
- **Wide-trail strategy:** instead of relying on the S-boxes to also diffuse, separate concerns — make the linear layer spread differences/masks across as many S-boxes as possible in the NEXT round, so any cheap (few-active-S-box) trail in one round is forced to activate many S-boxes in the next.
- **Branch number** of a linear map θ over bundles: B(θ) = min over nonzero input a of [ wt(a) + wt(θ(a)) ], where wt = number of nonzero bundles (bytes). It lower-bounds the number of active S-boxes across the substitution layers straddling θ in any trail. For a map on 4 bytes, B ≤ 5 (Singleton bound: an [n,k] MDS code over the bytes — here the 4→4 map's [8,4] code — has minimum distance ≤ n−k+1 = 5). B = 5 ⇒ **MDS** (Maximum Distance Separable).
- **MixColumns** = multiply each column by the fixed circulant MDS matrix over GF(2^8):
  rows [02 03 01 01 / 01 02 03 01 / 01 01 02 03 / 03 01 01 02]. Equivalently the column poly c(x)=03·x^3+01·x^2+01·x+02 multiplied mod x^4+1. This matrix is MDS ⇒ branch number exactly 5: any nonzero column difference with 1 active input byte produces ≥4 active output bytes (1+4=5), etc. Coefficients {01,02,03} chosen minimal (small ⇒ cheap: just xtime and one XOR) while still MDS. Inverse matrix has coefficients {09,0b,0d,0e} (heavier — decryption is slower, an accepted asymmetry).
- **ShiftRows**: cyclically shift row r left by r (rows 0,1,2,3 → 0,1,2,3 byte offsets). MixColumns mixes WITHIN a column; ShiftRows moves bytes BETWEEN columns so that the 4 bytes of any column come from 4 different columns of the previous state. This is what makes the per-column branch numbers compound across rounds.
- **The 4-round bound:** MixColumns gives B=5 active S-boxes minimum across a single column over the SubBytes layers it touches → over **two rounds** any (nontrivial) trail has ≥ 5 active S-boxes. ShiftRows' inter-column spreading makes the bound multiply: over **four rounds** any differential or linear trail has **≥ 25 = 5×5** active S-boxes. With max S-box diff prob 2^-6, a 4-round trail has prob ≤ (2^-6)^25 = 2^-150 << 2^-127 ⇒ no usable 4-round (hence no usable 8-round) differential or linear trail. That is the *provable, structural* security argument DES lacked.

### AddRoundKey + Key schedule
- AddRoundKey: XOR the 128-bit round key into the state. Makes it key-alternating (round transforms key-independent), which is exactly what lets you reason about trails independent of key.
- KeyExpansion: expand the cipher key into Nr+1 round keys (words w[i]). w[i] = w[i−Nk] ⊕ w[i−1], except every Nk-th word: w[i] = w[i−Nk] ⊕ SubWord(RotWord(w[i−1])) ⊕ Rcon[i/Nk]. RotWord = byte rotate; SubWord = S-box each byte (nonlinearity in the schedule, resists related-key/slide); Rcon[j] left byte = x^(j-1) in GF(2^8) (breaks symmetry between rounds, kills slide attacks). AES-256 adds an extra SubWord at i mod 8 = 4.

### CTR mode (high level)
- A block cipher only permutes 128-bit blocks. To encrypt a stream: **counter (CTR) mode** turns AES into a stream cipher. Keystream block_i = AES_K(nonce ‖ counter_i); ciphertext_i = plaintext_i ⊕ keystream_i. Counter increments per block. Only the *forward* AES is used (encrypt = decrypt structurally), fully parallel, random access. Needs a unique nonce per message (never reuse (key,counter)).

## Design decisions → why
- 4×4 byte state, 128-bit block: byte-orientation matches 8-bit S-box and GF(2^8); square geometry lets ShiftRows+MixColumns interlock for the wide-trail bound.
- S-box from inverse: provably min differential uniformity (4) and linear bias — arguable security.
- affine on top: invariant DC/LC properties, removes fixed points + algebraic simplicity.
- MixColumns MDS, coeffs {01,02,03}: max branch number (5) at minimum implementation cost.
- ShiftRows offsets (0,1,2,3): makes column-local diffusion spread globally → 25 active S-boxes/4 rounds.
- key-alternating XOR: lets the trail argument ignore the key.
- Rcon = x^(j-1): asymmetry between rounds, anti-slide.
- 10/12/14 rounds: enough trail-multiplications past the 25-active-S-box 4-round bound with margin.
- last round no MixColumns: makes encryption/decryption structurally alignable (so inverse cipher has same shape); MixColumns at the end would be peeled off trivially anyway.

## Code grounding
`code/aes_ref.py` (Bo Zhu / boppreh): s_box/inv_s_box tables, xtime, mix_single_column (the t/u XOR trick = the {02},{03} matrix), shift_rows, add_round_key, _expand_key (RotWord/SubWord/Rcon), encrypt_block/decrypt_block, encrypt_ctr/decrypt_ctr. Final answer.md code mirrors this.

## Three sources
1. PRIMARY: FIPS 197 (NIST, AES standard) + Daemen & Rijmen "The Wide Trail Design Strategy" (the design rationale of Rijndael). Read in full.
2. BACKGROUND: DES (Wikipedia/standard — 56-bit key, secret S-box, broken by brute force); differential cryptanalysis (Biham–Shamir, 2^47); linear cryptanalysis (Matsui, 2^43); GF(2^8); SPN.
3. EXPLAINERS: Wikipedia Rijndael MixColumns (MDS, branch number 5, 25 active S-boxes), Wikipedia Rijndael S-box (inverse + affine rationale, fixed points), web search syntheses.

## Unsourced / flagged
- The exact constants (m(x)=0x11B, S-box table, MixColumns coeffs, Rcon, affine matrix+0x63, Nr values) are all from FIPS 197 directly. Differential uniformity 4 / linear bias 2^-3 of the inverse: from explainers (Nyberg's result), consistent with FIPS's choice. No fabricated benchmark numbers. The EFF "Deep Crack 56 hours" figure is a well-known historical fact but only invoked qualitatively ("brute-forceable").
