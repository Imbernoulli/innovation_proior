# Synthesis — Merkle–Damgård + SHA-256

## Pain point
Need a public (unkeyed) hash mapping arbitrary-length messages to a fixed n-bit
digest, that is collision-resistant (CR): hard to find x≠y with H(x)=H(y). Also
want preimage / 2nd-preimage resistance. Problem: we only know how to design and
analyze *fixed-input-length* compression functions f:{0,1}^{n+b}→{0,1}^n. How to
lift a fixed-size CR primitive to arbitrary length WITHOUT new assumptions, and
with a *proof*?

Prior difficulty (Damgård intro): proofs got harder as message length grew. Early
provable CR hashes (claw-free permutations / modular squaring, Damgård [Da]) were
slow (RSA-per-message). Block-cipher hashes (Rabin: h_{i+1}=E_{x_i}(h_i)) were fast
but unproven / broken.

## The MD reduction (the theorem)
f:{0,1}^{n+b}→{0,1}^n CR. Pad message, split into b-bit blocks M_1..M_L (last
block(s) encode the bit-length — MD-strengthening). Iterate:
  H_0 = IV (fixed),  H_i = f(H_{i-1} ‖ M_i),  output H_L.
THEOREM: if f is CR then H is CR.
Proof (contrapositive / reduction): given a collision x≠x' with H(x)=H(x'), walk
the two chains from the LAST block backward. Final outputs equal: H_L = H'_{L'}.
At each step H_i=f(H_{i-1}‖M_i). If the two inputs to f at the matched step differ,
that pair IS an f-collision — done. If they're equal, step back one. Since lengths
are encoded (so equal final length ⇒ equal block count L=L'; differing length ⇒
the length block differs ⇒ collision at the length block), and x≠x', the inputs
cannot match all the way back to the IV (that would force x=x'). So some step
yields an f-collision. Contrapositive: f CR ⇒ H CR.
WHY length padding is essential: without it, m and m‖0...0 (trailing zeros) or two
messages of different block counts can chain to the same H without ever producing
an f-collision (e.g. prepend a block that maps IV→IV, or the suffix-freeness fails).
Damgård's exact device: append an extra block holding d = number of padding zeros,
so you can tell "padded with d zeros" apart from "ends in d zeros". FIPS uses the
cleaner 64-bit bit-length suffix (the strengthening): the padding map is
suffix-free / prefix-condition (Merkle–Damgård compliant): (i) M is a prefix of
pad(M), (ii) equal-length M give equal-length pad, (iii) different-length M give
different final blocks.

## Davies–Meyer — building f from a block cipher
We still need a concrete CR f. Rabin's f(a,b)=E_a(b) is NOT CR (encrypt with key
k, decrypt-pick with key k' gives a collision freely; also it's a permutation in b
so invertible). Fix: feed-forward — Davies–Meyer:
  f(H, M) = E_M(H) ⊕ H    (key = message block M, plaintext = chaining value H).
The ⊕H makes f non-invertible: you can't run the cipher backward to control the
output, because output = E_M(H) ⊕ H mixes input H back in. In the ideal-cipher
model, DM is CR: any attacker with q < 2^{l/2} queries finds a collision with prob
≤ q²/2^l (birthday bound). [Proof: each new query reveals a hash value uniform over
~2^l values; Pr[collide with a fixed earlier one] ≤ 2/(2^l−(i−1)); union over
C(q,2) ≤ q²/2 pairs ⇒ q²/2^l.] WHY ideal-cipher (not just strong PRP): we don't
know how to prove DM CR from PRP alone; need no related-key / no weak-key / random
even when key known.

## SHA-256's compression function (FIPS 180-4)
b = 512-bit block, n = 256-bit chaining value (8 × 32-bit words). DM with an
internal block cipher SHACAL-2 (key = 512-bit message block expanded to schedule;
plaintext = 256-bit state). 64 rounds.

Padding (5.1.1): append 0x80 (a single '1' bit + zeros), then zeros so that
len ≡ 448 mod 512, then 64-bit big-endian message bit-length. So total multiple of
512.

IV H^(0) (5.3.3): first 32 bits of fractional parts of √(first 8 primes):
6a09e667 bb67ae85 3c6ef372 a54ff53a 510e527f 9b05688c 1f83d9ab 5be0cd19.

K_t (4.2.2): first 32 bits of fractional parts of cube roots of first 64 primes.
(nothing-up-my-sleeve numbers ⇒ no hidden trapdoor in constants.)

Functions (4.1.2), all on 32-bit words, ROTR = rotate right, SHR = shift right:
  Ch(x,y,z) = (x∧y) ⊕ (¬x∧z)              -- choose
  Maj(x,y,z)= (x∧y) ⊕ (x∧z) ⊕ (y∧z)       -- majority
  Σ0(x) = ROTR2 ⊕ ROTR13 ⊕ ROTR22
  Σ1(x) = ROTR6 ⊕ ROTR11 ⊕ ROTR25
  σ0(x) = ROTR7 ⊕ ROTR18 ⊕ SHR3
  σ1(x) = ROTR17 ⊕ ROTR19 ⊕ SHR10

Message schedule (6.2.2): W_t = M_t for 0≤t≤15; for 16≤t≤63
  W_t = σ1(W_{t-2}) + W_{t-7} + σ0(W_{t-15}) + W_{t-16}  (mod 2^32).
This expands 16 → 64 words: diffuses each message word across many rounds so a
single-bit message change avalanches.

Round (6.2.2), working vars a..h = H^{(i-1)}:
  T1 = h + Σ1(e) + Ch(e,f,g) + K_t + W_t
  T2 = Σ0(a) + Maj(a,b,c)
  h=g; g=f; f=e; e=d+T1; d=c; c=b; b=a; a=T1+T2     (all mod 2^32)
After 64 rounds, feed-forward (the Davies–Meyer ⊕ realized as mod-2^32 add):
  H^{(i)}_j = H^{(i-1)}_j + (working var)_j.
Output after last block: concatenation of the 8 words = 256-bit digest.

WHY the round structure:
- Two "tracks": (a..d) updated through Σ0/Maj, (e..h) through Σ1/Ch. Only a and e
  get new injections each round; the rest are a shift register ⇒ cheap diffusion
  over 64 rounds.
- Σ (three rotations, no shift) gives intra-word diffusion / good avalanche;
  σ (two rotations + one shift) in the schedule — the shift breaks the pure
  rotation symmetry so the schedule isn't rotation-invariant.
- Ch / Maj are the only nonlinear (bitwise) mixers; combined with mod-2^32 add
  (carry = nonlinearity across bit positions) they defeat linear/differential
  cryptanalysis.
- + (mod 2^32) feed-forward instead of ⊕: makes the round function non-invertible
  as a map of the state (Davies–Meyer property) ⇒ can't run compression backward.

## Length-extension — structural weakness
Output H(M) = full final chaining value = the entire internal state after pad(M).
So an attacker who knows H(M) and |M| can set state = H(M), append any block(s) Y
(with their own padding/length), and compute H(pad(M) ‖ Y) WITHOUT knowing M. Lets
you forge H(secret ‖ msg ‖ Y) ⇒ naive MAC = H(key‖msg) is broken. (HMAC, or
truncated SHA-2 like SHA-512/256, or a finalization that hides the state, defeats
it.) This is NOT a collision break — the MD theorem still holds — it's that the
output equals the chaining state.

## Hash security goals / birthday (context facts)
- preimage: given y, find x with H(x)=y — ~2^n work.
- 2nd-preimage: given x, find x'≠x with H(x')=H(x) — ~2^n.
- collision: find any x≠x' colliding — birthday bound ~2^{n/2} (256→128-bit
  security). Generic, applies to any n-bit hash.

## Code (validated against hashlib): code/sha256_reference.py
pad → split into 64-byte blocks → compress (schedule + 64 rounds + feed-forward)
from IV → hex of 8 words. Passes vectors incl. "", "abc", long inputs.

## Three sources
1. PRIMARY: Damgård 1989 "A Design Principle for Hash Functions" (CRYPTO '89, LNCS
   435 pp.416-427) — refs/damgard1989.pdf (full text incl. reduction proof,
   Merkle meta-method note, Rabin/DES & ISO E_a(b)⊕b discussion). + NIST FIPS 180-4
   — refs/fips180-4.pdf (the SHA-256 spec).
2. BACKGROUND: Davies–Meyer ideal-cipher CR proof (IISc CSA notes, refs/
   davies_meyer_iisc.pdf); MD reduction + birthday (Harvard CS127 lec18, refs/
   harvard_cs127_lec18.pdf).
3. EXPLAINER: Wikipedia "Merkle–Damgård construction" (MD-strengthening conditions,
   length-extension).

UNSOURCED: none load-bearing. SHACAL-2 name and "nothing-up-my-sleeve" framing are
standard background, not asserted as derivation-time facts beyond what the
constants' definition (FIPS) supports.
