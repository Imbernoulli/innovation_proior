# Context: building a collision-resistant hash for arbitrary-length messages

## Research question

We want a single public function H that maps a message of *any* length to a short,
fixed-size digest — say n bits — and behaves like a "fingerprint": it should be
infeasible to find two different messages that share a digest. The fingerprint is
the workhorse behind digital signatures (you sign H(M), not M), data-integrity
checks, commitment schemes, and password storage, so its security is the security
of everything built on top of it.

"Public" is the crux: unlike a MAC there is no secret key, so the adversary knows
H completely and can evaluate it as much as we can. The three properties we need,
in increasing order of how easy they are to break:

- **Preimage resistance** — given a digest y, hard to find any M with H(M)=y.
- **Second-preimage resistance** — given M, hard to find M'≠M with H(M')=H(M).
- **Collision resistance** — hard to find *any* pair M≠M' with H(M)=H(M').

Collision resistance is the demanding one and the property signatures rest on (a
signer who finds a collision can repudiate). Two observations frame the problem.
First, the input is unbounded but the output is fixed, so collisions *exist* in
abundance — "hard to find" is the most we can ask, and we want a *proof* that
finding them is hard, reduced to a clean assumption. Second, we have no idea how
to design and analyze a function on unbounded input directly; every primitive we
know how to build and scrutinize takes a *fixed-size* input. The question is how
to lift a fixed-input building block to arbitrary length, ideally with the lifting
itself proved to preserve collision resistance.

## Background

**The birthday bound sets the bar.** For any n-bit output, a generic collision
search needs only about 2^{n/2} evaluations, not 2^n. Pick t random messages; the
number of unordered pairs is C(t,2)≈t²/2, each pair collides with probability about
2^{-n}, so the expected number of collisions is ≈ t²/2^{n+1}, which crosses 1 near
t≈2^{n/2}. Preimage and second-preimage searches, by contrast, cost about 2^n.
This birthday gap is why a hash aiming at, say, 128-bit collision security must
output 256 bits. It is a generic attack — it assumes nothing about the internals —
so it is the *ceiling*: no construction can do better, and the goal is a design
whose best attack is no better than this generic one.

**Fixed-input compression functions are the only thing we can analyze.** Call a
function f:{0,1}^{n+b}→{0,1}^n a *compression function*: it eats n+b bits and emits
n, shrinking by b bits per call. Collision resistance for such a fixed-shape f is a
self-contained combinatorial property we can study, model (e.g. in idealized
models), and even reduce to number-theoretic assumptions. The open problem of the
time is the *composition*: how to chain fixed-size compressions into an
arbitrary-length hash and prove the composite inherits collision resistance.

**Provable-but-slow vs. fast-but-unproven.** Two prior families framed the tension:

- *Number-theoretic, provable.* Hashes built from claw-free pairs of permutations
  or from modular squaring (Damgård's own earlier line) could be *proved*
  collision-resistant assuming a hard number-theoretic problem, but cost roughly an
  RSA operation per message block — far too slow for bulk data.
- *Cipher-based, fast.* Rabin's idea hashed by iterating a block cipher,
  h_{i+1}=E_{x_{i+1}}(h_i) with a fixed start h_0, keying the cipher with successive
  message blocks.

**Cipher-based compression.** Writing the per-block map as f(a,b)=E_a(b) (key a,
plaintext b) shows the structure: for any target c, solving E_a(b)=c for (a,b) is
trivial — pick any key a and set b=E_a^{-1}(c), since E_a is a permutation in its
data argument. The era's proposals to build a compression function out of a block
cipher all had to contend with this invertibility, and it was an open question
whether any of them could be argued collision-resistant from properties of E alone.

**Padding and block sequences.** A pre-method fact: when chaining compressions over
message blocks, the padding rule determines which bit sequences are fed to the chain.
Two messages that differ only by how the final block is filled can drive the chain
to the same final value if the padding is not carefully designed. Whatever a
composition argument needs from the padding, it depends on the padding being
*injective* in an appropriate sense.

**The ideal-cipher model.** For arguing collision resistance of a cipher-based
compression function, the relevant idealization treats the block cipher E as an
ideal cipher: every key k names an independent, uniformly random permutation
E(k,·), accessible only by querying an oracle for E and E^{-1}. This rules out
related-key behaviour (E(k,·) and E(k',·) are independent even if k,k' differ in
one bit), weak keys, and any structure that survives knowing the key. It is
stronger than assuming E is a strong pseudorandom permutation — but a strong-PRP
assumption alone is not known to suffice for collision resistance of the per-block
function.

**Word-level cryptographic primitives.** On the engineering side, the building
blocks for a software-friendly compression function over 32-bit words are: bitwise
AND/OR/NOT/XOR, circular rotations ROTR^n and logical shifts SHR^n, and addition
modulo 2^32 (whose carry chain is the cheap source of cross-bit nonlinearity).
"Nothing-up-my-sleeve" constants — derived from fractional parts of square/cube
roots of small primes — are the standard way to publish fixed constants that
visibly hide no trapdoor.

## Baselines

- **Iterated cipher hash (Rabin / DES-based).** h_0 fixed; h_{i+1}=E_{x_{i+1}}(h_i);
  output the last h. Core idea: reuse a trusted block cipher as the mixing engine.
  With a 64-bit block the chaining value is 64 bits wide.

- **Claw-free / modular-squaring hashes (Damgård's earlier construction).** Build
  collision resistance from claw-free permutation pairs (f_0,f_1): finding x≠y with
  f_0(x)=f_1(y) is hard. These were the first hashes with a real collision-resistance
  *proof* under a number-theoretic assumption, with proofs tailored per-construction.

- **Naor–Yung composition (independent).** Shows fixed-size hashes can be composed
  to compress arbitrary polynomial-length messages, including for a weaker
  "universal one-way" (target-collision) property built from any one-way
  permutation, with signature-scheme applications. The weaker-property version
  draws a *fresh independent instance* of the fixed-length hash per message block.

- **Merkle's "meta-method" (independent, concurrent).** The same chaining intuition —
  iterate a fixed compression over message blocks from a fixed IV.

## Evaluation settings

The natural yardsticks are not benchmark datasets but adversarial work factors and
spec conformance:

- **Generic attack costs** as the reference scale: ≈2^{n/2} for collisions, ≈2^n for
  (second-)preimages, against an n-bit digest. A construction is judged by whether
  its best known attack stays at this generic ceiling.
- **Reduction tightness**: how a collision in the whole hash translates into a
  collision (or preimage) in the underlying compression function — how many
  compression-collisions, and how much extra work, one composite collision costs.
- **Idealized-model query bounds**: for a cipher-based compression function, the
  number of ideal-cipher queries q an adversary may make versus its collision
  probability.
- **Conformance test vectors**: a candidate digest must match the published expected
  digests for fixed inputs (e.g. the empty string, the 3-byte string "abc", and
  longer multi-block inputs), bit for bit.

## Code framework

The primitives that already exist: 32-bit word arithmetic with rotations/shifts and
addition mod 2^32, byte/word (de)serialization in big-endian order, and a fixed
public IV. The shape of an iterated hash — pad, split into fixed-size blocks, fold
each block into a running chaining value from the IV, emit the final value — is the
known skeleton. The two slots the method must fill are the *padding rule* and the
*compression function* itself.

```python
WORD_MASK = 0xffffffff

def rotr(x, n):
    return ((x >> n) | (x << (32 - n))) & WORD_MASK

def shr(x, n):
    return x >> n

# Fixed public initialization vector: the chaining value before any block.
IV = []          # TODO: the fixed starting chaining value

def pad(message: bytes) -> bytes:
    # TODO: the padding rule that turns the message into a whole number of blocks.
    pass

def compress(state, block: bytes):
    # TODO: the fixed-input compression function f(state, block) -> new state.
    pass

def hash_message(message: bytes) -> str:
    state = list(IV)
    data = pad(message)
    for off in range(0, len(data), BLOCK_BYTES):   # TODO: BLOCK_BYTES
        state = compress(state, data[off:off + BLOCK_BYTES])
    return serialize(state)                         # TODO: hex of the words
```
