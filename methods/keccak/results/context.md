# Context: designing a cryptographic hash that behaves like a random oracle

## Research question

We want a cryptographic hash function: a map from arbitrary-length messages to fixed-length digests that is collision-resistant, preimage-resistant, and second-preimage-resistant — and, more demandingly, that exhibits *no* structural weakness a random oracle would not have. The deployed hash functions of the day (MD5, SHA-0/1, SHA-2, RIPEMD) are all built the same way, and that shared construction has begun to fail in two distinct senses at once. First, the compression functions at their core are falling to collision search (MD5 collisions are practical; SHA-1 collisions are within theoretical reach). Second, even granting a perfect compression function, the *construction wrapping it* leaks properties a random oracle does not have — most visibly, length-extension. The question is whether there is a different way to build a variable-input, variable-output hash whose security can be stated as a single compact claim ("as strong as a random oracle up to a stated bound") and whose only deviation from that ideal is something we can quantify and push arbitrarily far down with one parameter.

## Background

**The random-oracle ideal.** A random oracle (Bellare–Rogaway, 1993) maps each distinct input to an independent, uniformly random infinite output string; truncated to n bits it resists collisions in ~2^{n/2} queries and (second) preimages in ~2^n. It satisfies every security criterion anyone has proposed, and new criteria keep appearing (correlation-freeness, length-extension resistance, chosen-target-forced-prefix preimage resistance), so a single criterion — "behaves like a random oracle" — is the most economical specification a designer could hope to make.

**Iterated hashing and its intrinsic gap.** Practical hashes are iterated over a chaining value so a stream can be hashed on the fly without buffering the whole message. But a finite chaining state means *state collisions* exist: two messages M1, M2 reaching the same chaining value. From any such collision, M1‖N and M2‖N collide for every suffix N. A random oracle has no such property — even when RO(M1)=RO(M2), the outputs on M1‖N and M2‖N are independent. So an iterated hash with a finite state can *never* be a random oracle; the right question is not "is it a RO" but "how far is it, and can the gap be made negligible."

**The empirical failures that forced the issue.** Wang et al. demonstrated practical MD5 collisions in 2004, and collision attacks on SHA-1 below the 2^80 birthday bound followed (Wang, 2005). These are observed breaks of the *compression-function* layer of the dominant family. They prompted a public competition: a call for new hash algorithms was issued on 2 November 2007, submissions closed 31 October 2008 with 51 entries, narrowed to 14 (July 2009) and then to five finalists (December 2010). The competition is the backdrop into which a new construction must fit: it must be efficient, must come with a clean security argument, and should avoid the structural pitfalls that the prevailing construction exhibits.

**Length-extension as a structural (not compression-function) defect.** In the prevailing iterated design the emitted digest *is* the chaining state after the message (up to an optional finalization). Knowing the digest and the message length — but not the message — lets one resume the iteration and compute the digest of message‖padding‖suffix. The naive MAC H(K‖M) is forgeable this way. In practice this is patched with extra structure (HMAC, MGF1), which is a tell: the base construction is structurally leaky and must be wrapped to be safe.

**Indifferentiability.** Maurer et al. introduced indifferentiability; Coron et al. (2005) applied it to hashing: a construction C calling an ideal primitive F is indifferentiable from a random oracle if some simulator can mimic F using only RO access so that no distinguisher querying both interfaces (the construction *and* the primitive) can tell the two worlds apart. Coron et al. gave variants of the iterated construction (prefix-free encoding, chop-MD, NMAC/HMAC) that are indifferentiable from a RO when the compression function is modeled as an ideal fixed-input-length object — an ideal compression function or an ideal block cipher. The takeaway reframes hash design: instead of engineering a collision-resistant compression function, engineer a fixed-length primitive that *looks ideal*, and let a proven-sound construction lift it to a full hash.

## Baselines

**Merkle–Damgård (Merkle 1989, Damgård 1989).** Fix an initialization vector H_0. Pad the message (a 1 bit, zeros, and the encoded message length — "strengthening") and split into blocks M_1…M_t. Iterate H_i = f(H_{i-1}, M_i) with a fixed compression function f, and output H_t. Core guarantee: if f is collision-resistant and the padding is strengthened, the whole hash is collision-resistant — a collision in the hash forces a collision in f. This is its strength and the reason it dominates. Its gap: (1) collision-resistance is the *only* property the theorem transfers; other expectations (length-extension resistance, behaving like a RO under many criteria) are not guaranteed, and indeed length-extension holds because the output equals the chaining state; (2) it inherits whatever weaknesses the concrete f has, and concrete f's are now breaking.

**Davies–Meyer compression from a block cipher.** Build f from a block cipher E by h(H, M) = E_M(H) ⊕ H: encrypt the chaining value under the message block as key, then feed-forward XOR. This underlies MD5, SHA-1, and SHA-2. Under the ideal-cipher model the feed-forward makes it one-way and collision-resistant, and Coron et al.'s indifferentiable variants can be instantiated on top of it. Its gap: it demands a *keyed* primitive — a block cipher with a key schedule — bringing key-schedule cost and related-key concerns, and it still sits inside an iterated wrapper that must be made indifferentiable by extra encoding.

**Chop-MD / prefix-free / NMAC-HMAC (Coron et al. 2005).** Repairs of the iterated construction that achieve RO-indifferentiability: drop part of the output, prefix-free-encode the input, or nest two keyed calls. Each removes length-extension and yields a RO-indifferentiability proof, but as a *fixed-output-length* object built on an ideal compression function or ideal cipher; none natively produces arbitrary-length output, and each adds wrapper machinery rather than changing the underlying iteration.

**Permutation-based predecessors (PANAMA, RadioGatún, Grindahl).** A line of designs that hash by iterating a large fixed permutation/round function over a state rather than a keyed compression function. They show a permutation can drive a hash, but their security claims were ad hoc; what is missing is a construction with a *provable* reduction to the strength of the permutation.

## Evaluation settings

The yardstick is the security a truncated random oracle provides for a digest of length d: ~2^{d/2} work for a collision, ~2^d for a (second) preimage, and resistance to length-extension and related structural distinctions. A construction is judged by (a) a provable indistinguishability / indifferentiability bound against a RO, expressed in the number N of queries to the underlying primitive (including inverse queries when the primitive is invertible), and (b) the implied generic-attack complexities for collision, preimage, and second preimage. Standard known-answer test vectors (the digests of the empty string and of short fixed messages) serve as correctness checks. Efficiency is measured as throughput — work per input bit absorbed — on the fixed state, on both software and hardware. The structural primitive (a large fixed permutation on b bits) and the differential/linear-trail machinery used to argue its strength are pre-existing analysis tools; only the *outcomes* for the new design are out of scope here.

## Code framework

The primitives that already exist: fixed-width bitwise operations (rotate, XOR, AND, NOT) on machine words, little-endian packing of bytes into and out of a fixed-size state, and a generic iterated-hashing harness that pads an input and processes it block by block. The slots a new design will fill are the fixed-length primitive on the state and the variable-input/variable-output mode wrapped around it.

```python
B = 1600                      # state width in bits (known: we want a large fixed state)
LANES, W = 25, 64             # 5x5 lanes of 64-bit words

def rol64(a, n):
    return ((a >> (64 - (n % 64))) | (a << (n % 64))) & ((1 << 64) - 1)

def load64(b):                # little-endian bytes -> 64-bit lane
    return sum(b[i] << (8 * i) for i in range(8))

def store64(a):               # 64-bit lane -> little-endian bytes
    return list((a >> (8 * i)) & 0xFF for i in range(8))

def fixed_primitive(state):
    """A fixed-length transformation on the b-bit state.
    The contribution lives here: which transformation, and how its
    internal structure provides diffusion and nonlinearity. # TODO
    """
    pass

def pad(message_bytes, rate_in_bytes):
    """Extend the message to a multiple of the absorbed block size,
    injectively. # TODO: the exact rule."""
    pass

def hash(rate, capacity, input_bytes, output_len):
    """Variable-input, variable-output mode built on fixed_primitive.
    How input enters the state, what part of the state is exposed as
    output, and what part is held back — # TODO."""
    pass
```
