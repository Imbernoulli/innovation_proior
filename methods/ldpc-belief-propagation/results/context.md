## Research question

Can a family of error-correcting codes be both *good* (driving the probability of decoding error toward zero at a fixed rate below channel capacity, with the error exponent of the best codes) *and* practically decodable (encoding and decoding cost growing only modestly — ideally linearly — with the block length)? Shannon's 1948 channel-coding theorem guarantees that good codes exist for any rate below capacity, and Elias's 1955 random-coding refinement shows that *almost all* long linear codes are good. But both results are existence proofs: they say nothing about how to *decode* a particular long code without searching over an exponential number of codewords. By 1960 the field has good-but-undecodable random codes on one side and decodable-but-weak short algebraic codes on the other, and no construction that is simultaneously near-capacity and cheap to decode. The goal is a class of "random-like" codes whose very structure admits a decoder whose cost per bit does not blow up with block length.

## Background

**Shannon's channel-coding theorem (1948).** For a memoryless channel of capacity C, and any rate R < C, there exist block codes of length n with probability of decoding error that → 0 as n → ∞. The proof averages over a random ensemble of codes, so it is non-constructive and gives no decoder.

**Elias's random-coding exponent and linear codes (1955).** Elias sharpened this into matching upper and lower bounds on the smallest achievable error probability for codes of length n on the binary symmetric channel (BSC); these decrease exponentially in n for any R < C, and they coincide over a range of rates up to capacity. Two consequences: (a) reaching small error near capacity *requires* large n; (b) almost all randomly chosen codes are essentially as good as the best — "most codes are good." Elias further showed the special class of *linear* codes has the same average performance as fully random codes, so one may restrict to linear codes (simple encoding) without loss. This is the prevailing wisdom of the time: good codes are abundant; the hard part is decoding one.

**Linear block codes and parity checks.** An (n, k) binary linear code is a k-dimensional subspace of GF(2)^n. It is described either by a generator matrix G (k × n), codewords c = uG, or by a parity-check matrix H (m × n) with H c^T = 0 for every codeword; H is not unique. Each row of H is a parity-check equation — a subset of code positions whose modulo-2 sum must be zero. Encoding is cheap; decoding — finding the codeword closest to a noisy received vector — is the bottleneck.

**Maximum-likelihood / optimal decoding is intractable.** Given received y, the optimal decoder seeks the codeword maximizing P(y | c). For a general long code this means comparing against ~2^k codewords or solving a closest-vector problem; the cost is exponential in the code's dimension. This is exactly why Elias's "most codes are good" cannot be cashed in directly.

**The diagnostic tension known at the time.** For *all* then-known explicitly constructed code families (and for a parity-check matrix filled with independent equiprobable bits), the ratio of minimum distance to block length tends to 0 as n grows — they become asymptotically weak — while the genuinely random ensemble keeps a linear minimum distance but cannot be decoded. Structure that helps decoding seems to cost distance; distance seems to require unstructured randomness that defeats decoding.

**Probabilistic / soft-decision view of decoding.** Rather than first thresholding each received symbol into a hard bit and then decoding, one can keep the channel's *a posteriori* probability for each transmitted bit. Throwing away the soft information (e.g. converting a Gaussian channel to a BSC by per-symbol hard decisions) provably lowers the usable capacity, so a decoder that operates directly on per-bit a-posteriori probabilities has more to work with.

**Competing decoding ideas of the era.** Wozencraft's sequential decoding of tree/convolutional codes was the leading practical method, but its rate was believed to be bounded by the computational cut-off rate R0, below capacity. Threshold decoding (Massey) was simple but limited. None offered near-capacity performance at feasible, block-length-scalable cost.

## Baselines

- **Algebraic block codes (Hamming; BCH and Reed–Solomon).** Codes built from finite-field structure with guaranteed minimum distance and bounded-distance algebraic decoders. Core idea: design H/G so that syndromes algebraically locate a bounded number of errors. Gap: their minimum-distance-to-length ratio falls toward 0 as n grows, and bounded-distance decoding corrects far fewer errors than a maximum-likelihood decoder would; they sit well short of capacity for the long block lengths capacity demands.

- **Random linear codes (the Shannon/Elias ensemble).** Pick H (or G) with independent random bits. Core idea / strength: by Elias, these achieve the best error exponent and linear minimum distance with high probability. Gap: no decoder better than exponential search is known — exactly the object the theory praises and cannot use.

- **Convolutional codes with sequential decoding (Wozencraft, Fano).** A linear time-varying code whose tree structure permits a sequential search decoder. Strength: soft-decision capable, the dominant practical scheme. Gap: throughput collapses (decoding effort has infinite mean) above the computational cut-off rate R0 < C, so it cannot be pushed to capacity; performance also depends on a feedback/retransmission style of operation in hard cases.

- **Threshold decoding (Massey) and bit-by-bit majority logic.** Decide each bit from a small set of parity checks by majority/threshold. Strength: extremely simple, linear cost. Gap: weak — corrects far fewer errors than optimal, and the simple form ignores how one corrected bit could help resolve another through shared checks.

## Evaluation settings

- **Channels.** The binary symmetric channel (BSC) with crossover probability p0, used because the number of crossovers can be controlled exactly (eliminating channel-variation noise in measurements) and because it is the standard yardstick for comparing coding/decoding schemes. The binary-input additive white Gaussian noise (AWGN) channel with inputs ±a and noise variance σ², where the per-symbol log-likelihood ratio is proportional to the received value, used to test whether keeping soft channel information beats hard-thresholding to a BSC.
- **Code parameters.** Block length n, dimension k, rate R = k/n, and any structural quantities that control distance, decoding cost, and memory. Representative simulations should use long enough blocks that brute-force decoding is no longer plausible, while still fitting the available computing machinery.
- **Metrics.** Probability of decoding error / block failure as a function of channel quality (crossover count on the BSC; signal-to-noise ratio Eb/N0 in dB on the Gaussian channel); minimum-distance-to-block-length ratio as a function of n (asymptotic goodness); decoding cost measured as operations per bit per iteration and number of iterations to converge.
- **Protocol.** Monte-Carlo simulation of channel + decoder on a general-purpose computer of the period; repeat over many noise realizations and count decoding failures. Comparison reference is the theoretical maximum-likelihood error bound for an ensemble of the same rate.

## Code framework

The available primitives are linear algebra over GF(2), a channel simulator, and a generic iterative-decoder loop. The open slots are the construction of H and the local rule run by the decoder.

```python
import numpy as np

# --- already available: GF(2) linear algebra ---------------------------------
def gf2_matmul(A, B):
    return (A @ B) % 2

def syndrome(H, x):
    return (H @ x) % 2                         # zero <=> x is a codeword

def systematic_generator(H):
    """G (k x n) with H G^T = 0, by Gauss-Jordan over GF(2). Standard."""
    # ... Gauss-Jordan elimination ...
    pass

# --- already available: channels ---------------------------------------------
def bsc(c, p0, rng):
    flips = rng.random(c.shape) < p0
    return c ^ flips

def biawgn(c, sigma2, rng):
    bpsk = 1 - 2 * c                           # 0 -> +1, 1 -> -1
    return bpsk + np.sqrt(sigma2) * rng.standard_normal(c.shape)

def channel_llr_awgn(y, sigma2):
    return 2 * y / sigma2                      # per-bit a-posteriori LLR

# --- SLOT 1: how to construct the parity-check matrix ------------------------
def build_parity_check(n, *params, rng):
    # TODO: the structure we will design for H
    pass

# --- SLOT 2: the per-bit decoding rule run on H -----------------------------
def decode(H, channel_info, maxiter):
    # TODO: the iterative local rule that turns channel_info into bit estimates
    # generic skeleton that exists regardless of the rule:
    for _ in range(maxiter):
        # ... update per-bit estimates from the parity-check structure ...
        x = hard_decision(...)                 # noqa: F821
        if not syndrome(H, x).any():
            return x, True
    return x, False

# --- already available: experiment harness -----------------------------------
def run(n, *params, channel, level, trials, rng):
    H = build_parity_check(n, *params, rng=rng)
    G = systematic_generator(H)
    for _ in range(trials):
        u = rng.integers(0, 2, G.shape[0])
        c = gf2_matmul(u, G)
        received = channel(c, level, rng)
        # ... feed received -> channel_info -> decode -> count bit errors ...
        pass
```
