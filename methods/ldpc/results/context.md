# Context

## Research question

A channel corrupts the bits we send. We want to send k information bits as a longer block of n bits so that, even after the channel flips or noises some of them, the receiver recovers the original k bits with vanishingly small error probability — and we want to do this at the highest possible information rate R = k/n.

The existence side of this problem is settled. Shannon (1948) showed that every memoryless channel has a capacity C, and that for any rate R < C there exist block codes of length n whose probability of decoding error goes to zero as n → ∞ — in fact exponentially in n. His proof draws the code *at random*: pick 2^{nR} codewords uniformly from the 2^n binary strings, and decode a received word to the nearest (most likely) codeword. Averaged over this random ensemble, the error probability is small, so *some* code in the ensemble is at least that good.

Decoding a random code requires comparing the received word against all 2^{nR} codewords and keeping the most probable. That cost grows exponentially with n. The question, then, is: **can a family of codes have enough internal structure that decoding is feasible in practice, while still keeping the error probability low at a usable rate?**

## Background

**Linear codes and parity checks.** The standard way to give a code structure is to make it *linear* over GF(2). A linear code of length n is the null space of a parity-check matrix H: the codewords are exactly the binary vectors x with Hx = 0 (mod 2). Each row of H is one parity-check equation — a subset of bit positions constrained to sum to zero. If H is (n−k)×n of full rank, the code has rate R = k/n. Encoding and the membership test Hx = 0 are pure linear algebra over GF(2); decoding means inferring which codeword was sent given a noisy observation.

**Decoding is inference.** On a memoryless channel the received vector y gives, for each bit, a likelihood — on the binary symmetric channel (BSC) a crossover probability p, on the additive-white-Gaussian-noise (AWGN) channel with antipodal signalling x_n = ±1 a Gaussian density. Optimal decoding finds the codeword maximizing the posterior P(x | y) subject to Hx = 0. The information the channel hands us is genuinely *soft*: a per-bit a-posteriori probability, not just a hard 0/1 guess.

**The distance picture and its ensemble statistics.** The error-correcting power of a code is classically summarized by its minimum distance D — the smallest number of bit positions in which two codewords differ. A decoder that always returns the nearest codeword corrects any error pattern of weight up to ⌊(D−1)/2⌋. For an ensemble of randomly built parity-check matrices one can compute the *average* distance distribution rather than any single code's: the expected number of weight-ℓ codewords is N̄(ℓ) = (n choose ℓ) 2^{−n(1−R)}, because a fixed nonzero weight-ℓ pattern satisfies each of the n(1−R) independent parity checks with probability ½. Bounding the tail of this with Stirling gives an exponent H(δ) − (1−R) ln 2. For every δ below the root δ_0 satisfying H(δ_0) = (1−R) ln 2, the expected number of codewords of weight at most δn goes to zero exponentially, so the typical minimum distance grows linearly with n.

**Graphs of codes.** A parity-check matrix is equivalently a bipartite graph: one node per bit, one node per parity check, and an edge wherever H has a 1. This is a standard alternate view of H.

**Conditional independence on graphs.** A general fact about probabilistic models on graphs: if a graph has no cycles, then cutting any edge splits it into two parts whose evidence is conditionally independent. On a graph with cycles this independence is only approximate, and the failure is worse the shorter the cycles.

## Baselines

**Algebraic / bounded-distance codes (Hamming, BCH, Reed–Solomon).** These build H (or its dual) from algebraic structure — roots of polynomials over finite fields — so that an efficient algebraic procedure (syndrome computation, the Berlekamp–Massey / Peterson–Gorenstein–Zierler machinery) corrects *any* error pattern of weight up to a guaranteed t = ⌊(D−1)/2⌋, and nothing beyond. The decoder computes a syndrome from the received bits and corrects up to t flips. Decoding cost for BCH on the BSC grows roughly as the cube of the block length.

**Sequential decoding of convolutional codes (Wozencraft; Fano).** A probabilistic decoder that uses soft information, searching the code tree of a convolutional code guided by a likelihood metric. It achieves a probability of error bounded like e^{−αn} in the constraint length n for rates below capacity. The computation per decoded digit is a random variable that depends on the channel noise, and there is a practical rate ceiling — the computational cutoff rate R_comp < C — below which the average work per digit is bounded.

**Threshold decoding (Massey).** The simplest of the probabilistic schemes — shift registers, a few modulo-2 adders, and a threshold element forming a majority vote of parity checks on a bit. Cheap and fast, effective at relatively short constraint lengths.

## Evaluation settings

The natural yardsticks already exist. The two canonical channels are the binary symmetric channel, parameterized by a crossover probability p (each bit independently flipped with probability p), and the binary-input additive-white-Gaussian-noise channel, where each codeword bit is sent antipodally as x_n = ±1 and received as y_n = x_n + noise of variance σ². For the AWGN channel the operating point is reported as E_b/N_0 in decibels, the energy per information bit over the noise spectral density, related to the per-symbol signal-to-noise ratio through the rate R. The figure of merit is the decoded bit-error (or block-error) probability as a function of channel quality (p, or E_b/N_0), read against the Shannon limit for the given rate — the smallest channel quality at which rate-R reliable communication is information-theoretically possible. Complexity is measured as arithmetic operations per decoded bit per iteration and the number of iterations, as functions of block length n. Existing reference points on these axes are the convolutional codes with sequential decoding and the concatenated (convolutional + Reed–Solomon) constructions of the day.

## Code framework

The scaffold is a linear-code simulation harness: GF(2) matrix products, Gaussian elimination, a parity-check constructor, a generator constructor, a BPSK/AWGN channel, a decoder, and a parity-check stopping test. The constructor and the decoder are left open.

```python
import numpy as np

def binaryproduct(X, Y):
    return X.dot(Y) % 2

def gaussjordan(X):
    # row-reduce over GF(2); used to turn H into a generator G
    ...

def parity_check_matrix(n, *params, seed=None):
    """Return an (n-k) x n parity-check matrix H over GF(2).
    The structure of H is the design choice to be made."""
    pass  # TODO

def coding_matrix(H):
    # standard: derive a generator G from H so that H G = 0, x = G v
    ...

def encode(G, v, snr, seed=None):
    d = binaryproduct(G, v)
    x = (-1) ** d                      # bit 0 -> +1, bit 1 -> -1
    sigma = 10 ** (-snr / 20)
    return x + np.random.randn(*x.shape) * sigma

def decode(H, y, snr, maxiter=1000):
    """Infer the transmitted codeword from the noisy observation y.
    The decoding procedure is the design choice to be made."""
    pass  # TODO

def incode(H, x):
    return (binaryproduct(H, x) == 0).all()   # valid-codeword / stopping test
```
