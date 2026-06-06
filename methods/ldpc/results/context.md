# Context

## Research question

A channel corrupts the bits we send. We want to send k information bits as a longer block of n bits so that, even after the channel flips or noises some of them, the receiver recovers the original k bits with vanishingly small error probability — and we want to do this at the highest possible information rate R = k/n.

The existence side of this problem is settled. Shannon (1948) showed that every memoryless channel has a capacity C, and that for any rate R < C there exist block codes of length n whose probability of decoding error goes to zero as n → ∞ — in fact exponentially in n. His proof draws the code *at random*: pick 2^{nR} codewords uniformly from the 2^n binary strings, and decode a received word to the nearest (most likely) codeword. Averaged over this random ensemble, the error probability is small, so *some* code in the ensemble is at least that good.

The catch is entirely on the decoding side. A code drawn this way has no structure: the only way to decode is maximum-likelihood, comparing the received word against all 2^{nR} codewords and keeping the most probable. That cost grows exponentially with n — exactly the regime where Shannon's theorem says the codes get good. So capacity is *achievable in principle* by codes nobody can decode in practice.

The real question, then, is not "do good long codes exist?" — Shannon answered that — but: **is there a family of codes with enough internal structure that decoding is cheap (ideally growing only linearly with n per pass), while still keeping the error probability low at a usable rate?** A solution has to thread the gap between the structureless random codes (great error performance, hopeless decoding) and whatever structured families already exist (decodable, but as we will see, leaving error performance or rate on the table).

## Background

**Linear codes and parity checks.** The standard way to give a code structure is to make it *linear* over GF(2). A linear code of length n is the null space of a parity-check matrix H: the codewords are exactly the binary vectors x with Hx = 0 (mod 2). Each row of H is one parity-check equation — a subset of bit positions constrained to sum to zero. If H is (n−k)×n of full rank, the code has rate R = k/n. Encoding and the membership test Hx = 0 are pure linear algebra over GF(2); the difficulty is always *decoding*, i.e. inferring which codeword was sent given a noisy observation.

**Decoding is inference.** On a memoryless channel the received vector y gives, for each bit, a likelihood — on the binary symmetric channel (BSC) a crossover probability p, on the additive-white-Gaussian-noise (AWGN) channel with antipodal signalling x_n = ±1 a Gaussian density. Optimal decoding finds the codeword maximizing the posterior P(x | y) subject to Hx = 0. The information the channel hands us is genuinely *soft*: a per-bit a-posteriori probability, not just a hard 0/1 guess. Any decoder that throws the soft information away before it starts is leaving evidence on the table.

**The distance picture and its ensemble statistics.** The error-correcting power of a code is classically summarized by its minimum distance D — the smallest number of bit positions in which two codewords differ. A decoder that always returns the nearest codeword corrects any error pattern of weight up to ⌊(D−1)/2⌋. For an ensemble of randomly built parity-check matrices one can compute the *average* distance distribution rather than any single code's: the expected number of weight-ℓ codewords is N̄(ℓ) = (n choose ℓ) 2^{−n(1−R)}, because a fixed nonzero weight-ℓ pattern satisfies each of the n(1−R) independent parity checks with probability ½. Bounding the tail of this with Stirling shows the typical minimum distance grows linearly with n, with the linear coefficient δ_0 fixed by H(δ_0) = (1−R) ln 2 (the Gilbert bound). So *randomly* built parity-check codes already have good distance growth — the bottleneck was never distance, it was the decoder.

**Graphs of codes.** A parity-check matrix is equivalently a bipartite graph: one node per bit (variable node), one node per parity check (check node), an edge wherever H has a 1. Tanner (1981) made this graph view explicit, building long codes by wiring short subcodes onto a bipartite graph and decoding by passing partial results along the graph's edges, with the rate and minimum distance of the long code bounded in terms of the graph and the subcodes. The graph turns "decode the whole block" into "reconcile local constraints along edges" — a structural handle on the cost of decoding.

**Soft, iterative, probabilistic inference on graphs.** Pearl's belief propagation (1988) is a general algorithm for computing posterior marginals in a graphical model by passing "messages" — local probability summaries — between nodes along edges. Its defining property: on a graph with no cycles (a tree), the messages converge to the *exact* posterior marginals after one sweep. On a graph with cycles it is only an approximation, and its quality degrades as the cycles get shorter (as the girth shrinks). This is the lever that connects a sparse code graph to cheap near-optimal decoding: if a code's graph looks locally like a tree, local message passing computes nearly the right answer.

**Pre-method failure modes, recalled not measured.** Two phenomena about iterative graph decoders are established before any specific code is built, and they bound what such a decoder can do. First, *short cycles*: when the bit and check involved in a message lie on a short cycle, the "independent evidence" assumption underlying belief propagation is violated, a node's own past belief returns to it, and the marginals it computes are biased. Second, and more subtly, certain small, specially-structured subsets of bits — sets whose induced subgraph attaches to only a few odd-degree checks (later called trapping sets or near-codewords) — can hold an iterative decoder in a wrong but self-consistent state even when the channel noise is mild, so the residual error probability stops falling as fast as the distance would predict and flattens into a floor at high signal-to-noise ratio. Both are properties of the *graph*, knowable by inspecting it, and both say the same thing: an iterative local decoder is near-optimal only to the extent the graph is locally tree-like and free of small bad substructures.

## Baselines

**Algebraic / bounded-distance codes (Hamming, BCH, Reed–Solomon).** These build H (or its dual) from algebraic structure — roots of polynomials over finite fields — so that an efficient algebraic procedure (syndrome computation, the Berlekamp–Massey / Peterson–Gorenstein–Zierler machinery) corrects *any* error pattern of weight up to a guaranteed t = ⌊(D−1)/2⌋, and nothing beyond. Two limitations matter. (i) The decoder operates on *hard decisions*: it computes a syndrome from the received bits and corrects up to t flips, with no way to use the channel's per-bit a-posteriori probabilities — a structural feature of algebraic, as opposed to probabilistic, decoding. (ii) Because it corrects up to t and abandons everything past t, at long block lengths the error probability is dominated by the patterns just beyond t, and the resulting performance sits well short of capacity. Decoding cost for BCH on the BSC grows roughly as the cube of the block length.

**Sequential decoding of convolutional codes (Wozencraft; Fano).** A probabilistic decoder that *does* use soft information, searching the code tree of a convolutional code guided by a likelihood metric. It achieves a probability of error bounded like e^{−αn} in the constraint length n for rates below capacity. Its weaknesses are operational: the computation per decoded digit is a *random variable* that occasionally explodes (a buffer/waiting-line problem at the decoder), and there is a practical rate ceiling — the computational cutoff rate R_comp < C — below which the *average* work per digit is bounded but above which it is not. A single decoding error can also throw the decoder off track for a long burst.

**Threshold decoding (Massey).** The simplest of the probabilistic schemes — shift registers, a few modulo-2 adders, and a threshold element forming a majority vote of parity checks on a bit. Cheap and fast, but effective only at relatively short constraint lengths, with higher error probability and less flexibility than sequential decoding.

The gap each of these leaves open is the same one the random-coding theorem points at: the algebraic codes are decodable but throw away soft information and stop at a fixed distance; the convolutional/sequential schemes use soft information but pay with unbounded or rate-limited computation. None of them is a long block code that is simultaneously cheap to decode *and* able to exploit the full a-posteriori information at a rate close to capacity.

## Evaluation settings

The natural yardsticks already exist. The two canonical channels are the binary symmetric channel, parameterized by a crossover probability p (each bit independently flipped with probability p), and the binary-input additive-white-Gaussian-noise channel, where each codeword bit is sent antipodally as x_n = ±1 and received as y_n = x_n + noise of variance σ². For the AWGN channel the operating point is reported as E_b/N_0 in decibels, the energy per information bit over the noise spectral density, related to the per-symbol signal-to-noise ratio through the rate R. The figure of merit is the decoded bit-error (or block-error) probability as a function of channel quality (p, or E_b/N_0), read against the Shannon limit for the given rate — the smallest channel quality at which rate-R reliable communication is information-theoretically possible. Complexity is measured as arithmetic operations per decoded bit per iteration and the number of iterations, as functions of block length n. Existing reference points on these axes are the convolutional codes with sequential decoding and the concatenated (convolutional + Reed–Solomon) constructions of the day.

## Code framework

The pieces that already exist: GF(2) linear algebra (matrix products mod 2, Gaussian elimination), a noisy-channel simulator, and a generic iterative decoding loop that stops when the parity checks are satisfied. What does **not** yet exist is the choice of *which* parity-check matrix to use and *what messages* to pass on its graph — those are the slots to fill.

```python
import numpy as np

# --- known: GF(2) linear algebra -------------------------------------------
def binaryproduct(X, Y):
    return X.dot(Y) % 2

def gaussjordan(X):
    # row-reduce over GF(2); used to turn H into a generator G
    ...

# --- the slot: which parity-check matrix? ----------------------------------
def parity_check_matrix(n, *params, seed=None):
    """Return an (n-k) x n parity-check matrix H over GF(2).
    The structure of H is the design choice to be made."""
    pass  # TODO

def coding_matrix(H):
    # standard: derive a generator G from H so that H G = 0, x = G v
    ...

# --- known: channel simulator (BPSK over AWGN) -----------------------------
def encode(G, v, snr, seed=None):
    d = binaryproduct(G, v)
    x = (-1) ** d                      # bit 0 -> +1, bit 1 -> -1
    sigma = 10 ** (-snr / 20)
    return x + np.random.randn(*x.shape) * sigma

# --- the slot: how to decode? ----------------------------------------------
def decode(H, y, snr, maxiter=1000):
    """Infer the transmitted codeword from the noisy observation y.
    What quantities live on the edges of H's graph, and how do the
    variable side and the check side update them, is the contribution."""
    pass  # TODO

def incode(H, x):
    return (binaryproduct(H, x) == 0).all()   # valid-codeword / stopping test
```
