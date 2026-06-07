# Context — Decoding Convolutional Codes, and How Cheaply Their Error Probability Can Be Bounded

## Research question

A message of data symbols over a finite field `GF(q)` is encoded by a sliding shift register into a longer stream of channel symbols, sent over a memoryless noisy channel, and must be recovered at the receiver. The encoder is a *convolutional* (tree) code: at each step a new data symbol enters a `K`-stage register and `v` linear combinations of the register contents are emitted, so each data symbol influences `K` consecutive blocks of output. Such codes were conjectured to outperform block codes of the same length, but the case rested on the performance of one decoder — Fano sequential decoding — whose analysis runs through the code *tree* and whose decoding effort is a random variable with, above a certain rate, unbounded mean.

The precise problem is twofold and intertwined. First, *quantify* the error-correcting power of an optimal convolutional code as a clean function of its constraint length `N` — an error exponent `E(R)` such that the decoding error probability behaves like `exp(−N·E(R))` — so that the conjectured superiority over block codes becomes a theorem with an explicit exponent, not a belief. Second, to make that bound provable, exhibit a decoding procedure whose error probability can actually be *computed and bounded*, rather than the sequential decoder whose erratic computation makes its error behavior hard to pin down. A satisfying answer must produce a decoder simple enough to analyze by elementary union-bound arguments, deterministic in its per-step work, and tight enough that its error exponent meets the best known upper bound for the code class.

## Background

**Convolutional (tree) codes (Elias, 1955).** Peter Elias introduced convolutional/tree codes for the discrete memoryless channel and conjectured that, for a given length, they could beat block codes. The encoder is a `K`-stage shift register over `GF(q)` with `v` inner-product computers, each forming a fixed linear combination (a column of a generator matrix `G`) of the register contents; after each data symbol shifts in, `v` channel symbols are produced. The result is naturally drawn as a *tree*: from each branching node there are `q` branches, one per possible next data symbol, each carrying `v` channel symbols. The transmission rate is `R = (ln q)/v` nats per channel symbol; the constraint length in channel symbols is `N = Kv`. After `L` data symbols, `K−1` zero symbols are fed in to return the register to a known state.

**The structural property that everything later hinges on.** Because only the `K` most recent data symbols sit in the register at any moment, the encoder's output depends only on those `K` symbols. Two paths in the tree that *diverge* at some branch and then carry identical data symbols for `K` consecutive branches will, from that point on, contain identical register contents and therefore emit identical channel symbols — the two paths *converge*. Equivalently, only the last `K−1` data symbols (the register state) matter for the future; the deep tree folds back on itself into a finite-state graph. Two paths whose data symbols are never identical for `K` consecutive branches stay *totally distinct* (their channel symbols are statistically independent under random coding — a property established by Reiffen within the first constraint length, and extendable by a generator modification due to Massey).

**Sequential decoding and its discontents.** The known way to decode these codes was *sequential decoding* (Wozencraft & Reiffen, 1961; Fano, 1963). It explores the code tree, advancing along promising branches and backtracking when a path's metric falls, guided by the Fano metric. Its decisive limitation is computational: the number of branch computations is a random variable whose distribution is Pareto, with an expected value per branch that is bounded only for rates below the *computational cutoff rate* `R0 = E0(1)`, and *unbounded* above it. So sequential decoding has predictable average computation only for `R < R0`, its running time and storage are erratic near and above the cutoff, and its error analysis (Yudkin) is correspondingly delicate.

**The error exponent for an optimal convolutional code.** The first quantitative confirmation of Elias's conjecture was due to Yudkin (1964), who obtained an *upper* bound on the error probability of an optimal convolutional code, with exponent
```
E(R) = E0(rho),    R = E0(rho)/rho,   0 < rho <= 1,
```
attained when Fano sequential decoding is used, where `E0(rho)` is Gallager's random-coding function
```
E0(rho) = max_p  −ln  sum_y [ sum_x p(x) p(y|x)^{1/(1+rho)} ]^{1+rho}.
```
`E0(rho)` is increasing and concave on most channels, with `E0(0) = 0` and `E0'(0) = C` (channel capacity). A *lower* bound on the error probability, independent of the decoder, was still missing for convolutional codes; for *block* codes the lower bound came from the sphere-packing bound (Shannon–Gallager–Berlekamp) at high rates and a tighter expurgated/straight-line bound at low rates.

**Likelihood, logarithms, and the binary symmetric channel.** Over a memoryless channel a path's likelihood is the product of per-branch conditional densities `p(y_k | x_k)`; products of small numbers underflow and are not additive, so one works with the log-likelihood, a *sum* of per-branch terms `ln p(y_k | x_k)`. For the binary symmetric channel with crossover `p < 1/2`, `ln p(y|x)` is an affine decreasing function of the number of disagreements between received and transmitted symbols, so *maximizing likelihood is identical to minimizing Hamming distance* — the negative log-likelihood cost becomes an integer count of bit disagreements after constants are dropped.

**The combinatorial obstacle to optimal decoding.** Maximum-likelihood decoding of a length-`L` tree means comparing the likelihoods of all `q^L` paths — exponential in the tree length. This is hopeless both to execute and, more pressingly for the goal here, to *analyze*: there is no clean handle on the error probability of a brute comparison of exponentially many paths.

## Baselines

**Optimal (maximum-likelihood) decoding by exhaustive path comparison.** Core idea: compute the likelihood `p(y | path)` of every one of the `q^L` paths through the length-`L` tree and pick the largest. It is exactly optimal — it minimizes block error probability for equiprobable messages. Algorithm: enumerate `q^L` paths, score each by its log-likelihood (a sum of `L` branch metrics), take the max. Cost: `Θ(q^L)` — exponential in the tree length. Gap: utterly impractical, and (the operative point here) analytically intractable — no elementary bound on the error probability of comparing exponentially many correlated paths. It is the standard of correctness any cheaper decoder is measured against, not something one can run or analyze directly.

**Fano sequential decoding (Wozencraft & Reiffen 1961; Fano 1963).** Core idea: treat decoding as a guided search through the code tree, moving forward along high-metric branches and backtracking when the running Fano metric drops below an adaptive threshold. Algorithm: maintain a current node and threshold; step forward if the metric stays above threshold, else back up and try alternatives, lowering the threshold as needed. Cost: the number of computations per decoded branch is a random variable with a Pareto distribution; its expectation is finite only for `R < R0 = E0(1)` and unbounded for `R >= R0`. It achieves Yudkin's error exponent `E(R) = E0(rho)`, `R = E0(rho)/rho`. Gap: its computation and buffer occupancy are erratic and can overflow near and above the cutoff rate; and because the work is a heavy-tailed random variable, its error behavior is awkward to bound cleanly. A decoder with *deterministic* per-step work would both run predictably and admit an elementary error analysis.

**Bit-by-bit / threshold decoding (Massey).** Core idea: decode each information symbol from a small fixed window of received symbols using algebraic (e.g. majority-logic / threshold) rules. Algorithm: form syndromes over a constraint-length window and threshold them. Cost: low and fixed per symbol. Gap: it is decisively *suboptimal* — it does not use the full received sequence to decide each symbol, so its error exponent falls well short of the maximum-likelihood exponent. It buys predictable cost at the price of correcting power.

## Evaluation settings

The yardstick is the error exponent: for a code of constraint length `N` channel symbols at rate `R` over a memoryless channel, how fast does the decoding error probability `P_E` decay, written as the negative exponent `E(R)` in `P_E ≈ exp(−N·E(R))`, and how does it compare to the block-code exponent of the same length `N`. The reference channels are the binary symmetric channel (parametrized by crossover probability `p`), the additive white Gaussian noise channel (the model for deep-space links), and the abstract "very noisy" channel where upper and lower bounds can be compared cleanly. The analytic instruments are the Gallager random-coding function `E0(rho)` for upper bounds and the sphere-packing / straight-line lower bounds for block codes, combined by union-bound arguments; the relevant rate landmarks are capacity `C`, the cutoff rate `R0 = E0(1)`, and `C/2`. A second axis is decoding cost: the number of likelihood computations per decoded branch, and the storage required, as a function of constraint length `K` and rate — to be measured against the per-block cost of maximum-likelihood decoding of an equivalent block code. Correctness of any candidate decoder is checked against the true transmitted sequence over the channel.

## Code framework

The primitives already available: a `GF(q)` (here binary) convolutional encoder built from a shift register and fixed linear combiners; a memoryless channel that corrupts symbols (the binary symmetric channel, which flips bits independently); the per-branch log-likelihood, whose negative-cost form for the BSC is the Hamming distance between received and hypothesized branch symbols after constants are dropped; and the observation that a path's log-likelihood or path cost is the *sum* of its branch metrics. The scaffold is the encoder plus an empty decoder slot that must turn a noisy received stream back into the message.

```python
K = 3                       # branch constraint length (shift-register length)
G = [0b111, 0b101]          # generator taps (v outputs per branch)

def _parity(x):
    return bin(x).count("1") & 1

def _encode_branch(state, bit):
    window = (bit << (K - 1)) | state
    out = tuple(_parity(window & g) for g in G)
    next_state = window >> 1
    return next_state, out

def conv_encode(bits):
    state = 0
    coded = []
    for b in list(bits) + [0] * (K - 1):   # flush to a known final state
        ns, out = _encode_branch(state, b)
        coded.extend(out)
        state = ns
    return coded

def _hamming(a, b):
    return sum(x != y for x, y in zip(a, b))

def decode(coded):
    # TODO: recover the most-likely message from the noisy received stream,
    #       given that BSC negative log-likelihood cost is Hamming distance
    #       and that a path's metric is the sum of its branch metrics.
    pass  # TODO
```
