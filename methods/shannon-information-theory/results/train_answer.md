We want a *general* theory of communication — not another modulation scheme to set alongside PCM and PPM, but the thing underneath all of them, the standard against which any scheme can be measured and called good or wasteful. To get there we need two numbers that do not yet exist on any firm footing: how much a source is actually producing, and how much a channel can actually carry. The existing tools each reach toward one of these and stop short. Nyquist's $W = K\log m$ establishes the logarithm as the natural measure of signaling speed, but only for equiprobable code values, with no source model and no noise. Hartley's $H = \log(s^n) = n\log s$ takes the decisive conceptual step — strip the *meaning* of a message and keep only that it was selected from a set of possibilities — but it assumes the symbols are equally likely and independent, which real sources never are: English is not $26$ equiprobable independent letters, E is everywhere and Q is rare and almost always followed by U. Hartley's measure massively overcounts what an English source produces, because the receiver can guess so much of it; the redundancy is enormous, measurable, and exactly the structure Hartley discarded. Boltzmann and Gibbs already hold the form $-k\sum p_i\log p_i$ as the entropy of a statistical state, but it lives in physics and has not been read as a measure of information. Wiener and Kolmogorov model signal and noise as random processes and solve optimal filtering in mean square, but that is an estimation theory for recovering a waveform — it never quantifies how much information a source generates or how much a channel can convey. None of them answers all three questions together: a meaning-free measure of information, the exact compressibility of an arbitrary statistical source, and the capacity of a noisy channel with the rate and reliability that are simultaneously achievable.

I propose to build the whole theory on a single quantity, the **entropy**, and the method is **Shannon information theory**. The starting commitment is Hartley's, made rigid: the significant fact about a message is not what it means but that it was selected from a set, because the transmitter must be built at design time to handle every possible selection. So information is about probabilities, not semantics. The reason the measure must be logarithmic is forced rather than conventional: a measure of information should be additive in the way physical resources are additive — two relays should hold twice what one holds, two channels carry twice the rate, doubling the time should double the count — and a function with $f(M_1 M_2) = f(M_1)+f(M_2)$ is a logarithm. Choosing base $2$ makes one selection between two equally likely things the unit, the **bit**, which is exactly what a two-state device holds. But the real source is not a list of equally likely options; it is a stochastic process, an ergodic Markov chain over a finite alphabet, emitting symbols with probabilities that depend on what came before. Ergodicity is what makes "per symbol" well posed: the time average along one long sequence equals the ensemble average, so a single long sample is representative.

The defining quantity is fixed not by analogy to physics but by demanding three properties of any honest measure of choice $H(p_1,\dots,p_n)$: it is continuous in the $p_i$; for the uniform case $p_i = 1/n$ it increases with $n$ (more equally likely alternatives is more choice); and it is invariant to how a choice is *staged*, so that decomposing a selection into a coarse choice followed by a finer one, with the later branches weighted by how often they are reached, gives the same total — concretely $H(\tfrac12,\tfrac13,\tfrac16) = H(\tfrac12,\tfrac12) + \tfrac12 H(\tfrac23,\tfrac13)$. These three force the form uniquely. Write $A(n) = H(1/n,\dots,1/n)$. Staging a choice among $s^m$ equally likely things as $m$ successive choices among $s$ gives $A(s^m) = m\,A(s)$. To compare $A(s)$ and $A(t)$ for arbitrary $s,t$, squeeze $s^m \le t^n < s^{m+1}$ for large $n$; taking ordinary logs gives $\big|\tfrac{\log t}{\log s} - \tfrac{m}{n}\big| < \tfrac1n$, while monotonicity of $A$ applied to the same inequalities gives $\big|\tfrac{A(t)}{A(s)} - \tfrac{m}{n}\big| < \tfrac1n$; both ratios are pinned to the same $m/n$ within $1/n$ for arbitrarily large $n$, so $A(t) = K\log t$ with $K > 0$. For commensurable probabilities $p_i = n_i/\sum_j n_j$, manufacture a uniform choice among $\sum_j n_j$ outcomes and decompose it two ways — first pick block $i$ with probability $p_i$, then pick uniformly among its $n_i$ members — so the grouping rule gives $A(\sum_j n_j) = H(p_1,\dots,p_n) + \sum_i p_i A(n_i)$. Substituting $A = K\log$ and using $\sum_i p_i = 1$,

$$H = K\log\!\Big(\textstyle\sum_j n_j\Big) - K\sum_i p_i \log n_i = -K\sum_i p_i \log\!\frac{n_i}{\sum_j n_j} = -K\sum_i p_i \log p_i,$$

and continuity carries this to incommensurable $p_i$. So the measure is forced to be

$$H = -K\sum_i p_i \log p_i,$$

with $K$ a choice of unit; take $K=1$ and base-$2$ logs and $H$ is in bits. This is exactly the entropy of Boltzmann and Gibbs, but its justification here is the three properties, not the physical resemblance. It has the right behavior: $H=0$ only when one outcome is certain; $H \le \log n$ with equality iff uniform; and $H(x,y) \le H(x)+H(y)$ with equality iff $x,y$ are independent, because $H(x,y) = H(x) + H_x(y)$ and conditioning can only reduce uncertainty, $H_x(y) \le H(y)$. That last inequality is the redundancy of English appearing as a number — a correlated source has strictly less entropy than the sum of its marginals.

Entropy earns the title "rate of producing information" only by controlling compression, and that connection runs through the typical set. Take length-$N$ sequences. By the law of large numbers a typical sequence has symbol $i$ appearing about $p_i N$ times, so its probability is about $\prod_i p_i^{p_i N}$, whose base-$2$ log is $\sum_i (p_i N)\log p_i = -NH$; the probability is $\approx 2^{-HN}$. Since the typical sequences carry essentially all the probability and are nearly equiprobable, there are about $2^{HN}$ of them, out of the $s^N = 2^{N\log s}$ conceivable ones. Stated as the limit it is: for any $\varepsilon,\delta>0$ and $N$ large, the sequences split into a junk set of total probability $<\varepsilon$ and a typical set on which $\big|-\tfrac1N\log p - H\big| < \delta$; equivalently $\tfrac1N\log n(q) \to H$ where $n(q)$ counts the most probable sequences accumulating probability $q$. To name one of $\approx 2^{HN}$ typical sequences needs $\approx HN$ bits, i.e. $H$ bits per symbol; fewer than $N(H-\delta)$ bits would supply fewer than $2^{N(H-\delta)}$ labels for $\approx 2^{NH}$ nearly equiprobable candidates, and the atypical sequences cost almost nothing because they almost never occur. So the smallest achievable average description length per symbol is exactly $H$: **entropy is the compression limit**. Over a noiseless channel of capacity $C$ bits/second the source can be sent at an average rate approaching $C/H$ symbols/second and no faster, the converse being immediate since the one-to-one transmitter makes the channel-input entropy equal to the source entropy, which cannot exceed $C$.

Noise is the harder half, and the first instinct — measure throughput by counting symbols received correctly — is a trap. If the channel is so noisy that the output is independent of the input, about half the received bits agree with what was sent purely by chance, and the naive count credits half the bits as transmitted when in fact *nothing* got through. The correct quantity to subtract is the uncertainty about the input that *remains after the output is seen*, the part of the message that failed to arrive, and we already have a name for it: the conditional entropy $H_y(x)$, the **equivocation**. The actual rate of transmission is

$$R = H(x) - H_y(x).$$

On the broken channel $y$ tells nothing about $x$, so $H_y(x) = H(x)$ and $R=0$, exactly right; on a binary channel where a received $0$ means a sent $0$ with probability $.99$, the equivocation is $-[.99\log.99 + .01\log.01] \approx .081$ bits/symbol, so of $1000$ symbols/second about $919$ really get through. Equivocation is *the* right toll, not merely a plausible one, by a side argument: an observer who sees both sent and received messages and corrects the receiver over an auxiliary noiseless channel must specify, for each received block, which of the $\approx 2^{T H_y(x)}$ possible inputs was sent — about $T\,H_y(x)$ bits per block, so a correction channel of capacity $H_y(x)$ suffices and nothing smaller works. The **channel capacity** is then the best rate the channel can be driven to, optimized over how it is fed:

$$C = \max_{\text{input sources}}\big[\,H(x) - H_y(x)\,\big].$$

The folklore said that driving the error toward zero forces unbounded redundancy and a rate sinking to zero, which would make $C$ uninteresting. It is false, and the typical-set picture run on both ends shows why. Drive the channel with the maximizing source over blocks of length $T$: about $2^{TH(x)}$ typical inputs, about $2^{TH(y)}$ typical outputs, and each received output has a fan of about $2^{TH_y(x)}$ input causes that could reasonably have produced it. To send a source of rate $R<C$ we need $2^{TR}$ messages per block, and rather than construct a clever spread-out codebook — which is exactly what is hard — we *average over all codebooks*: assign the $2^{TR}$ messages to typical inputs completely at random and compute the average error probability, because if the average is below $\varepsilon$ then some particular codebook beats $\varepsilon$, proving a good code exists without building it. The probability that a given input point is one of the messages is $2^{TR}/2^{TH(x)} = 2^{T(R-H(x))}$, so the probability that no competing message lands in the fan of size $2^{TH_y(x)}$ is

$$P = \big(1 - 2^{T(R-H(x))}\big)^{2^{T H_y(x)}}.$$

Using $R < C = H(x)-H_y(x)$, write $R - H(x) = -H_y(x) - \eta$ with $\eta>0$; the expected number of competing messages in the fan is $2^{TH_y(x)}\cdot 2^{-TH_y(x)-T\eta} = 2^{-T\eta} \to 0$, so $P\to 1$ and the error probability $\to 0$ at the *fixed* rate $R$. The $\eta>0$ carries everything: there is room for about $2^{T(H(x)-H_y(x))} = 2^{TC}$ messages before the fans of size $2^{TH_y(x)}$ fill the $2^{TH(x)}$ typical inputs, so below capacity the fans pack without overlap and every output points back to a unique message — sphere-packing in the space of typical sequences — while above capacity overlap is forced and confusion is inevitable. The converse closes it: a source of entropy $C+a$ per unit time sent with equivocation $a-\varepsilon$ would deliver rate $(C+a)-(a-\varepsilon) = C+\varepsilon > C$, contradicting $C$ being the maximum of $H(x)-H_y(x)$. So both numbers exist and both are entropy: the source compresses to $H$ bits/symbol and no further, and reliable communication is possible at every rate below $C$ and impossible above it.

A small computation lets the source half breathe — the entropy of an empirical distribution and a binary prefix code, built by repeatedly merging the two least likely subtrees, whose average length lands within one bit of that entropy floor:

```python
import heapq
import math
from collections import Counter
from itertools import count

def empirical_distribution(symbols):
    n = len(symbols)
    if n == 0:
        raise ValueError("sample must be non-empty")
    return {s: c / n for s, c in Counter(symbols).items()}

def information_measure(distribution):
    """H = -sum p log2 p: bits/symbol, the compression limit."""
    if any(p < 0 for p in distribution.values()):
        raise ValueError("probabilities must be non-negative")
    return -sum(p * math.log2(p) for p in distribution.values() if p > 0)

def prefix_code(distribution):
    """Binary prefix code made by repeatedly merging the two least likely subtrees."""
    serial = count()
    heap = []
    for sym, p in distribution.items():
        if p < 0:
            raise ValueError("probabilities must be non-negative")
        if p > 0:
            heap.append([p, next(serial), {sym: ""}])
    if not heap:
        raise ValueError("distribution must have positive mass")
    heapq.heapify(heap)
    if len(heap) == 1:
        return heap[0][2]
    while len(heap) > 1:
        p_lo, _, lo = heapq.heappop(heap)
        p_hi, _, hi = heapq.heappop(heap)
        merged = {sym: "0" + bits for sym, bits in lo.items()}
        merged.update({sym: "1" + bits for sym, bits in hi.items()})
        heapq.heappush(heap, [p_lo + p_hi, next(serial), merged])
    return heap[0][2]

def average_length(code, distribution):
    return sum(p * len(code[s]) for s, p in distribution.items() if p > 0)

def source_coding_check(symbols):
    dist = empirical_distribution(symbols)
    H = information_measure(dist)
    code = prefix_code(dist)
    L = average_length(code, dist)
    return H, L, code

if __name__ == "__main__":
    text = "the source coding limit is the entropy of the source"
    H, L, code = source_coding_check(text)
    print(f"H = {H:.3f} bits/symbol; avg length = {L:.3f} bits/symbol")
    assert L + 1e-12 >= H
    assert L < H + 1 + 1e-12
```
