Let me start where the trouble actually is. I want a *general* theory of communication — not another modulation scheme, but the thing underneath all of them. People keep inventing systems that trade bandwidth for signal-to-noise — PCM, PPM — and each is analyzed on its own terms, and I have no way to say what any of them is really *worth*, or how close it comes to the best possible. To say that I need two numbers that don't yet exist: how much is a source actually producing, and how much can a channel actually carry. And I have nothing rigorous behind either word. So let me try to build them from scratch.

The first thing I keep tripping over is *meaning*. A message usually means something — it refers to apples, or to a gun's bearing, or to nothing at all if it's noise. But the meaning is irrelevant to the engineering problem, and worse, it isn't even well defined: the same string means different things to different receivers depending on what they already share. Hartley already saw this clearly back in '28 — he insisted a measure of information be based on physical rather than psychological considerations, and he was right. So I'll commit to it hard: the significant fact about a message is not what it means but that it was *selected from a set of possible messages*. The transmitter has to be built to handle every possible selection, because at design time we don't know which one will actually be sent. That's the whole game — choice from a set of possibilities. Strip everything else.

Good. Now if there are M equally likely messages and I pick one, how much information is that? Hartley's answer is the logarithm of M, and I want to re-derive *why* the logarithm, because if I just accept it I won't trust the generalization later. Take two independent systems, one with M1 possibilities and one with M2. Used together they have M1·M2 possibilities. I want the information of the pair to be the *sum* of the two informations — two punched cards should hold twice what one holds, two identical channels should carry twice what one carries. A function f with f(M1·M2) = f(M1) + f(M2) is forced to be a logarithm. And it lines up with everything else: add one relay to a bank and you double the number of states but add exactly 1 to the base-2 log; double the available time and you roughly square the number of possible messages, doubling the log. The engineering parameters — relays, time, bandwidth — all move linearly with the log of the number of possibilities, never with the number itself. So the logarithm isn't a convention, it's the only thing that's additive in the way resources are additive. Pick base 2; then one selection between two equally likely things is the unit. A two-state device — a relay, a flip-flop — holds exactly one of these. I'll call the unit a bit; Tukey suggested the word and it's right.

So far this is just Hartley dressed up. Now the gap that's been bothering me: Hartley's measure assumes the symbols are equally likely and independent, and *real sources are nothing like that*. English isn't 26 equiprobable independent letters. E is everywhere, Q is rare, and after Q comes U almost surely. If I just count log(26) per letter I am wildly overcounting what an English source actually produces, because the receiver can guess a lot of it. The redundancy is enormous and it's *measurable* — I can build it up myself: take first-order letter frequencies and generate random text, it already looks vaguely word-shaped; take digram then trigram statistics and it drifts steadily toward English. The statistics are the structure, and the structure is exactly what Hartley threw away by assuming equiprobability.

So the source is not a list of equally likely options. The source is a *stochastic process* — it emits symbols with probabilities that depend on what came before. A Markov chain, in the discrete case. This is the move I keep coming back to as the real one: model the message itself as a random process. Wiener and Kolmogorov modeled the *noise* as random, and Wiener modeled the signal as random for the filtering problem, but as a thing to *estimate*, a waveform to pull out of noise — not as a *source of information* whose output rate I want to measure. The object I care about isn't "what is the true waveform behind this noisy data," it's "at what rate is this process generating choices I'd have to communicate." Different question. So: a source is an ergodic Markov process over a finite alphabet, and I want a number — call it H — that measures how much uncertainty, or choice, there is per symbol. For ergodic processes the time average along one long sequence equals the ensemble average, so a single long sample is representative; that's what makes "per symbol" well posed.

Now, what should H of a set of probabilities p1, …, pn actually be? Hartley's log(n) only covers the uniform case. I don't want to just guess −Σ p log p by analogy and hope. Let me instead write down the properties any honest measure of "amount of choice" must have, and see what they force.

One: H should be continuous in the probabilities. A tiny change in the p's shouldn't make the uncertainty jump.

Two: if the n outcomes are equally likely, p_i = 1/n, then H should *increase* with n. More equally likely alternatives is more choice. Call this special case A(n) = H(1/n, …, 1/n); I'm asking A to be monotone increasing.

Three — and this is the one with teeth — it shouldn't matter how I *stage* the choice. If I make a selection by first making a coarse choice and then a finer one, the total uncertainty has to be the same as making it in one shot, with the later choices weighted by how often I reach them. Concretely: choosing among {1/2, 1/3, 1/6} should equal first flipping a fair coin {1/2, 1/2}, and then, only on the second branch (which happens half the time), choosing {2/3, 1/3}. So H(1/2,1/3,1/6) = H(1/2,1/2) + (1/2)·H(2/3,1/3). Information is invariant to how you decompose the decision. This is just demanding that the bookkeeping be consistent.

Let me see whether these three pin H down. Start with the uniform case under the grouping rule. A choice among s^m equally likely things can be staged as m successive choices among s equally likely things each — write the outcome in base s, you choose one digit at a time. Grouping says the total equals the sum of the stages: A(s^m) = m·A(s). Same for any base: A(t^n) = n·A(t). Now I want to compare A(s) and A(t) for arbitrary s, t. Squeeze t^n between consecutive powers of s: choose n large, find m with s^m ≤ t^n < s^{m+1}. Take logs (ordinary logs) and divide by n log s: m/n ≤ log t / log s < m/n + 1/n, so log t / log s and m/n differ by less than 1/n, arbitrarily small. Now use monotonicity of A: from s^m ≤ t^n < s^{m+1} I get A(s^m) ≤ A(t^n) ≤ A(s^{m+1}), i.e. m·A(s) ≤ n·A(t) ≤ (m+1)·A(s). Divide by n·A(s): m/n ≤ A(t)/A(s) ≤ m/n + 1/n. So A(t)/A(s) also sits within 1/n of m/n. Both A(t)/A(s) and log t / log s are pinned to the same m/n within 1/n, for arbitrarily large n — so they're equal. Hence A(t) = K log t, with K a positive constant (positive because A is increasing). The uniform case is forced to be a logarithm, exactly Hartley, derived rather than assumed.

Now the general, unequal case. Suppose the probabilities are commensurable: p_i = n_i / Σn_j with integer n_i. Here's the trick — manufacture a uniform choice and decompose it two ways. Imagine Σn_j equally likely fine outcomes. I can reach one by a two-stage choice: first pick block i with probability p_i = n_i/Σn_j, then pick uniformly among the n_i outcomes in that block. By the grouping rule, the single-stage uncertainty equals the first-stage uncertainty plus the average of the second stages:

    A(Σn_j) = H(p_1, …, p_n) + Σ_i p_i · A(n_i).

But A is now known to be K log, so K log(Σn_j) = H + K Σ_i p_i log n_i. Solve for H:

    H = K log(Σn_j) − K Σ_i p_i log n_i = −K Σ_i p_i [log n_i − log(Σn_j)] = −K Σ_i p_i log(n_i/Σn_j) = −K Σ_i p_i log p_i.

(using Σ_i p_i = 1 so K log(Σn_j) = K Σ_i p_i log(Σn_j).) For incommensurable p_i, approximate by rationals and let continuity carry it. So the three requirements force, uniquely,

    H = −K Σ_i p_i log p_i,

and K is just a choice of unit; take K = 1 with base-2 logs and H is in bits.

Now I stare at this and it's familiar. −Σ p log p, with p a probability over states — that's the entropy of statistical mechanics, Boltzmann's H, the Gibbs entropy of a distribution over phase-space cells. I've read enough Tolman to recognize it on sight; this is an old form. I don't need anyone to christen it for me — the function I derived *is* entropy, the same quantity that's maximal for the uniform distribution and additive over independent systems, and I'll call it that. Good that the names agree, but the justification isn't the analogy; the justification is that the three properties leave no other choice, and beyond that, that the definition is going to *earn* its keep in the theorems that follow.

Let me check the properties fall out. H = 0 only when one p_i is 1 and the rest 0 — certainty, no choice, as it must be. For fixed n, H is maximal and equals log n when all p_i = 1/n — maximum uncertainty is the uniform case, also right. And independence: H(x,y) ≤ H(x) + H(y), with equality exactly when x and y are independent, since conditioning can only remove uncertainty, H(x,y) = H(x) + H_x(y) and H_x(y) ≤ H(y). So a correlated source has *less* entropy than the sum of its marginals — which is precisely the redundancy of English showing up as a number. The entropy of a Markov source is the average over states of the per-symbol entropy of its transition distribution, and for English it's well below log 26. Now I have the first of my two numbers: the rate at which a source produces information.

But "rate of producing information" is still just a name unless I can show this H is *the* thing that controls how compactly the source can be represented. So I need to connect H to actual coding. Take long sequences of length N from the source. By the law of large numbers, in a typical sequence each symbol i appears about p_i·N times, so the probability of a typical sequence is about Π_i p_i^{p_i N}. With base-2 logs, the log of that probability is Σ_i (p_i N) log p_i = −N·H, so the probability itself is ≈ 2^{−HN}. And since the typical sequences carry essentially all the probability and each has about the same probability 2^{−HN}, there must be about 2^{HN} of them. Out of the s^N = 2^{N log s} conceivable sequences, only about 2^{HN} ever really occur, and they're nearly equiprobable. The rest form a set of vanishing total probability.

Let me state it as the limit it is. For any ε, δ > 0 and N large enough, the sequences split into a junk set of total probability < ε, and a typical set on which |(−1/N) log p − H| < δ. Equivalently, if I sort sequences by probability and take just enough of the most probable ones to accumulate probability q (any 0 < q < 1), the count n(q) satisfies (1/N) log n(q) → H. The number of "reasonably probable" sequences grows like 2^{HN} regardless of where I draw the line. So for almost all purposes I can pretend there are exactly 2^{HN} messages of length N, each with probability 2^{−HN}.

Compression now writes itself. There are ≈ 2^{HN} typical sequences; to name one I need ≈ HN bits, so H bits per symbol. If I tried to use fewer than N(H − δ) bits for the high-probability part, I would have fewer than 2^{N(H − δ)} short binary labels available, while the set I must cover has about 2^{NH} nearly equiprobable candidates. The atypical sequences I handle with a longer escape code; they cost almost nothing because they almost never happen. So the smallest achievable average description length per symbol is exactly H. Stated against a channel: a source of entropy H bits/symbol fed through a noiseless channel of capacity C bits/second can be sent at an average rate approaching C/H symbols/second, and no faster. The converse is immediate — the channel input entropy per second equals the source entropy (the transmitter is one-to-one), and that can't exceed the channel's own capacity, so H′ ≤ C and the symbol rate H′/H ≤ C/H. Entropy is the compression limit, full stop. As an explicit code reaching it, I can order messages by probability and assign binary labels whose lengths track log(1/p); the average length stays at or just above H for one-symbol coding and closes to H per symbol over long blocks.

That handles a clean channel. Now the hard half: noise. With noise the channel output no longer determines the input — the same received string could have come from several sent strings. My first instinct is to measure how much got through by counting symbols received correctly, but that's a trap, and the trap is sharp. Suppose the channel is so noisy that the output is statistically independent of the input — say each received bit is 0 or 1 by a coin flip no matter what I sent. Then about half my received bits "agree" with what I sent purely by chance, and the naive count would credit me with transmitting half the bits. But I've transmitted *nothing*: I'd get the same agreement by throwing the channel away and flipping a coin at the receiver. So raw throughput overcounts, badly, and I need the correction.

What's the right correction? The thing I should subtract is exactly the uncertainty that *remains about the input after the output is seen* — because that's the part of the message the receiver still doesn't know, the information that failed to arrive. And I already have a measure of remaining uncertainty: conditional entropy. H_y(x), the entropy of the input given the output, averaged over outputs. Call it the equivocation. Then the actual rate of transmission is

    R = H(x) − H_y(x).

Check it on the broken channel: input and output independent means knowing y tells me nothing about x, so H_y(x) = H(x), and R = 0. Exactly right — zero, as it must be. Check the nearly-clean binary channel where a received 0 means the sent bit was 0 with probability .99: H_y(x) = −[.99 log .99 + .01 log .01] ≈ .081 bits/symbol, so out of 1000 symbols/second I'm really getting 1000 − 81 = 919 through. The equivocation is precisely the toll noise charges.

I can pin equivocation down as *the* right measure, not just a plausible one, with a side argument. Put an observer who sees both the sent message and the received one; let the observer send correction data over an auxiliary noiseless channel so the receiver can fix its errors. How big must that correction channel be? Over a long block of T, for each received message there are about 2^{T H_y(x)} inputs that could reasonably have produced it, so the observer must specify which one — about T·H_y(x) bits per block, i.e. a correction channel of capacity H_y(x) suffices to repair all but a vanishing fraction of errors, and nothing smaller works. So H_y(x) is exactly the missing information, and R = H(x) − H_y(x) is forced.

Now define the channel's capacity as the best rate it can be driven to:

    C = max over input sources of [H(x) − H_y(x)].

Capacity is a property of the channel, so I optimize over how I feed it. For a noiseless channel H_y(x) = 0 and this reduces to the maximum input entropy — consistent with the noiseless definition.

Here's where I expect to lose something, and where it turns out I don't. The folklore says: to push the error probability toward zero on a noisy channel you must pile on redundancy — repeat the message, vote on the copies — and as the error requirement tightens the redundancy grows without bound and the *rate* sinks toward zero. If that were true there'd be no single capacity, only a rate that depends on how much error you'll tolerate. Let me see whether it's actually true, because if it is, C is far less interesting than I hoped.

It is not true, and the reason is the typical-set picture again, now on both ends. Drive the channel with the maximizing source S0 and look at length-T blocks. The inputs that occur are about 2^{T H(x)} typical ones. The outputs that occur are about 2^{T H(y)} typical ones. A typical input has its own cloud of likely outputs, but decoding turns on the reverse question: once an output is received, how many typical inputs could reasonably have caused it? That reverse fan has about 2^{T H_y(x)} inputs. That is the ambiguity the code must prevent from containing two messages at once.

Now I want to send a source of rate R < C, so 2^{T R} distinct messages per block. The instinct is to *design* a clever set of 2^{T R} input blocks spread far apart. But explicit construction is exactly what's hard, so don't construct anything — *average over all codes*. Assign the 2^{T R} messages to input blocks completely at random, drawn from the typical inputs, and compute the *average* error probability over this whole ensemble of random codebooks. If the average error is below ε, then at least one particular codebook in the ensemble has error below ε — so a good code must exist, even though I never built it.

Compute the average. A message gets sent as some input; the channel produces some typical output y1; the receiver looks at the fan of ~2^{T H_y(x)} inputs that could have caused y1 and declares the message correct if the true input is the only message-codeword in that fan. An error happens only if some *other* message also landed, at random, inside that fan. So: 2^{T R} messages scattered uniformly at random over 2^{T H(x)} typical inputs; the probability that a particular input point is one of the messages is 2^{T R}/2^{T H(x)} = 2^{T(R − H(x))}. The probability that *none* of the points in the fan of size 2^{T H_y(x)} is a competing message is

    P = (1 − 2^{T(R − H(x))})^{2^{T H_y(x)}}.

Now use R < C = H(x) − H_y(x). Then R − H(x) < −H_y(x), so write R − H(x) = −H_y(x) − η with η > 0. The base becomes 1 − 2^{−T H_y(x) − T η}, raised to the power 2^{T H_y(x)}. Multiply exponent and the small quantity: the number of trials is 2^{T H_y(x)} and the per-trial failure probability is 2^{−T H_y(x)}·2^{−T η}, so the expected number of collisions is 2^{T H_y(x)}·2^{−T H_y(x) − T η} = 2^{−T η} → 0. Hence P → 1: no competitor lands in the fan, and the probability of error → 0 as T → ∞, at the *fixed* rate R. That's the whole thing. Reliability and a positive rate at the same time — the redundancy stays bounded, the error still vanishes, and the gap I was afraid of never opens, as long as R stays below C.

The η > 0 is doing all the work, and it's worth saying out loud what it means. The fan around each output has 2^{T H_y(x)} input causes; I'm allowed about 2^{T(H(x) − H_y(x))} = 2^{TC} messages before the fans start colliding, because 2^{TC} fans of size 2^{T H_y(x)} just fill the 2^{T H(x)} typical inputs. Below capacity the fans pack in without overlap and every output points back to a unique message; at capacity they're touching; above it they must overlap and confusion is forced. That's the geometry — sphere-packing in the space of typical sequences, the discrete shadow of the noise-sphere-hardening picture I'd been carrying around from the continuous case.

The converse closes it. If I try to transmit a source of entropy C + a per unit time and claim an equivocation as low as a − ε, then the delivered rate would be (C + a) − (a − ε) = C + ε, which contradicts C being the *maximum* possible value of H(x) − H_y(x). So you cannot beat C: above it, nature charges an equivocation at least equal to the excess. There is room to make uncertainty vanish only up to C; past it the uncertainty is forced on you.

So both numbers exist and both are entropy. Let me trace the chain once. Strip meaning: a message is a selection from a set, so information is about probabilities, not semantics. Demand the measure be additive ⇒ logarithm ⇒ bits. Real sources are statistical, so model the source as an ergodic Markov process and ask for a per-symbol uncertainty H(p) obeying continuity, monotonicity in the uniform case, and invariance to staging the choice ⇒ those three force H = −Σ p log p uniquely, which is exactly the entropy of statistical mechanics. Long blocks concentrate on ≈ 2^{HN} nearly-equiprobable typical sequences, so the source compresses to H bits/symbol and no further — entropy is the compression limit. Add noise: the right toll is the equivocation H_y(x), the leftover uncertainty about the input, so the achievable rate is H(x) − H_y(x); maximizing over inputs defines the capacity C. And the typical-set fans, counted under a random code and averaged, show that for every rate below C reliable communication is possible with error → 0 at fixed rate, while above C it is impossible. Two definitions and two theorems, and the same quantity — entropy — sits at the center of both.

A small computation just to see the source half breathe — the uncertainty of a distribution and a prefix code whose average length lands on the same floor:

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
    # H = -sum p log2 p, the unique additive measure of uncertainty.
    if any(p < 0 for p in distribution.values()):
        raise ValueError("probabilities must be non-negative")
    return -sum(p * math.log2(p) for p in distribution.values() if p > 0)

def prefix_code(distribution):
    # Merge the two least likely subtrees until one prefix tree remains.
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
        return heap[0][2]                   # no binary choice is needed
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
    print(f"H = {H:.3f} bits/symbol; avg code length = {L:.3f} bits/symbol")
    assert L + 1e-12 >= H
    assert L < H + 1 + 1e-12
```
