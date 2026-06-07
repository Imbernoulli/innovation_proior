OK, let me think this through from scratch, because the thing I am actually stuck on is teaching. I stand in front of an information-theory class and I have to explain why convolutional codes are worth caring about — why Elias was right that a sliding shift register beats a block code of the same length. And the only honest answer I can give is "because when you run Fano sequential decoding on them, Yudkin showed the error probability has this nice exponent." That answer is unsatisfying to me and impossible to my students. The exponent is buried inside the analysis of a decoder that searches a tree, backs up, advances, with a Fano metric and an adaptive threshold, and whose *number of computations is a random variable* — Pareto-distributed, finite mean only below the cutoff rate `R0`, unbounded above it. I cannot put that on a blackboard and have anyone feel why the code is good. The decoder is in the way of the truth about the code. So let me set aside the goal of decoding well, and ask a smaller, cleaner question: is there a decoder I can *analyze* — one whose error probability I can bound with a union bound on the back of an envelope — even if it is not the decoder anyone would build? If I had that, I could read the code's error exponent off the analysis, and the superiority over block codes would become a theorem instead of a story.

So what makes the convolutional code tick, structurally, before any decoder? Let me draw the tree. A data symbol enters a `K`-stage register; `v` linear combinations of the register contents come out; the next symbol enters; and so on. From every node, `q` branches, one per next data symbol. After `L` symbols the tree has `q^L` leaves. That exponential is exactly the trouble. But stare at the register. At any instant the output depends only on the `K` symbols currently in it. Push that one step further: suppose two different paths diverged somewhere back, but then they happen to carry the *same* `K` consecutive data symbols. After those `K` symbols, both registers hold identical contents — the divergence has been shifted clear out the far end of the register. From that point on the two paths emit *identical* channel symbols forever. They have converged. They are, as far as the future is concerned, the same path.

That is the property. Only the last `K−1` data symbols — the register state — carry any memory; everything older has fallen off the end. So the tree is a wasteful drawing for this purpose. It looks like it branches forever into `q^L` distinct leaves, but paths with the same last `K−1` symbols are in the same state class for all future channel symbols. The finite object has `q^{K−1}` state classes repeated at every depth, and the depth axis just marks time. That drawing already feels more teachable than the full tree, because it does not explode. But more than teachable: the collapse is a structural fact about the code, and a decoder that respects it might be cheap enough to analyze.

Now, optimal decoding. Maximum likelihood: among all paths, pick the one most likely to have produced the received sequence `y`. Over a memoryless channel the likelihood of a path is the product of per-branch densities `p(y_k | x_k)`. Products of small numbers underflow and, worse, they do not add, so the natural thing is the logarithm: the log-likelihood of a path is the *sum* of branch terms `ln p(y_k | x_k)`. Define a branch metric `m_k = ln p(y_k | x_k)`; the score of a path is `sum_k m_k`, and a likelihood decoder maximizes that sum. For the binary symmetric channel with crossover probability `p < 1/2`, a branch of `v` bits with Hamming distance `d` has log-likelihood `v ln(1-p) + d ln(p/(1-p))`. The first term is constant and the second coefficient is negative, so maximizing log-likelihood is exactly minimizing total Hamming distance. If I use log-likelihood I compare maxima; if I use Hamming cost I compare minima. The decisions are the same.

But the optimal decision is "compare all `q^L` paths and take the best sum." That is the wall. `q^L` is exponential in the tree length; I can neither run it nor, which is what I actually care about right now, analyze it — there is no clean bound on the error probability of a brute comparison among exponentially many correlated paths. I need to get the `q^L` down to something I can sum a union bound over.

Take any state class `s` at branch level `K` — that is, fix the last `K−1` data symbols. Many paths from the origin arrive at `s`; each has its own accumulated likelihood. Two paths that both end in `s` share the same register contents, so any common future suffix adds the same future likelihood terms to both. That suggests a severe elimination rule that the error analysis can follow: for each state class, keep only the largest-likelihood path that reaches it and discard the other paths in that class. At the first step I compare, for every fixed `(a_2, ..., a_K)`, the `q` paths `(0, a_2, ..., a_K)`, `(1, a_2, ..., a_K)`, ..., `(q−1, a_2, ..., a_K)` and preserve the survivor. At the next step I extend each survivor by `q` new branches and compare the groups with common `(a_3, ..., a_{K+1})`. Then I repeat.

So I do not keep `q^L` paths. At each branch level, for each of the `q^{K−1}` state classes, I keep exactly *one* path: the highest-likelihood survivor retained for that class. The population swells by a factor of `q` when I extend the survivors and shrinks by the same factor when I compare the groups. The number of stored paths stays at `q^{K−1}`, independent of the tree length.

I have to keep my claim modest. The exhaustive decoder compares all `q^L` final paths. My procedure discards paths after local `q`-way comparisons and is built so that the event "the correct path is discarded" has a structure I can count. I am not going to sell it as the full optimum decoder. I want to earn the word "optimum" only in the asymptotic error-exponent sense, after the bound is on the page.

The recursion writes itself once I name the survivor metric. Let `M_K(s)` be the accumulated log-likelihood of the survivor into state class `s` at level `K`. A path to `s` at level `K` is a survivor to some predecessor class `s'` at level `K−1`, followed by the branch `s' → s` with branch metric `m_K(s', s)`. The retained metric is `M_K(s) = max_{s'} [ M_{K-1}(s') + m_K(s', s) ]`. At each level, for each state class, I look at the branches entering it, *add* the branch metric to the predecessor's survivor metric, *compare* across the entering branches, and *select* the maximum. If I use BSC Hamming costs instead, the same add-compare-select step selects the minimum. The work per level is `q^{K−1}` state classes times `q` entering branches, a fixed `q^K` regardless of how long the tree is. The exponential in `L` is gone; what is left is exponential only in the constraint length `K`, which is a small fixed parameter of the code, not the message length.

Said procedurally, the same thing reads: at each step extend every survivor by its `q` branches, then group the resulting paths by the state class they land in — equivalently, by the last `K−1` data symbols — and in each group keep only the maximum-likelihood one. "Survivor" is the right word for the kept path. Steady state: exactly one survivor per state class, `q^{K−1}` of them, forever.

The honest framing is still: this is a decoder simple enough to analyze, built to expose the code's exponent, not a decoder I am claiming is practical. The error event is now concrete. The correct all-zero path is lost only if it loses one of these survivor comparisons.

There is one more thing the recursion does not yet give me: I want the decoded *sequence*, not just the best final metric. The metric tells me how good the winning path is; it does not tell me which data symbols it used. So with each survivor I store, at each step, the branch I chose — the predecessor state and the input symbol that produced the maximum. At the end I walk these stored choices backward, predecessor by predecessor, reading off the input symbols in reverse, and flip them around to get the message. Because I drive the encoder back to the all-zero state with `K−1` flushing zeros at the end, the traceback starts from state 0.

Now to the reason I built any of this — the error bound, the thing for the class. I want `P_E`, the probability the decoder picks the wrong path, and I want it as `exp(−N·E(R))` with `N = Kv` the constraint length in channel symbols. Without loss of generality let the correct path be all-zeros. The correct path is eliminated only if, at some comparison, it loses to an *adversary* — a path that diverged from it some branches back and is being compared against it now. By the merging property, an adversary that diverged and then matched the correct path for `K` symbols would have re-converged and would not be a distinct competitor; so the adversaries that can actually beat the correct path at a given step are the ones that diverged `K + l` branches back and stayed *totally distinct* from it — never matching for `K` in a row. Count them: at most `q − 1` diverged exactly `K` branches back, and at most `(q − 1)^2 q^{l−1}` diverged `K + l` branches back, for `l = 1, 2, …`, each carrying `N + (ln q / R) l` channel symbols. The crucial fact is that totally-distinct paths have *independent* channel symbols (Reiffen's property, extended over the whole tree by Massey's generator modification), so each pairwise contest is a clean two-codeword hypothesis test on independent symbols, and a group of them is exactly a *block-code* decoding problem.

So the probability the correct path loses at one step is a union over adversaries, and each term is bounded by the random-coding error probability of a block code with that many words and that many symbols. The zeroth term is a block code of `q − 1` possible adversaries of length `N`, contributing at most `(q−1)^rho exp[−N E0(rho)]`. The `l`-th term, for `l >= 1`, has at most `(q−1)^2 q^{l−1}` possible adversaries and length `N + (ln q/R)l`, so it contributes at most `[(q−1)^2 q^{l−1}]^rho exp{−[N + (ln q/R)l] E0(rho)}`. Now `eps = E0(rho) − rho R` is positive, and `0 < rho <= 1` lets me upper-bound the coefficients by one leading `(q−1)` times the geometric tail `q^{-l eps/R}`. The per-step error is therefore `P(j+1) < [(q−1)/(1 − q^{-eps/R})] exp[−N E0(rho)]`.

This is independent of `j`, so a second union bound over the `L` survivor-comparison steps gives `P_E < [L(q − 1)/(1 − q^{-eps/R})] exp[−N E(R)]`, with `E(R) = E0(rho)` and, as the slack `eps` is allowed to shrink, `R = E0(rho)/rho` for `0 < rho <= 1`. Below the cutoff-rate endpoint the same upper-bound statement uses the `rho = 1` plateau; the clean asymptotic tightness claim is the one above `R0` for nonpathological channels. The `L` out front is harmless: it inflates the bound only linearly, while the exponent kills it exponentially in `N`, so I can let the tree be as long as I like.

Two things make this the payoff I wanted. First, this exponent `E(R) = E0(rho)` with `R = E0(rho)/rho` is *exactly* Yudkin's upper bound for Fano sequential decoding — but I have derived it for a decoder whose per-branch work is a fixed `q^K`, deterministic, not a heavy-tailed random variable with unbounded mean above `R0`. The same error performance, with predictable computation, by an analysis a student can follow: extend, compare in groups, keep survivors, union-bound the contests. Second, I can now set this convolutional exponent beside the block-code exponent of the same length `N` and *see* the superiority. Compute the cost honestly: decoding `L` branches costs about `L q^K` branch-likelihood computations, i.e. `(L/K) N q^K` symbol computations; the equivalent block code of length `N` carries `q^K` words and a maximum-likelihood decoder must compute `N q^K` symbol likelihoods per block, repeated `L/K` times — the *same* count. Same decoding complexity, and yet for the convolutional code, near capacity the exponent is *linear* in `(C − R)` rather than quadratic, and on a very noisy channel at `R = R0 = C/2` the convolutional exponent is almost six times the block-code exponent. That is the sentence I have been wanting to write on the board: for a given decoding complexity, the convolutional code wins, and here is the exponent that proves it.

I will note, with no false modesty, that the decoder I built to *prove* this is itself impractical: it keeps exponentially many survivor histories as `K` grows. That is fine, because it is meant first to make the code's error probability boundable and the proof teachable. It contributes to understanding convolutional codes and sequential decoding through its sheer simplicity of mechanization and analysis.

Let me land it on something runnable, so the procedure is concrete and not just an exponent. Take the simplest interesting case: binary `q = 2`, rate `1/2` (`v = 2` outputs per branch), constraint length `K = 3`, the classic generators `g1 = 1 + D + D^2` and `g2 = 1 + D^2` — in octal `(7, 5)`. The register has `K−1 = 2` memory cells, so `2^2 = 4` states. I build the finite-state tables once: for each state and each input bit, the next state and the two output bits. Encoding flushes `K−1 = 2` zeros at the end so the path ends in state 0. The channel is the binary symmetric channel — it flips bits — so the branch cost is Hamming distance, and the survivor cost is the smallest accumulated Hamming distance into each state. The decoder is the add-compare-select recursion plus the stored-survivor traceback I described.

```python
# Convolutional encoder + hard-decision decoder over a BSC.
# Rate 1/2, constraint length K=3, generators (7,5) octal:
#   g1 = 1 + D + D^2 = 0b111,  g2 = 1 + D^2 = 0b101.

K = 3                       # branch constraint length (register length)
N_STATES = 1 << (K - 1)     # 4 states = contents of the 2 memory cells
G = [0b111, 0b101]          # generator taps over the K-bit window

def _parity(x):
    return bin(x).count("1") & 1

def _encode_branch(state, bit):
    # window = newest bit (MSB) then the two memory cells; each generator is
    # a mod-2 inner product with the window -> one output bit.
    window = (bit << (K - 1)) | state
    out = tuple(_parity(window & g) for g in G)
    next_state = window >> 1          # shift; the oldest cell falls off the end
    return next_state, out

# Precompute the finite-state graph the deep tree folds into.
NEXT_STATE = [[0, 0] for _ in range(N_STATES)]
OUTPUT     = [[(), ()] for _ in range(N_STATES)]
PREDECESSORS = [[] for _ in range(N_STATES)]
for s in range(N_STATES):
    for b in (0, 1):
        ns, out = _encode_branch(s, b)
        NEXT_STATE[s][b], OUTPUT[s][b] = ns, out
        PREDECESSORS[ns].append((s, b, out))

def conv_encode(bits):
    state, coded = 0, []
    for b in list(bits) + [0] * (K - 1):   # flush -> path ends in state 0
        ns, out = _encode_branch(state, b)
        coded.extend(out); state = ns
    return coded

def _hamming(a, b):
    return sum(x != y for x, y in zip(a, b))

def decode(coded):
    n = len(G)                              # 2 output bits per branch
    if len(coded) % n:
        raise ValueError("coded stream length must be a multiple of branch size")
    T = len(coded) // n                     # branch stages
    INF = float("inf")

    # survivor metric per state: smallest accumulated Hamming distance into it.
    path_metric = [INF] * N_STATES
    path_metric[0] = 0                      # known initial state

    # store, per (stage, state), the chosen predecessor and input bit, so the
    # decoded SEQUENCE can be traced back -- the metric alone is not enough.
    survivor = [[None] * N_STATES for _ in range(T)]

    for t in range(T):
        r = coded[t * n:(t + 1) * n]        # received branch symbols
        new_metric = [INF] * N_STATES
        for ns in range(N_STATES):
            for s, b, out in PREDECESSORS[ns]:
                if path_metric[s] == INF:
                    continue
                # ADD branch metric to the survivor metric, then COMPARE/SELECT
                # the best path entering this next state.
                cand = path_metric[s] + _hamming(r, out)
                if cand < new_metric[ns]:
                    new_metric[ns] = cand
                    survivor[t][ns] = (s, b)
        path_metric = new_metric

    # traceback: the flushed encoder ends in state 0; walk survivors backward.
    state = 0
    bits = []
    for t in reversed(range(T)):
        choice = survivor[t][state]
        if choice is None:
            raise ValueError("no survivor reaches the required final state")
        prev, b = choice
        bits.append(b); state = prev
    bits.reverse()
    return bits[:-(K - 1)]                   # drop the flush bits

if __name__ == "__main__":
    import random
    random.seed(0)
    msg = [random.randint(0, 1) for _ in range(20)]
    coded = conv_encode(msg)
    received = coded[:]
    for i in random.sample(range(len(received)), 3):   # BSC: flip 3 bits
        received[i] ^= 1
    decoded = decode(received)
    print("recovered exactly:", decoded == msg)
```

So I end where I started. I cannot teach the superiority of convolutional codes cleanly if the only proof runs through Fano sequential decoding, whose computation is an unbounded random variable and whose error analysis is opaque; so I build an analyzable nonsequential decoder. The register's finiteness folds the `q^L`-leaf tree into `q^{K−1}` state classes; at each level I extend the survivors, add the branch metric, compare the paths entering each state class, select one survivor, and store enough history to trace the decoded sequence back. That recursion has fixed `q^K` work per step independent of tree length, and its error probability, union-bounded over totally distinct adversaries and summed over the `L` steps, gives `P_E < [L(q−1)/(1 − q^{-eps/R})] exp(−N E(R))` with `E(R) = E0(rho)`, `R = E0(rho)/rho` in the vanishing-slack limit. This is the exponent I needed: the same exponent Yudkin obtained through sequential decoding, now attached to a deterministic survivor decoder simple enough to put on the board.
