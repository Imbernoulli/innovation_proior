The problem I set myself is to decode a convolutional code sent over a memoryless noisy channel — recover the most-likely transmitted data sequence from a corrupted received stream — and to do it with a procedure whose error probability I can actually bound. That last clause is the real motive. Elias conjectured that a sliding shift-register code beats a block code of the same length, but the only confirmation rested on running Fano sequential decoding and reading Yudkin's exponent off its analysis. That is unsatisfying. Sequential decoding searches the code tree, advancing along high-metric branches and backing up when the running Fano metric drops below an adaptive threshold; its number of branch computations is a Pareto-distributed random variable whose mean is finite only below the computational cutoff rate $R_0 = E_0(1)$ and unbounded at and above it. A decoder whose work is a heavy-tailed random variable is awkward to bound and impossible to teach. The exhaustive maximum-likelihood alternative is no better for my purpose: comparing all $q^L$ paths through a length-$L$ tree is exponential in the message length and, more to the point, there is no clean handle on the error probability of a brute comparison among exponentially many correlated paths. Massey's bit-by-bit threshold decoding has fixed low cost but is decisively suboptimal — it decides each symbol from a small window rather than the full received sequence, so its exponent falls well short of maximum likelihood. What I want is a decoder simple enough that an elementary union bound pins down its error probability, deterministic in its per-step work, and tight enough that its exponent meets the best known bound for the code class.

The method I propose is the Viterbi algorithm. Its foundation is a structural fact about the encoder, prior to any decoder. At any instant the $v$ output symbols depend only on the $K$ data symbols currently in the register; everything older has fallen off the end. So if two paths diverge somewhere and then happen to carry the same $K$ consecutive data symbols, both registers hold identical contents — the divergence has been shifted clear out the far end — and from that point on they emit identical channel symbols forever. They *merge*. Only the last $K-1$ data symbols, the register state, carry any memory. This means the $q^L$-leaf tree is a wasteful drawing: paths sharing the last $K-1$ symbols are in the same state class for all future channel symbols, so the tree folds into a finite-state trellis of just $q^{K-1}$ states repeated at every time step, with the depth axis marking time. That collapse is what makes both a cheap decoder and a clean proof possible.

The decoding rule that respects the collapse is to keep exactly one survivor per state. Over a memoryless channel a path's likelihood is the product of per-branch densities $p(y_k \mid x_k)$; products underflow and do not add, so I work with the log-likelihood, which makes a path's score the *sum* of branch metrics $m_k = \ln p(y_k \mid x_k)$. The key observation is that any two paths ending in the same state share identical register contents, so any common future suffix adds the same future likelihood to both. Whichever of them is behind now stays behind forever — so I may discard it immediately. Concretely, at each step I extend every survivor by all $q$ possible input symbols, group the resulting extended paths by the state they land in (equivalently by their last $K-1$ data symbols), and in each group keep only the largest-likelihood path. The population swells by a factor $q$ on extension and shrinks by $q$ on grouping, so in steady state there are exactly $q^{K-1}$ survivors, independent of the tree length. Naming the survivor metric $M_t(s)$ as the accumulated log-likelihood of the survivor into state $s$ at time $t$, the recursion writes itself: a path to $s$ is a survivor to some predecessor $s'$ extended by the branch $s' \to s$ with metric $m_t(s', s)$, so

$$M_t(s) = \max_{s'}\;\bigl[\,M_{t-1}(s') + m_t(s', s)\,\bigr].$$

This is the add-compare-select step — *add* the branch metric to each predecessor's survivor metric, *compare* across the branches entering the state, *select* the maximum. The work per step is $q^{K-1}$ states times $q$ entering branches, a fixed $q^K$ regardless of how long the tree is; the exponential in $L$ is gone, and what remains is exponential only in the small fixed constraint length $K$. The recursion as stated only yields the best final metric, not the decoded symbols, so with each survivor I also store the chosen predecessor and input symbol at each (state, time); at the end I trace these choices backward to read off the message in reverse. Because $K-1$ flushing zeros are fed in to drive the encoder to the all-zero state, the traceback starts deterministically from state 0.

Two design choices earn their place. Working in log-likelihood rather than likelihood is not just numerical hygiene — it is what turns the path score into an additive sum so the max distributes over predecessors and the recursion becomes local. And for the binary symmetric channel with crossover $p < 1/2$, a $v$-bit branch at Hamming distance $d$ has log-likelihood $v\ln(1-p) + d\ln\!\bigl(p/(1-p)\bigr)$; the first term is constant across branches and the coefficient of $d$ is negative, so maximizing log-likelihood is identically minimizing total Hamming distance. The hard-decision implementation therefore uses accumulated Hamming distance as a negative-log-likelihood cost and selects the minimum — the same add-compare-select with $\max$ replaced by $\min$.

What makes this the payoff I wanted is the error analysis the structure now permits. Take the correct path to be all-zeros without loss of generality; it is lost only if at some comparison it loses to an adversary that diverged from it some branches back. By the merging property an adversary that diverged and then matched for $K$ symbols would have re-converged and is no competitor, so the dangerous adversaries are the *totally distinct* ones that never match for $K$ in a row: at most $q-1$ diverged exactly $K$ branches back, and at most $(q-1)^2 q^{l-1}$ diverged $K+l$ branches back for $l \ge 1$, those spanning $N + (\ln q / R)\,l$ channel symbols. Totally distinct paths have independent channel symbols (Reiffen's property within the first constraint length, extended over the whole tree by Massey's generator modification), so each pairwise contest is a clean two-codeword hypothesis test and a group of them is exactly a block-code decoding problem. Bounding each by Gallager's random-coding error probability, the zeroth term contributes at most $(q-1)^\rho \exp[-N E_0(\rho)]$ and the $l$-th term at most $[(q-1)^2 q^{l-1}]^\rho \exp\{-[N + (\ln q/R)l]E_0(\rho)\}$. With $\varepsilon = E_0(\rho) - \rho R > 0$ and $0 < \rho \le 1$, the coefficients collapse into a leading $(q-1)$ times a geometric tail $q^{-l\varepsilon/R}$, so the per-step error is below $[(q-1)/(1 - q^{-\varepsilon/R})]\exp[-N E_0(\rho)]$, independent of the step. A second union bound over the $L$ comparison steps gives

$$P_E \;<\; \frac{L(q-1)}{1 - q^{-\varepsilon/R}}\;\exp\!\bigl[-N\,E(R)\bigr],\qquad E(R) = E_0(\rho),\quad R = \frac{E_0(\rho)}{\rho},\quad 0 < \rho \le 1,$$

where $N = Kv$ and $E_0(\rho) = \max_p\, -\ln \sum_y \bigl[\sum_x p(x)\,p(y\mid x)^{1/(1+\rho)}\bigr]^{1+\rho}$. The factor $L$ inflates the bound only linearly while the exponent kills it exponentially in $N$, so the tree may be as long as I like. This is precisely Yudkin's exponent for Fano sequential decoding — but now attained by a decoder whose per-branch work is a fixed, deterministic $q^K$ rather than an unbounded-mean random variable. Decoding $L$ branches costs about $L q^K$ branch-likelihood computations, the same symbol count as maximum-likelihood decoding the equivalent length-$N$ block code repeatedly; at equal decoding complexity the convolutional exponent is linear in $(C-R)$ near capacity rather than quadratic, and on a very noisy channel at $R = R_0 = C/2$ it is almost six times the block-code exponent. The decoder I built to prove this keeps exponentially many survivor histories as $K$ grows, so it is itself impractical at large $K$; its worth is that it makes the code's error probability boundable by an argument a student can follow — extend, group, select survivors, union-bound the contests.

For a concrete instance: binary $q=2$, rate $1/2$ ($v=2$ outputs per branch), constraint length $K=3$, generators $g_1 = 1 + D + D^2$ and $g_2 = 1 + D^2$, i.e. $(7,5)$ octal, giving $2^{K-1} = 4$ states. The trellis tables are built once; encoding flushes $K-1=2$ zeros so the path ends in state 0; the channel flips bits, so the branch cost is Hamming distance and the survivor cost is the smallest accumulated Hamming distance into each state.

```python
K = 3                       # branch constraint length (register length)
N_STATES = 1 << (K - 1)     # 4 states = contents of the 2 memory cells
G = [0b111, 0b101]          # generators (7,5) octal: 1+D+D^2, 1+D^2

def _parity(x):
    return bin(x).count("1") & 1

def _encode_branch(state, bit):
    window = (bit << (K - 1)) | state          # newest bit is the MSB
    out = tuple(_parity(window & g) for g in G) # each generator: mod-2 inner product
    next_state = window >> 1                    # shift; oldest cell falls off
    return next_state, out

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
    for b in list(bits) + [0] * (K - 1):       # flush -> path ends in state 0
        ns, out = _encode_branch(state, b)
        coded.extend(out); state = ns
    return coded

def _hamming(a, b):
    return sum(x != y for x, y in zip(a, b))

def decode(coded):
    n = len(G)                                  # output bits per branch
    if len(coded) % n:
        raise ValueError("coded stream length must be a multiple of branch size")
    T = len(coded) // n
    INF = float("inf")
    path_metric = [INF] * N_STATES
    path_metric[0] = 0                          # known initial state
    survivor = [[None] * N_STATES for _ in range(T)]
    for t in range(T):
        r = coded[t * n:(t + 1) * n]
        new_metric = [INF] * N_STATES
        for ns in range(N_STATES):
            for s, b, out in PREDECESSORS[ns]:
                if path_metric[s] == INF:
                    continue
                cand = path_metric[s] + _hamming(r, out)           # ADD
                if cand < new_metric[ns]:                          # COMPARE/SELECT
                    new_metric[ns] = cand
                    survivor[t][ns] = (s, b)
        path_metric = new_metric
    state = 0                                                       # known final state
    bits = []
    for t in reversed(range(T)):                                   # TRACEBACK
        choice = survivor[t][state]
        if choice is None:
            raise ValueError("no survivor reaches the required final state")
        prev, b = choice
        bits.append(b); state = prev
    bits.reverse()
    return bits[:-(K - 1)]                                         # drop flush bits

if __name__ == "__main__":
    import random
    random.seed(0)
    msg = [random.randint(0, 1) for _ in range(20)]
    coded = conv_encode(msg)
    received = coded[:]
    for i in random.sample(range(len(received)), 3):               # BSC: flip 3 bits
        received[i] ^= 1
    print("recovered exactly:", decode(received) == msg)
```
