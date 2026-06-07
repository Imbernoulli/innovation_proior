# The Viterbi Algorithm

## Problem

Decode a convolutional code transmitted over a memoryless noisy channel: recover the most-likely transmitted data sequence from a corrupted received stream, and do it with a procedure whose error probability can be bounded cleanly enough to prove that convolutional codes outperform block codes of the same length. Optimal (maximum-likelihood) decoding compares all `q^L` paths through the length-`L` code tree — exponential in the message length, and intractable to analyze. Sequential decoding (Fano) attains the right exponent, but its number of computations is a random variable with unbounded mean at and above the cutoff rate `R0`.

## Key idea

A convolutional encoder's output depends only on the `K` most recent data symbols (the shift-register state). Two paths that carry identical data symbols for `K` consecutive branches re-enter identical register states and emit identical channel symbols forever after — they *merge*. So the `q^L`-leaf tree can be folded into a finite-state trellis with only `q^{K−1}` states, repeated at each time step.

The decoder keeps **one survivor per state**: at each step it extends the current survivors by all `q` possible input symbols, groups the resulting paths by the next `K−1`-symbol state, and keeps the largest-likelihood path in each group. This leaves exactly `q^{K−1}` survivors after each step and gives a deterministic comparison structure whose error probability can be union-bounded.

Working with log-likelihoods makes the path metric a *sum* of branch metrics. For log-likelihood metrics, the survivor metric obeys the **add-compare-select** recursion
```
M_t(s) = max over predecessors s' of [ M_{t-1}(s') + m_t(s', s) ],
```
and storing the chosen predecessor at each (state, time) lets a **traceback** recover the decoded sequence. Over a BSC with crossover `p < 1/2`, log-likelihood is an affine decreasing function of Hamming distance, so the hard-decision implementation uses accumulated Hamming distance as a negative log-likelihood cost and selects the minimum.

## Cost and error exponent

Per step the work is `q^{K−1}` states × `q` entering branches = `q^K`, **independent of the tree length `L`** (exponential only in the small fixed constraint length `K`, not in the message). The correct path is lost only when it loses a comparison to a *totally distinct* adversary. There are at most `q − 1` adversaries diverging exactly `K` branches back, and at most `(q − 1)^2 q^{l−1}` adversaries diverging `K + l` branches back for `l >= 1`; those longer adversaries span `N + (ln q/R)l` channel symbols. Totally distinct paths have independent channel symbols, so each contest is bounded as a block-code hypothesis test. Union-bounding over adversaries and over the `L` steps with Gallager's random-coding bound gives
```
P_E  <  [ L(q − 1) / (1 − q^{-eps/R}) ] · exp[ −N · E(R) ],   eps = E0(rho) − rho R > 0,
E(R) = E0(rho); in the vanishing-slack exponent curve, R = E0(rho) / rho,   0 < rho <= 1,
```
where `N = Kv` is the constraint length in channel symbols and `E0(rho) = max_p −ln sum_y [ sum_x p(x) p(y|x)^{1/(1+rho)} ]^{1+rho}`. This exponent matches Yudkin's bound for sequential decoding, but is now attained by a decoder with *deterministic* per-branch cost. Setting it beside the block-code exponent at equal decoding complexity shows the convolutional code wins: near capacity its exponent is linear in `(C − R)` rather than quadratic, and on a very noisy channel at `R = R0 = C/2` it is almost six times the block-code exponent.

## Algorithm

1. Build the trellis: for each state and input symbol, the next state and the emitted channel symbols.
2. Initialize the survivor metric to 0 at the known start state, infinity elsewhere.
3. For each received branch, for each next state, **add** the branch cost (Hamming distance for the BSC) to each predecessor's survivor cost, **compare** the candidates entering that state, **select** the minimum, and record the surviving predecessor and input.
4. After flushing the encoder to the known zero final state, **trace back** the recorded survivors from state 0 to read off the decoded data sequence.

## Working code

Hard-decision decoder for the rate-1/2, `K=3`, `(7,5)`-octal convolutional code over a binary symmetric channel. The structure — next-state/output tables, an accumulated path metric per state, Hamming branch metrics, one stored survivor predecessor per (state, time), and a traceback — mirrors the canonical CommPy convolutional-coding implementation.

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

Running the demo prints `recovered exactly: True` for the seeded 20-bit message with 3 channel flips.
