# Low-Density Parity-Check (LDPC) codes and iterative belief-propagation (sum-product) decoding

## Problem

Shannon's theorem guarantees that random block codes of rate R < C reach vanishing error probability as block length n → ∞, but a random code can only be decoded by an exponential maximum-likelihood search. The goal: a structured code family that keeps near-capacity error performance while decoding in time linear in n per iteration, using the channel's soft (a-posteriori) information.

## Key idea

Define the code by a **sparse** parity-check matrix H over GF(2): codewords are the x with Hx = 0, where each column of H has a small fixed weight j (each bit is in j checks) and each row a small fixed weight k (each check involves k bits). This is a **regular (n, j, k) LDPC code**, rate R ≥ 1 − j/k.

Sparsity is the whole point. View H as a bipartite bits-and-checks graph (bit nodes, parity-check nodes, edges = the 1s of H). Because j and k are small constants, the neighborhood of any node is a **tree** out to depth ~ log n. On a tree, **belief propagation** — passing soft messages between bit and check nodes — computes the *exact* posterior of each bit; on the real (slightly cyclic) graph it is a near-optimal approximation, costing O(n) per iteration with logarithmically many iterations. Keep j ≥ 3 so the minimum distance grows linearly with n (j = 2 gives only logarithmic distance), and design H with large girth so the tree approximation holds as long as possible.

## The algorithm (log-domain sum-product / BP)

Work with log-likelihood ratios L = ln(P(bit=0)/P(bit=1)).

- **Channel LLRs (seed).** BPSK over AWGN: send x = (−1)^bit (bit 0 → +1, bit 1 → −1), receive y = x + noise of variance σ². Channel LLR L_c = 2y/σ². (BSC with crossover p: ±ln((1−p)/p).)
- **Check → variable (horizontal), tanh rule.** For check i and bit j in it, combine the *other* incoming messages:
  L_{i→j} = 2·artanh( ∏_{j' ∈ check i, j' ≠ j} tanh(L_{j'→i}/2) ) = ln( (1+X)/(1−X) ),  X = ∏ tanh(L/2).
  (Follows from: the probability of an even number of 1s among independent bits is [1 + ∏(1−2P)]/2, and 1−2P = tanh(L/2).)
- **Variable → check (vertical), sum rule.** For bit j and check i on it, add the channel LLR and the *other* check messages:
  L_{j→i} = L_c[j] + Σ_{i' ∈ checks(j), i' ≠ i} L_{i'→j}.
- **Posterior + stop.** L_post[j] = L_c[j] + Σ_{i ∈ checks(j)} L_{i→j}; decide bit j = 0 if L_post[j] > 0, else 1. If the hard-decided x satisfies Hx = 0 (mod 2) the decoder halts (a valid codeword is self-certifying); otherwise iterate up to a cap, declaring a *detected* failure if the cap is hit.

Every message omits the edge it is sent on, so only **extrinsic** information flows — no node feeds its own belief back to itself.

## Code

```python
import warnings
import numpy as np

def check_random_state(seed):
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, np.random.RandomState):
        return seed
    return np.random.RandomState(seed)

def binaryproduct(X, Y):
    return X.dot(Y) % 2

def gaussjordan(X, change=False):
    A = np.copy(X).astype(int)
    m, n = A.shape
    if change:
        P = np.identity(m, dtype=int)
    pivot_old = -1
    for j in range(n):
        pivot = np.argmax(A[pivot_old + 1:m, j]) + pivot_old + 1
        if A[pivot, j]:
            pivot_old += 1
            if pivot_old != pivot:
                A[[pivot_old, pivot]] = A[[pivot, pivot_old]]
                if change:
                    P[[pivot_old, pivot]] = P[[pivot, pivot_old]]
            for i in range(m):
                if i != pivot_old and A[i, j]:
                    A[i] = np.abs(A[i] - A[pivot_old])
                    if change:
                        P[i] = np.abs(P[i] - P[pivot_old])
        if pivot_old == m - 1:
            break
    return (A, P) if change else A

def parity_check_matrix(n, d_v, d_c, seed=None):
    """Regular (n, d_v, d_c) LDPC matrix by Gallager's construction:
    stack d_v column-permuted blocks of a base block with d_c consecutive 1s/row."""
    if d_v <= 1:
        raise ValueError("d_v must be at least 2.")
    if d_c <= d_v:
        raise ValueError("d_c must be greater than d_v.")
    if n % d_c:
        raise ValueError("d_c must divide n for a regular matrix.")
    rng = check_random_state(seed)
    n_equations = (n * d_v) // d_c               # = d_v * n / d_c checks
    block_size = n_equations // d_v
    block = np.zeros((block_size, n), dtype=int)
    for i in range(block_size):
        block[i, i*d_c:(i+1)*d_c] = 1            # d_c consecutive 1s
    H = np.empty((n_equations, n), dtype=int)
    H[:block_size] = block
    for i in range(1, d_v):
        H[i*block_size:(i+1)*block_size] = rng.permutation(block.T).T
    return H                                      # weight d_v per col, d_c per row

def coding_matrix(H):
    n_equations, n = H.shape
    H_columns, tQ = gaussjordan(H.T, change=True)
    H_reduced = gaussjordan(H_columns.T)
    n_bits = n - H_reduced.sum()
    Y = np.zeros((n, n_bits), dtype=int)
    Y[n - n_bits:, :] = np.identity(n_bits, dtype=int)
    return binaryproduct(tQ.T, Y)

def encode(G, v, snr, seed=None):
    """G v over GF(2) -> BPSK -> AWGN.  snr = 10 log10(1/sigma^2) dB."""
    rng = check_random_state(seed)
    d = binaryproduct(G, v)
    x = (-1) ** d                                 # bit 0 -> +1, bit 1 -> -1
    sigma = 10 ** (-snr / 20)
    return x + rng.randn(*x.shape) * sigma

def incode(H, x):
    return (binaryproduct(H, x) == 0).all()

def decode(H, y, snr, maxiter=1000):
    """Log-domain belief propagation (sum-product)."""
    y = np.asarray(y, dtype=float)
    m, n = H.shape
    var = 10 ** (-snr / 10)                        # sigma^2
    Lc = 2 * y / var                              # channel LLR, log P(0)/P(1)
    Lq = np.zeros((m, n))                          # variable -> check
    Lr = np.zeros((m, n))                          # check -> variable
    checks = [np.where(H[i])[0] for i in range(m)]   # bits in each check
    bits   = [np.where(H[:, j])[0] for j in range(n)]# checks on each bit

    for it in range(maxiter):
        # horizontal: tanh check rule, omit the receiving bit
        for i in range(m):
            for j in checks[i]:
                src = Lq[i] if it > 0 else Lc
                X = 1.0
                for jp in checks[i]:
                    if jp != j:
                        X *= np.tanh(0.5 * src[jp])
                num, denom = 1 + X, 1 - X
                if num == 0:
                    Lr[i, j] = -1.0
                elif denom == 0:
                    Lr[i, j] = 1.0
                else:
                    Lr[i, j] = np.log(num / denom)   # 2*artanh(X)
        # vertical: sum of channel + other checks, omit this check
        for j in range(n):
            for i in bits[j]:
                Lq[i, j] = Lc[j] + sum(Lr[ip, j] for ip in bits[j] if ip != i)
        # posterior + hard decision + stopping test
        L_post = Lc + np.array([Lr[bits[j], j].sum() for j in range(n)])
        x = (L_post <= 0).astype(int)              # positive LLR -> bit 0
        if incode(H, x):
            break
    else:
        warnings.warn("Decoding stopped before convergence.")
    return x
```

The decoder returns as soon as the hard decision satisfies all parity checks; if the cap is reached, the last hard decision is a detected non-converged attempt rather than a certified codeword.

## Why it works

- **Sparse H ⇒ tree-like neighborhoods ⇒ near-exact BP at O(n)/iteration.** Few 1s per row/column keep the bits-and-checks graph locally cycle-free out to depth ~ log n; BP is exact on trees and a good approximation otherwise.
- **Soft information, fully used.** Unlike bounded-distance algebraic decoders, BP consumes per-bit a-posteriori LLRs and improves with iteration.
- **Regular, j ≥ 3.** Uniform degrees give tractable analysis and uniform hardware; column weight ≥ 3 guarantees minimum distance growing linearly with n.
- **Extrinsic messages + Hx = 0 stop.** Omitting the receiving edge prevents self-reinforcement; the parity-check test makes a successful decode self-certifying and failures detectable.
- **Caveats by graph structure.** Short cycles (small girth) bias the messages, so good code design keeps the local graph tree-like for as many tiers as possible.
