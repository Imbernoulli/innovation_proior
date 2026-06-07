# Low-Density Parity-Check Codes and Belief-Propagation Decoding

## Problem

Shannon's theorem guarantees codes that approach capacity, and the random-coding argument shows almost all long linear codes are good — but optimal (maximum-likelihood) decoding of a long unstructured code is exponential, while the short algebraic codes that *are* decodable lose distance and sit far from capacity. We want a code that is simultaneously good (linear minimum distance, near-capacity) and decodable at low cost (ideally linear in block length).

## Key idea

Make the parity-check matrix **sparse**: each bit participates in a small fixed number of checks (column weight `j`), each check ties a small fixed number of bits (row weight `k`), with `j, k = O(1)`. Sparsity does three jobs at once:

1. The matrix has `O(n)` ones, so a decoder that sweeps its nonzeros runs in linear time per iteration.
2. The ensemble of such matrices is still **asymptotically good**: for `j ≥ 3`, the typical minimum distance grows linearly with `n` (rate `R ≈ 1 − j/k`). `j = 2` is too weak (low-weight codewords from short cycles), so `j ≥ 3`.
3. Sparsity means long cycles, so each bit's local neighborhood in the bipartite (bit ↔ check) graph looks like a tree out to many levels.

On a tree, the exact per-bit a-posteriori probabilities are computed by a local rule. Run that local rule on the real (loopy but locally tree-like) graph as an iterative approximation: **belief propagation / the sum-product algorithm**. The residual correlations from cycles are weak because the cycles are long, so the iteration is a low-cost approximation to maximum-likelihood decoding.

## The decoder (sum-product, log domain)

Work in log-likelihood ratios `L = log(P(bit=0)/P(bit=1))`. Channel input on a binary-input AWGN channel (BPSK `0→+1, 1→−1`, noise variance `σ²`): `L_c = 2y/σ²`.

The check rule rests on one identity (Lemma): for independent bits with `P(bit_l = 1) = P_l`, the probability of an **even** number of ones is `(1 + ∏(1 − 2P_l))/2`, and `1 − 2P_l = tanh(L_l/2)`. A parity check is satisfied iff its bits have an even number of ones, so:

- **Check → bit (horizontal), extrinsic:** `L_{r,m→n} = 2·atanh( ∏_{n'∈N(m)\n} tanh(L_{q,n'→m}/2) )`, i.e. `log( (1+∏)/(1−∏) )` with the product over the *other* bits in the check.
- **Bit → check (vertical), extrinsic:** `L_{q,n→m} = L_c(n) + Σ_{m'∈M(n)\m} L_{r,m'→n}` — channel log-ratio plus the *other* checks' verdicts (independent evidence adds in the log domain).
- **Posterior:** `L_n = L_c(n) + Σ_{m∈M(n)} L_{r,m→n}`; hard bit `x_n = [L_n < 0]`.

Every message is **extrinsic** (excludes the edge it is sent along) so the tree-independence the lemma assumes stays honest. Initialize bit→check messages to `L_c`. Iterate; after each sweep, if the syndrome `H x = 0`, `x` is a codeword — stop. Cap the iterations and declare failure otherwise (so it never returns a wrong codeword silently). Cost per iteration is `O(jn)`; number of useful iterations grows ~`log n`.

## Code construction

Regular `(n, j, k)` matrix: a base block of `n/k` rows with consecutive runs of `k` ones (one per column), stacked with `j−1` independent random column-permutations of it. Result: column weight `j`, row weight `k`, random-like. A systematic generator `G` with `H Gᵀ = 0` comes from Gauss-Jordan over GF(2).

## Working code

```python
import numpy as np

def parity_check_matrix(n, d_v, d_c, rng):
    """Sparse regular H: stack d_v column-permutations of a consecutive-ones block.
    Column weight d_v (bit in d_v checks), row weight d_c (check on d_c bits)."""
    assert n % d_c == 0 and d_c > d_v >= 2
    m = n * d_v // d_c
    rows = m // d_v
    block = np.zeros((rows, n), dtype=int)
    for i in range(rows):
        block[i, i * d_c:(i + 1) * d_c] = 1
    H = [block]
    for _ in range(1, d_v):
        H.append(rng.permutation(block.T).T)
    return np.vstack(H)

def systematic_generator(H):
    """G (k x n) with H G^T = 0 over GF(2), via Gauss-Jordan."""
    m, n = H.shape
    A = H.copy() % 2
    r = 0
    for c in range(n):
        piv = np.where(A[r:, c] == 1)[0]
        if not len(piv):
            continue
        A[[r, piv[0] + r]] = A[[piv[0] + r, r]]
        for rr in range(m):
            if rr != r and A[rr, c]:
                A[rr] ^= A[r]
        r += 1
        if r == m:
            break
    rank = r
    pivot_cols = [int(np.where(A[i] == 1)[0][0]) for i in range(rank)]
    nonpivot = [c for c in range(n) if c not in pivot_cols]
    order = pivot_cols + nonpivot
    P = A[:rank, order][:, rank:]
    k = n - rank
    G = np.zeros((k, n), dtype=int)
    G[:, :rank] = P.T
    G[:, rank:] = np.eye(k, dtype=int)
    G = G[:, np.argsort(order)]
    assert not ((G @ H.T) % 2).any()
    return G

def bp_decode(H, Lc, maxiter=100):
    """Log-domain sum-product. Lc = per-bit channel log-likelihood ratios."""
    m, n = H.shape
    bits_in_check = [np.where(H[i] == 1)[0] for i in range(m)]
    checks_on_bit = [np.where(H[:, j] == 1)[0] for j in range(n)]
    Lq = np.zeros((m, n)); Lr = np.zeros((m, n))
    for i in range(m):
        Lq[i, bits_in_check[i]] = Lc[bits_in_check[i]]
    for _ in range(maxiter):
        for i in range(m):                                   # horizontal (check->bit)
            ni = bits_in_check[i]
            t = np.tanh(0.5 * Lq[i, ni])
            for j in ni:
                prod = np.clip(np.prod(t[ni != j]), -1 + 1e-12, 1 - 1e-12)
                Lr[i, j] = np.log((1 + prod) / (1 - prod))
        for j in range(n):                                   # vertical (bit->check)
            mj = checks_on_bit[j]; s = Lr[mj, j].sum()
            for i in mj:
                Lq[i, j] = Lc[j] + s - Lr[i, j]
        L = Lc + np.array([Lr[checks_on_bit[j], j].sum() for j in range(n)])
        x = (L < 0).astype(int)
        if not ((H @ x) % 2).any():
            return x, True
    return x, False

if __name__ == "__main__":
    rng = np.random.RandomState(0)
    n, d_v, d_c = 96, 3, 6
    H = parity_check_matrix(n, d_v, d_c, rng)
    G = systematic_generator(H); k = G.shape[0]
    for snr_db in [1.0, 2.0, 3.0, 4.0]:
        raw = dec = fail = 0
        for s in range(40):
            r = np.random.RandomState(s)
            u = r.randint(0, 2, k); c = (u @ G) % 2
            sigma2 = 10 ** (-snr_db / 10) / (2 * (k / n))
            y = (1 - 2 * c) + np.sqrt(sigma2) * r.randn(n)
            Lc = 2 * y / sigma2
            raw += int(((y < 0).astype(int) != c).sum())
            xhat, ok = bp_decode(H, Lc); dec += int((xhat != c).sum()); fail += (0 if ok else 1)
        print(f"rate={k/n:.3f} snr={snr_db:.0f}dB raw_bit_err={raw} decoded_bit_err={dec} fail={fail}")
```

The harness prints raw hard-decision bit errors, decoded bit errors, and decoding failures so the decoder can be checked directly on the toy AWGN example.
