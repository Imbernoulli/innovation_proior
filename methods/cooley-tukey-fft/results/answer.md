# The Fast Fourier Transform (Cooley–Tukey)

## Problem

Compute all $N$ discrete Fourier coefficients of $N$ samples,

$$X(j) = \sum_{k=0}^{N-1} A(k)\, W^{jk}, \qquad W = e^{-2\pi i/N}, \qquad j = 0,\dots,N-1,$$

faster than the naive $O(N^2)$ matrix-vector product $X = F A$ with $F_{jk} = W^{jk}$.

The immediate pressure is large spectral computation: a visible three-dimensional helium-3 spin-periodicity transform on power-of-two axes, and behind it the larger problem of analyzing seismometer time series around the Soviet Union for nuclear-test-ban verification without on-site inspection.

## Key idea

The matrix $F$ is entirely powers of one root of unity, so it is hugely redundant. Split the sum on the index $k$ to expose shorter DFTs and recombine them with a single multiplier.

**Radix-2 decimation in time.** For even $N$, split the inputs by parity ($k = 2m$ and $k = 2m+1$). Using $W^2 = e^{-2\pi i/(N/2)}$,

$$X(j) = \underbrace{\sum_{m} A(2m)\,W_{N/2}^{\,jm}}_{E(j)} + W^{j}\underbrace{\sum_{m} A(2m+1)\,W_{N/2}^{\,jm}}_{O(j)},$$

where $E,O$ are length-$N/2$ DFTs of the even and odd subsequences, and $W^j$ is the **twiddle factor**. Both half-transforms are periodic with period $N/2$, and $W^{j+N/2} = -W^j$, so one twiddle multiply yields two outputs — the **butterfly**:

$$X(j) = E(j) + W^{j}O(j), \qquad X(j+N/2) = E(j) - W^{j}O(j), \qquad j = 0,\dots,\tfrac N2-1.$$

**Recursion and cost.** Apply the split recursively. The cost obeys $T(N) = 2T(N/2) + O(N)$, and by the master theorem $T(N) = O(N\log N)$: $\log_2 N$ passes, each $O(N)$. The base case $N=1$ is the identity.

**Bit reversal.** Splitting repeatedly by parity routes each input by its bits least-significant-first; the sample landing at leaf position $p$ is the input at index $\mathrm{rev}(p)$ (bit-reversal). Permuting a working array into bit-reversed order first lets the transform be built bottom-up by doubling passes whose butterflies overwrite two slots at a time.

**General composite $N$.** Nothing requires radix 2. For $N = r_1 r_2$, group inputs by residue modulo $r_1$ and split outputs compatibly: $k = k_0 + r_1 k_1$, $j = j_0 + r_2 j_1$:

$$X(j_0 + r_2 j_1) = \sum_{k_0=0}^{r_1-1} W_{r_1}^{\,j_1 k_0}\, W^{\,j_0 k_0}\Big[\sum_{k_1=0}^{r_2-1} A(k_0 + r_1 k_1)\,W_{r_2}^{\,j_0 k_1}\Big],$$

i.e. $r_1$ DFTs of length $r_2$, a twiddle $W^{j_0 k_0}$, then $r_2$ DFTs of length $r_1$ — cost $N(r_1+r_2)$ up to linear twiddle work. With $r_1=2$, this reduces exactly to the even/odd butterfly above. Recursing to the prime factorization gives cost $N\sum_i p_i$ ($N$ times the sum of prime factors), which is $2N\log_2 N$ when $N = 2^L$. The method favors highly composite $N$ and this factorization alone degrades to $O(N^2)$ when $N$ is prime. Zero-padding is useful when the application accepts a longer transform, but it is not the exact original prime-length DFT. (Twiddle factors distinguish this from the coprime-factor / CRT variant, which has no cross term but cannot recurse on $N = 2^L$.) The inverse transform is the same computation with $W^{-1} = e^{+2\pi i/N}$ and a $1/N$ normalization.

## Algorithm (radix-2, iterative butterflies)

1. Permute a working array into bit-reversed index order. The permutation can be implemented by disjoint swaps.
2. For $L = 2, 4, \dots, N$ (a pass per stage): set $W_L = e^{-2\pi i/L}$; for each length-$L$ block, run a twiddle $w$ from $1$ multiplying by $W_L$ each step, and for $r = 0,\dots,L/2-1$ apply the butterfly $u = a[i{+}r]$, $t = w\cdot a[i{+}r{+}L/2]$, $a[i{+}r] = u+t$, $a[i{+}r{+}L/2] = u-t$.

Computing $W_L$ once per pass (not per butterfly) keeps it to $\log_2 N$ transcendental evaluations.

## Code

```python
import cmath

def root_of_unity(N, sign=-1):
    return cmath.exp(sign * 2j * cmath.pi / N)

def dft_direct(A, sign=-1):
    """Baseline O(N^2) DFT: X(j) = sum_k A(k) W^{jk}, W = exp(sign*2pi i/N)."""
    N = len(A)
    W = root_of_unity(N, sign)
    return [sum(A[k] * W ** (j * k) for k in range(N)) for j in range(N)]

def _validate_length(n):
    if n < 1 or n & (n - 1):
        raise ValueError("length must be a power of two")

def transform_recursive(A, sign=-1):
    """Recursive radix-2 decimation-in-time transform."""
    N = len(A)
    _validate_length(N)
    if N == 1:
        return [A[0]]                       # one-point DFT is the identity
    E = transform_recursive(A[0::2], sign)  # even-indexed subsequence
    O = transform_recursive(A[1::2], sign)  # odd-indexed subsequence
    X = [0j] * N
    W = root_of_unity(N, sign)
    for j in range(N // 2):
        t = W ** j * O[j]                   # W^j * O(j)
        X[j]          = E[j] + t            # X(j)
        X[j + N // 2] = E[j] - t            # X(j+N/2), since W^{N/2} = -1
    return X

def _reorder_for_iteration(A):
    A = list(A)
    n = len(A)
    _validate_length(n)
    bits = n.bit_length() - 1
    for p in range(n):
        q = int(format(p, '0{}b'.format(bits))[::-1], 2)
        if q > p:                           # disjoint swaps on the working copy
            A[p], A[q] = A[q], A[p]
    return A

def transform_iterative(A, sign=-1):
    """Iterative radix-2 FFT on a bit-reversed working copy. N must be a power of two."""
    A = _reorder_for_iteration(A)
    n = len(A)
    L = 2
    while L <= n:
        wL = root_of_unity(L, sign)          # root for length-L blocks
        for start in range(0, n, L):
            w = 1 + 0j
            for r in range(L // 2):
                u = A[start + r]
                t = w * A[start + r + L // 2]
                A[start + r]          = u + t       # butterfly E + W^r O
                A[start + r + L // 2] = u - t       # butterfly E - W^r O
                w *= wL
        L <<= 1
    return A

def inverse_transform(X):
    """Inverse: conjugate root (sign=+1) and 1/N normalization."""
    n = len(X)
    return [v / n for v in transform_iterative(X, sign=+1)]
```
