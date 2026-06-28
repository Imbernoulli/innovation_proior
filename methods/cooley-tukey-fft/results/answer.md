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

Single-file C++17. It reads `N` (a power of two) and `N` complex samples as `re im` pairs from stdin, and writes the `N` coefficients `X(j)`, one `re im` pair per line. `dft_direct` is kept as the small-N oracle and `inverse_transform` as the conjugate-root wrapper.

```cpp
// Cooley-Tukey radix-2 FFT.
// Reads from stdin: N (a power of two) then N complex samples as "re im" pairs.
// Writes to stdout: the N DFT coefficients X(j) = sum_k A(k) W^{jk}, W = e^{-2*pi*i/N},
// one "re im" pair per line.
#include <bits/stdc++.h>
using namespace std;

using cd = complex<double>;
const double PI = acos(-1.0);

// O(N^2) baseline / small-N oracle: X(j) = sum_k A(k) W^{jk}, W = exp(sign*2*pi*i/N).
vector<cd> dft_direct(const vector<cd>& A, int sign = -1) {
    int N = (int)A.size();
    cd W = exp(cd(0.0, sign * 2.0 * PI / N));
    vector<cd> X(N);
    for (int j = 0; j < N; ++j) {
        cd acc = 0.0;
        for (int k = 0; k < N; ++k) acc += A[k] * pow(W, (double)(j * k));
        X[j] = acc;
    }
    return X;
}

// Route input to leaf order: sample at index rev(p) goes to position p.
// Reversal is an involution, so the swaps are disjoint and done in place.
void bit_reverse(vector<cd>& a) {
    int n = (int)a.size();
    int bits = __builtin_ctz((unsigned)n); // log2 n, n a power of two
    for (int p = 0; p < n; ++p) {
        int q = 0;
        for (int b = 0; b < bits; ++b)
            if (p & (1 << b)) q |= 1 << (bits - 1 - b);
        if (q > p) swap(a[p], a[q]);
    }
}

// Iterative radix-2 FFT: bit-reverse a working copy, then run log2 N doubling
// passes of in-place butterflies. N must be a power of two. sign=-1 forward, +1 inverse.
vector<cd> transform(vector<cd> A, int sign = -1) {
    int n = (int)A.size();
    bit_reverse(A);
    for (int L = 2; L <= n; L <<= 1) {            // pass builds length-L = 2^s transforms
        cd wL = exp(cd(0.0, sign * 2.0 * PI / L)); // root of unity for this block length
        for (int start = 0; start < n; start += L) {
            cd w = 1.0;                            // running twiddle, propagated by multiplication
            for (int r = 0; r < L / 2; ++r) {
                cd u = A[start + r];
                cd t = w * A[start + r + L / 2];   // twiddle times the upper half
                A[start + r]         = u + t;      // butterfly: E + W^r O
                A[start + r + L / 2] = u - t;      // butterfly: E - W^r O
                w *= wL;
            }
        }
    }
    return A;
}

// Inverse: conjugate root (sign=+1) and 1/N normalization.
vector<cd> inverse_transform(const vector<cd>& X) {
    int n = (int)X.size();
    vector<cd> a = transform(X, +1);
    for (cd& v : a) v /= (double)n;
    return a;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int N;
    if (!(cin >> N)) return 0;
    if (N < 1 || (N & (N - 1))) {                  // length must be a power of two
        cerr << "length must be a power of two\n";
        return 1;
    }
    vector<cd> A(N);
    for (int i = 0; i < N; ++i) {
        double re, im;
        cin >> re >> im;
        A[i] = cd(re, im);
    }
    vector<cd> X = transform(A, -1);
    cout << fixed << setprecision(6);
    for (int j = 0; j < N; ++j)
        cout << X[j].real() << ' ' << X[j].imag() << '\n';
    return 0;
}
```
