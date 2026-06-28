The discrete Fourier transform is the workhorse for extracting frequency content from sampled signals, but its direct definition as a matrix-vector product costs N² complex multiplications. For the million-point records that arise in seismology, spectroscopy, and three-dimensional solid-state calculations, that quadratic cost is the difference between a feasible computation and one that would consume weeks of machine time. The matrix entries are not arbitrary numbers: every one of them is a power of the single primitive root of unity W = e^{-2πi/N}. That redundancy means most of the arithmetic in the direct method is recomputation, and the right way to exploit it is to express a long transform in terms of genuinely shorter transforms rather than merely grouping equal trigonometric values.

Earlier speedups did not change the order of growth. Term-grouping schemes collect coefficients that multiply the same sine or cosine and add them before multiplying, which saves a constant factor but still leaves the count at Θ(N²). Runge's doubling rule and the Danielson–Lanczos identity did split a length-N transform into two length-N/2 transforms, but they were presented as special halving or doubling devices rather than a uniform machine procedure for general composite lengths. What was missing was a single recursive factorization, with explicit twiddle factors and index bookkeeping, that a computer could run for any highly composite N.

The method is the Cooley–Tukey Fast Fourier Transform. It takes the root-of-unity structure seriously and factors the transform size. For the radix-2 case that covers power-of-two lengths, split the input by parity into even-indexed and odd-indexed subsequences. Because W² is the primitive root of unity for length N/2, the even half becomes a length-N/2 DFT E and the odd half becomes another length-N/2 DFT O, with a single cross factor W^j that splices them together. The half-transforms are periodic with period N/2, and because W^{N/2} equals -1, one twiddle multiply W^j·O(j) yields two outputs at once: X(j) = E(j) + W^j·O(j) and X(j+N/2) = E(j) - W^j·O(j). This add-and-subtract pair is the butterfly. Recursing on E and O gives the cost recurrence T(N) = 2T(N/2) + O(N), which solves to T(N) = O(N log N). At a million points the reduction is by a factor of roughly fifty thousand.

The recursion has a clean iterative form. Repeatedly splitting by parity routes each input sample according to its bits read least-significant-first, so the sample that ends up at leaf position p is the input whose index is the bit-reversal of p. If the working array is first permuted into bit-reversed order, the leaves are adjacent one-point transforms and the result can be built bottom-up with log₂ N passes of in-place butterflies. Each pass doubles the transform length, computes the root of unity for that block size once, and propagates a running twiddle through the butterflies by multiplication. This is the standard in-place radix-2 FFT.

The same factorization generalizes beyond radix 2. For N = r₁r₂, group inputs by residue modulo r₁ and split outputs compatibly; the inner stage computes r₁ DFTs of length r₂, the outer stage computes r₂ DFTs of length r₁, and the cross term W^{j₀k₀} is the twiddle that couples them. Recursing to the prime factorization N = p₁p₂⋯pₗ gives cost N(p₁ + p₂ + ⋯ + pₗ). When every prime is 2 this is 2N log₂ N; when N is prime the factorization has no smaller pieces and the method falls back to direct Θ(N²) evaluation. For scientific applications that can tolerate a longer transform, zero-padding to a composite length is the usual escape.

The program reads `N` (a power of two) and `N` complex samples as `re im` pairs from stdin, and writes the `N` coefficients `X(j)`, one `re im` pair per line. The transform is in-place radix-2 on a bit-reversed working copy; `dft_direct` and `inverse_transform` are carried as the small-N oracle and the conjugate-root wrapper.

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
