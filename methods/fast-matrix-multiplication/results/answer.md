# Strassen's algorithm and the tensor-rank viewpoint on matrix multiplication

## Problem

Multiply two n×n matrices over a field. The schoolbook algorithm evaluates c_ij = Σ_k a_ik b_kj
in Θ(n³) arithmetic operations. The question is whether o(n³) is possible — i.e. whether the
exponent ω = inf{β : n×n matrices can be multiplied in O(n^β) operations} is strictly below 3.

## Key idea

Recursively splitting each matrix into four (n/2)×(n/2) blocks reduces an n×n product to **block
products**, and the running time is T(n) = (#block products)·T(n/2) + O(n²), which solves to
Θ(n^{log₂(#block products)}). The naive eight block products give log₂ 8 = 3 (no gain). A single
algebraic identity that multiplies two 2×2 matrices with **seven** products instead of eight —
and, crucially, never multiplies two A-entries or two B-entries together, so it remains valid when
the entries are non-commuting matrix blocks — recurses to

    T(n) = 7·T(n/2) + O(n²)  ⟹  T(n) = Θ(n^{log₂ 7}),   log₂ 7 ≈ 2.807.

The deeper reframing: over a fixed field, a bilinear algorithm with r multiplications is exactly
a decomposition of the matrix-multiplication tensor t into r rank-one tensors u⊗v⊗w. The minimum
such r is the **rank** R(t). Matrix tensors multiply,
⟨k,m,n⟩ ⊗ ⟨k',m',n'⟩ = ⟨kk',mm',nn'⟩, and rank is submultiplicative, so R(⟨2,2,2⟩) ≤ 7 yields
R(⟨2^L,2^L,2^L⟩) ≤ 7^L and ω ≤ log₂ 7. The recursion is a tensor power; the base rank bound
becomes the exponent bound.

## The 2×2 identity (seven multiplications)

For A = [[A11,A12],[A21,A22]], B = [[B11,B12],[B21,B22]] (entries may be matrix blocks):

    P1 = (A11 + A22)(B11 + B22)
    P2 = (A21 + A22) B11
    P3 = A11 (B12 − B22)
    P4 = A22 (B21 − B11)
    P5 = (A11 + A12) B22
    P6 = (A21 − A11)(B11 + B12)
    P7 = (A12 − A22)(B21 + B22)

    C11 = P1 + P4 − P5 + P7
    C12 = P3 + P5
    C21 = P2 + P4
    C22 = P1 − P2 + P3 + P6

Seven multiplications, eighteen additions/subtractions. Every P_k is (an A-side combination) ×
(a B-side combination), so no factor is ever reordered and the identity holds for non-commuting
entries — which is what licenses the recursion.

Expanding the recombinations gives the four block products exactly:

    P1 + P4 − P5 + P7
      = (A11B11 + A11B22 + A22B11 + A22B22)
        + (A22B21 − A22B11) − (A11B22 + A12B22)
        + (A12B21 + A12B22 − A22B21 − A22B22)
      = A11B11 + A12B21 = C11

    P3 + P5
      = (A11B12 − A11B22) + (A11B22 + A12B22)
      = A11B12 + A12B22 = C12

    P2 + P4
      = (A21B11 + A22B11) + (A22B21 − A22B11)
      = A21B11 + A22B21 = C21

    P1 − P2 + P3 + P6
      = (A11B11 + A11B22 + A22B11 + A22B22)
        − (A21B11 + A22B11) + (A11B12 − A11B22)
        + (A21B11 + A21B12 − A11B11 − A11B12)
      = A21B12 + A22B22 = C22

## Algorithm

Pad to even size whenever a recursive level needs it; recurse via the seven products down to
a crossover leaf size (below which the eighteen additions plus recursion overhead lose to a tuned
cubic kernel); call the cubic kernel at the leaf; unpad. Asymptotic cost Θ(n^{log₂ 7}). Numerical
stability is worse than the classical algorithm in floating point because the extra subtractions
introduce more cancellation, so practical use depends on the field, precision, and crossover.

## Code

Single-file C++17. Reads `n` and then two n×n integer matrices A and B from stdin (row-major,
whitespace-separated) and prints C = A·B to stdout. Recurses via the seven block products down to
a cubic leaf kernel, pads odd sizes up to the next even size, and accumulates in `long long`.

```cpp
// Reads n, then two n x n integer matrices A and B from stdin (row-major,
// whitespace-separated); prints C = A*B (n rows, space-separated) to stdout.
// Recursive Strassen (7 block products) with a cubic leaf kernel; long long
// accumulation to avoid overflow.
#include <bits/stdc++.h>
using namespace std;

using ll = long long;
using Mat = vector<vector<ll>>;

static const int LEAF = 64;

// Cubic leaf kernel / ground truth: C = A*B for two s x s matrices.
Mat matmul_naive(const Mat& A, const Mat& B) {
    int s = (int)A.size();
    Mat C(s, vector<ll>(s, 0));
    for (int i = 0; i < s; i++)
        for (int k = 0; k < s; k++) {
            ll a = A[i][k];
            if (a == 0) continue;
            for (int j = 0; j < s; j++)
                C[i][j] += a * B[k][j];
        }
    return C;
}

Mat add(const Mat& A, const Mat& B) {
    int s = (int)A.size();
    Mat C(s, vector<ll>(s));
    for (int i = 0; i < s; i++)
        for (int j = 0; j < s; j++) C[i][j] = A[i][j] + B[i][j];
    return C;
}

Mat sub(const Mat& A, const Mat& B) {
    int s = (int)A.size();
    Mat C(s, vector<ll>(s));
    for (int i = 0; i < s; i++)
        for (int j = 0; j < s; j++) C[i][j] = A[i][j] - B[i][j];
    return C;
}

// Recursive Strassen for an s x s matrix where s is even (or <= LEAF).
Mat strassen(const Mat& X, const Mat& Y) {
    int n = (int)X.size();
    if (n <= LEAF || (n & 1)) return matmul_naive(X, Y);

    int m = n / 2;
    Mat a(m, vector<ll>(m)), b(m, vector<ll>(m)), c(m, vector<ll>(m)), d(m, vector<ll>(m));
    Mat e(m, vector<ll>(m)), f(m, vector<ll>(m)), g(m, vector<ll>(m)), h(m, vector<ll>(m));
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++) {
            a[i][j] = X[i][j];     b[i][j] = X[i][j + m];
            c[i][j] = X[i + m][j]; d[i][j] = X[i + m][j + m];
            e[i][j] = Y[i][j];     f[i][j] = Y[i][j + m];
            g[i][j] = Y[i + m][j]; h[i][j] = Y[i + m][j + m];
        }

    // Seven products: each is (A-side combination) x (B-side combination),
    // A always on the left, so the identity lifts to non-commuting blocks.
    Mat p1 = strassen(add(a, d), add(e, h));   // (A11+A22)(B11+B22)
    Mat p2 = strassen(add(c, d), e);           // (A21+A22) B11
    Mat p3 = strassen(a, sub(f, h));           // A11 (B12-B22)
    Mat p4 = strassen(d, sub(g, e));           // A22 (B21-B11)
    Mat p5 = strassen(add(a, b), h);           // (A11+A12) B22
    Mat p6 = strassen(sub(c, a), add(e, f));   // (A21-A11)(B11+B12)
    Mat p7 = strassen(sub(b, d), add(g, h));   // (A12-A22)(B21+B22)

    // Recombine; cross terms cancel exactly by the Strassen identity.
    Mat c11 = add(sub(add(p1, p4), p5), p7);   // C11 = p1 + p4 - p5 + p7
    Mat c12 = add(p3, p5);                      // C12 = p3 + p5
    Mat c21 = add(p2, p4);                      // C21 = p2 + p4
    Mat c22 = add(add(sub(p1, p2), p3), p6);    // C22 = p1 - p2 + p3 + p6

    Mat C(n, vector<ll>(n));
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++) {
            C[i][j] = c11[i][j];     C[i][j + m] = c12[i][j];
            C[i + m][j] = c21[i][j]; C[i + m][j + m] = c22[i][j];
        }
    return C;
}

// Pad to the next even size when needed so the halving always works.
Mat recursive_matmul(const Mat& A, const Mat& B) {
    int n = (int)A.size();
    if (n == 0) return Mat();
    if (n <= LEAF) return matmul_naive(A, B);

    int s = (n & 1) ? n + 1 : n;
    if (s == n) return strassen(A, B);

    Mat Ap(s, vector<ll>(s, 0)), Bp(s, vector<ll>(s, 0));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) { Ap[i][j] = A[i][j]; Bp[i][j] = B[i][j]; }
    Mat Cp = strassen(Ap, Bp);
    Mat C(n, vector<ll>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) C[i][j] = Cp[i][j];
    return C;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    Mat A(n, vector<ll>(n)), B(n, vector<ll>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) cin >> A[i][j];
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) cin >> B[i][j];

    Mat C = recursive_matmul(A, B);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (j) cout << ' ';
            cout << C[i][j];
        }
        cout << '\n';
    }
    return 0;
}
```
