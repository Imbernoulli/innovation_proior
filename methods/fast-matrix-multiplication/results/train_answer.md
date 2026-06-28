The problem is to multiply two n×n matrices faster than the schoolbook Θ(n³) bound. At first glance the cubic cost seems unavoidable: each output entry c_ij is the sum Σ_k a_ik b_kj, so there are n³ monomial products a_ik b_kj and one naturally believes each must be formed explicitly. Naive divide-and-conquer does not help either—splitting each matrix into four (n/2)×(n/2) blocks and recursing on the eight block products gives T(n)=8T(n/2)+O(n²), whose exponent log₂8=3 simply reproduces cubic time. The lesson from Karatsuba multiplication for polynomials and integers shows that the intuition is wrong: a bilinear form can sometimes be assembled from combined products whose unwanted cross-terms cancel, but the polynomial case does not directly transfer to matrices because the four output blocks are four separate bilinear forms and because the blocks do not commute.

The lever for beating cubic time is the number of recursive block multiplications, not the additions. For a block recursion with r multiplications the recurrence is T(n)=rT(n/2)+O(n²), so the exponent is log₂r. Eight gives 3; any r<8 breaks the cubic barrier. The question becomes whether two 2×2 block matrices can be multiplied with fewer than eight products while respecting non-commutativity—every intermediate product must be (a linear combination of A-blocks) times (a linear combination of B-blocks), never an A-block on the right of a B-block or a product of two blocks from the same matrix.

The new method is Strassen's algorithm. It multiplies two 2×2 block matrices using exactly seven bilinear products and recombines them with additions and subtractions to recover all four output blocks. The seven products are P1=(A11+A22)(B11+B22), P2=(A21+A22)B11, P3=A11(B12−B22), P4=A22(B21−B11), P5=(A11+A12)B22, P6=(A21−A11)(B11+B12), and P7=(A12−A22)(B21+B22). Each Pk has an A-side combination on the left and a B-side combination on the right, so the identity remains valid when the blocks themselves are matrices. The four outputs are then C11=P1+P4−P5+P7, C12=P3+P5, C21=P2+P4, and C22=P1−P2+P3+P6. Expanding these combinations formally cancels every unwanted monomial, leaving exactly the eight required block products. Because the seven products can be substituted for the eight naive block products at every level of recursion, the running time becomes T(n)=7T(n/2)+O(n²), giving Θ(n^{log₂7}) with log₂7≈2.807.

Strassen's algorithm is therefore not just a clever base case; it is a tensor-rank witness. A bilinear multiplication algorithm with r products is a decomposition of the matrix-multiplication tensor into r rank-one tensors u⊗v⊗w, and the minimum such r is the tensor rank. Matrix tensors multiply by taking tensor powers, and rank is submultiplicative, so a rank-7 decomposition of the 2×2×2 base tensor yields rank O(n^{log₂7}) for n×n matrices. This is why saving one multiplication at the base level compounds across all recursive levels and lowers the exponent. In practice one switches back to a tuned cubic kernel below a crossover size, because the extra additions and recursion overhead make Strassen's method advantageous only for sufficiently large matrices. Floating-point implementations also incur somewhat larger backward error than the classical algorithm because the additional subtractions create more cancellation, but over exact fields the algorithm is algebraically exact.

The program below reads `n` and then two n×n integer matrices A and B from stdin (row-major, whitespace-separated) and prints C = A·B to stdout. It recurses via the seven block products down to a cubic leaf kernel, pads odd sizes up to the next even size, and accumulates in `long long` to avoid overflow.

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
