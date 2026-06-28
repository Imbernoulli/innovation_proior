I propose the canonical name Lenstra–Lenstra–Lovász lattice basis reduction, or simply LLL reduction, for the problem of turning an arbitrary integer basis of a lattice into a basis whose vectors are short and nearly orthogonal while keeping every step inside the same lattice. The input is a lattice L of rank n in R^m, presented as n linearly independent integer vectors b_1, ..., b_n. The legal moves are exactly the unimodular ones: add an integer multiple of one basis vector to another, swap two basis vectors, or negate a vector. Each such move is represented by an integer matrix of determinant plus or minus one, so the lattice itself never changes. The goal is to reach, in time polynomial in n and in the bit-size of the input, a basis whose first vector is provably short relative to the true shortest nonzero vector of L.

The difficulty is that there is no canonical short basis. The same lattice has infinitely many bases, related by GL_n(Z), and ordinary Euclidean lengths of the basis vectors are not invariant under those moves. What is invariant is the covolume d(L), the square root of the determinant of the Gram matrix of inner products. So any useful notion of quality must be one the unimodular group cannot inflate away. The right invariant quantities come from Gram-Schmidt orthogonalization. From the given ordered basis we build an orthogonal sequence b*_1, ..., b*_n by projecting each b_i orthogonally onto the orthogonal complement of the span of the earlier vectors. The coefficients are mu_{i,j} = <b_i, b*_j> / <b*_j, b*_j> for j < i. Because the Gram-Schmidt vectors are pairwise orthogonal, the product of their lengths equals the covolume: prod_i |b*_i| = d(L). Moreover, any nonzero lattice vector x = sum r_i b_i, with integers r_i, can be rewritten as x = sum r'_i b*_i, and the change of basis is upper triangular with ones on the diagonal. Hence for the largest index i with r_i nonzero we have r'_i = r_i, a nonzero integer, and therefore |x| >= |b*_i|. So the Gram-Schmidt lengths are the real currency of the geometry: they lower-bound the lengths of lattice vectors that reach each index, and their product is fixed by the lattice alone.

A good basis, then, is one whose Gram-Schmidt lengths decay slowly. If |b*_i| stays comparable to |b*_{i-1}| all along, then every nonzero lattice vector is at least roughly |b*_1|, and b_1, whose length starts at |b*_1|, is essentially as short as anything in L. Two conditions capture this. The first is size-reduction: for every j < i, the Gram-Schmidt coefficient mu_{i,j} should satisfy |mu_{i,j}| <= 1/2. This says that b_i has been cleaned of integer multiples of earlier basis vectors as far as possible; the best single move is b_i <- b_i - round(mu_{i,j}) b_j, which lands the coefficient in the interval [-1/2, 1/2]. Size-reduction does not change any b*_i, so it is free in the Gram-Schmidt-length currency. The second condition is the Lovasz condition with a parameter delta in (1/4, 1): |b*_i + mu_{i,i-1} b*_{i-1}|^2 >= delta |b*_{i-1}|^2. Because b*_i is orthogonal to b*_{i-1}, this is equivalent to |b*_i|^2 >= (delta - mu_{i,i-1}^2) |b*_{i-1}|^2. It says that the projected rank-two pair formed by b_{i-1} and b_i is already reduced enough in the sense of Gauss-Lagrange rank-two reduction; no adjacent swap would improve the leading Gram-Schmidt length by more than a factor of delta.

The constant delta is the crucial relaxation that makes polynomial time possible. If we required every swap to give any improvement at all, a swap could shrink the leading length by a factor arbitrarily close to one, and we would have no bound on the number of swaps. By insisting on a definite factor delta < 1, each useful swap multiplies a global potential by less than delta. Define d_i = det(<b_j, b_l>)_{1 <= j,l <= i} = prod_{j <= i} |b*_j|^2, the squared covolume of the rank-i sublattice spanned by the first i vectors, and define D = prod_{i=1}^{n-1} d_i. Size-reduction leaves every b*_i unchanged, so it leaves D unchanged. A swap of b_{k-1} and b_k only affects d_{k-1}: the pair's covolume is invariant, and all d_i with i different from k-1 are untouched. When the Lovasz condition fails, the new d_{k-1} is less than delta times the old one, so D drops by a factor below delta. For an integral basis each d_i is a positive integer, hence D >= 1, while initially |b_i|^2 <= B gives D <= B^{n(n-1)/2}. Therefore the number of swaps is at most O(n^2 log B), and the algorithm terminates in polynomial time. Over the reals the same argument works using Minkowski's convex-body bound to give a positive lower bound on D, but the integer floor is the clean polynomial guarantee.

The standard choice is delta = 3/4. Then, after size-reduction, mu_{i,i-1}^2 <= 1/4, and the Lovasz condition implies |b*_i|^2 >= (1/2) |b*_{i-1}|^2. From this slow decay one obtains the classical quality guarantees: the product of the raw basis lengths satisfies prod_i |b_i| <= 2^{n(n-1)/4} d(L), and the first vector satisfies |b_1| <= 2^{(n-1)/4} d(L)^{1/n}. More importantly, relative to the true shortest vector lambda_1(L), we get |b_1| <= 2^{(n-1)/2} lambda_1(L). The same argument bounds each b_j by 2^{(n-1)/2} times the j-th successive minimum, so the whole reduced basis is short, not merely its first vector. The general quality factor per index is 4/(4 delta - 1); delta = 3/4 gives the clean base-two bounds above. Pushing delta toward one improves the approximation factor but weakens the per-swap potential drop, so the polynomial-time bound blows up; choosing delta = 3/4 balances the two.

The algorithm is implemented as a single sweep with a pointer k, starting at k = 2 in one-based indexing (k = 1 in zero-based code). The invariant is that the prefix before k is already fully size-reduced and satisfies the Lovasz condition at every internal step. At each k, we first size-reduce b_k only against its immediate predecessor b_{k-1}; this is the only coefficient the upcoming test depends on. Then we test the Lovasz condition. If it holds, we finish size-reducing b_k against b_{k-2}, ..., b_1 and advance k by one. If it fails, we swap b_{k-1} and b_k, which shrinks the potential D by the required factor, and retreat k by one to recheck the previous pair, since the swap may have disturbed it. When k reaches n + 1 the entire basis is reduced and we return it. For exactness the whole computation is carried out in integers: instead of tracking the rational mu and |b*_i|^2 directly, carry the integer determinants d_i = prod_{j<=i}|b*_j|^2 (with d_0 = 1) and the integer-scaled coefficients lambda_{i,j} = d_{j+1} * mu_{i,j}, recovered from the Gram matrix by the integer Cholesky recurrence. The Lovasz test |b*_k|^2 >= (delta - mu_{k,k-1}^2)|b*_{k-1}|^2 with delta = 3/4 then clears denominators to the all-integer form 4 d_{k+1} d_{k-1} >= 3 d_k^2 - 4 lambda_{k,k-1}^2, and size-reduction's round(mu_{k,j}) becomes round(lambda_{k,j} / d_{j+1}). All denominators stay controlled and the algorithm remains polynomial. The self-contained single-file C++17 program below reads the basis from stdin and writes the reduced basis to stdout.

```cpp
// LLL lattice basis reduction (single-file, stdin -> stdout).
// Reads: n m, then n rows of m integers (an integer basis b_1..b_n of a lattice
// in Z^m). Writes: the LLL-reduced basis (delta = 3/4), n rows of m integers.
//
// All arithmetic is exact in integers: the algorithm carries the Gram-Schmidt
// data as the integer determinants d_i = prod_{j<=i}|b*_j|^2 together with the
// integer-scaled coefficients lambda_{i,j} = d_j * mu_{i,j}, so no floating point
// and no rationals are needed (de Weger's integer LLL). __int128 guards the
// products against overflow of the intermediate quantities.

#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
typedef __int128 lll_t;

static int n, m;
static vector<vector<ll>> B;     // basis rows, integers
static vector<vector<ll>> G;     // Gram matrix <b_i,b_j>
static vector<ll> d;             // d[0..n], d[i] = prod_{j<=i}|b*_j|^2 (d[0]=1)
static vector<vector<ll>> lam;   // lam[i][j] = d[j+1]*mu_{i,j} for j<i (integers)

static ll dot(const vector<ll>& u, const vector<ll>& v) {
    lll_t s = 0;
    for (int t = 0; t < m; t++) s += (lll_t)u[t] * v[t];
    return (ll)s; // assumed to fit in ll for the test inputs
}

// Round a rational num/den to nearest integer, ties to even (matching the
// round() used in the reasoning's size-reduction so b_i loses the integer part
// of mu and lands |mu| in [-1/2, 1/2]).
static ll nint(ll num, ll den) {
    if (den < 0) { num = -num; den = -den; }
    // floor division
    ll q = num / den, r = num - q * den;
    if (r < 0) { q -= 1; r += den; }   // now 0 <= r < den, q = floor(num/den)
    ll twice = 2 * r;
    if (twice > den) q += 1;
    else if (twice == den && (q & 1)) q += 1; // tie -> round to even
    return q;
}

// Recompute Gram matrix, d[], and lambda[] from scratch from B (exact).
static void recompute() {
    G.assign(n, vector<ll>(n, 0));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            G[i][j] = dot(B[i], B[j]);

    d.assign(n + 1, 0);
    d[0] = 1;
    lam.assign(n, vector<ll>(n, 0));
    // Cholesky-like integer recurrence (Gram-Schmidt over the Gram matrix):
    // u_{i,j} = G[i][j] - sum_{p<j} lam[j][p]*u_{i,p}/d[p]   (built iteratively)
    // with lam[i][j] = u_{i,j} (the d_{j+1}-scaled coefficient) and
    // d[i+1] = u_{i,i}.
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            lll_t u = G[i][j];
            for (int p = 0; p < j; p++) {
                // u = (d[p+1]*u - lam[i][p]*lam[j][p]) / d[p]
                u = (lll_t)d[p + 1] * u - (lll_t)lam[i][p] * lam[j][p];
                u /= d[p];
            }
            if (j < i) lam[i][j] = (ll)u;
            else d[i + 1] = (ll)u;
        }
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    if (!(cin >> n >> m)) return 0;
    B.assign(n, vector<ll>(m, 0));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            cin >> B[i][j];

    recompute();

    // delta = 3/4. The Lovasz test |b*_k|^2 >= (delta - mu_{k,k-1}^2)|b*_{k-1}|^2
    // becomes, after multiplying by d[k-1]^2 and using |b*_k|^2 = d[k+1]/d[k],
    // |b*_{k-1}|^2 = d[k]/d[k-1], mu_{k,k-1} = lam[k][k-1]/d[k]:
    //   4 * d[k+1] * d[k-1] >= (3 * d[k]^2 - 4 * lam[k][k-1]^2),
    // all in integers (here k is the 0-based pointer, predecessor k-1).
    int k = 1;
    while (k < n) {
        // size-reduce b_k against b_{k-1} only, before the test
        if (2 * llabs(lam[k][k - 1]) > d[k]) {
            ll r = nint(lam[k][k - 1], d[k]);
            for (int t = 0; t < m; t++) B[k][t] -= r * B[k - 1][t];
            recompute();
        }

        lll_t lhs = (lll_t)4 * d[k + 1] * d[k - 1];
        lll_t rhs = (lll_t)3 * d[k] * d[k] - (lll_t)4 * lam[k][k - 1] * lam[k][k - 1];
        if (lhs >= rhs) {
            // Lovasz holds: finish size-reducing b_k against b_{k-2}..b_0, advance
            for (int j = k - 2; j >= 0; j--) {
                if (2 * llabs(lam[k][j]) > d[j + 1]) {
                    ll r = nint(lam[k][j], d[j + 1]);
                    for (int t = 0; t < m; t++) B[k][t] -= r * B[j][t];
                    recompute();
                }
            }
            k += 1;
        } else {
            // swap b_{k-1}, b_k; potential drops by factor < 3/4; retreat
            swap(B[k], B[k - 1]);
            recompute();
            k = max(k - 1, 1);
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            cout << B[i][j];
            if (j + 1 < m) cout << ' ';
        }
        cout << '\n';
    }
    return 0;
}
```

On the worked example, feeding `3 3 / 1 1 1 / -1 0 2 / 3 5 6` on stdin prints the reduced basis `0 1 0`, `1 0 1`, `-1 0 2`.
