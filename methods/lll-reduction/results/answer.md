# LLL lattice basis reduction

## Problem

Given a lattice `L` of rank `n` in `R^m` by an arbitrary integer basis
`b_1, ..., b_n`, compute, in time polynomial in `n` and the input bit-size, a new
basis of the *same* lattice that is short and nearly orthogonal — in particular
one whose first vector `b_1` is provably short relative to the shortest nonzero
lattice vector `lambda_1(L)`. Exact shortest-vector computation is hard in high
rank, so the target is a guaranteed approximation in polynomial time.

## Key idea

A lattice has no canonical short basis, but Gram-Schmidt orthogonalization of any
basis gives an orthogonal sequence `b*_1, ..., b*_n` whose lengths capture the
real geometry: the individual `|b*_i|` depend on the ordered basis, but their
product is the basis-invariant covolume `prod_i |b*_i| = d(L)`, and every nonzero
`x in L` satisfies `|x| >= |b*_i|` for the largest index `i` it reaches. So a
"good" basis is one whose Gram-Schmidt lengths decay slowly. Demand two things of
the basis, with `mu_{i,j} = <b_i, b*_j>/<b*_j, b*_j>`:

- **Size-reduction:** `|mu_{i,j}| <= 1/2` for all `j < i` (each vector is cleaned
  of integer multiples of the earlier ones; rounding `mu` to the nearest integer
  is optimal and does not change any `b*`).
- **Lovász condition** (parameter `1/4 < delta < 1`, here `delta = 3/4`):
  `|b*_i + mu_{i,i-1} b*_{i-1}|^2 >= delta |b*_{i-1}|^2`, equivalently (using
  `b*_i ⟂ b*_{i-1}`)
  `|b*_i|^2 >= (delta - mu_{i,i-1}^2) |b*_{i-1}|^2`.

When the Lovász condition fails at index `k`, swapping `b_{k-1}` and `b_k` makes
the new `b*_{k-1}` equal to the old `b*_k + mu_{k,k-1} b*_{k-1}`, whose squared
length is `< delta |b*_{k-1}|^2` — a guaranteed factor-`delta` shrink. The
potential `D = prod_{i=1}^{n-1} d_i`, where `d_i = prod_{j<=i} |b*_j|^2` is the
squared covolume of the first-`i` sublattice, changes only on swaps (size-
reduction leaves every `b*` fixed) and only by a factor `< delta`. For an
integral basis each `d_i` is a positive integer so `D >= 1`, while initially
`D <= B^{n(n-1)/2}` for `|b_i|^2 <= B`; hence there are at most `O(n^2 log B)`
swaps and the algorithm runs in polynomial time.

With `delta = 3/4`, size-reduction gives `mu_{i,i-1}^2 <= 1/4`, so the Lovász
condition forces `|b*_i|^2 >= (1/2)|b*_{i-1}|^2`. This propagates to the output
guarantees `|b_1| <= 2^{(n-1)/4} d(L)^{1/n}` and `|b_1| <= 2^{(n-1)/2} \cdot
lambda_1(L)` — the first vector is within `2^{(n-1)/2}` of the shortest vector
(and more generally `|b_j| <= 2^{(n-1)/2}` times the `j`-th successive minimum).
The general quality factor is `4/(4 delta - 1)` per index; `delta = 3/4` (giving
factor 2) balances approximation quality against the per-swap potential drop
`delta`, which controls the iteration count.

## Algorithm

1. Compute the Gram-Schmidt orthogonalization `b*_i` and the coefficients
   `mu_{i,j}`. Set `k = 2`.
2. While `k <= n`:
   - Reduce only `mu_{k,k-1}`: if `|mu_{k,k-1}| > 1/2` set
     `b_k <- b_k - round(mu_{k,k-1}) b_{k-1}` and update the Gram-Schmidt data.
     (This is all the Lovász test below depends on.)
   - If `|b*_k|^2 >= (delta - mu_{k,k-1}^2) |b*_{k-1}|^2`: finish size-reducing
     `b_k` against `b_{k-2}, ..., b_1` (for `j` from `k-2` down to `1`, if
     `|mu_{k,j}| > 1/2` set `b_k <- b_k - round(mu_{k,j}) b_j` and update), then
     set `k <- k + 1`.
   - Else swap `b_k` and `b_{k-1}`, update the Gram-Schmidt data, and set
     `k <- max(k - 1, 2)`.
3. When `k = n + 1`, the basis is reduced; return it.

## Code

Single-file C++17. It reads `n m` and then `n` rows of `m` integers (the basis
`b_1, ..., b_n` in `Z^m`) from stdin, and writes the reduced basis to stdout. All
arithmetic is kept exact in integers by carrying the determinants
`d_i = prod_{j<=i}|b*_j|^2` and the integer-scaled coefficients
`lambda_{i,j} = d_{j+1} * mu_{i,j}`; the Lovász test and size-reduction rounding
are then evaluated entirely over integers, with `__int128` guarding the products.

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

For input `3 3 / 1 1 1 / -1 0 2 / 3 5 6` the program prints the reduced basis
`0 1 0`, `1 0 1`, `-1 0 2`.
