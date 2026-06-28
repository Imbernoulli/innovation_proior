# Tutte Matrix Matching

For a finite simple graph `G = (V, E)` with `V = {1, ..., n}`, odd `n` immediately means no perfect matching, and the skew-symmetric determinant is identically zero. For even `n`, assign an independent variable `x_ij` to each edge `{i, j}` with `i < j`. Define the skew-symmetric matrix `T(G)` by

```text
T_ij =  x_ij   if i < j and {i, j} in E
T_ij = -x_ji   if i > j and {i, j} in E
T_ij =  0      otherwise.
```

Changing the orientation convention only changes signs of variables, not whether the determinant is the zero polynomial.

For even `n`, the Pfaffian of `T(G)` expands over all pairings of the vertices:

```text
Pf(T) = sum epsilon * product of the paired entries,
epsilon in {+1, -1}.
```

There is no extra numeric factor. A term survives exactly when every chosen pair is an edge, so surviving terms are exactly perfect matchings. Distinct perfect matchings produce distinct monomials in independent variables, so they cannot cancel.

Thus

```text
G has a perfect matching
<=> Pf(T(G)) is not the zero polynomial
<=> det(T(G)) is not the zero polynomial
<=> T(G) has full generic rank.
```

A randomized decision test instantiates each `x_ij` independently from a large finite field and computes rank or determinant. Full rank proves a perfect matching exists. If all trials are singular, the correct report is one-sided-error nonexistence: for a graph with a perfect matching, one trial fails with probability at most `n / |S|` when sampling from a set `S`, and independent repetition reduces that risk.

The single-file C++ program below matches this decision test: it reads `n m` and the `m` edges from stdin, handles odd `n`, builds the signed skew matrix over a prime field, and prints `YES` on a full-rank trial, otherwise `NO`. It does not construct the matching or implement the later Rabin-Vazirani maximum-matching search.

```cpp
// Randomized Tutte-matrix perfect-matching test over a prime field.
// Reads: first line "n m" (n vertices numbered 0..n-1, m edges); then m lines
// "u v". Prints "YES" if G has a perfect matching, otherwise "NO" (one-sided
// error: a NO can be wrong only with probability driven below any threshold by
// the trial count, while a YES is always correct).

#include <bits/stdc++.h>
using namespace std;

static const long long PRIME = 2147483647LL; // 2^31 - 1, a Mersenne prime

// Modular exponentiation, used for the field inverse via Fermat's little theorem.
long long pow_mod(long long base, long long exp, long long mod) {
    long long result = 1 % mod;
    base %= mod;
    if (base < 0) base += mod;
    while (exp > 0) {
        if (exp & 1) result = (__int128)result * base % mod;
        base = (__int128)base * base % mod;
        exp >>= 1;
    }
    return result;
}

// Rank of an n x n matrix over the prime field by Gaussian elimination.
int rank_mod(vector<vector<long long>> a, long long prime) {
    int n_rows = (int)a.size();
    int n_cols = n_rows ? (int)a[0].size() : 0;
    int rank = 0;
    for (int col = 0; col < n_cols && rank < n_rows; ++col) {
        int pivot = -1;
        for (int row = rank; row < n_rows; ++row) {
            if (((a[row][col] % prime) + prime) % prime) { pivot = row; break; }
        }
        if (pivot == -1) continue;
        swap(a[rank], a[pivot]);
        long long inv = pow_mod(a[rank][col], prime - 2, prime);
        for (long long &v : a[rank]) v = (__int128)((v % prime + prime) % prime) * inv % prime;
        for (int row = 0; row < n_rows; ++row) {
            if (row == rank) continue;
            long long factor = ((a[row][col] % prime) + prime) % prime;
            if (factor) {
                for (int c = 0; c < n_cols; ++c) {
                    long long sub = (__int128)factor * a[rank][c] % prime;
                    a[row][c] = ((a[row][c] - sub) % prime + prime) % prime;
                }
            }
        }
        ++rank;
    }
    return rank;
}

// True on a full-rank random instantiation of the Tutte matrix; loops are
// ignored because they cannot take part in a perfect matching.
bool has_perfect_matching(int n, const vector<pair<int,int>> &edges,
                          int trials, long long prime, mt19937_64 &rng) {
    if (n % 2) return false;
    for (int t = 0; t < trials; ++t) {
        vector<vector<long long>> mat(n, vector<long long>(n, 0));
        for (auto [u, v] : edges) {
            if (u == v) continue;
            long long value = (long long)(rng() % (unsigned long long)(prime - 1)) + 1; // 1..prime-1
            mat[u][v] = value;
            mat[v][u] = (prime - value) % prime;
        }
        if (rank_mod(mat, prime) == n) return true;
    }
    return false;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    vector<pair<int,int>> edges(m);
    for (int i = 0; i < m; ++i) {
        int u, v;
        cin >> u >> v;
        edges[i] = {u, v};
    }

    mt19937_64 rng(0x9e3779b97f4a7c15ULL);
    bool yes = has_perfect_matching(n, edges, 8, PRIME, rng);
    cout << (yes ? "YES" : "NO") << "\n";
    return 0;
}
```
