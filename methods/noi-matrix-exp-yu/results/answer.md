# Matrix power for walk counting and linear recurrences

## Problem

Count the number of walks of length exactly $N$ from a source $s$ to a target $t$ in a
directed graph on $d$ vertices, or evaluate a $d$-term linear recurrence
$a_n = c_1 a_{n-1} + \dots + c_d a_{n-d}$ at index $N$, with $N$ up to $10^{18}$ and
answers taken modulo $m$.

## Key idea

One step of the process is multiplication by a **fixed $d \times d$ transition matrix
$T$**, identical at every index. Stacking $d$ consecutive states into a column vector
$v_n = (a_n, a_{n-1}, \dots, a_{n-d+1})^\top$, the recurrence becomes $v_n = T\,v_{n-1}$
with

$$T = \begin{pmatrix} c_1 & c_2 & \cdots & c_d \\ 1 & 0 & \cdots & 0 \\ 0 & 1 & \cdots & 0 \\ \vdots & & \ddots & \vdots \\ 0 & \cdots & 1 & 0 \end{pmatrix}
\quad(\text{companion matrix}),$$

so $v_N = T^{\,N-(d-1)} v_{d-1}$ and $a_N$ is its top entry. For walk counting $T$ is just
the edge-count (adjacency) matrix $G$ with $G[i][j] = $ number of edges $i \to j$; by the
midpoint identity $G^k[i][j] = \sum_l G^{k-1}[i][l]\,G[l][j]$ counts length-$k$ walks, so the
answer is the single entry $(G^N)[s][t]$.

Matrix multiplication $C[i][j] = \sum_k A[i][k]\,B[k][j]$ is **associative**
($(AB)C = A(BC)$, by reordering the finite double sum $\sum_l\sum_k A[i][k]B[k][l]C[l][j]
= \sum_k\sum_l$), so the $N$-fold product $T^N$ is well defined and may be regrouped. A
forward scan computing it as $T \cdot T \cdots T$ costs $N$ products and is hopeless at
$N = 10^{18}$.

**Binary (fast) exponentiation.** Using $T^N = (T^{\lfloor N/2\rfloor})^2$ (times $T$ once
more if $N$ is odd), walk the bits of $N$: keep an accumulator starting at the identity
matrix $I$, square a running power $T, T^2, T^4, \dots$ once per bit, and fold it into the
accumulator on each set bit. This reaches exponent $N$ in $O(\log N)$ matrix products.

**Modulus.** Reduction mod $m$ commutes with $+$ and $\times$, so every matrix entry is kept
in $[0, m)$ throughout; the exact (exponentially large) counts are never formed.

## Complexity

One $d \times d$ product is $O(d^3)$; binary exponentiation does $O(\log N)$ of them, for
$O(d^3 \log N)$ time and $O(d^2)$ space. At $N = 10^{18}$ that is about sixty bit rounds,
with at most one extra fold per round, instead of $10^{18}$ scan steps.

## Code

Single-file C++17 reading from stdin: the first token selects the form — `W` then `d s t N m` and the $d\times d$ edge-count table prints the length-$N$ walk count $s\to t$; `L` then `d N m`, the $d$ coefficients and $d$ initial terms prints $a_N$ — all reduced mod $m$, with one product widened to `__int128` so it cannot overflow before reduction.

```cpp
// Reads from stdin either a walk-counting query or a linear-recurrence query and
// prints the single integer answer mod m to stdout.
//   First token = mode: "W" (walks) or "L" (linrec).
//   W:  d  s  t  N  m,  then a d x d edge-count table adj[i][j] (# edges i->j);
//       prints the number of length-N walks s -> t, mod m.
//   L:  d  N  m,  then d coefficients c_1..c_d, then d initial terms a_0..a_{d-1};
//       prints a_N for  a_n = c_1 a_{n-1} + ... + c_d a_{n-d},  mod m.
// N may be as large as 1e18; everything is kept reduced mod m via O(d^3 log N)
// fast matrix exponentiation (entries stay in [0,m), products fit in __int128).
#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
typedef vector<vector<ll>> Mat;

static ll MOD;

// (p x q) by (q x r) product, every entry reduced mod MOD.
Mat mul(const Mat &A, const Mat &B) {
    int p = A.size(), q = B.size(), r = B[0].size();
    Mat C(p, vector<ll>(r, 0));
    for (int i = 0; i < p; i++) {
        for (int k = 0; k < q; k++) {
            ll a = A[i][k];
            if (a == 0) continue;
            const vector<ll> &Bk = B[k];
            vector<ll> &Ci = C[i];
            for (int j = 0; j < r; j++) {
                Ci[j] = (Ci[j] + (__int128)a * Bk[j]) % MOD;
            }
        }
    }
    return C;
}

Mat identity(int d) {
    Mat I(d, vector<ll>(d, 0));
    for (int i = 0; i < d; i++) I[i][i] = 1 % MOD;
    return I;
}

// M^N mod MOD by binary exponentiation; O(d^3 log N) ring multiplications.
Mat matpow(Mat M, ll N) {
    int d = M.size();
    Mat R = identity(d);
    for (auto &row : M) for (auto &x : row) x %= MOD;
    while (N > 0) {
        if (N & 1) R = mul(R, M);
        M = mul(M, M);
        N >>= 1;
    }
    return R;
}

// Number of length-N walks s -> t in a directed graph, mod MOD.
ll walks(const Mat &adj, int s, int t, ll N) {
    return matpow(adj, N)[s][t];
}

// Evaluate a_n = sum_j coeffs[j]*a_{n-1-j} at index N, given init = [a_0..a_{d-1}].
ll linrec(const vector<ll> &coeffs, const vector<ll> &init, ll N) {
    int d = coeffs.size();
    if (N < d) return ((init[N] % MOD) + MOD) % MOD;
    Mat T(d, vector<ll>(d, 0));
    for (int j = 0; j < d; j++) T[0][j] = ((coeffs[j] % MOD) + MOD) % MOD; // top row = coefficients
    for (int i = 1; i < d; i++) T[i][i - 1] = 1 % MOD;                     // sub-diagonal shift
    Mat v0(d, vector<ll>(1, 0));
    for (int i = 0; i < d; i++) v0[i][0] = ((init[d - 1 - i] % MOD) + MOD) % MOD; // (a_{d-1},...,a_0)^T
    Mat vN = mul(matpow(T, N - (d - 1)), v0);
    return vN[0][0];
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    string mode;
    if (!(cin >> mode)) return 0;
    if (mode == "W" || mode == "w") {
        int d, s, t; ll N;
        cin >> d >> s >> t >> N >> MOD;
        Mat adj(d, vector<ll>(d, 0));
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++) cin >> adj[i][j];
        cout << walks(adj, s, t, N) << "\n";
    } else { // "L" linear recurrence
        int d; ll N;
        cin >> d >> N >> MOD;
        vector<ll> coeffs(d), init(d);
        for (int i = 0; i < d; i++) cin >> coeffs[i];
        for (int i = 0; i < d; i++) cin >> init[i];
        cout << linrec(coeffs, init, N) << "\n";
    }
    return 0;
}
```
