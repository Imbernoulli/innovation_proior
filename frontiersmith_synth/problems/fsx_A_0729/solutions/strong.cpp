// TIER: strong
// Character-sum construction + product-patching.
//
// 1. Precompute the primes q = 3 (mod 4) up to n (only these admit a Paley
//    tournament: edge a->b iff (b-a) mod q is a quadratic residue; q = 3 mod 4
//    guarantees exactly one of d, -d is a residue, so this is a valid
//    tournament, and it is exactly regular).
// 2. Find the largest m <= n that is a product (with repetition) of such
//    primes, via a reachability DP, and recover one witnessing factor list.
// 3. Build the size-m tournament by taking the LEXICOGRAPHIC (tensor) product
//    of the individual Paley tournaments across the factor list -- this is
//    again exactly regular (out-degree (m-1)/2 for every vertex), and its
//    codegree statistics inherit most of each factor's character-sum
//    cancellation.
// 4. If m < n, recursively build a tournament on the remaining n-m vertices
//    the same way, and splice it on: cross edges between the base block
//    [0,m) and the patch block [m,n) are decided by a fixed alternating-
//    parity rule, which keeps every vertex's cross contribution within one
//    of an even split regardless of which side it's on.
//
// This composes the algebraic construction beyond its native prime domain to
// cover arbitrary odd n, tracking exactly how the defect degrades as the
// patched fraction of n grows.
#include <bits/stdc++.h>
using namespace std;

static bool isPrime(int x) {
    if (x < 2) return false;
    for (int p = 2; (long long)p * p <= x; p++)
        if (x % p == 0) return false;
    return true;
}

static vector<int> goodPrimes(int maxN) {
    vector<int> r;
    for (int p = 3; p <= maxN; p++)
        if (isPrime(p) && p % 4 == 3) r.push_back(p);
    return r;
}

static vector<int> quadraticResidues(int q) {
    vector<char> is(q, 0);
    for (int x = 1; x < q; x++) is[(long long)x * x % q] = 1;
    vector<int> r;
    for (int x = 1; x < q; x++)
        if (is[x]) r.push_back(x);
    return r;
}

// M[i][j] = 1 if i beats j
static vector<vector<char>> buildPaley(int q) {
    auto qr = quadraticResidues(q);
    vector<char> isqr(q, 0);
    for (int x : qr) isqr[x] = 1;
    vector<vector<char>> M(q, vector<char>(q, 0));
    for (int i = 0; i < q; i++)
        for (int j = 0; j < q; j++)
            if (i != j) {
                int d = ((j - i) % q + q) % q;
                if (isqr[d]) M[i][j] = 1;
            }
    return M;
}

static vector<vector<char>> lexProduct(const vector<vector<char>>& M1, const vector<vector<char>>& M2) {
    int n1 = (int)M1.size(), n2 = (int)M2.size();
    int n = n1 * n2;
    vector<vector<char>> M(n, vector<char>(n, 0));
    for (int a1 = 0; a1 < n1; a1++)
        for (int b1 = 0; b1 < n2; b1++)
            for (int a2 = 0; a2 < n1; a2++)
                for (int b2 = 0; b2 < n2; b2++) {
                    if (a1 == a2 && b1 == b2) continue;
                    int u = a1 * n2 + b1, v = a2 * n2 + b2;
                    if (a1 != a2)
                        M[u][v] = M1[a1][a2];
                    else
                        M[u][v] = M2[b1][b2];
                }
    return M;
}

static vector<int> PRIMES;

static vector<vector<char>> buildStrong(int n) {
    if (n <= 1) return vector<vector<char>>(n, vector<char>(n, 0));

    // reachability DP: reach[v] = true if v is a product (w/ repetition) of
    // primes in PRIMES; parent[v] = the last prime multiplied in.
    vector<char> reach(n + 1, 0);
    vector<int> parent(n + 1, -1);
    reach[1] = 1;
    for (int v = 1; v <= n; v++) {
        if (!reach[v]) continue;
        for (int p : PRIMES) {
            if ((long long)v * p > n) break;
            if (!reach[v * p]) {
                reach[v * p] = 1;
                parent[v * p] = p;
            }
        }
    }
    int best = 1;
    for (int v = n; v >= 1; v--)
        if (reach[v]) { best = v; break; }

    vector<int> factors;
    int cur = best;
    while (cur > 1) {
        int p = parent[cur];
        factors.push_back(p);
        cur /= p;
    }

    vector<vector<char>> base;
    if (factors.empty()) {
        base = vector<vector<char>>(1, vector<char>(1, 0)); // single-vertex identity block
    } else {
        base = buildPaley(factors[0]);
        for (size_t k = 1; k < factors.size(); k++) {
            base = lexProduct(base, buildPaley(factors[k]));
        }
    }

    int m = (int)base.size();
    if (m == n) return base;

    auto patch = buildStrong(n - m);
    int pn = (int)patch.size();

    vector<vector<char>> full(n, vector<char>(n, 0));
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++) full[i][j] = base[i][j];
    for (int i = 0; i < pn; i++)
        for (int j = 0; j < pn; j++) full[m + i][m + j] = patch[i][j];
    for (int u = 0; u < m; u++)
        for (int vp = 0; vp < pn; vp++) {
            bool uBeats = ((u + vp) % 2 == 0);
            full[u][m + vp] = uBeats ? 1 : 0;
            full[m + vp][u] = uBeats ? 0 : 1;
        }
    return full;
}

int main() {
    int n;
    scanf("%d", &n);

    PRIMES = goodPrimes(n);
    auto M = buildStrong(n);

    string line;
    for (int i = 0; i < n - 1; i++) {
        int len = n - 1 - i;
        line.resize(len);
        for (int k = 0; k < len; k++) {
            int j = i + 1 + k;
            line[k] = M[i][j] ? '1' : '0';
        }
        printf("%s\n", line.c_str());
    }
    return 0;
}
