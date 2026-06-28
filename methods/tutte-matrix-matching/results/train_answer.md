A perfect matching asks for a global pairing of all of a graph's vertices into disjoint edges, and the trouble is that local compatibility does not compose. A graph can carry many plausible partial pairings and still strand its last few vertices, an odd number of vertices kills the question outright, and the genuine obstruction in the general case is parity after deletion: removing a set of vertices can leave too many odd components, each of which must export an edge to be covered. Petersen's regular-graph factorizations, with their alternating color changes along closed polygons, give structure but only for special degree patterns and only as operations on already-decomposed graphs. What I want instead is a single criterion that keeps the whole pairing problem in view at once, without ever committing to a particular partial matching, without a case analysis over the ways a greedy search can get stuck, that nonetheless separates genuinely different pairings and respects the parity obstruction. The natural place to look is the classical algebra of skew-symmetric arrays, whose facts — odd-order skew determinants vanish, even-order ones are perfect squares, and the square root expands over index pairings — are true before any graph is inserted and seem to be describing matchings already.

I propose encoding the graph in what I call the Tutte matrix. Number the vertices $V = \{1,\dots,n\}$. If $n$ is odd, no pairing can cover every vertex and we stop. For even $n$, assign an independent indeterminate $x_{ij}$ to each edge $\{i,j\}$ with $i<j$, and build the skew-symmetric matrix $T(G)$ by
$$T_{ij} = \begin{cases} x_{ij} & i<j,\ \{i,j\}\in E \\ -x_{ji} & i>j,\ \{i,j\}\in E \\ 0 & \text{otherwise.}\end{cases}$$
The diagonal and every non-edge are zero, and the lower triangle is the negated transpose of the upper, so $T(G)$ is skew-symmetric by construction. The choice of which orientation gets the positive sign is free: flipping it merely negates some variables, which cannot change whether the resulting determinant is identically zero, so nothing of substance rides on the convention. What matters is the zeros — they are the filter that makes illegal pairs vanish.

The reason this works is the Pfaffian. For an even skew-symmetric matrix the determinant is the square of the Pfaffian, and the Pfaffian is precisely the signed sum over all ways of partitioning the indices into unordered pairs,
$$\mathrm{Pf}(T) = \sum_{\text{pairings}} \varepsilon \prod (\text{paired entries}), \qquad \varepsilon\in\{+1,-1\},$$
with no extra numeric factor out front — each pairing contributes exactly one product, signed by the parity of its listed sequence. This is exactly the combinatorial shape of a perfect matching before the graph has decided which pairs are legal, which is why I do not first ask which matching to choose; I let the Pfaffian try every pairing and let the matrix decide. A raw determinant would not do, because its permutation expansion carries cycles and directions whose signs belong to directed arrangements rather than to unordered pairs; the Pfaffian is the object whose native terms are already pairings. Now inspect a surviving term. It selects pairs covering all $n$ vertices; if any selected pair is a non-edge, that entry is $0$ and the whole product dies, and if every selected pair is an edge then those pairs are pairwise disjoint and cover everything — that is a perfect matching. So the surviving monomials are not merely related to perfect matchings, they *are* the perfect matchings, each written as the product of its edge variables.

The one real worry is cancellation: if two matchings appeared with opposite signs, a numeric sum could erase them. This is exactly why the variables must be independent. Two distinct perfect matchings differ in at least one edge, so their products are distinct monomials in the polynomial ring, and distinct monomials over the integers cannot cancel regardless of sign. Hence $\mathrm{Pf}(T(G))$ is the zero polynomial precisely when $G$ has no perfect matching. Passing to the determinant adds no new combinatorics: $\det(T(G)) = \mathrm{Pf}(T(G))^2$, and squaring may erase signs but over an integral polynomial ring it preserves the boundary between zero and nonzero. Folding in the odd-order case, the same symbolic determinant decides every finite simple graph:
$$G \text{ has a perfect matching} \iff \mathrm{Pf}(T(G))\not\equiv 0 \iff \det(T(G))\not\equiv 0 \iff T(G) \text{ has full generic rank.}$$

What makes this algorithmic, without ever expanding the determinant symbolically, is that rank over a field is computable by elimination once the variables hold concrete values. Substitute each $x_{ij}$ independently from a large set $S$ in a field. If there is no perfect matching the determinant polynomial is identically zero, so every substitution yields a singular matrix. If there is one, the determinant is a nonzero polynomial of degree $n$, and by polynomial identity testing a random substitution fails to give full rank with probability at most $n/|S|$; independent repetition multiplies that false-negative risk down. The test is therefore one-sided and asymmetric. Full rank at a single random point is decisive, because a nonzero evaluation proves the polynomial is not identically zero and a matching must exist. Singularity after one trial is not decisive, since a nonzero polynomial can still vanish at a particular point, especially over a small field — so repeated singular trials are reported as nonexistence only with one-sided error. The implementation below mirrors exactly this decision consequence and nothing more: it returns False immediately for odd $n$, instantiates the signed skew matrix at random values over a prime field, computes the modular rank by Gaussian elimination, returns True the moment a trial is full rank, and returns False if every trial is singular. It deliberately does not construct a witness matching or implement the later Rabin–Vazirani maximum-matching search; the conceptual core is the whole point — encode all vertex pairings in one Pfaffian, use the graph's zeros to kill illegal pairings, and use independent edge variables to keep legal pairings from cancelling.

The program below reads a graph from stdin — first line `n m` (n vertices numbered `0..n-1`, m edges), then m lines `u v` — and prints `YES` if a perfect matching exists, otherwise `NO`.

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
