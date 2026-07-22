// TIER: strong
// The insight: total reconfiguration cost is dominated by HOW OFTEN the expensive
// lines flip, not by the raw flip count. So nest the cheap lines inside and toggle
// the costly ones nearly monotonically:
//   1. Sort lines by cost (most expensive first) and recursively bisect the
//      configuration set by the most expensive line that actually splits it. The
//      costliest line flips at most once at the top level, the next at most twice,
//      etc. -- a weighted Gray nesting. Path cost is reversal-invariant, so at
//      every join we freely pick the orientation (8 options) minimizing the join.
//   2. Deterministic exact-delta refinement: 2-opt reversals (only the two cut
//      edges change cost; a virtual all-off start node lets prefix reversals fix
//      the setup edge) interleaved with adjacent-swap sweeps until convergence.
#include <bits/stdc++.h>
using namespace std;

int N, M;
vector<long long> C;
vector<unsigned long long> cfg;
int nb;                       // number of 8-bit tables needed
long long tab[8][256];        // per-byte-position weighted popcount table

static inline long long wdiff(unsigned long long x, unsigned long long y) {
    unsigned long long d = x ^ y;
    long long s = 0;
    for (int b = 0; b < nb; b++) s += tab[b][(d >> (8 * b)) & 255ULL];
    return s;
}
static inline long long wsetup(unsigned long long x) {
    long long s = 0;
    for (int b = 0; b < nb; b++) s += tab[b][(x >> (8 * b)) & 255ULL];
    return s;
}

vector<int> ord_bits;         // line indices sorted by cost (desc), ties by index

// Recursive weighted-Gray nesting. Returns an ordering of idxs.
vector<int> nest(const vector<int>& idxs, int p) {
    if (idxs.size() <= 1) return idxs;
    int split = -1;
    for (int q = p; q < M; q++) {
        int b = ord_bits[q];
        bool z = false, o = false;
        for (int i : idxs) {
            if ((cfg[i] >> b) & 1ULL) o = true; else z = true;
            if (z && o) { split = q; break; }
        }
        if (split >= 0) break;
    }
    if (split < 0) return idxs;   // configs identical on all remaining lines
    int b = ord_bits[split];
    vector<int> A, B;
    A.reserve(idxs.size());
    B.reserve(idxs.size());
    for (int i : idxs) ((cfg[i] >> b) & 1ULL ? B : A).push_back(i);
    vector<int> sa = nest(A, split + 1);
    vector<int> sb = nest(B, split + 1);
    // 8 orientation options (path reversal is cost-invariant internally);
    // pick the one with the cheapest join edge; ties -> lowest option index.
    long long best = -1;
    int bestk = 0;
    for (int k = 0; k < 8; k++) {
        vector<int>& L = (k & 4) ? sb : sa;
        vector<int>& R = (k & 4) ? sa : sb;
        int lt = (k & 1) ? L.front() : L.back();
        int rh = (k & 2) ? R.back() : R.front();
        long long c = wdiff(cfg[lt], cfg[rh]);
        if (best < 0 || c < best) { best = c; bestk = k; }
    }
    vector<int> L = (bestk & 4) ? sb : sa;
    vector<int> R = (bestk & 4) ? sa : sb;
    if (bestk & 1) reverse(L.begin(), L.end());
    if (bestk & 2) reverse(R.begin(), R.end());
    L.insert(L.end(), R.begin(), R.end());
    return L;
}

vector<int> perm;

// edge cost; a == -1 is the virtual all-off start, b == -1 is the free end
static inline long long ec(int a, int b) {
    if (a < 0 && b < 0) return 0;
    if (a < 0) return wsetup(cfg[b]);
    if (b < 0) return 0;
    return wdiff(cfg[a], cfg[b]);
}

bool twoOptPass() {
    bool any = false;
    for (int i = -1; i < N - 1; i++) {
        int a = (i < 0) ? -1 : perm[i];
        int b = perm[i + 1];
        long long e1 = ec(a, b);
        for (int j = i + 2; j < N; j++) {
            int c = perm[j];
            int d = (j + 1 < N) ? perm[j + 1] : -1;
            long long delta = ec(a, c) + ec(b, d) - e1 - ec(c, d);
            if (delta < 0) {
                reverse(perm.begin() + i + 1, perm.begin() + j + 1);
                any = true;
                b = perm[i + 1];
                e1 = ec(a, b);
            }
        }
    }
    return any;
}

bool adjPass() {
    bool any = false;
    for (int i = 0; i < N - 1; i++) {
        int a = (i > 0) ? perm[i - 1] : -1;
        int x = perm[i], y = perm[i + 1];
        int d = (i + 2 < N) ? perm[i + 2] : -1;
        long long before = ec(a, x) + ec(x, y) + ec(y, d);
        long long after = ec(a, y) + ec(y, x) + ec(x, d);
        if (after < before) {
            swap(perm[i], perm[i + 1]);
            any = true;
        }
    }
    return any;
}

int main() {
    if (scanf("%d %d", &N, &M) != 2) return 1;
    C.assign(M, 0);
    for (int j = 0; j < M; j++) scanf("%lld", &C[j]);
    cfg.assign(N, 0);
    char buf[128];
    for (int i = 0; i < N; i++) {
        scanf("%127s", buf);
        for (int j = 0; j < M; j++)
            if (buf[j] == '1') cfg[i] |= (1ULL << j);
    }
    nb = (M + 7) / 8;
    memset(tab, 0, sizeof(tab));
    for (int b = 0; b < nb; b++)
        for (int v = 0; v < 256; v++)
            for (int t = 0; t < 8; t++)
                if ((v >> t) & 1) {
                    int bit = 8 * b + t;
                    if (bit < M) tab[b][v] += C[bit];
                }

    ord_bits.resize(M);
    iota(ord_bits.begin(), ord_bits.end(), 0);
    stable_sort(ord_bits.begin(), ord_bits.end(),
                [](int a, int b) { return C[a] > C[b]; });

    vector<int> all(N);
    iota(all.begin(), all.end(), 0);
    perm = nest(all, 0);

    for (int it = 0; it < 12; it++) {
        bool c1 = twoOptPass();
        bool c2 = adjPass();
        if (!c1 && !c2) break;
    }

    for (int i = 0; i < N; i++)
        printf("%d%c", perm[i], i + 1 == N ? '\n' : ' ');
    return 0;
}
