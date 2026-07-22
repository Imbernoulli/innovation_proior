// TIER: strong
// The insight: reformulate "minimum probes" as MINIMUM COMPATIBLE-CUBE
// PARTITIONING (equivalently, coloring the conflict graph where two faults
// conflict iff they disagree on some shared position).  For boolean partial
// assignments, PAIRWISE compatibility inside a group implies the whole group
// is jointly satisfiable by one probe (every position gets at most one
// value across the group).  So instead of asking "which OTHER RAW FAULTS
// does this 0-filled probe happen to also satisfy", we ask "which faults are
// STRUCTURALLY COMPATIBLE with each other" -- this is exactly reasoning
// about the hidden quotient (equivalence/orbit) structure instead of raw
// faults, and it is what collapses an entire symmetric orbit (pairwise
// compatible because their positions are disjoint) into a single probe, on
// top of the dominance collapse the greedy already finds.
//
// Heuristic: process faults most-constrained-first (largest cube first, a
// Welsh-Powell-style ordering) and FIRST-FIT merge each into the first
// currently-open probe it is compatible with (checked against the probe's
// ACCUMULATED constraints, not just the original seeding fault); otherwise
// open a new probe.
#include <bits/stdc++.h>
using namespace std;
typedef unsigned long long u64;

static int K, N, W;
static vector<vector<u64>> one_, zero_;

static inline void setbit(vector<u64> &v, int p) { v[p >> 6] |= (1ULL << (p & 63)); }
static inline int popcnt(const vector<u64> &v) {
    int s = 0; for (u64 x : v) s += __builtin_popcountll(x); return s;
}
static inline bool compatible(const vector<u64> &probeOne, const vector<u64> &probeZero,
                               const vector<u64> &fOne, const vector<u64> &fZero) {
    for (int w = 0; w < W; w++) {
        if (probeOne[w] & fZero[w]) return false;
        if (probeZero[w] & fOne[w]) return false;
    }
    return true;
}

int main() {
    scanf("%d %d", &K, &N);
    W = (K + 63) / 64;
    one_.assign(N, vector<u64>(W, 0));
    zero_.assign(N, vector<u64>(W, 0));
    for (int i = 0; i < N; i++) {
        int c; scanf("%d", &c);
        for (int k = 0; k < c; k++) {
            int p, v; scanf("%d %d", &p, &v);
            if (v == 1) setbit(one_[i], p); else setbit(zero_[i], p);
        }
    }

    vector<int> order(N);
    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [](int a, int b) {
        int ca = popcnt(one_[a]) + popcnt(zero_[a]);
        int cb = popcnt(one_[b]) + popcnt(zero_[b]);
        if (ca != cb) return ca > cb;
        return a < b;
    });

    vector<vector<u64>> probeOne, probeZero;
    for (int idx : order) {
        int found = -1;
        for (int p = 0; p < (int)probeOne.size(); p++) {
            if (compatible(probeOne[p], probeZero[p], one_[idx], zero_[idx])) { found = p; break; }
        }
        if (found < 0) {
            probeOne.push_back(one_[idx]);
            probeZero.push_back(zero_[idx]);
        } else {
            for (int w = 0; w < W; w++) {
                probeOne[found][w] |= one_[idx][w];
                probeZero[found][w] |= zero_[idx][w];
            }
        }
    }

    int P = (int)probeOne.size();
    printf("%d\n", P);
    string probe(K, '0');
    for (int p = 0; p < P; p++) {
        fill(probe.begin(), probe.end(), '0');
        for (int w = 0; w < W; w++) {
            u64 bits = probeOne[p][w];
            while (bits) {
                int b = __builtin_ctzll(bits);
                int pos = w * 64 + b;
                if (pos < K) probe[pos] = '1';
                bits &= bits - 1;
            }
        }
        probe.push_back('\n');
        fwrite(probe.data(), 1, probe.size(), stdout);
        probe.pop_back();
    }
    return 0;
}
