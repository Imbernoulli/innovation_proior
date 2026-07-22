// TIER: greedy
// The obvious "try harder" recipe: standard weighted greedy SET COVER.
// Candidate probes are generated the natural way -- for each fault, a probe
// that fixes exactly that fault's required positions and 0-fills everything
// else.  Coverage of each candidate against every fault is computed by REAL
// simulation (so it correctly finds fault-dominance: a candidate for a
// "superset" cube automatically also satisfies any "subset" cube).  Then it
// repeatedly picks the not-yet-used candidate covering the most still-
// uncovered faults, until everything is covered.
//
// What it does NOT do: deliberately search for a NON-zero-filled probe that
// satisfies several UNRELATED faults on disjoint positions at once.  Because
// symmetric "orbit" faults require bit=1 on positions the 0-filled candidates
// leave at 0, one candidate essentially never covers another orbit member by
// accident -- this recipe needs about one probe per orbit member, far more
// than the single probe an orbit actually needs.
#include <bits/stdc++.h>
using namespace std;
typedef unsigned long long u64;

static int K, N, W, NW;
static vector<vector<u64>> one_, zero_;

static inline void setbit(vector<u64> &v, int p) { v[p >> 6] |= (1ULL << (p & 63)); }
static inline bool satisfies(const vector<u64> &probe, const vector<u64> &oneReq, const vector<u64> &zeroReq) {
    for (int w = 0; w < W; w++) {
        if ((probe[w] & oneReq[w]) != oneReq[w]) return false;
        if ((probe[w] & zeroReq[w]) != 0ULL) return false;
    }
    return true;
}

int main() {
    scanf("%d %d", &K, &N);
    W = (K + 63) / 64;
    NW = (N + 63) / 64;
    one_.assign(N, vector<u64>(W, 0));
    zero_.assign(N, vector<u64>(W, 0));
    for (int i = 0; i < N; i++) {
        int c; scanf("%d", &c);
        for (int k = 0; k < c; k++) {
            int p, v; scanf("%d %d", &p, &v);
            if (v == 1) setbit(one_[i], p); else setbit(zero_[i], p);
        }
    }

    // covers[i] = bitset over faults: which faults candidate i (=one_[i], 0-fill) satisfies
    vector<vector<u64>> covers(N, vector<u64>(NW, 0));
    for (int i = 0; i < N; i++) {
        // candidate i's probe bits ARE exactly one_[i] (rest 0)
        for (int j = 0; j < N; j++) {
            if (satisfies(one_[i], one_[j], zero_[j])) covers[i][j >> 6] |= (1ULL << (j & 63));
        }
    }

    vector<u64> uncovered(NW, 0);
    for (int j = 0; j < N; j++) uncovered[j >> 6] |= (1ULL << (j & 63));
    vector<char> selected(N, 0);
    vector<int> chosen;
    int remaining = N;

    while (remaining > 0) {
        int best = -1, bestGain = -1;
        for (int i = 0; i < N; i++) {
            if (selected[i]) continue;
            int gain = 0;
            for (int w = 0; w < NW; w++) gain += __builtin_popcountll(covers[i][w] & uncovered[w]);
            if (gain > bestGain) { bestGain = gain; best = i; }
        }
        selected[best] = 1;
        chosen.push_back(best);
        for (int w = 0; w < NW; w++) uncovered[w] &= ~covers[best][w];
        remaining = 0;
        for (int w = 0; w < NW; w++) remaining += __builtin_popcountll(uncovered[w]);
    }

    printf("%d\n", (int)chosen.size());
    string probe(K, '0');
    for (int idx : chosen) {
        fill(probe.begin(), probe.end(), '0');
        for (int w = 0; w < W; w++) {
            u64 bits = one_[idx][w];
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
