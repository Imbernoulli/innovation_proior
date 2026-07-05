// TIER: strong
// Nearest-feasible construction, then two improvements the greedy lacks:
//   (1) precedence-safe 2-opt local search to shorten the ordering, and
//   (2) prize-collecting removal: drop a served relay whenever splicing out its two
//       sites saves more travel than paying its skip penalty.
// The two phases are alternated. Deterministic.
#include <bits/stdc++.h>
using namespace std;

static int n, N;
static vector<long long> X, Y, pen;

static inline long long D(int a, int b) {
    long long dx = X[a] - X[b], dy = Y[a] - Y[b];
    return (long long)llround(sqrt((double)(dx * dx + dy * dy)));
}

static long long travel(const vector<int>& seq) {
    long long f = 0; int prev = 0;
    for (int v : seq) { f += D(prev, v); prev = v; }
    f += D(prev, 0);
    return f;
}

static long long totalF(const vector<int>& seq) {
    long long f = travel(seq);
    vector<char> in(N + 1, 0);
    for (int v : seq) in[v] = 1;
    for (int i = 1; i <= n; i++)
        if (!(in[i] && in[n + i])) f += pen[i];
    return f;
}

// After a 2-opt reversal all nodes remain present; only check pickup-before-delivery.
static bool feasibleOrder(const vector<int>& seq) {
    vector<int> pos(N + 1, -1);
    for (int r = 0; r < (int)seq.size(); r++) pos[seq[r]] = r;
    for (int i = 1; i <= n; i++)
        if (pos[i] >= 0 && pos[n + i] >= 0 && pos[i] >= pos[n + i]) return false;
    return true;
}

static void twoOpt(vector<int>& seq, int maxPasses) {
    bool improved = true; int pass = 0;
    while (improved && pass < maxPasses) {
        improved = false; pass++;
        int m = seq.size();
        for (int i = 0; i < m - 1; i++) {
            int a = (i == 0) ? 0 : seq[i - 1];
            for (int j = i + 1; j < m; j++) {
                int b = (j == m - 1) ? 0 : seq[j + 1];
                long long delta = D(a, seq[j]) + D(seq[i], b)
                                - D(a, seq[i]) - D(seq[j], b);
                if (delta < 0) {
                    reverse(seq.begin() + i, seq.begin() + j + 1);
                    if (feasibleOrder(seq)) { improved = true; a = (i == 0) ? 0 : seq[i - 1]; }
                    else reverse(seq.begin() + i, seq.begin() + j + 1);
                }
            }
        }
    }
}

static void removals(vector<int>& seq) {
    bool improved = true;
    while (improved) {
        improved = false;
        vector<char> in(N + 1, 0);
        for (int v : seq) in[v] = 1;
        long long curF = totalF(seq);
        int bestRelay = -1; long long bestF = curF;
        for (int i = 1; i <= n; i++) {
            if (!(in[i] && in[n + i])) continue;    // already unserved
            vector<int> cand;
            cand.reserve(seq.size());
            for (int v : seq) if (v != i && v != n + i) cand.push_back(v);
            long long f = totalF(cand);
            if (f < bestF) { bestF = f; bestRelay = i; }
        }
        if (bestRelay != -1) {
            vector<int> ns;
            for (int v : seq) if (v != bestRelay && v != n + bestRelay) ns.push_back(v);
            seq.swap(ns);
            improved = true;
        }
    }
}

int main() {
    if (scanf("%d", &n) != 1) return 0;
    N = 2 * n;
    X.assign(N + 1, 0); Y.assign(N + 1, 0); pen.assign(n + 1, 0);
    scanf("%lld %lld", &X[0], &Y[0]);
    for (int i = 1; i <= n; i++)
        scanf("%lld %lld %lld %lld %lld", &X[i], &Y[i], &X[n + i], &Y[n + i], &pen[i]);

    // nearest-feasible serve-all construction
    vector<char> visited(N + 1, 0);
    vector<int> seq; seq.reserve(N);
    int cur = 0;
    for (int step = 0; step < N; step++) {
        int best = -1; long long bestD = LLONG_MAX;
        for (int v = 1; v <= N; v++) {
            if (visited[v]) continue;
            bool allowed = (v <= n) ? true : (bool)visited[v - n];
            if (!allowed) continue;
            long long d = D(cur, v);
            if (d < bestD) { bestD = d; best = v; }
        }
        visited[best] = 1; seq.push_back(best); cur = best;
    }

    // alternate improvement phases
    twoOpt(seq, 8);
    removals(seq);
    twoOpt(seq, 6);
    removals(seq);
    twoOpt(seq, 4);

    printf("%d\n", (int)seq.size());
    string line;
    for (size_t r = 0; r < seq.size(); r++) { if (r) line += ' '; line += to_string(seq[r]); }
    printf("%s\n", line.c_str());
    return 0;
}
