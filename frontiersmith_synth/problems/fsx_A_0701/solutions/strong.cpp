// TIER: strong
// Insight: a genuine lift forces a UNIFORM fiber size, so k0 must divide N. Restrict the
// search to divisors of N, and for each candidate k run local-search hill-climbing that
// swaps vertices between class/sheet slots to maximize how many real edges are directly
// explained by a compact top-frequency base-edge set (a coordinate-ascent / EM-style
// dilation-local-search over the group-quotient embedding), then keep the best candidate.
#include <bits/stdc++.h>
using namespace std;

static int N, M, n_g, KG;
static vector<int> EA, EB;
static vector<vector<int>> adj;

long long canonKey(int ca, int cb, int delta, int n) {
    if (ca > cb) { swap(ca, cb); delta = (n - delta) % n; }
    else if (ca == cb) { int alt = (n - delta) % n; delta = min(delta, alt); }
    return ((long long)ca * 1001 + cb) * 1001 + delta;
}
void decodeKey(long long key, int& ca, int& cb, int& g) {
    g = (int)(key % 1001); key /= 1001;
    cb = (int)(key % 1001); key /= 1001;
    ca = (int)key;
}

int matchedEdge(int a, int b, vector<int>& cls, vector<int>& sheet,
                unordered_set<long long>& S, int n) {
    int ca = cls[a], cb = cls[b];
    int delta = ((sheet[b] - sheet[a]) % n + n) % n;
    return S.count(canonKey(ca, cb, delta, n)) ? 1 : 0;
}

long long sumMatchedAround(int x, int y, vector<int>& cls, vector<int>& sheet,
                            unordered_set<long long>& S, int n) {
    long long tot = 0;
    bool xyAdj = false;
    for (int nb : adj[x]) {
        if (nb == y) { xyAdj = true; continue; }
        tot += matchedEdge(x, nb, cls, sheet, S, n);
    }
    for (int nb : adj[y]) {
        if (nb == x) continue;
        tot += matchedEdge(y, nb, cls, sheet, S, n);
    }
    if (xyAdj) tot += matchedEdge(x, y, cls, sheet, S, n);
    return tot;
}

// Runs local search for a given divisor k; returns (proxyCoverage, k, n, cls, sheet, edges).
struct Candidate { double coverage; int k, n; vector<int> cls, sheet; vector<array<int,3>> edges; };

Candidate runForK(int k, mt19937& rng) {
    int n = N / k;
    vector<int> cls(N), sheet(N);
    vector<int> perm(N);
    iota(perm.begin(), perm.end(), 0);
    shuffle(perm.begin(), perm.end(), rng);
    for (int t = 0; t < N; t++) {
        int v = perm[t];
        cls[v] = t / n;
        sheet[v] = t % n;
    }

    const int PASSES = 4;
    const int EMAX_LOCAL = 64;
    unordered_set<long long> S;
    for (int pass = 0; pass < PASSES; pass++) {
        unordered_map<long long, int> freq;
        freq.reserve(M * 2);
        for (int i = 0; i < M; i++) {
            int a = EA[i], b = EB[i];
            int delta = ((sheet[b] - sheet[a]) % n + n) % n;
            freq[canonKey(cls[a], cls[b], delta, n)]++;
        }
        vector<pair<int,long long>> byCount;
        byCount.reserve(freq.size());
        for (auto& kv : freq) byCount.push_back({kv.second, kv.first});
        sort(byCount.begin(), byCount.end(), greater<pair<int,long long>>());
        S.clear();
        for (int i = 0; i < (int)byCount.size() && i < EMAX_LOCAL; i++) S.insert(byCount[i].second);

        int trials = min(3 * N, 30000);
        for (int t = 0; t < trials; t++) {
            int x = rng() % N, y = rng() % N;
            if (x == y) continue;
            long long before = sumMatchedAround(x, y, cls, sheet, S, n);
            swap(cls[x], cls[y]);
            swap(sheet[x], sheet[y]);
            long long after = sumMatchedAround(x, y, cls, sheet, S, n);
            if (after <= before) {
                // revert
                swap(cls[x], cls[y]);
                swap(sheet[x], sheet[y]);
            }
        }
    }

    // Final frequency table + coverage on the converged assignment.
    unordered_map<long long, int> freq;
    for (int i = 0; i < M; i++) {
        int a = EA[i], b = EB[i];
        int delta = ((sheet[b] - sheet[a]) % n + n) % n;
        freq[canonKey(cls[a], cls[b], delta, n)]++;
    }
    vector<pair<int,long long>> byCount;
    byCount.reserve(freq.size());
    for (auto& kv : freq) byCount.push_back({kv.second, kv.first});
    sort(byCount.begin(), byCount.end(), greater<pair<int,long long>>());
    int take = min((int)byCount.size(), EMAX_LOCAL);
    long long covered = 0;
    vector<array<int,3>> edges;
    for (int i = 0; i < take; i++) {
        covered += byCount[i].first;
        int ca, cb, g;
        decodeKey(byCount[i].second, ca, cb, g);
        edges.push_back({ca, cb, g});
    }
    Candidate cand;
    cand.coverage = (double)covered / (double)M;
    cand.k = k; cand.n = n; cand.cls = cls; cand.sheet = sheet; cand.edges = edges;
    return cand;
}

int main() {
    scanf("%d %d", &N, &M);
    EA.resize(M); EB.resize(M);
    adj.assign(N, {});
    for (int i = 0; i < M; i++) {
        int u, v;
        scanf("%d %d", &u, &v);
        u--; v--;
        EA[i] = u; EB[i] = v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    vector<int> divisors;
    for (int k = 2; k <= 20 && k <= N; k++) if (N % k == 0) divisors.push_back(k);
    if (divisors.empty()) divisors.push_back(min(20, N));

    mt19937 rng(987654321u);
    Candidate best;
    best.coverage = -1;
    for (int k : divisors) {
        Candidate c = runForK(k, rng);
        if (c.coverage > best.coverage + 1e-9 ||
            (fabs(c.coverage - best.coverage) < 1e-9 && (best.coverage < 0 || c.k < best.k))) {
            best = c;
        }
    }

    printf("%d %d\n", best.k, best.n);
    for (int i = 0; i < N; i++) printf("%d %d\n", best.cls[i], best.sheet[i]);
    printf("%d\n", (int)best.edges.size());
    for (auto& e : best.edges) printf("%d %d %d\n", e[0], e[1], e[2]);
    return 0;
}
