// TIER: strong
// "singer-cyclic-construction": when N=H*W factors as N=n^2+n+1 with gcd(H,W)=1, a perfect
// (N,n+1,1) difference set exists in Z_N (Singer's theorem, for n a prime power). We recover
// one by backtracking search directly in Z_N (small N here, so this is fast), then fold it
// onto the torus via the CRT ring map r -> (r mod H, r mod W) -- a ring isomorphism when
// gcd(H,W)=1, so differences transport exactly and the folded set is automatically Sidon on
// the torus with the maximum possible mark count and full coverage=1. We then exhaustively
// try every shift t in Z_N and every unit u in Z_N (both preserve the difference-set property)
// to pick the translate/scaling that maximizes the weighted sum. This reaches a global
// algebraic ceiling invisible to any incremental collision-avoidance heuristic. When no such
// factorization exists (or the search doesn't find a set within its node budget), we fall
// back to a randomized multi-start greedy local search directly on the torus, and report
// whichever candidate scores higher under the problem's own objective.
#include <bits/stdc++.h>
using namespace std;

static int H, W;
static long long N;
static vector<vector<long long>> wgt;
static double LAMBDA;

double objective(const vector<pair<int, int>> &marks) {
    long long m = marks.size();
    long long S = 0;
    for (auto &p : marks) S += wgt[p.first][p.second];
    double coverage = (N > 1) ? ((double)m * (double)(m - 1)) / (double)(N - 1) : 0.0;
    return (double)S * (1.0 + LAMBDA * coverage);
}

bool feasible(const vector<pair<int, int>> &marks) {
    set<pair<int, int>> cells;
    set<long long> diffs;
    for (auto &p : marks)
        if (!cells.insert(p).second) return false;
    int m = marks.size();
    for (int a = 0; a < m; a++)
        for (int b = 0; b < m; b++) {
            if (a == b) continue;
            int dr = ((marks[a].first - marks[b].first) % H + H) % H;
            int dc = ((marks[a].second - marks[b].second) % W + W) % W;
            long long id = (long long)dr * W + dc;
            if (!diffs.insert(id).second) return false;
        }
    return true;
}

// Backtracking search for a perfect difference set of size k in Z_n (0 is fixed WLOG).
vector<long long> findDiffSet(long long n, int k, long long nodeBudget) {
    vector<char> used(n, 0);
    vector<long long> chosen = {0};
    long long nodes = 0;
    function<bool(long long, int)> rec = [&](long long last, int depth) -> bool {
        nodes++;
        if (nodes > nodeBudget) return false;
        if (depth == k) return true;
        for (long long cand = last + 1; cand < n; cand++) {
            vector<long long> newly;
            bool ok = true;
            for (long long a : chosen) {
                long long d1 = ((cand - a) % n + n) % n;
                long long d2 = ((a - cand) % n + n) % n;
                if (d1 == 0 || d2 == 0 || used[d1] || used[d2]) { ok = false; break; }
                newly.push_back(d1);
                newly.push_back(d2);
            }
            if (!ok) continue;
            for (long long d : newly) used[d] = 1;
            chosen.push_back(cand);
            if (rec(cand, depth + 1)) return true;
            chosen.pop_back();
            for (long long d : newly) used[d] = 0; // reset
            if (nodes > nodeBudget) return false;
        }
        return false;
    };
    if (rec(0, 1)) return chosen;
    return {};
}

long long gcdll(long long a, long long b) { return b == 0 ? a : gcdll(b, a % b); }

vector<pair<int, int>> trySinger() {
    if (gcdll(H, W) != 1) return {};
    // solve k^2-k+1 = N for integer k>=2
    double kk = (1.0 + sqrt(4.0 * (double)N - 3.0)) / 2.0;
    long long k = llround(kk);
    if (k < 2) return {};
    if (k * k - k + 1 != N) return {};
    vector<long long> D = findDiffSet(N, (int)k, 400000);
    if (D.empty()) return {};

    // best over shifts t in [0,N) and units u in [1,N) with gcd(u,N)=1
    vector<pair<int, int>> best;
    double bestObj = -1;
    for (long long t = 0; t < N; t++) {
        for (long long u = 1; u < N; u++) {
            if (gcdll(u, N) != 1) continue;
            vector<pair<int, int>> cand;
            cand.reserve(D.size());
            for (long long d : D) {
                long long r = ((u * d + t) % N + N) % N;
                cand.push_back({(int)(r % H), (int)(r % W)});
            }
            double obj = objective(cand);
            if (obj > bestObj) { bestObj = obj; best = cand; }
        }
    }
    if (!best.empty() && feasible(best)) return best;
    return {};
}

vector<pair<int, int>> randomizedGreedy(mt19937_64 &rng) {
    vector<tuple<double, int, int>> cells;
    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++) {
            double noise = uniform_real_distribution<double>(0.85, 1.15)(rng);
            cells.push_back({(double)wgt[i][j] * noise, i, j});
        }
    sort(cells.begin(), cells.end(), [](auto &a, auto &b) { return get<0>(a) > get<0>(b); });

    vector<pair<int, int>> marks;
    set<long long> usedDiff;
    auto diffId = [&](int dr, int dc) -> long long { return (long long)dr * W + dc; };
    for (auto &[wt, r, c] : cells) {
        vector<long long> newly;
        set<long long> localNew; // catches collisions introduced BY this candidate itself,
                                  // including a self-negating offset (id1==id2 for one pair)
                                  // and collisions across two different existing marks.
        bool ok = true;
        for (auto &mk : marks) {
            int dr1 = ((r - mk.first) % H + H) % H;
            int dc1 = ((c - mk.second) % W + W) % W;
            int dr2 = ((mk.first - r) % H + H) % H;
            int dc2 = ((mk.second - c) % W + W) % W;
            long long id1 = diffId(dr1, dc1), id2 = diffId(dr2, dc2);
            if (usedDiff.count(id1) || usedDiff.count(id2) || localNew.count(id1) || localNew.count(id2) || id1 == id2) {
                ok = false;
                break;
            }
            localNew.insert(id1);
            localNew.insert(id2);
            newly.push_back(id1);
            newly.push_back(id2);
        }
        if (!ok) continue;
        for (long long id : newly) usedDiff.insert(id);
        marks.push_back({r, c});
    }
    return marks;
}

int main() {
    long long lnum, lden;
    cin >> H >> W >> lnum >> lden;
    LAMBDA = (double)lnum / (double)lden;
    N = (long long)H * W;
    wgt.assign(H, vector<long long>(W));
    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++) cin >> wgt[i][j];

    vector<pair<int, int>> best;
    double bestObj = -1;

    // every candidate is re-verified with the full O(m^2) feasible() check before it is
    // allowed to compete on objective value -- defense in depth against any bug upstream.
    vector<pair<int, int>> singer = trySinger();
    if (!singer.empty() && feasible(singer)) {
        double obj = objective(singer);
        if (obj > bestObj) { bestObj = obj; best = singer; }
    }

    mt19937_64 rng(987654321ULL);
    int restarts = (N <= 200) ? 400 : (N <= 1000 ? 150 : 60);
    for (int it = 0; it < restarts; it++) {
        vector<pair<int, int>> cand = randomizedGreedy(rng);
        if (!feasible(cand)) continue;
        double obj = objective(cand);
        if (obj > bestObj) { bestObj = obj; best = cand; }
    }

    if (best.empty()) {
        // absolute fallback: single heaviest cell
        int br = 0, bc = 0;
        long long bv = -1;
        for (int i = 0; i < H; i++)
            for (int j = 0; j < W; j++)
                if (wgt[i][j] > bv) { bv = wgt[i][j]; br = i; bc = j; }
        best = {{br, bc}};
    }

    cout << best.size() << "\n";
    for (auto &p : best) cout << p.first << " " << p.second << "\n";
    return 0;
}
