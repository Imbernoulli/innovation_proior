// Checker for fsx_A_0701 "Fold a Big Network onto a Small Labeled One"
#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

static const int KMAX = 20;
static const int NMAX = 1000;
static const int EMAX = 64;
static const int PENALTY = 8;

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int N = inf.readInt();
    int M = inf.readInt();
    vector<int> ea(M), eb(M);
    for (int i = 0; i < M; i++) {
        ea[i] = inf.readInt() - 1;
        eb[i] = inf.readInt() - 1;
    }

    int k = ouf.readInt(1, KMAX, "k");
    int n = ouf.readInt(1, NMAX, "n");

    vector<int> b(N), s(N);
    vector<int> invSlot(k * n, -1);
    for (int i = 0; i < N; i++) {
        b[i] = ouf.readInt(0, k - 1, "b_i");
        s[i] = ouf.readInt(0, n - 1, "s_i");
        int slot = b[i] * n + s[i];
        if (invSlot[slot] != -1) {
            quitf(_wa, "duplicate (class,sheet) slot (%d,%d) used by vertices %d and %d",
                  b[i], s[i], invSlot[slot] + 1, i + 1);
        }
        invSlot[slot] = i;
    }

    int E = ouf.readInt(0, EMAX, "E");
    vector<vector<pair<int,int>>> trans(k); // trans[class] = list of (otherClass, advance)
    for (int j = 0; j < E; j++) {
        int p = ouf.readInt(0, k - 1, "p_j");
        int q = ouf.readInt(0, k - 1, "q_j");
        int g = ouf.readInt(0, n - 1, "g_j");
        trans[p].push_back({q, g});
        trans[q].push_back({p, (n - g) % n});
    }

    if (!ouf.seekEof()) quitf(_wa, "trailing tokens in output");

    // BFS from every base class over states (class, sheetDelta mod n), capped at PENALTY hops.
    vector<vector<int>> dist(k, vector<int>(k * n, -1));
    for (int src = 0; src < k; src++) {
        vector<int>& d = dist[src];
        int start = src * n + 0;
        d[start] = 0;
        deque<int> q;
        q.push_back(start);
        while (!q.empty()) {
            int cur = q.front(); q.pop_front();
            int dc = d[cur];
            if (dc >= PENALTY) continue;
            int curClass = cur / n, curDelta = cur % n;
            for (auto& tr : trans[curClass]) {
                int nc = tr.first;
                int nd = (curDelta + tr.second) % n;
                int nxt = nc * n + nd;
                if (d[nxt] == -1) {
                    d[nxt] = dc + 1;
                    q.push_back(nxt);
                }
            }
        }
    }

    if (M <= 0) quitf(_fail, "checker internal error: M<=0");

    long long sumDilation = 0;
    for (int i = 0; i < M; i++) {
        int a = ea[i], bb = eb[i];
        int ca = b[a], cb = b[bb];
        int delta = ((s[bb] - s[a]) % n + n) % n;
        int d = dist[ca][cb * n + delta];
        int dilation = (d == -1) ? PENALTY : min(d, PENALTY);
        sumDilation += dilation;
    }

    double F = (double)sumDilation / (double)M;
    double B = (double)PENALTY;
    double sc = min(1000.0, 100.0 * B / max(F, 1e-9));
    quitp(sc / 1000.0, "OK F=%.6f B=%.6f Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
