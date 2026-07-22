#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int n = inf.readInt();
    int m = inf.readInt();
    int pageCap = inf.readInt();
    int L = inf.readInt();
    int T = inf.readInt();

    for (int i = 0; i < m; i++) { inf.readInt(); inf.readInt(); } // paddock adjacency: unused by scoring

    vector<int> trace(T);
    for (int i = 0; i < T; i++) trace[i] = inf.readInt(1, n, "trace_node");

    vector<int> page(n + 1, 0);
    vector<int> cnt(n + 1, 0);
    for (int i = 1; i <= n; i++) {
        int p = ouf.readInt(1, n, "shelf_id");
        page[i] = p;
        cnt[p]++;
        if (cnt[p] > pageCap) quitf(_wa, "shelf %d holds more than pageCap=%d animals", p, pageCap);
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing tokens in participant output");

    auto simulateReloads = [&](const vector<int>& pg) -> long long {
        vector<int> cache; // recency order, front = LRU, back = MRU
        cache.reserve(L);
        long long reloads = 0;
        for (int i = 0; i < T; i++) {
            int sh = pg[trace[i]];
            int pos = -1;
            for (int k = 0; k < (int)cache.size(); k++) if (cache[k] == sh) { pos = k; break; }
            if (pos >= 0) {
                cache.erase(cache.begin() + pos);
                cache.push_back(sh);
            } else {
                reloads++;
                if ((int)cache.size() >= L) cache.erase(cache.begin());
                cache.push_back(sh);
            }
        }
        return reloads;
    };

    long long F = simulateReloads(page);

    vector<int> basePage(n + 1, 0);
    for (int i = 1; i <= n; i++) basePage[i] = 1 + (i - 1) / pageCap;
    long long B = simulateReloads(basePage);
    if (B < 1) B = 1;

    double sc = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
