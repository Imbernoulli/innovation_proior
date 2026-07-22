#include "testlib.h"
#include <vector>
#include <cmath>
using namespace std;

// Replays the cue sheet against a fixed room assignment `color` (1-indexed actors ->
// 1-indexed rooms) using a per-room LRU rack set of capacity K. Returns total slow
// changes (misses).
static long long simulate(int P, int S, int K, int T, const vector<int> &trace,
                           const vector<int> &color) {
    // racks[r] holds up to K actor ids, front = most-recently-used.
    vector<vector<int>> racks(S + 1);
    for (int r = 1; r <= S; r++) racks[r].reserve(K);
    long long misses = 0;
    for (int t = 0; t < T; t++) {
        int a = trace[t];
        int r = color[a];
        vector<int> &rk = racks[r];
        int pos = -1;
        for (int i = 0; i < (int)rk.size(); i++) {
            if (rk[i] == a) { pos = i; break; }
        }
        if (pos >= 0) {
            // hit: move to front (most-recently-used)
            int v = rk[pos];
            rk.erase(rk.begin() + pos);
            rk.insert(rk.begin(), v);
        } else {
            misses++;
            if ((int)rk.size() >= K) rk.pop_back(); // evict least-recently-used
            rk.insert(rk.begin(), a);
        }
    }
    return misses;
}

int main(int argc, char *argv[]) {
    registerTestlibCmd(argc, argv);

    int P = inf.readInt();
    int S = inf.readInt();
    int K = inf.readInt();
    int T = inf.readInt();

    vector<int> trace(T);
    for (int t = 0; t < T; t++) trace[t] = inf.readInt(1, P, "a_t");

    vector<int> color(P + 1, 0);
    for (int i = 1; i <= P; i++) {
        color[i] = ouf.readInt(1, S, "c_i");
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing tokens after P assignments");

    long long F = simulate(P, S, K, T, trace, color);

    // Internal baseline: the "do-nothing" reference assignment (address-modulo coloring).
    vector<int> base(P + 1, 0);
    for (int i = 1; i <= P; i++) base[i] = ((i - 1) % S) + 1;
    long long B = simulate(P, S, K, T, trace, base);
    if (B < 1) B = 1;

    double sc = 100.0 * (double)B / (double)max((long long)1, F);
    if (sc > 1000.0) sc = 1000.0;

    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
