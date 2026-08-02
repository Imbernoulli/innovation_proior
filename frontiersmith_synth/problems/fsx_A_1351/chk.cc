#include "testlib.h"
#include <vector>
#include <utility>
#include <cstdio>
using namespace std;

// ---- Game rules (see statement.txt) --------------------------------------
// Each test file holds K independent "arenas". In an arena, You and Rival
// alternately claim distinct vertices out of n; whoever ends up owning BOTH
// endpoints of some poison edge, by their own claims, loses that arena
// immediately. If every vertex gets claimed with nobody ever doing that, the
// arena is a draw. Rival's algorithm (fixed, deterministic) is fully
// specified in the statement. Outcome value per arena: You lose -> 0,
// draw -> 1, Rival loses (You "win") -> 2.
// ---------------------------------------------------------------------------

struct Arena {
    int n, m, first;
    vector<vector<int>> adj; // 1..n
};

static vector<Arena> arenas;

static void readArenas() {
    int K = inf.readInt();
    arenas.resize(K);
    for (auto &ar : arenas) {
        ar.n = inf.readInt();
        ar.m = inf.readInt();
        ar.first = inf.readInt();
        ar.adj.assign(ar.n + 1, {});
        for (int j = 0; j < ar.m; j++) {
            int u = inf.readInt(1, ar.n);
            int v = inf.readInt(1, ar.n);
            ar.adj[u].push_back(v);
            ar.adj[v].push_back(u);
        }
    }
}

// Simulate one arena given a policy (pair[], prio[]); returns outcome value.
static double simulateArena(const Arena &ar, const vector<int> &pr, const vector<int> &prio) {
    int n = ar.n;
    vector<int> owner(n + 1, 0);      // 0 none, 1 you, 2 rival
    vector<char> claimed(n + 1, 0);
    int lastRival = -1;
    int mover = (ar.first == 0) ? 1 : 2;
    int totalClaimed = 0;

    while (totalClaimed < n) {
        int choice = -1;
        if (mover == 1) {
            if (lastRival != -1 && pr[lastRival] != 0 && !claimed[pr[lastRival]])
                choice = pr[lastRival];
            else {
                for (int k = 1; k <= n && choice == -1; k++) {
                    int v = prio[k];
                    if (!claimed[v]) choice = v;
                }
            }
            claimed[choice] = 1; owner[choice] = 1; totalClaimed++;
            for (int nb : ar.adj[choice]) if (owner[nb] == 1) return 0.0;   // You complete a poison edge
        } else {
            int best = -1, bestDeg = -1;
            for (int v = 1; v <= n; v++) {
                if (claimed[v]) continue;
                bool safe = true;
                for (int nb : ar.adj[v]) if (owner[nb] == 2) { safe = false; break; }
                if (!safe) continue;
                int deg = (int)ar.adj[v].size();
                if (deg > bestDeg) { bestDeg = deg; best = v; }
            }
            if (best == -1) {
                for (int v = 1; v <= n; v++) if (!claimed[v]) { best = v; break; }
            }
            choice = best;
            claimed[choice] = 1; owner[choice] = 2; totalClaimed++;
            lastRival = choice;
            for (int nb : ar.adj[choice]) if (owner[nb] == 2) return 2.0;   // Rival completes a poison edge
        }
        mover = 3 - mover;
    }
    return 1.0; // draw
}

// Parse a full policy (pair[]/prio[] for every arena) from a stream (participant
// output via ouf, or the checker's own trivial baseline construction).
static bool readPolicyFromOuf(vector<vector<int>> &prAll, vector<vector<int>> &prioAll) {
    prAll.assign(arenas.size(), {});
    prioAll.assign(arenas.size(), {});
    for (size_t ai = 0; ai < arenas.size(); ai++) {
        int n = arenas[ai].n;
        vector<int> pr(n + 1, 0), prio(n + 1, 0);
        for (int v = 1; v <= n; v++) pr[v] = ouf.readInt(0, n, "pair value");
        vector<char> seen(n + 1, 0);
        for (int k = 1; k <= n; k++) {
            int x = ouf.readInt(1, n, "prio value");
            if (seen[x]) quitf(_wa, "prio array is not a permutation (arena %d, repeated value %d)", (int)ai + 1, x);
            seen[x] = 1;
            prio[k] = x;
        }
        prAll[ai] = pr;
        prioAll[ai] = prio;
    }
    return true;
}

int main(int argc, char *argv[]) {
    setName("avoidance claiming game scorer");
    registerTestlibCmd(argc, argv);

    readArenas();

    // --- participant's policy ---
    vector<vector<int>> prAll, prioAll;
    readPolicyFromOuf(prAll, prioAll);
    if (!ouf.seekEof()) quitf(_wa, "trailing data in output");

    double F = 0.0;
    for (size_t ai = 0; ai < arenas.size(); ai++)
        F += simulateArena(arenas[ai], prAll[ai], prioAll[ai]);

    // --- checker's own trivial baseline: never react, claim vertices in
    // plain ascending index order ---
    double B = 0.0;
    for (size_t ai = 0; ai < arenas.size(); ai++) {
        int n = arenas[ai].n;
        vector<int> pr(n + 1, 0), prio(n + 1, 0);
        for (int k = 1; k <= n; k++) prio[k] = k;
        B += simulateArena(arenas[ai], pr, prio);
    }
    if (B < 1.0) B = 1.0;

    double sc = min(1000.0, 100.0 * F / B);
    quitp(sc / 1000.0, "OK F=%.3f B=%.3f Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
