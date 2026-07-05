#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---------------------------------------------------------------
    // Structure ladder. testId 1 is tiny (near example scale), growing
    // to a large adversarial instance by testId 10.
    //   n  : number of observations
    //   K  : number of latent signature-clusters
    //   W  : width of the signature universe [0,W)
    //   P  : signature size (spikes per signature)
    //   mut: signature mutations per cluster step  (controls the gradient)
    //   phaseRange : per-observation random time offset
    //   noise: random extra firings per observation
    // ---------------------------------------------------------------
    int n         = 8 + 45 * (testId - 1);          // 8 .. 413
    if (n > 450) n = 450;
    int K         = 3 + testId;                     // 4 .. 13
    if (K > n) K = n;
    int W         = 40 + 4 * testId;                // 44 .. 80
    int P         = 10 + testId;                    // 11 .. 20
    if (P > W - 2) P = W - 2;
    int mut       = max(2, P / 5);                  // 2 .. 4
    int phaseRange= 30 + 5 * testId;                // 35 .. 80
    int noise     = 3 + (testId % 4);               // 3 .. 6
    int L         = W + phaseRange + 8;             // < 400
    if (L > 400) L = 400;

    // adversarial flavor selection
    bool skew   = (testId % 3 == 1);   // one dominant cluster (clumping trap)
    bool bridge = (testId % 3 == 2);   // a far cluster pair made near-identical (decoy)
    bool needle = (testId % 4 == 0);   // one near-orthogonal cluster (valuable separator)

    // ---------------------------------------------------------------
    // Build cluster signatures as a random walk in "set space":
    //   S[0] random; S[c] = S[c-1] with `mut` spikes resampled.
    // Overlap |S[c] ∩ S[c']| then decays smoothly with |c - c'|,
    // bottoming out near the random floor P*P/W.
    // ---------------------------------------------------------------
    vector<set<int>> S(K);
    {
        set<int> cur;
        while ((int)cur.size() < P) cur.insert(rnd.next(0, W - 1));
        S[0] = cur;
        for (int c = 1; c < K; c++) {
            // remove `mut` random members, add `mut` fresh ones
            vector<int> mem(cur.begin(), cur.end());
            shuffle(mem.begin(), mem.end());
            for (int t = 0; t < mut && !mem.empty(); t++) {
                cur.erase(mem.back()); mem.pop_back();
            }
            while ((int)cur.size() < P) {
                int x = rnd.next(0, W - 1);
                if (!cur.count(x)) cur.insert(x);
            }
            S[c] = cur;
        }
    }

    // BRIDGE decoy: force two distant clusters to be nearly identical, so a
    // heuristic that only looks at "how different the raw patterns look" is
    // fooled into thinking those two are safe to place adjacently.
    if (bridge && K >= 4) {
        int a = 0, b = K - 1;
        set<int> merged = S[a];
        // copy most of S[a] into S[b], keeping a couple of distinct spikes
        vector<int> vb(S[b].begin(), S[b].end());
        set<int> nb = S[a];
        // reintroduce 2 unique-ish spikes so they are not byte-identical
        for (int t = 0; t < 2; t++) {
            int x = rnd.next(0, W - 1);
            nb.insert(x);
        }
        while ((int)nb.size() > P) { auto it = nb.begin(); advance(it, rnd.next(0,(int)nb.size()-1)); nb.erase(it); }
        while ((int)nb.size() < P) { int x = rnd.next(0, W-1); nb.insert(x); }
        S[b] = nb;
    }

    // NEEDLE: make one middle cluster nearly disjoint from all others by
    // sampling it from a shifted-up universe band [W, 2W) region mapped back.
    int needleCluster = -1;
    if (needle && K >= 3) {
        needleCluster = K / 2;
        set<int> nc;
        // use the odd residues to minimize overlap with the (mostly dense) walk
        while ((int)nc.size() < P) {
            int x = (rnd.next(0, W / 2 - 1) * 2 + 1) % W;
            nc.insert(x);
        }
        S[needleCluster] = nc;
    }

    // ---------------------------------------------------------------
    // Assign observations to clusters and choose block sizes.
    // The INPUT order is laid out as consecutive same-cluster blocks so
    // the identity schedule is deliberately bad (adjacent = same cluster
    // => full-signature coupling ~ P). Good schedules interleave clusters.
    // ---------------------------------------------------------------
    vector<int> clusterOf;
    {
        // base sizes
        vector<int> sz(K, 0);
        int rem = n;
        for (int c = 0; c < K; c++) { sz[c] = 1; rem--; }   // at least 1 each
        // distribute the rest
        if (skew) {
            // dump the bulk into cluster 0 (a clumping trap) then spread a bit
            int big = rem * 3 / 5;
            sz[0] += big; rem -= big;
        }
        while (rem > 0) { sz[rnd.next(0, K - 1)]++; rem--; }
        for (int c = 0; c < K; c++)
            for (int t = 0; t < sz[c]; t++)
                clusterOf.push_back(c);
    }
    // (clusterOf is already in block order 0,0,..,1,1,.. -> bad identity)

    // ---------------------------------------------------------------
    // Materialize each observation's firing pattern:
    //   A_i = { delta_i + p : p in S[c_i] }  U  {noise firings}
    // The shared signature guarantees a high-coupling alignment for
    // same/near clusters; noise adds per-observation variation so the
    // problem is a genuine per-node TSP-path, not pure cluster labeling.
    // ---------------------------------------------------------------
    vector<vector<int>> pattern(n);
    for (int i = 0; i < n; i++) {
        int c = clusterOf[i];
        int delta = rnd.next(0, phaseRange);
        set<int> firings;
        for (int p : S[c]) firings.insert(delta + p);
        for (int t = 0; t < noise; t++) firings.insert(rnd.next(0, L - 1));
        // cap firings at 40 (constraint) — drop extras deterministically
        pattern[i] = vector<int>(firings.begin(), firings.end());
        while ((int)pattern[i].size() > 40) pattern[i].pop_back();
    }

    // ---------------------------------------------------------------
    printf("%d %d\n", n, L);
    for (int i = 0; i < n; i++) {
        printf("%d", (int)pattern[i].size());
        for (int x : pattern[i]) printf(" %d", x);
        printf("\n");
    }
    return 0;
}
