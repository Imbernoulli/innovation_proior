#include "testlib.h"
#include <vector>
#include <algorithm>
#include <cstdio>
using namespace std;

struct Seg {
    vector<int> toks;
};

int main(int argc, char *argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    static const int Svals[10]  = {3, 4, 5, 7, 9, 12, 15, 18, 21, 24};
    static const int Kvals[10]  = {1, 1, 2, 1, 2, 1, 3, 1, 2, 1};
    double t = (testId - 1) / 9.0;

    int S = Svals[testId - 1];
    int K = Kvals[testId - 1];

    // Planted alternation clusters: S clusters, each with (K+1) members that ring-alternate.
    // Member j (0..K) of cluster c (0..S-1) gets id = c + 1 + j*S -- spaced exactly S apart.
    // This means: under ANY assignment rule that treats all (K+1)*S cluster ids as one
    // frequency-tied block and spreads ties in id order (address-modulo, or naive
    // frequency-sorted round robin), every cluster's members land in the SAME room --
    // even though a genuinely alternation-aware solver sees the ring edges and keeps them
    // apart. All members of every cluster share EXACTLY the same reference count (ties are
    // load-bearing, not incidental).
    int clusterIdMax = S * (K + 1);

    int nBursts = (int)(4 + t * 7 + 0.5);        // 4 .. 11 (kept modest so it doesn't drown the wildcard signal)
    int repeatsPerBurst = 3;                     // ring cycles per burst
    int burstLen = repeatsPerBurst * (K + 1);
    int clusterFreqPerMember = nBursts * repeatsPerBurst;

    // Planted heavy-traffic pairs: Sh = floor(S/2) pairs, member ids exactly S apart (same
    // address-modulo-collision trick as the clusters, so the checker's baseline is GUARANTEED
    // to seat both members of every pair in the same room). Unlike the clusters, every heavy
    // actor gets a DISTINCT total reference count (strictly decreasing), and there are at
    // most S of them total -- so a frequency-first packer (which always seats the next actor
    // into whichever room currently has the least load) provably never has to reuse a room
    // while placing them, and GUARANTEES all heavies land in distinct rooms. Their traffic is
    // split into several scattered chunks (not one contiguous block), so an actual collision
    // -- which only the baseline suffers -- costs real, repeated misses instead of a free ride.
    int Sh = max(1, S / 2);
    int noisePoolSize = max(4, 2 * S);
    int wildPoolSize = max(6, 3 * S);
    int heavyIdBase = clusterIdMax;
    int heavyIdTop = heavyIdBase + Sh + S; // highest id used by a pair's "+S" member

    // Ladder-scaled actor pool P: comfortably above what the trap needs, extra ids are inert
    // filler (never referenced, harmless, lets P scale with the ladder as required).
    int Pladder = (int)(30 + t * (1500 - 30));
    int Pneeded = heavyIdTop + wildPoolSize + noisePoolSize + 10;
    int P = max(Pladder, Pneeded);
    if (P > 1500) P = 1500;
    if (P < Pneeded) P = Pneeded; // safety, should not trigger given the cap above holds

    int heavyFreqBase = 6 * clusterFreqPerMember + 4 * Sh + 40; // clearly out-frequencies clusters

    vector<int> pool;
    pool.reserve(P - heavyIdTop);
    for (int i = heavyIdTop + 1; i <= P; i++) pool.push_back(i);
    for (int i = (int)pool.size() - 1; i > 0; i--) {
        int j = rnd.next(0, i);
        swap(pool[i], pool[j]);
    }
    int off = 0;
    vector<int> wildIds;
    for (int i = off; i < (int)pool.size() && (int)wildIds.size() < wildPoolSize; i++)
        wildIds.push_back(pool[i]);
    off += (int)wildIds.size();
    if (wildIds.empty()) wildIds.push_back(pool.empty() ? P : pool[0]);
    vector<int> noisePool;
    for (int i = off; i < (int)pool.size() && (int)noisePool.size() < noisePoolSize; i++)
        noisePool.push_back(pool[i]);
    if (noisePool.empty()) noisePool.push_back(pool.empty() ? P : pool[0]);

    // --- build segments ---
    vector<Seg> segs;

    // cluster ring bursts
    for (int c = 0; c < S; c++) {
        vector<int> ring(K + 1);
        for (int j = 0; j <= K; j++) ring[j] = c + 1 + j * S;
        for (int b = 0; b < nBursts; b++) {
            Seg sg;
            sg.toks.reserve(burstLen);
            for (int r = 0; r < repeatsPerBurst; r++)
                for (int j = 0; j <= K; j++) sg.toks.push_back(ring[j]);
            segs.push_back(move(sg));
        }
    }

    // heavy pairs: build the (id, freq) list, freq strictly decreasing across ALL 2*Sh
    // heavies so a frequency-first packer never sees a tie among them.
    vector<pair<int, int>> heavyActors; // (id, run length)
    heavyActors.reserve(2 * Sh);
    for (int k = 0; k < Sh; k++) {
        int idA = heavyIdBase + k + 1;
        int idB = idA + S;
        int freqA = heavyFreqBase - 2 * k;
        int freqB = heavyFreqBase - 2 * k - 1;
        heavyActors.push_back({idA, freqA});
        heavyActors.push_back({idB, freqB});
    }

    // Each heavy's traffic is split into several scattered chunks (not one contiguous
    // block), so an actual collision -- which only the baseline suffers -- costs real,
    // repeated misses instead of a free ride.
    int nChunks = 6;
    for (auto &pr : heavyActors) {
        int id = pr.first, run = pr.second;
        if (run < nChunks * 2) run = nChunks * 2;
        int base = run / nChunks;
        int rem = run - base * nChunks;
        for (int c = 0; c < nChunks; c++) {
            int len = base + (c < rem ? 1 : 0);
            if (len < 1) continue;
            Seg sg;
            sg.toks.assign(len, id);
            segs.push_back(move(sg));
        }
    }

    // wildcard groups: partition wildIds into disjoint alternating groups (each actor
    // belongs to exactly one group, unlike the shared-pool clusters). Each group gets its
    // own RANDOMLY varying burst length, so -- unlike the exactly-tied clusters -- distinct
    // groups end up with distinct (untied) total reference counts. Group ids are otherwise
    // uncontrolled (random draw), so an address-modulo assignment has no way to keep a
    // group's members apart; but because group weights are untied, even a plain "sort
    // actors by raw reference count, greedily spread" pass naturally separates each small
    // group into distinct rooms (the same mechanism that isolates heavies) -- a real,
    // honestly-earned edge for ANY frequency-aware strategy, not only the alternation-miner.
    int wildRounds = 6; // re-partition wildIds into fresh random groupings several times
    for (int round = 0; round < wildRounds; round++) {
        vector<int> shuffled = wildIds;
        for (int i = (int)shuffled.size() - 1; i > 0; i--) {
            int j = rnd.next(0, i);
            swap(shuffled[i], shuffled[j]);
        }
        int idx = 0;
        int Wn = (int)shuffled.size();
        while (idx < Wn) {
            if (Wn - idx < 2) break;
            int g = rnd.next(2, min(Wn - idx, K + 3));
            vector<int> grp(shuffled.begin() + idx, shuffled.begin() + idx + g);
            idx += g;
            int reps = rnd.next(6, 20 + (int)(t * 60));
            int L = reps * g;
            Seg sg;
            sg.toks.reserve(L);
            for (int r = 0; r < reps; r++)
                for (int m = 0; m < g; m++) sg.toks.push_back(grp[m]);
            segs.push_back(move(sg));
        }
    }

    // bounded-pool background noise (kept small so it doesn't create an unavoidable
    // strategy-independent miss floor: repeats from a small pool mostly hit once warm)
    int structuredTotal = 0;
    for (auto &sg : segs) structuredTotal += (int)sg.toks.size();
    int noiseBudget = structuredTotal / 6 + 20;
    while (noiseBudget > 0) {
        int L = min(noiseBudget, rnd.next(4, 16));
        Seg sg;
        sg.toks.reserve(L);
        for (int i = 0; i < L; i++) sg.toks.push_back(noisePool[rnd.next(0, (int)noisePool.size() - 1)]);
        segs.push_back(move(sg));
        noiseBudget -= L;
    }

    // shuffle segment order so structure is genuinely scattered across the whole trace
    for (int i = (int)segs.size() - 1; i > 0; i--) {
        int j = rnd.next(0, i);
        swap(segs[i], segs[j]);
    }

    vector<int> trace;
    trace.reserve(structuredTotal + noisePoolSize * 20);
    for (auto &sg : segs)
        for (int x : sg.toks) trace.push_back(x);

    int T = (int)trace.size();
    if (T > 15000) { trace.resize(15000); T = 15000; }
    if (T < 1) { trace.push_back(1); T = 1; }

    printf("%d %d %d %d\n", P, S, K, T);
    for (int i = 0; i < T; i++) {
        printf("%d%c", trace[i], (i + 1 < T) ? ' ' : '\n');
    }
    return 0;
}
