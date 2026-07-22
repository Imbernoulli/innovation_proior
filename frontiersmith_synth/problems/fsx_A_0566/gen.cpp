#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// "Binding the Atlas"  (generator)  family: biased-fanout-search-tree
//
// N tiles in FIXED order 1..N are bound into a multiway search tree. The objective
// mixes three levers (see chk.cc): depth-biasing (hot tiles want shallow pages),
// fanout vs. line count (ceil(f/B)), and scan contiguity (a swept band wants to sit
// on few fat pages, not be shattered into singletons).
//
// The obvious recipe is a UNIFORM B-tree (pack size-B leaves, balance by count). It
// is depth-blind: it never floats a hot tile shallow. Two planted structures make it
// far from a co-designed binding:
//   (A) a PLATEAU of hot, lightly-scanned tiles SCATTERED across the order. A good
//       binding floats them to shallow pages (a biased/Huffman-ish tree); the uniform
//       B-tree leaves them at full depth. Sized so the plateau lands at moderate depth
//       (not depth 1) -- the score ceiling stays open above the reference solutions.
//   (B) a hot, HEAVILY-SCANNED contiguous band. It looks hot, so a naive frequency-only
//       binding isolates its tiles -> shatters the band -> every crossing scan pays many
//       touches. The co-design keeps the band PACKED and only floats LIGHTLY-scanned
//       hot tiles. (Rewards reading the scans, not just the demands.)
//
// Trap (A-dominant): t4,t6,t9 ; scan trap (B) added on t5,t8,t10 ; t1..t3,t7 milder.
// Output:  N B S ; w_1..w_N on one line ; S lines "lo hi c".
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]){
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]);

    int Narr[11] = {0,  18,  200,  600, 1500, 3000, 5000, 7000, 9000, 11000, 12000};
    int Barr[11] = {0,   2,    4,    3,    4,    6,    4,    8,    4,     4,     5};
    // skew (plateau A) present on all but t2 ; scan-band (B) on t5,t8,t10.
    bool skewA[11] = {false, true, false, true, true, true, true, true, true, true, true};
    bool bandB[11] = {false, true, false, false, false, true, false, false, true, false, true};

    int N = Narr[t], B = Barr[t];
    vector<ll> w(N + 1, 0);
    // light background demand everywhere
    for (int i = 1; i <= N; i++) w[i] = rnd.next(1, 4);

    vector<array<ll,3>> scans;   // lo, hi, c

    const ll HW = 100000;        // hot plateau height

    // ---- band range (B): a contiguous heavily-scanned hot block ----
    int A = 0, Bend = -1;
    if (bandB[t]){
        int band = min(N / 3, max(3 * B, (int)(6 * B + rnd.next(0, 4 * B))));
        A = rnd.next(1, max(1, N - band + 1));
        Bend = min(N, A + band - 1);
        for (int p = A; p <= Bend; p++)
            w[p] = max(w[p], (ll)(HW * (60 + rnd.next(0, 40)) / 100));   // 0.6..1.0*HW
        int ns = min(3500, max(50, N / 3));
        for (int i = 0; i < ns; i++){
            int a = rnd.next(A, Bend), b = rnd.next(A, Bend);
            if (a > b) swap(a, b);
            scans.push_back({(ll)a, (ll)b, (ll)rnd.next(150, 900)});
        }
    }

    // ---- plateau (A): scattered hot, lightly-scanned tiles ----
    if (skewA[t]){
        // K sized so the plateau sits at moderate depth under a biased tree
        int K = min(max(1, N / 4), max(6, (int)llround(0.045 * N)));
        // pick K distinct tiles outside the band
        vector<int> pool;
        for (int p = 1; p <= N; p++) if (p < A || p > Bend) pool.push_back(p);
        for (int i = 0; i < K && i < (int)pool.size(); i++){
            int j = i + rnd.next(0, (int)pool.size() - 1 - i);
            swap(pool[i], pool[j]);
        }
        for (int i = 0; i < K && i < (int)pool.size(); i++)
            w[pool[i]] = max(w[pool[i]], (ll)(HW * (55 + rnd.next(0, 45)) / 100));
    }

    // ---- background light scans (broad, cheap) ----
    {
        int q = min(1200, max(4, N / 20));
        for (int i = 0; i < q; i++){
            int len = rnd.next(1, max(2, N / 30));
            int lo = rnd.next(1, max(1, N - len + 1));
            int hi = min(N, lo + len - 1);
            scans.push_back({(ll)lo, (ll)hi, (ll)rnd.next(1, 6)});
        }
    }

    int Sn = (int)scans.size();
    printf("%d %d %d\n", N, B, Sn);
    for (int i = 1; i <= N; i++) printf("%lld%c", w[i], i == N ? '\n' : ' ');
    for (auto &s : scans) printf("%lld %lld %lld\n", s[0], s[1], s[2]);
    return 0;
}
