#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// "One rule paints the whole mural"  (generator)  family: ca-target-painter
//
// We emit a binary TARGET image on an N x N grid plus a step count T, a bounded
// central seed BOX (rows [r0,r0+b), cols [c0,c0+b)) and a seed budget K.
//
// PLANTED STRUCTURE (never revealed to the solver): the target is produced by
// actually RUNNING an outer-totalistic 2-state cellular automaton -- a random
// SPARSE seed of <=K live cells placed inside the box, evolved forward T steps
// under one library rule.  Two properties are enforced by resampling:
//   (1) a LARGE fraction of the target's live cells lie OUTSIDE the seed box, so
//       they can never be placed by hand -- they MUST be grown by the rule's
//       dynamics.  This is the trap: memorizing the picture into the seed is
//       impossible (seed is box-bounded + <=K), you must find a rule that
//       self-organizes the mural.
//   (2) at least K target cells lie inside the box (so the checker's do-nothing
//       baseline B = K is well defined and the trivial reference hits ratio 0.1).
//
// Several testIds plant with EXOTIC rules (Replicator, Seeds, gnarl, 34-Life,
// Diamoeba) whose dynamics the famous {identity, Conway, fill} rules a greedy
// coder reaches for cannot emulate at all -- those are the far-from-strong traps.
//
// Output:  N K T r0 c0 b
//          then N lines of N chars ('0'/'1') = target.
// -----------------------------------------------------------------------------

static inline int MK(initializer_list<int> cs){ int m=0; for(int c:cs) m|=(1<<c); return m; }

int main(int argc, char* argv[]){
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    double f = (testId - 1) / 9.0;

    // ---- rule library (Bmask, Smask over neighbor-count 0..8) ----
    // index: 0 identity,1 Conway(B3/S23),2 fill(B3/S012345678),3 Seeds(B2/S),
    //        4 Replicator(B1357/S1357),5 gnarl(B1/S1),6 34-Life(B34/S34),
    //        7 HighLife(B36/S23),8 Diamoeba(B35678/S5678),9 DayNight(B3678/S34678)
    vector<int> Bm = { MK({}), MK({3}), MK({3}), MK({2}), MK({1,3,5,7}), MK({1}),
                       MK({3,4}), MK({3,6}), MK({3,5,6,7,8}), MK({3,6,7,8}) };
    vector<int> Sm = { MK({0,1,2,3,4,5,6,7,8}), MK({2,3}), MK({0,1,2,3,4,5,6,7,8}),
                       MK({}), MK({1,3,5,7}), MK({1}), MK({3,4}), MK({2,3}),
                       MK({5,6,7,8}), MK({3,4,6,7,8}) };

    // planting rule per testId (1..10); traps use exotic dynamics
    int plantTbl[10] = { 2, 1, 7, 4, 3, 6, 5, 9, 8, 4 };
    int rIdx = plantTbl[(testId - 1) % 10];

    int N  = 12 + (int)llround(f * 28.0);          // 12..40
    int b  = 5  + (int)llround(f * 7.0);           // 5..12
    int K  = 6  + (int)llround(f * 24.0);          // 6..30
    int T  = 3  + (int)llround(f * 5.0);           // 3..8
    if (b > N - 2) b = N - 2;
    if (K > (b * b) * 2 / 5) K = (b * b) * 2 / 5;  // keep seed sparse vs box
    if (K < 3) K = 3;
    int r0 = (N - b) / 2, c0 = (N - b) / 2;

    auto evolve = [&](const vector<pair<int,int>>& seed, int Bmask, int Smask,
                      int steps, vector<uint8_t>& out){
        vector<uint8_t> g(N * N, 0), h(N * N, 0);
        for (auto& p : seed) g[p.first * N + p.second] = 1;
        for (int s = 0; s < steps; s++){
            for (int r = 0; r < N; r++) for (int c = 0; c < N; c++){
                int cnt = 0;
                for (int dr = -1; dr <= 1; dr++) for (int dc = -1; dc <= 1; dc++){
                    if (!dr && !dc) continue;
                    int nr = r + dr, nc = c + dc;
                    if (nr < 0 || nr >= N || nc < 0 || nc >= N) continue;
                    cnt += g[nr * N + nc];
                }
                int self = g[r * N + c];
                int nxt = self ? ((Smask >> cnt) & 1) : ((Bmask >> cnt) & 1);
                h[r * N + c] = (uint8_t)nxt;
            }
            g.swap(h);
        }
        out.swap(g);
    };

    auto stats = [&](const vector<uint8_t>& g, int& total, int& box, int& outside){
        total = box = outside = 0;
        for (int r = 0; r < N; r++) for (int c = 0; c < N; c++) if (g[r * N + c]){
            total++;
            bool in = (r >= r0 && r < r0 + b && c >= c0 && c < c0 + b);
            if (in) box++; else outside++;
        }
    };

    vector<uint8_t> best; int bestScore = -1;
    auto tryPlant = [&](int Bmask, int Smask, int attempts, double outFrac,
                        double dLo, double dHi){
        for (int a = 0; a < attempts; a++){
            int m = rnd.next(max(2, K / 2), K);       // sparse seed, <=K
            vector<pair<int,int>> seed;
            set<pair<int,int>> seen;
            int guard = 0;
            while ((int)seed.size() < m && guard++ < 4000){
                int r = rnd.next(r0, r0 + b - 1);
                int c = rnd.next(c0, c0 + b - 1);
                if (seen.insert({r, c}).second) seed.push_back({r, c});
            }
            vector<uint8_t> g;
            evolve(seed, Bmask, Smask, T, g);
            int tot, bx, out;
            stats(g, tot, bx, out);
            double dens = (double)tot / (N * N);
            if (dens < dLo || dens > dHi) continue;
            if (bx < K) continue;
            if (tot > 0 && (double)out / tot < outFrac) continue;
            // prefer richer patterns (more total, satisfying constraints)
            int sc = tot;
            if (sc > bestScore){ bestScore = sc; best = g; }
            if (a > attempts / 3 && bestScore >= 0) break; // good enough, deterministic
        }
    };

    tryPlant(Bm[rIdx], Sm[rIdx], 900, 0.35, 0.12, 0.55);
    if (bestScore < 0)  // fallback: fill rule always keeps box full & spreads out
        tryPlant(Bm[2], Sm[2], 900, 0.20, 0.12, 0.70);
    if (bestScore < 0){ // last resort: a plain grown blob under Conway-fill
        vector<pair<int,int>> seed;
        for (int r = r0; r < r0 + b && (int)seed.size() < K; r++)
            for (int c = c0; c < c0 + b && (int)seed.size() < K; c++)
                seed.push_back({r, c});
        vector<uint8_t> g; evolve(seed, Bm[2], Sm[2], T, g); best = g;
    }

    printf("%d %d %d %d %d %d\n", N, K, T, r0, c0, b);
    for (int r = 0; r < N; r++){
        string line(N, '0');
        for (int c = 0; c < N; c++) if (best[r * N + c]) line[c] = '1';
        printf("%s\n", line.c_str());
    }
    return 0;
}
