#include "testlib.h"
#include <vector>
#include <set>
#include <cstdlib>
#include <algorithm>
using namespace std;

// Generator for "Compiling a Task Graph on a Tiny Scratchpad".
//
// Each cluster is a chain of csize tasks. The chain ROOT touches one shared "hot"
// page (round-robin over the H hot pages) plus the cluster's private signature page;
// the body tasks touch the signature page plus scattered background pages. Roots have
// no predecessor, so they are all schedulable up front.
//
// Tasks are emitted in an INTERLEAVED order (round-robin across clusters), so the
// identity order 1..N reuses each signature page only every K accesses -> poor cold
// locality (a beatable reference baseline).
//
// The trap: clustering each chain together restores signature locality but stretches a
// hot page's reuse distance across whole clusters, so a locality-greedy schedule reloads
// the hot pages once per cluster. The insight is to SHAPE the reuse-distance histogram:
// pack the (root) hot touches together so the few hot pages stay resident, then run the
// cluster bodies -- keeping both signature locality AND the hot pages hot.
int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int T = atoi(argv[1]);
    int K, csize, H, C, Bpool, Rcross;
    double pExtra;
    bool skew = false;
    if (T == 1)      { K=5;   csize=4;  H=5;  C=4;  Bpool=4;   pExtra=0.25; Rcross=1;   }
    else if (T == 2) { K=24;  csize=5;  H=9;  C=7;  Bpool=16;  pExtra=0.25; Rcross=8;   }
    else if (T == 3) { K=40;  csize=6;  H=10; C=8;  Bpool=24;  pExtra=0.30; Rcross=12;  }
    else if (T == 4) { K=70;  csize=6;  H=13; C=9;  Bpool=40;  pExtra=0.25; Rcross=24;  }
    else if (T == 5) { K=60;  csize=7;  H=12; C=9;  Bpool=34;  pExtra=0.30; Rcross=420; } // dense DAG (less freedom)
    else if (T == 6) { K=110; csize=7;  H=14; C=10; Bpool=60;  pExtra=0.30; Rcross=30;  } // sparse, high freedom
    else if (T == 7) { K=160; csize=8;  H=16; C=10; Bpool=90;  pExtra=0.25; Rcross=45;  }
    else if (T == 8) { K=230; csize=8;  H=20; C=11; Bpool=130; pExtra=0.30; Rcross=60;  }
    else if (T == 9) { K=300; csize=8;  H=22; C=11; Bpool=160; pExtra=0.30; Rcross=80; skew=true; } // skewed hotness
    else             { K=440; csize=9;  H=28; C=12; Bpool=240; pExtra=0.25; Rcross=120; } // largest

    // tuning override (unset in the harness -> the table above is authoritative/deterministic)
    if (const char* gp = getenv("GP")) {
        double pe; int rc;
        if (sscanf(gp, "%d %d %d %d %d %lf %d", &K, &csize, &H, &C, &Bpool, &pe, &rc) == 7) {
            pExtra = pe; Rcross = rc;
        }
    }

    int N = K * csize;
    int hotBase = 0, sigBase = H, bgBase = H + K;
    int M = H + K + Bpool;

    vector<set<int>> tp(N);
    for (int g = 0; g < N; g++) {
        int c = g % K, slot = g / K;
        tp[g].insert(sigBase + c);                 // cluster signature page
        if (slot == 0) {                           // chain root: touch one hot page
            int h;
            if (skew) { double r = rnd.next(0.0, 1.0); h = (int)(H * r * r); if (h >= H) h = H - 1; }
            else h = c % H;
            tp[g].insert(hotBase + h);
        } else if (Bpool > 0 && rnd.next(0.0, 1.0) < pExtra) {
            tp[g].insert(bgBase + rnd.next(0, Bpool - 1));   // body: scattered background
        }
    }

    vector<pair<int,int>> edges;
    for (int c = 0; c < K; c++)                     // within-cluster chain (u<v under interleaving)
        for (int j = 0; j + 1 < csize; j++)
            edges.push_back({ (c + j*K) + 1, (c + (j+1)*K) + 1 });
    set<pair<int,int>> es(edges.begin(), edges.end());
    int placed = 0, tries = 0;
    while (placed < Rcross && tries < Rcross * 25) {
        tries++;
        int a = rnd.next(0, N - 2);
        int b = rnd.next(a + 1, N - 1);
        if (es.insert({ a + 1, b + 1 }).second) { edges.push_back({ a + 1, b + 1 }); placed++; }
    }

    int E = (int)edges.size();
    printf("%d %d %d %d\n", N, M, C, E);
    for (int g = 0; g < N; g++) {
        printf("%d", (int)tp[g].size());
        for (int p : tp[g]) printf(" %d", p);
        printf("\n");
    }
    for (auto& e : edges) printf("%d %d\n", e.first, e.second);
    return 0;
}
