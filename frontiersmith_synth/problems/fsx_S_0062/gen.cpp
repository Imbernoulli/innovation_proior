#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // difficulty / structure ladder
    int G   = 30 + 26 * (testId - 1);       // grid dimension (t1:30 ... t10:264, < 300)
    int nA  = 6  + 60 * (testId - 1);       // number of artifacts (t1:6 ... t10:546)
    int r   = 2  + (testId - 1) / 3;        // coverage radius (t1:2 ... t10:5)
    if (r > 12) r = 12;
    int nClusters = max(2, nA / 12);        // ~12 artifacts per (well-separated) cluster
    int spread = r + 2;                     // cluster a bit wider than r -> a single beacon
                                            // covers only part of it, so the cheapest cover
                                            // needs several beacons and never yields a >10x cut

    struct Site { int row, col; ll cost; int art; };
    vector<Site> sites;

    // cluster centers
    vector<pair<int,int>> ctr(nClusters);
    for (int c = 0; c < nClusters; c++)
        ctr[c] = { rnd.next(0, G - 1), rnd.next(0, G - 1) };

    auto clampv = [&](int v){ return max(0, min(G - 1, v)); };

    // artifacts, distributed round-robin over clusters
    for (int i = 0; i < nA; i++) {
        int c  = i % nClusters;
        int rr = clampv(ctr[c].first  + rnd.next(-spread, spread));
        int cc = clampv(ctr[c].second + rnd.next(-spread, spread));
        sites.push_back({ rr, cc, (ll)rnd.next(20, 60), 1 });
    }

    // one cheap candidate near each cluster center
    for (int c = 0; c < nClusters; c++) {
        int rr = clampv(ctr[c].first  + rnd.next(-1, 1));
        int cc = clampv(ctr[c].second + rnd.next(-1, 1));
        sites.push_back({ rr, cc, (ll)rnd.next(25, 45), 0 });
    }

    // extra candidate sites (some near clusters, some fully random)
    int extra = nA / 2 + 2;
    for (int i = 0; i < extra; i++) {
        if (rnd.next(0, 1)) {
            int c  = rnd.next(0, nClusters - 1);
            int rr = clampv(ctr[c].first  + rnd.next(-spread, spread));
            int cc = clampv(ctr[c].second + rnd.next(-spread, spread));
            sites.push_back({ rr, cc, (ll)rnd.next(30, 70), 0 });
        } else {
            sites.push_back({ rnd.next(0, G - 1), rnd.next(0, G - 1),
                              (ll)rnd.next(40, 80), 0 });
        }
    }

    shuffle(sites.begin(), sites.end());

    int N = (int)sites.size();
    printf("%d %d\n", N, r);
    for (auto& s : sites)
        printf("%d %d %lld %d\n", s.row, s.col, s.cost, s.art);
    return 0;
}
