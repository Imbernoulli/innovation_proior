#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Thermal-aware fixed-die floorplanning generator.
// testId is a difficulty/structure ladder:
//   t=1 : tiny (example scale)
//   t growing -> larger N, denser/sparser netlists, more hotspots, tighter/looser die.
// Structure: blocks are partitioned into GROUPS via a random permutation, so group members are
// SCATTERED in index order (=> the index-order shelf baseline scatters every cluster => large HPWL,
// leaving real headroom). Most nets connect blocks WITHIN a group (planted clusters); a fraction
// are cross-group noise nets (traps). A handful of high-power hotspots are sprinkled across groups,
// with some groups holding two hotspots (thermal tension inside a cluster you still want to pack).

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]);

    // ---- size ladder ----
    int Ntab[11] = {0, 4, 20, 40, 60, 90, 120, 160, 200, 250, 300};
    int N = (t >= 1 && t <= 10) ? Ntab[t] : 60;

    // block dims + base power
    vector<int> w(N), h(N), p(N);
    int maxw = 1, maxh = 1;
    long long area = 0;
    for (int i = 0; i < N; i++) {
        w[i] = rnd.next(4, 20);
        h[i] = rnd.next(4, 20);
        p[i] = rnd.next(1, 4);
        maxw = max(maxw, w[i]);
        maxh = max(maxh, h[i]);
        area += (long long)w[i] * h[i];
    }

    // squarish die with ~3x area slack (room to rearrange for wirelength)
    int side = (int)ceil(sqrt(3.0 * (double)area));
    int W = max(side, maxw + 2);

    // simulate index-order shelf to guarantee vertical fit, then give a little slack
    auto shelfHeight = [&](int Wd) {
        long long x = 0, y = 0; int rowh = 0;
        for (int i = 0; i < N; i++) {
            if (x + w[i] > Wd) { y += rowh; x = 0; rowh = 0; }
            x += w[i];
            rowh = max(rowh, h[i]);
        }
        return y + rowh;
    };
    long long Hneed = shelfHeight(W);
    int H = (int)max((long long)side, Hneed) + rnd.next(maxh, 3 * maxh);
    if (H < (int)Hneed) H = (int)Hneed + maxh; // safety

    // ---- hotspots ----
    int nHot = min(N / 2, max(3, N / 18));
    // choose hotspot blocks
    vector<int> perm(N);
    for (int i = 0; i < N; i++) perm[i] = i;
    shuffle(perm.begin(), perm.end());
    for (int i = 0; i < nHot; i++) p[perm[i]] = rnd.next(18, 40);

    // ---- groups (scattered in index order) ----
    int gs;                    // group size varies with t to make chain-like vs blob-like clusters
    if (t % 3 == 1)      gs = max(2, min(N, 3));      // tiny groups (chain-ish, trap for BFS)
    else if (t % 3 == 2) gs = max(2, min(N, 14));     // fat clusters
    else                 gs = max(2, min(N, 8));      // medium
    int G = (N + gs - 1) / gs;
    // group id per block, using a fresh scatter permutation
    vector<int> gperm(N);
    for (int i = 0; i < N; i++) gperm[i] = i;
    shuffle(gperm.begin(), gperm.end());
    vector<vector<int>> groups(G);
    for (int idx = 0; idx < N; idx++) {
        int blk = gperm[idx];
        groups[idx / gs].push_back(blk);
    }

    // ---- nets ----
    // density varies with t
    double dens = (t % 2 == 0) ? 1.6 : 0.9;
    int M = max(1, min(800, (int)(dens * N)));
    // fraction of cross-group noise nets
    double noiseFrac = 0.12;

    // build nets first into a buffer (need count M exactly)
    vector<vector<int>> nets;
    nets.reserve(M);
    for (int e = 0; e < M; e++) {
        bool noise = (rnd.next(0, 999) < (int)(noiseFrac * 1000)) && (N >= 2);
        vector<int> members;
        if (noise) {
            // random 2..3 blocks anywhere
            int k = rnd.next(2, min(3, N));
            set<int> s;
            while ((int)s.size() < k) s.insert(rnd.next(0, N - 1));
            members.assign(s.begin(), s.end());
        } else {
            // pick a group with >=2 members
            int g = rnd.next(0, G - 1);
            int tries = 0;
            while ((int)groups[g].size() < 2 && tries < 8) { g = rnd.next(0, G - 1); tries++; }
            if ((int)groups[g].size() < 2) {
                // fallback: random pair
                int a = rnd.next(0, N - 1), b = rnd.next(0, N - 1);
                while (b == a) b = rnd.next(0, N - 1);
                members = {a, b};
            } else {
                int gsz = groups[g].size();
                int k = rnd.next(2, min(8, gsz));
                // sample k distinct members of the group
                vector<int> pool = groups[g];
                shuffle(pool.begin(), pool.end());
                members.assign(pool.begin(), pool.begin() + k);
            }
        }
        sort(members.begin(), members.end());
        members.erase(unique(members.begin(), members.end()), members.end());
        if ((int)members.size() < 2) {
            int a = rnd.next(0, N - 1), b = rnd.next(0, N - 1);
            while (b == a) b = rnd.next(0, N - 1);
            members = {min(a, b), max(a, b)};
        }
        nets.push_back(members);
    }

    // thermal radius: within ~side (doubled units) hotspots interact
    int R = max(4, side + rnd.next(-side / 4, side / 4));

    // ---- emit ----
    printf("%d %d %d %d %d\n", W, H, N, (int)nets.size(), R);
    for (int i = 0; i < N; i++) printf("%d %d %d\n", w[i], h[i], p[i]);
    for (auto& mem : nets) {
        printf("%d", (int)mem.size());
        for (int b : mem) printf(" %d", b + 1);
        printf("\n");
    }
    return 0;
}
