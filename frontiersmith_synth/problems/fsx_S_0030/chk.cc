#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int H, R;
vector<ll> C;                 // 1..R
vector<vector<int>> V, D;     // [1..H][1..R]

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    H = inf.readInt();
    R = inf.readInt();
    C.assign(R + 1, 0);
    for (int j = 1; j <= R; j++) C[j] = inf.readInt();
    V.assign(H + 1, vector<int>(R + 1, 0));
    D.assign(H + 1, vector<int>(R + 1, 0));
    for (int i = 1; i <= H; i++)
        for (int j = 1; j <= R; j++) {
            V[i][j] = inf.readInt();
            D[i][j] = inf.readInt();
        }

    // ---- internal baseline B: deterministic first-fit-by-index (value-blind) ----
    vector<ll> rem = C;
    ll B = 0;
    for (int i = 1; i <= H; i++) {
        for (int j = 1; j <= R; j++) {
            if ((ll)D[i][j] <= rem[j]) {
                rem[j] -= D[i][j];
                B += V[i][j];
                break;
            }
        }
    }
    if (B <= 0) quitf(_fail, "bad instance: baseline B=%lld <= 0", B);

    // ---- read & validate the participant's assignment ----
    vector<ll> used(R + 1, 0);
    ll F = 0;
    for (int i = 1; i <= H; i++) {
        int a = ouf.readInt(0, R, "assignment");
        if (a != 0) {
            used[a] += D[i][a];
            F += V[i][a];
        }
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    for (int j = 1; j <= R; j++)
        if (used[j] > C[j])
            quitf(_wa, "route %d over capacity: used %lld > C=%lld", j, used[j], C[j]);

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
