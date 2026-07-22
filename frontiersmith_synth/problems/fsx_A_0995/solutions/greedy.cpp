// TIER: greedy
// The obvious single-pass recipe: fix every region's color to the FIRST
// entry of its tolerance class (never uses the recoloring freedom), then,
// PER LAYER INDEPENDENTLY, build a nearest-neighbour tour over that fixed
// color assignment (start at the layer's first listed region, repeatedly
// hop to whichever unvisited region is cheapest to reach from the current
// nozzle color). This is the textbook "greedy TSP" move and looks locally
// optimal layer by layer -- but it never looks at what color the previous
// layer ended on, nor what color the next layer will start with, so the
// carryover transition between layers is entirely invisible to it, and a
// region whose tolerance class happens to list the wrong-tier color first
// is never recolored away.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int L, C;
    scanf("%d %d", &L, &C);
    vector<vector<ll>> P(C, vector<ll>(C));
    for (int i = 0; i < C; i++) for (int j = 0; j < C; j++) scanf("%lld", &P[i][j]);

    for (int l = 0; l < L; l++) {
        int R; scanf("%d", &R);
        vector<vector<int>> cls(R);
        for (int r = 0; r < R; r++) {
            int k; scanf("%d", &k);
            cls[r].resize(k);
            for (int i = 0; i < k; i++) scanf("%d", &cls[r][i]);
        }
        vector<int> color(R);
        for (int r = 0; r < R; r++) color[r] = cls[r][0];

        vector<char> used(R, 0);
        vector<int> order;
        order.reserve(R);
        int cur = 0;
        used[0] = 1;
        order.push_back(0);
        for (int step = 1; step < R; step++) {
            int best = -1;
            ll bd = -1;
            for (int r = 0; r < R; r++) {
                if (used[r]) continue;
                ll d = P[color[cur]][color[r]];
                if (best == -1 || d < bd || (d == bd && r < best)) { bd = d; best = r; }
            }
            used[best] = 1;
            order.push_back(best);
            cur = best;
        }

        for (int i = 0; i < R; i++) printf("%d%c", order[i], i + 1 == R ? '\n' : ' ');
        for (int i = 0; i < R; i++) printf("%d%c", color[order[i]], i + 1 == R ? '\n' : ' ');
    }
    return 0;
}
