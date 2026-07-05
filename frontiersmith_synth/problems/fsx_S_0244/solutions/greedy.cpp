// TIER: greedy
// Standalone-benefit greedy: serve request i iff its solo dock round-trip cost
// (dock->shelf->chute->dock) is strictly less than its skip penalty w_i.
// Emit served requests as consecutive depth-1 pick/deliver pairs, ordered by
// nearest shelf from the current position (a simple chain). Always LIFO-feasible
// (stack depth 1 <= H since H>=1).
#include <bits/stdc++.h>
using namespace std;
int main() {
    int P, H; long long x0, y0;
    if (!(cin >> P >> H)) return 0;
    cin >> x0 >> y0;
    vector<long long> px(P), py(P), dx(P), dy(P), w(P);
    for (int i = 0; i < P; i++) cin >> px[i] >> py[i] >> dx[i] >> dy[i] >> w[i];

    auto manh = [](long long ax,long long ay,long long bx,long long by){
        return llabs(ax-bx)+llabs(ay-by);
    };

    vector<int> chosen;
    for (int i = 0; i < P; i++) {
        long long solo = manh(x0,y0,px[i],py[i]) + manh(px[i],py[i],dx[i],dy[i])
                       + manh(dx[i],dy[i],x0,y0);
        if (w[i] > solo) chosen.push_back(i);
    }

    // order chosen by nearest shelf greedily from current position
    vector<char> used(chosen.size(), 0);
    long long cx = x0, cy = y0;
    vector<int> order;
    for (size_t k = 0; k < chosen.size(); k++) {
        int best = -1; long long bd = LLONG_MAX;
        for (size_t j = 0; j < chosen.size(); j++) {
            if (used[j]) continue;
            long long dd = manh(cx,cy,px[chosen[j]],py[chosen[j]]);
            if (dd < bd) { bd = dd; best = (int)j; }
        }
        used[best] = 1; order.push_back(chosen[best]);
        cx = dx[chosen[best]]; cy = dy[chosen[best]]; // end at its chute
    }

    printf("%d\n", (int)order.size()*2);
    for (int idx : order) {
        printf("0 %d\n", idx+1);
        printf("1 %d\n", idx+1);
    }
    return 0;
}
