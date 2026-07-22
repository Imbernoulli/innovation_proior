// TIER: trivial
#include <bits/stdc++.h>
using namespace std;

struct Basin { int r, c, d, node; };
struct Pump { int r, c, cost, reach, node; };

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int H, W, N, P;
    if (!(cin >> H >> W >> N >> P)) return 0;
    vector<string> grid(H);
    for (int i = 0; i < H; i++) cin >> grid[i];
    auto node = [&](int r, int c) { return (r - 1) * W + c; };

    vector<Basin> basins(N + 1);
    for (int i = 1; i <= N; i++) {
        cin >> basins[i].r >> basins[i].c >> basins[i].d;
        basins[i].node = node(basins[i].r, basins[i].c);
    }
    vector<Pump> pumps(P + 1);
    for (int j = 1; j <= P; j++) {
        cin >> pumps[j].r >> pumps[j].c >> pumps[j].cost >> pumps[j].reach;
        pumps[j].node = node(pumps[j].r, pumps[j].c);
    }

    cout << P;
    for (int j = 1; j <= P; j++) cout << ' ' << j;
    cout << '\n';

    for (int i = 1; i <= N; i++) {
        int pick = 1;
        for (int j = 1; j <= P; j++) {
            if (pumps[j].node == basins[i].node && pumps[j].reach == 0) {
                pick = j;
                break;
            }
        }
        cout << pick << " 0 " << basins[i].node << '\n';
    }
    return 0;
}
