// TIER: invalid
#include <bits/stdc++.h>
using namespace std;
int main() {
    int H, W;
    long long lnum, lden;
    cin >> H >> W >> lnum >> lden;
    long long v;
    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++) cin >> v;
    // Deliberately infeasible: three collinear equally-spaced cells share a repeated
    // difference vector (0, W-1) between consecutive pairs -- the checker must reject this.
    int c2 = (W >= 3) ? 2 : (W - 1);
    cout << 3 << "\n" << 0 << " " << 0 << "\n" << 0 << " " << (W > 1 ? 1 : 0) << "\n" << 0 << " " << c2 << "\n";
    return 0;
}
