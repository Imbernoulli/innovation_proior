// TIER: invalid
#include <bits/stdc++.h>
using namespace std;

// Deliberately infeasible: places a block at row H+1, outside the grid.
int main() {
    int H, W, M, Thot, Tamb, Kcond, Kloss, BW, BC;
    cin >> H >> W >> M >> Thot >> Tamb >> Kcond >> Kloss >> BW >> BC;
    int cb = BC + BW / 2;
    cout << 1 << "\n";
    cout << (H + 1) << " " << cb << "\n";
    return 0;
}
