// TIER: trivial
#include <bits/stdc++.h>
using namespace std;

// Mirrors the checker's own internal baseline exactly: the single cheapest
// feasible radiator -- ONE block attached to the wall's center column.
int main() {
    int H, W, M, Thot, Tamb, Kcond, Kloss, BW, BC;
    cin >> H >> W >> M >> Thot >> Tamb >> Kcond >> Kloss >> BW >> BC;
    int cb = BC + BW / 2;
    cout << 1 << "\n";
    cout << 1 << " " << cb << "\n";
    return 0;
}
