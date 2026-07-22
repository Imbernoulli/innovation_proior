// TIER: trivial
#include <bits/stdc++.h>
using namespace std;
int main() {
    int H, W;
    long long lnum, lden;
    cin >> H >> W >> lnum >> lden;
    int br = 0, bc = 0;
    long long best = -1;
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            long long v;
            cin >> v;
            if (v > best) { best = v; br = i; bc = j; }
        }
    }
    cout << 1 << "\n" << br << " " << bc << "\n";
    return 0;
}
