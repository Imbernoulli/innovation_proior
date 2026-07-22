// TIER: invalid
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// Deliberately infeasible: outputs a single occupied cell at the very top row
// (x=1, y=H) with nothing beneath it anywhere -- it has no support chain to
// bedrock at all (H is always >= 4 in this problem's tests, so row H != row 1).
// The checker must reject this and score 0.

int main() {
    ll W, H, S, M, Fx, Fy;
    cin >> W >> H >> S >> M >> Fx >> Fy;
    printf("1\n1 %lld\n", H);
    return 0;
}
