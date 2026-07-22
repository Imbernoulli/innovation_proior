// TIER: invalid
// Deliberately infeasible: draws from well 1 before it has ever received
// anything (avail=0 < vol), which the checker must reject with score 0.
#include <bits/stdc++.h>
using namespace std;

int main(){
    long long W, M, Vmax, Vcap, stepCost, stockAccessCost, maxOps;
    cin >> W >> M >> Vmax >> Vcap >> stepCost >> stockAccessCost >> maxOps;
    for (long long i = 0; i < W; i++){ long long c, v; cin >> c >> v; }
    cout << 1 << "\n";
    cout << 1 << " " << 2 << " " << 1 << "\n"; // src=well1 (empty), dst=well2, vol=1
    return 0;
}
