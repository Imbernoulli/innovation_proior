// TIER: trivial
// Empty protocol -- reproduces the checker's own do-nothing baseline exactly.
#include <bits/stdc++.h>
using namespace std;

int main(){
    long long W, M, Vmax, Vcap, stepCost, stockAccessCost, maxOps;
    if (!(cin >> W >> M >> Vmax >> Vcap >> stepCost >> stockAccessCost >> maxOps)) return 0;
    for (long long i = 0; i < W; i++){
        long long c, v; cin >> c >> v;
    }
    cout << 0 << "\n";
    return 0;
}
