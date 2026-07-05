// TIER: invalid
// Deliberately infeasible: emits a pair with loop length 0 (j = i+1), which violates the
// minimum-loop rule. The checker must reject this and score 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int L;
    if (!(cin >> L)) return 0;
    string S; cin >> S;
    int P; cin >> P;
    for (int k = 0; k < P; k++) { int p, v; cin >> p >> v; }
    // one illegal pair: (1,2) has loop length 0 < 3
    cout << 1 << "\n" << 1 << " " << 2 << "\n";
    return 0;
}
