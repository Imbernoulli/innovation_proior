// TIER: trivial
// Even split of the ink budget across all T checkpoints, a single stamp
// (k=1) per checkpoint. Deliberately mirrors the checker's own internal
// baseline construction.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int T; ll B, S1, S2;
    cin >> T >> B >> S1 >> S2;
    ll m = B / T;
    for (int i = 1; i <= T; i++) cout << m << " " << 1 << "\n";
    return 0;
}
