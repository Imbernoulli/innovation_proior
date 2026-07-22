// TIER: invalid
// Deliberately infeasible: gives every checkpoint the FULL budget B (so
// sum(m_i) = T*B, far over the shared budget) -- must score 0.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int T; ll B, S1, S2;
    cin >> T >> B >> S1 >> S2;
    for (int i = 1; i <= T; i++) cout << B << " " << 3 << "\n";
    return 0;
}
