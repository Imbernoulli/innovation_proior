// TIER: invalid
// Deliberately infeasible: claims to patrol 1 edge but names an out-of-range
// edge index. The checker's bounded read must reject this with score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m, K; long long B;
    cin >> n >> m >> K >> B;
    // consume the rest of the input (not required, but harmless)
    cout << 1 << "\n" << (m + 12345) << "\n";
    return 0;
}
