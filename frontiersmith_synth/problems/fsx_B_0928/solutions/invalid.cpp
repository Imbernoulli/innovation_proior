// TIER: invalid
// Deliberately infeasible: claims ship 1 with entry tick == exit tick and an
// anchor point far outside any basin, which must be rejected regardless of
// the input, so this must score 0 on every test.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int W, H, K;
    cin >> W >> H >> K;
    for (int i = 0; i < K; i++) { long long x; cin >> x; }
    int N;
    cin >> N;
    for (int i = 0; i < N; i++) { long long a, b, c, d; cin >> a >> b >> c >> d; }

    cout << 1 << "\n";
    cout << 1 << " " << -999999 << " " << -999999 << " " << 1 << " " << 1 << "\n";
    return 0;
}
