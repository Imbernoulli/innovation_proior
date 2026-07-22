// TIER: invalid
// Deliberately infeasible: connect village 1 to every other village, blowing
// the per-village degree cap d (n-1 >> d for every test in this family).
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, k, d; long long W;
    if (!(cin >> n >> k >> d >> W)) return 0;
    vector<long long> x(n);
    for (auto &v : x) cin >> v;

    cout << (n - 1) << "\n";
    for (int i = 2; i <= n; i++) cout << 1 << ' ' << i << "\n";
    return 0;
}
