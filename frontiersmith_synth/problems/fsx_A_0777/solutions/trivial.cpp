// TIER: trivial
// Do-nothing baseline: connect the villages into the plain path along the
// ridge (village i -- village i+1). Always feasible (degree<=2<=d, and its
// wirelength is exactly the ridge span, which is far under budget). This is
// exactly the checker's internal baseline construction B.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, k, d; long long W;
    if (!(cin >> n >> k >> d >> W)) return 0;
    vector<long long> x(n);
    for (auto &v : x) cin >> v;

    cout << (n - 1) << "\n";
    for (int i = 1; i < n; i++) cout << i << ' ' << (i + 1) << "\n";
    return 0;
}
