#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<long long> x(n), y(n);
    for (int i = 0; i < n; i++) cin >> x[i] >> y[i];

    // best holds twice the maximum triangle area (a non-negative integer).
    // With |coord| <= 1e9, edge differences reach 2e9 and the cross-product
    // products reach ~4e18, so all of this MUST be long long.
    long long best = 0;
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            for (int k = j + 1; k < n; k++) {
                long long abx = x[j] - x[i];
                long long aby = y[j] - y[i];
                long long acx = x[k] - x[i];
                long long acy = y[k] - y[i];
                long long cross = abx * acy - acx * aby; // = 2 * signed area
                long long twiceArea = llabs(cross);
                if (twiceArea > best) best = twiceArea;
            }

    cout << best << "\n";
    return 0;
}
