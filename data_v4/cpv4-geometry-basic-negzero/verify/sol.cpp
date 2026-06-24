#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) { return 0; }              // no header at all -> nothing to do
    vector<long long> x(n), y(n);
    for (int i = 0; i < n; i++) cin >> x[i] >> y[i];

    if (n < 2) {                                 // no ordered pair i<j exists
        cout << "NONE" << "\n";
        return 0;
    }

    // best signed area*2 over ordered pairs i<j: cross(P[i],P[j]) = x_i*y_j - x_j*y_i.
    // Must start from a REAL pair, not 0, because every cross product can be negative.
    long long best = LLONG_MIN;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            long long cr = x[i] * y[j] - x[j] * y[i];
            if (cr > best) best = cr;
        }
    }

    cout << best << "\n";
    return 0;
}
