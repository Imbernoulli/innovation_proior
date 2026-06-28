#include <bits/stdc++.h>
using namespace std;

// Count lattice points strictly interior to a simple polygon.
// Pick's theorem: A = I + B/2 - 1  =>  I = A - B/2 + 1.
// With doubled area  S = 2A  (exact integer via shoelace) and boundary count B,
// I = (S - B) / 2 + 1.  S can reach ~2e23, so accumulate it in __int128.
int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;

    vector<long long> x(n), y(n);
    for (int i = 0; i < n; i++) cin >> x[i] >> y[i];

    __int128 twiceArea = 0; // signed doubled area via shoelace
    long long boundary = 0; // total boundary lattice points

    for (int i = 0; i < n; i++) {
        int j = (i + 1) % n;
        // shoelace cross term  x_i*y_j - x_j*y_i  (fits in __int128)
        twiceArea += (__int128)x[i] * y[j] - (__int128)x[j] * y[i];
        // boundary lattice points on edge i->j = gcd(|dx|, |dy|)
        long long dx = llabs(x[j] - x[i]);
        long long dy = llabs(y[j] - y[i]);
        boundary += std::__gcd(dx, dy);
    }

    if (twiceArea < 0) twiceArea = -twiceArea; // S = |2A|

    // interior = (S - B) / 2 + 1   (S - B is always even for integer polygons)
    __int128 interior = (twiceArea - (__int128)boundary) / 2 + 1;

    // print the __int128 result
    if (interior == 0) {
        cout << 0 << "\n";
        return 0;
    }
    bool neg = interior < 0;
    if (neg) interior = -interior;
    string s;
    while (interior > 0) {
        int d = (int)(interior % 10);
        s.push_back((char)('0' + d));
        interior /= 10;
    }
    if (neg) s.push_back('-');
    reverse(s.begin(), s.end());
    cout << s << "\n";
    return 0;
}
