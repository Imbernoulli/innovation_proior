#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long L;
    if (!(cin >> n >> L)) return 0;
    vector<long long> w(n);
    for (auto &x : w) cin >> x;

    sort(w.begin(), w.end());

    // Two pointers from both ends: pair the lightest free rider with the
    // heaviest free rider whenever they fit together (sum <= L). If they do
    // not fit, the heaviest cannot ride with anyone (everyone else is >= the
    // lightest), so drop it and keep trying with the next-heaviest.
    int lo = 0, hi = n - 1;
    long long pairs = 0;
    while (lo < hi) {
        if (w[lo] + w[hi] <= L) {
            pairs++;
            lo++;
            hi--;
        } else {
            hi--;
        }
    }

    cout << pairs << "\n";
    return 0;
}
