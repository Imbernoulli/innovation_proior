#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;             // no n -> nothing to decide
    vector<int> a(n);
    long long total = 0;
    for (auto &x : a) { cin >> x; total += x; }

    // If the total is odd it can never split into two equal halves.
    if (total % 2 != 0) { cout << "NO" << "\n"; return 0; }

    long long half = total / 2;            // target sum for one subset

    // reach[s] = can some subset of the processed elements sum to exactly s.
    // Boolean subset-sum DP over [0, half]; reach[0] true (empty subset).
    vector<char> reach(half + 1, 0);
    reach[0] = 1;
    for (int i = 0; i < n; i++) {
        int v = a[i];
        // Iterate s downward so each element is used at most once.
        for (long long s = half; s >= v; s--) {
            if (reach[s - v]) reach[s] = 1;
        }
    }

    cout << (reach[half] ? "YES" : "NO") << "\n";
    return 0;
}
