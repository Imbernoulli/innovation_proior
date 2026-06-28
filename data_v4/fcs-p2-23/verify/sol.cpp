#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;             // n = 0 (or empty input) -> first player gets 0
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    if (n == 0) { cout << 0 << "\n"; return 0; }

    // diff[i][j] = (current player's total) - (opponent's total) under optimal play
    //             on the subarray a[i..j]. The mover takes a[i] or a[j], then the
    //             opponent becomes the mover on the smaller interval, so the sign flips:
    //   diff[i][j] = max( a[i] - diff[i+1][j], a[j] - diff[i][j-1] ).
    // Base: diff[i][i] = a[i] (one stone, the mover takes it).
    vector<vector<long long>> diff(n, vector<long long>(n, 0));
    long long total = 0;
    for (int i = 0; i < n; i++) { diff[i][i] = a[i]; total += a[i]; }

    for (int len = 2; len <= n; len++) {
        for (int i = 0; i + len - 1 < n; i++) {
            int j = i + len - 1;
            long long takeLeft  = a[i] - diff[i + 1][j];
            long long takeRight = a[j] - diff[i][j - 1];
            diff[i][j] = max(takeLeft, takeRight);
        }
    }

    // first + second = total ; first - second = diff[0][n-1]
    // => first = (total + diff[0][n-1]) / 2 . The parity always divides evenly.
    long long first = (total + diff[0][n - 1]) / 2;
    cout << first << "\n";
    return 0;
}
